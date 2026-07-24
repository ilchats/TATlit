import os, json, time, math, argparse
from types import SimpleNamespace
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
C=SimpleNamespace(vocab=32000,n_layer=12,n_embd=768,n_head=12,n_kv=4,head_dim=64,intermediate=2048,block=1024,rope_base=10000,eps=1e-5)
class RMSNorm(nn.Module):
    def __init__(s,d,eps): super().__init__(); s.w=nn.Parameter(torch.ones(d)); s.eps=eps
    def forward(s,x):
        dt=x.dtype; x=x.float(); x=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+s.eps); return (x*s.w.float()).to(dt)
def build_rope(hd,T,base,dev):
    inv=1.0/(base**(torch.arange(0,hd,2,device=dev).float()/hd)); t=torch.arange(T,device=dev).float(); f=torch.outer(t,inv); return torch.cos(f),torch.sin(f)
def rope(x,cos,sin):
    D=x.shape[-1]; T=x.shape[2]; x1=x[...,:D//2]; x2=x[...,D//2:]; c=cos[:T].view(1,1,T,D//2); s=sin[:T].view(1,1,T,D//2); return torch.cat([x1*c-x2*s,x2*c+x1*s],-1)
class Attn(nn.Module):
    def __init__(s,c):
        super().__init__(); s.nh=c.n_head; s.nkv=c.n_kv; s.hd=c.head_dim
        s.wq=nn.Linear(c.n_embd,s.nh*s.hd,bias=False); s.wk=nn.Linear(c.n_embd,s.nkv*s.hd,bias=False); s.wv=nn.Linear(c.n_embd,s.nkv*s.hd,bias=False); s.wo=nn.Linear(s.nh*s.hd,c.n_embd,bias=False)
    def forward(s,x,cos,sin):
        B,T,_=x.shape
        q=s.wq(x).view(B,T,s.nh,s.hd).transpose(1,2); k=s.wk(x).view(B,T,s.nkv,s.hd).transpose(1,2); v=s.wv(x).view(B,T,s.nkv,s.hd).transpose(1,2)
        q=rope(q,cos,sin); k=rope(k,cos,sin); rep=s.nh//s.nkv; k=k.repeat_interleave(rep,1); v=v.repeat_interleave(rep,1)
        o=F.scaled_dot_product_attention(q,k,v,is_causal=True); return s.wo(o.transpose(1,2).reshape(B,T,s.nh*s.hd))
class MLP(nn.Module):
    def __init__(s,c): super().__init__(); s.w1=nn.Linear(c.n_embd,c.intermediate,bias=False); s.w3=nn.Linear(c.n_embd,c.intermediate,bias=False); s.w2=nn.Linear(c.intermediate,c.n_embd,bias=False)
    def forward(s,x): return s.w2(F.silu(s.w1(x))*s.w3(x))
class Block(nn.Module):
    def __init__(s,c): super().__init__(); s.n1=RMSNorm(c.n_embd,c.eps); s.attn=Attn(c); s.n2=RMSNorm(c.n_embd,c.eps); s.mlp=MLP(c); s.drop=nn.Dropout(getattr(c,'dropout',0.0))
    def forward(s,x,cos,sin): x=x+s.drop(s.attn(s.n1(x),cos,sin)); return x+s.drop(s.mlp(s.n2(x)))
class GPT(nn.Module):
    def __init__(s,c):
        super().__init__(); s.c=c; s.wte=nn.Embedding(c.vocab,c.n_embd); s.blocks=nn.ModuleList([Block(c) for _ in range(c.n_layer)]); s.norm_f=RMSNorm(c.n_embd,c.eps); s.lm_head=nn.Linear(c.n_embd,c.vocab,bias=False); s.lm_head.weight=s.wte.weight
        s.apply(s._init)
        for n,p in s.named_parameters():
            if n.endswith('wo.weight') or n.endswith('w2.weight'): nn.init.normal_(p,0,0.02/math.sqrt(2*c.n_layer))
        s.cos=None; s.sin=None
    def _init(s,m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,0.02)
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,0.02)
    def forward(s,idx,tgt=None):
        if s.cos is None or s.cos.device!=idx.device: s.cos,s.sin=build_rope(s.c.head_dim,s.c.block,s.c.rope_base,idx.device)
        x=s.wte(idx)
        for b in s.blocks: x=b(x,s.cos,s.sin)
        logits=s.lm_head(s.norm_f(x)); loss=None
        if tgt is not None: loss=F.cross_entropy(logits.float().view(-1,logits.size(-1)),tgt.view(-1))
        return logits,loss
class Loader:
    def __init__(s,bins,w,block,dev):
        s.d={L:np.memmap(p,dtype=np.uint16,mode='r') for L,p in bins.items() if os.path.exists(p)}
        s.langs=[L for L in w if L in s.d]; s.p=np.array([w[L] for L in s.langs],float); s.p/=s.p.sum(); s.block=block; s.dev=dev; s.tn={L:int(len(s.d[L])*0.998) for L in s.langs}; s.val_langs=[L for L in s.langs if (len(s.d[L])-s.tn[L])>s.block+1]
    def _win(s,L,lo,hi):
        i=np.random.randint(lo,max(lo+1,hi)); wd=np.asarray(s.d[L][i:i+s.block+1],np.int64); return wd[:-1],wd[1:]
    def batch(s,bs,split='train'):
        ch=np.random.choice(len(s.langs),bs,p=s.p); xs=np.empty((bs,s.block),np.int64); ys=np.empty((bs,s.block),np.int64)
        for j,ci in enumerate(ch): L=s.langs[ci]; lo,hi=((0,s.tn[L]-s.block-1) if split=='train' else (s.tn[L],len(s.d[L])-s.block-1)); xs[j],ys[j]=s._win(L,lo,hi)
        return torch.from_numpy(xs).to(s.dev,non_blocking=True),torch.from_numpy(ys).to(s.dev,non_blocking=True)
    def batch_lang(s,L,bs):
        xs=np.empty((bs,s.block),np.int64); ys=np.empty((bs,s.block),np.int64); lo,hi=s.tn[L],len(s.d[L])-s.block-1
        for j in range(bs): xs[j],ys[j]=s._win(L,lo,hi)
        return torch.from_numpy(xs).to(s.dev),torch.from_numpy(ys).to(s.dev)
def lr_at(step,total,warm,peak,ff=0.1):
    if step<warm: return peak*step/max(1,warm)
    pr=(step-warm)/max(1,total-warm); return peak*(ff+(1-ff)*0.5*(1+math.cos(math.pi*min(1,pr))))
def main():
    ap=argparse.ArgumentParser()
    for k,v in [('max-steps',19000),('micro',4),('accum',32),('block',4096),('warmup',300),('ckpt-every',500),('val-every',500),('snap-every',2500)]: ap.add_argument('--'+k,type=int,default=v)
    ap.add_argument('--peak-lr',type=float,default=4e-4); ap.add_argument('--wandb',action='store_true'); ap.add_argument('--no-compile',action='store_true'); ap.add_argument('--resume',action='store_true'); ap.add_argument('--run-name',default='abl')
    ap.add_argument('--mix',required=True); ap.add_argument('--data-dir',default='/workspace/ablation/bins')
    ap.add_argument('--out-dir',default='/workspace/ablation/ckpt'); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--init-from',default=''); ap.add_argument('--dropout',type=float,default=0.0)
    a=ap.parse_args(); C.block=a.block; C.dropout=a.dropout
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    os.makedirs(a.out_dir,exist_ok=True)
    mix=json.load(open(a.mix))["target_proportions"]
    ld=Loader({L:f"{a.data_dir}/{L}.bin" for L in mix},mix,C.block,'cuda')
    m=GPT(C).cuda(); npar=sum(p.numel() for p in m.parameters()); mc=m if a.no_compile else torch.compile(m)
    opt=torch.optim.AdamW(m.parameters(),lr=a.peak_lr,betas=(0.9,0.95),weight_decay=0.1,fused=True)
    start=0; ckp=f"{a.out_dir}/{a.run_name}_last.pt"
    if a.init_from:
        _c=torch.load(a.init_from,map_location='cuda',weights_only=False)
        m.load_state_dict(_c['model']); print(f'INIT_FROM {a.init_from} dropout={a.dropout}',flush=True)
    if a.resume and os.path.exists(ckp):
        ck=torch.load(ckp,map_location='cuda'); m.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); start=ck['step']+1; print(f"RESUMED @ {start}",flush=True)
    print(f"params {npar/1e6:.1f}M | {a.micro}x{a.accum}x{C.block}={a.micro*a.accum*C.block} tok/step | langs {ld.langs} | start {start}",flush=True)
    if a.wandb:
        import wandb; wandb.init(project="ilTAT-ablation",name=a.run_name,id=a.run_name,resume="allow",config={'params_M':npar/1e6,'block':C.block,'tok_per_step':a.micro*a.accum*C.block,'peak_lr':a.peak_lr,'max_steps':a.max_steps,'mix':mix})
    @torch.no_grad()
    def validate(nb=4):
        m.eval(); r={}
        for L in ld.val_langs:
            t=0.0
            for _ in range(nb):
                x,y=ld.batch_lang(L,a.micro)
                with torch.autocast('cuda',dtype=torch.bfloat16): _,l=m(x,y)
                t+=l.item()
            r[L]=t/nb
        m.train(); return r
    m.train(); t0=time.time(); td=0
    for step in range(start,a.max_steps+1):
        lr=lr_at(step,a.max_steps,a.warmup,a.peak_lr)
        for g in opt.param_groups: g['lr']=lr
        opt.zero_grad(set_to_none=True); la=0.0
        for _ in range(a.accum):
            x,y=ld.batch(a.micro)
            with torch.autocast('cuda',dtype=torch.bfloat16): _,loss=mc(x,y)
            (loss/a.accum).backward(); la+=loss.item()/a.accum
        gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); td+=a.micro*a.accum*C.block
        if step%10==0:
            tps=td/(time.time()-t0); print(f"step {step} loss {la:.3f} lr {lr:.2e} gnorm {gn:.2f} tok/s {tps:.0f} mem {torch.cuda.max_memory_allocated()/1e9:.1f}G",flush=True)
            if a.wandb: wandb.log({'train/loss':la,'lr':lr,'grad_norm':float(gn),'tokens':td,'tok_per_s':tps},step=step)
        if a.val_every and step>0 and step%a.val_every==0:
            v=validate(); print("  val "+" ".join(f"{L}={v[L]:.3f}" for L in v),flush=True)
            if a.wandb: wandb.log({**{f"val/{L}_loss":v[L] for L in v},"val/tat_ppl":math.exp(min(v['tat'],20))},step=step)
        if a.ckpt_every and step>0 and step%a.ckpt_every==0:
            ck={'model':m.state_dict(),'opt':opt.state_dict(),'step':step,'cfg':vars(C)}; torch.save(ck,ckp)
            if a.snap_every and step%a.snap_every==0: torch.save(ck,f"{a.out_dir}/{a.run_name}_step{step}.pt")
    print("___DONE_TRAIN___",flush=True)
if __name__=="__main__": main()
