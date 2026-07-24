#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert TATlit (custom nanoGPT-style checkpoint) -> Hugging Face LlamaForCausalLM.

Low-footprint: holds ONE model copy at a time (peak ~3-4 GB RAM), writes the output
to the external drive (next to the weights), in bf16 (~1 GB). Nothing is uploaded until
the logit-by-logit check against your original GPT passes.

    python3 "/Volumes/ORICO_ILSHAT/06 Татарская ЛЛМ/_mac_run/convert_tatlit_to_hf.py"
"""
import argparse, os, sys, json, shutil, gc
import torch

BASE  = "/Volumes/ORICO_ILSHAT/06 Татарская ЛЛМ"
PROBE = "Татар теле төрки телләрнең кыпчак тармагына керә."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",    default=f"{BASE}/_weights_backup/TATlit.pt")
    ap.add_argument("--spm",     default=f"{BASE}/_mac_run/kypchak_unigram_48k.model")
    ap.add_argument("--gpt-src", default=f"{BASE}/_mac_run/train_ablation.py")
    ap.add_argument("--out",     default=f"{BASE}/_mac_run/TATlit_hf")
    ap.add_argument("--dtype",   default="bf16", choices=["bf16", "fp32"])
    a = ap.parse_args()

    from transformers import LlamaConfig, LlamaForCausalLM
    import sentencepiece as spm

    # 1. load checkpoint, then drop the container to free memory early
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    print("checkpoint top-level keys:", list(ck.keys()))
    sd  = ck["model"] if "model" in ck else ck.get("state_dict", ck)
    cfg = ck.get("cfg")
    if cfg is None:
        sys.exit("No 'cfg' in the checkpoint — cannot read the architecture. Keys above.")
    if not isinstance(cfg, dict):
        cfg = vars(cfg)
    del ck; gc.collect()
    print("cfg:", cfg)

    V, L, H = cfg["vocab"], cfg["n_layer"], cfg["n_embd"]
    NH, NKV, HD = cfg["n_head"], cfg["n_kv"], cfg["head_dim"]
    INT, BLK = cfg["intermediate"], cfg["block"]
    ROPE, EPS = cfg.get("rope_base", 10000), cfg.get("eps", 1e-5)

    kw = {}
    if HD != H // NH:
        kw["head_dim"] = HD
    config = LlamaConfig(
        vocab_size=V, hidden_size=H, intermediate_size=INT,
        num_hidden_layers=L, num_attention_heads=NH, num_key_value_heads=NKV,
        max_position_embeddings=BLK, rope_theta=float(ROPE),
        rms_norm_eps=float(EPS), tie_word_embeddings=True,
        attention_bias=False, mlp_bias=False, hidden_act="silu", **kw,
    )

    sp = spm.SentencePieceProcessor(model_file=a.spm)
    ids = torch.tensor([sp.encode(PROBE)[:64]])

    # 2. ORIGINAL model first: get reference logits, then free it
    import torch.nn as nn, torch.nn.functional as F, math
    from types import SimpleNamespace
    src = open(a.gpt_src, encoding="utf-8").read()
    ns = {"torch": torch, "nn": nn, "F": F, "math": math}
    exec(src[src.index("class RMSNorm"): src.index("class Loader")], ns)
    Corig = SimpleNamespace(**cfg)
    if not hasattr(Corig, "dropout"): Corig.dropout = 0.0
    orig = ns["GPT"](Corig); orig.load_state_dict(sd); orig = orig.float().eval()
    with torch.no_grad():
        lo = orig(ids)[0].clone()
    del orig; gc.collect()

    # 3. HF model: remap keys, load, then free the source tensors
    new = {"model.embed_tokens.weight": sd["wte.weight"]}
    for i in range(L):
        p, q = f"blocks.{i}.", f"model.layers.{i}."
        new[q+"input_layernorm.weight"]          = sd[p+"n1.w"]
        new[q+"self_attn.q_proj.weight"]         = sd[p+"attn.wq.weight"]
        new[q+"self_attn.k_proj.weight"]         = sd[p+"attn.wk.weight"]
        new[q+"self_attn.v_proj.weight"]         = sd[p+"attn.wv.weight"]
        new[q+"self_attn.o_proj.weight"]         = sd[p+"attn.wo.weight"]
        new[q+"post_attention_layernorm.weight"] = sd[p+"n2.w"]
        new[q+"mlp.gate_proj.weight"]            = sd[p+"mlp.w1.weight"]
        new[q+"mlp.up_proj.weight"]              = sd[p+"mlp.w3.weight"]
        new[q+"mlp.down_proj.weight"]            = sd[p+"mlp.w2.weight"]
    new["model.norm.weight"] = sd["norm_f.w"]
    new["lm_head.weight"]    = sd.get("lm_head.weight", sd["wte.weight"])

    model = LlamaForCausalLM(config)
    missing, unexpected = model.load_state_dict(new, strict=False)
    missing = [m for m in missing if m != "lm_head.weight"]
    print("missing (should be []):", missing)
    print("unexpected (should be []):", unexpected)
    if missing or unexpected:
        sys.exit("Key mismatch — stop and send me this output.")
    del new, sd; gc.collect()
    model = model.float().eval()

    # 4. equivalence check
    with torch.no_grad():
        lh = model(ids).logits
    dmax  = (lo - lh).abs().max().item()
    agree = (lo.argmax(-1) == lh.argmax(-1)).float().mean().item()
    model_ok = agree > 0.999 and dmax < 1e-1
    print(f"\nmax|Δlogit|={dmax:.2e}   next-token argmax agreement={agree:.4f}")
    print("MODEL EQUIVALENCE:", "PASS ✅" if model_ok else "FAIL ❌")
    if not model_ok:
        sys.exit("Model check FAILED — do NOT upload, send me the whole output.")

    # 5. save (bf16 by default to keep the file ~1 GB)
    if a.dtype == "bf16":
        model = model.to(torch.bfloat16)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    del model; gc.collect()

    # 6. tokenizer, best-effort (never fatal)
    try:
        from transformers import LlamaTokenizer
        tok = LlamaTokenizer(vocab_file=a.spm)
        for attr, val in (("add_bos_token", False), ("add_eos_token", False)):
            try: setattr(tok, attr, val)
            except Exception: pass
        tok.save_pretrained(a.out)
        try:
            exact = sp.encode(PROBE) == tok(PROBE, add_special_tokens=False)["input_ids"]
        except Exception:
            exact = "unknown"
        tok_note = f"LlamaTokenizer saved (byte-exact vs sentencepiece: {exact})"
    except Exception as e:
        shutil.copy(a.spm, os.path.join(a.out, "tokenizer.model"))
        json.dump(
            {"tokenizer_class": "LlamaTokenizer", "model_max_length": BLK,
             "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>",
             "add_bos_token": False, "add_eos_token": False},
            open(os.path.join(a.out, "tokenizer_config.json"), "w"), ensure_ascii=False)
        tok_note = f"LlamaTokenizer unavailable ({e!r}); copied raw tokenizer.model + minimal config"

    sz = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)) / 1e9
    print("\nwrote to:", a.out, f"({sz:.2f} GB on the external drive)")
    print("files:", sorted(os.listdir(a.out)))
    print("tokenizer:", tok_note)
    print(f"""
======================================================================
MODEL: PASS ✅  — safe to upload. HF_TOKEN is already set in this terminal.

  hf upload ilchats/TATlit "{a.out}" . --repo-type model
  python3 -c "from huggingface_hub import HfApi; HfApi().super_squash_history(repo_id='ilchats/TATlit')"

Optional, drop the raw pickle so only safetensors remains, then re-squash:
  python3 -c "from huggingface_hub import HfApi; HfApi().delete_file('TATlit.pt','ilchats/TATlit')"

You can delete the folder {a.out} afterwards; it is not needed once uploaded.
======================================================================
""")

if __name__ == "__main__":
    main()
