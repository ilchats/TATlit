#!/usr/bin/env python3
"""
Прогнать модель (TATlit или арку абляции) по TatBLiMP на маке — CPU или Apple MPS.
Это ИНФЕРЕНС, GPU-кластер не нужен: один проход по 1251 паре коротких предложений.

Запуск:
  pip install torch sentencepiece
  python3 score_tatlit_mac.py [чекпойнт.pt] [токенизатор.model] [бенч.jsonl]

Дефолты (если запустить без аргументов): s2v3_best.pt + kypchak_unigram_48k.model + tatBLIMP_1.0.jsonl
  TATlit:  python3 score_tatlit_mac.py s2v3_best.pt          kypchak_unigram_48k.model tatBLIMP_1.0.jsonl
  арка:    python3 score_tatlit_mac.py B_kypchak_s0_step9000.pt neutral6_32k.model     tatBLIMP_1.0.jsonl
Если MPS капризничает: TATLIT_DEV=cpu python3 score_tatlit_mac.py ...
"""
import sys, os, json, math
from types import SimpleNamespace
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import sentencepiece as spm

HERE  = os.path.dirname(os.path.abspath(__file__))
CKPT  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "s2v3_best.pt")
TOK   = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "kypchak_unigram_48k.model")
BENCH = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "tatBLIMP_1.0.jsonl")

DEV = os.environ.get("TATLIT_DEV") or ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"устройство: {DEV}")
print(f"чекпойнт:   {CKPT}")
print(f"токенизатор:{TOK}")
print(f"бенч:       {BENCH}\n")

# класс модели берём из train_ablation.py (та же архитектура, что на поде)
src = open(os.path.join(HERE, "train_ablation.py")).read()
exec(src[src.index("class RMSNorm"): src.index("class Loader")], globals())

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
C = SimpleNamespace(**ck["cfg"])
if not hasattr(C, "dropout"): C.dropout = 0.0
m = GPT(C); m.load_state_dict(ck["model"]); m = m.float().eval().to(DEV)
sp = spm.SentencePieceProcessor(model_file=TOK)
print(f"параметров: {sum(p.numel() for p in m.parameters())/1e6:.0f}M | словарь {C.vocab} | block {C.block}")

@torch.no_grad()
def lp(s):
    ids = sp.encode(s)[:C.block]
    if len(ids) < 2: return 0.0
    x = torch.tensor([ids], device=DEV)
    lg, _ = m(x)
    l = torch.log_softmax(lg[0, :-1].float(), -1); t = x[0, 1:]
    return l[range(t.shape[0]), t].sum().item()

pairs = [json.loads(l) for l in open(BENCH, encoding="utf-8") if l.strip()]
ph = defaultdict(lambda: [0, 0]); ok = 0
for i, p in enumerate(pairs):
    g = lp(p["good"]) > lp(p["bad"]); ok += g
    r = ph[p["phenomenon"]]; r[0] += g; r[1] += 1
    if (i + 1) % 200 == 0: print(f"  ..{i+1}/{len(pairs)}", flush=True)

print(f"\nВСЕГО: {ok/len(pairs):.4f}   ({ok}/{len(pairs)})")
rc = ph.get("13_relative_clauses")
if rc: print(f"RC (относительные придаточные): {rc[0]/rc[1]:.3f}")
print("\nпо явлениям:")
for k in sorted(ph):
    print(f"  {k:34} {ph[k][0]/ph[k][1]:.3f}  ({ph[k][0]}/{ph[k][1]})")
