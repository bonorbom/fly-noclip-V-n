#!/usr/bin/env python3
"""
resave_ckpt.py - Debug Agent "Moc Loi", patch D-01 (dot 2).

Van de: ckpt/bonllm.pt do code train CU luu chi co config={'vocab_size': 130}.
chat.py load duoc chi vi cac gia tri con lai trung ngau nhien voi default cua
Config trong model.py. Doi default la lech kien truc am tham (checklist 3.7).

Script nay:
  1. Suy config DAY DU tu chinh state_dict (khong doan mo):
     vocab_size/n_embd tu token_embedding.weight, block_size tu
     position_embedding.weight, n_layer tu so block, n_head tu head_size,
     bias (cua Linear) tu viec key '...key.bias'/'lm_head.bias' co ton tai hay
     khong — KHONG dung any('.bias') vi LayerNorm luon co .bias rieng (v1 cua
     script nay da bi assert chan dung o day). dropout khong co trong state_dict
     -> lay default Config (0.1); dropout khong anh huong eval/generate.
  2. Doi chieu voi Config mac dinh: bat buoc khop het (tru vocab_size lay tu file).
  3. Ghi de ckpt/bonllm.pt voi config day du (weights giu nguyen 100%).
     Ban goc da backup o ckpt/bonllm-orig.pt truoc khi chay script nay.
  4. Tu kiem chung: load lai ca 2 file, so tung tensor bang torch.equal.

Chay: ./venv/bin/python team/resave_ckpt.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from model import Config  # noqa: E402

CKPT = os.path.join(ROOT, "ckpt", "bonllm.pt")
ORIG = os.path.join(ROOT, "ckpt", "bonllm-orig.pt")


def main():
    assert os.path.exists(ORIG), "Chua backup! cp ckpt/bonllm.pt ckpt/bonllm-orig.pt truoc."
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model"]
    print("config cu trong file:", ck.get("config"))

    vocab_size, n_embd = sd["token_embedding.weight"].shape
    block_size = sd["position_embedding.weight"].shape[0]
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("blocks.")})
    n_layer = len(layers)
    head_size = sd["blocks.0.attn.heads.0.key.weight"].shape[0]
    assert n_embd % head_size == 0
    n_head = n_embd // head_size
    bias = ("blocks.0.attn.heads.0.key.bias" in sd) or ("lm_head.bias" in sd)

    dflt = Config()
    full = {
        "vocab_size": int(vocab_size),
        "block_size": int(block_size),
        "n_layer": int(n_layer),
        "n_head": int(n_head),
        "n_embd": int(n_embd),
        "dropout": float(dflt.dropout),   # khong luu trong state_dict; chi dung khi train
        "bias": bool(bias),
    }
    print("config suy tu state_dict:", full)
    for k in ("block_size", "n_layer", "n_head", "n_embd", "bias"):
        assert full[k] == getattr(dflt, k), f"Lech default o {k}: {full[k]} != {getattr(dflt, k)}"
    print("doi chieu Config mac dinh: KHOP (block_size/n_layer/n_head/n_embd/bias)")

    torch.save({"model": sd, "config": full}, CKPT)
    print("da ghi de:", CKPT)

    # kiem chung: tensors phai giong het ban goc
    new = torch.load(CKPT, map_location="cpu", weights_only=False)
    old = torch.load(ORIG, map_location="cpu", weights_only=False)
    assert new["config"] == full, new["config"]
    assert set(new["model"].keys()) == set(old["model"].keys())
    bad = [k for k in old["model"] if not torch.equal(new["model"][k], old["model"][k])]
    assert not bad, f"tensor lech: {bad}"
    print(f"kiem chung weights: {len(old['model'])} tensor, lech: 0 -> GIONG HET 100%")
    print("config moi trong file:", new["config"])


if __name__ == "__main__":
    main()
