#!/usr/bin/env python3
"""
restore_model.py - Giai ma ckpt/bonllm.pt tu cac part text ckpt/parts/part*.txt.
Ly do: GitHub connector chi day duoc file text (UTF-8) va gioi han kich thuoc moi lan day,
nen file weights nhi phan 4.38MB duoc luu tren repo duoi dang base64 chia 8 part.
Script nay noi cac part lai, giai ma thanh bonllm.pt de chat.py dung.
Tu dong chay boi .devcontainer khi mo Codespace; chay tay: python3 restore_model.py
"""
import base64
from pathlib import Path

CKPT = Path(__file__).resolve().parent / "ckpt"
PT = CKPT / "bonllm.pt"
PARTS = CKPT / "parts"
SINGLE = CKPT / "bonllm.pt.b64"


def restore() -> Path:
    if PT.exists():
        print(f"Da co {PT} ({PT.stat().st_size:,} B) - khong can giai ma.")
        return PT
    if SINGLE.exists():
        b64 = SINGLE.read_text()
    else:
        files = sorted(PARTS.glob("part*.txt"))
        assert files, "Khong thay ckpt/parts/part*.txt - repo chua day du file?"
        b64 = "".join(f.read_text() for f in files)
    raw = base64.b64decode(b64)
    PT.write_bytes(raw)
    print(f"Da giai ma {len(b64):,} ky tu base64 -> {PT} ({len(raw):,} B)")
    return PT


if __name__ == "__main__":
    restore()
