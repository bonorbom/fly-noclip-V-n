"""
Chat / test BonLLM mini da train.
Chay:  ./venv/bin/python chat.py                -> chat tuong tac
       ./venv/bin/python chat.py --test         -> chay bo prompt test san
"""
import argparse, json, os, sys, unicodedata
import torch
from model import BonGPT, Config

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    # Ban tren GitHub: weights luu dang text base64 chia part o ckpt/parts/
    # -> tu giai ma thanh bonllm.pt neu chua co (xem restore_model.py).
    _pt = os.path.join(HERE, "ckpt/bonllm.pt")
    if not os.path.exists(_pt):
        try:
            import restore_model
            restore_model.restore()
            print("[chat] Da giai ma bonllm.pt tu ckpt/parts/")
        except Exception as e:
            print(f"[chat] Chua co bonllm.pt va khong giai ma duoc: {e}")
    meta = json.load(open(os.path.join(HERE, "ckpt/meta.json"), encoding="utf-8"))
    ckpt = torch.load(os.path.join(HERE, "ckpt/bonllm.pt"), map_location="cpu", weights_only=False)
    cfg = Config()
    for k, v in ckpt["config"].items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    model = BonGPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    stoi = meta["stoi"]
    itos = {int(k): v for k, v in meta["itos"].items()}
    return model, stoi, itos


def answer(model, stoi, itos, user_text, temperature=0.7, max_new=160):
    # Patch Debug dot 2 (D-05): max_new 90 -> 160. Bang chung: 52/720 dong bot
    # trong corpus (7.2%) dai hon 90 ky tu, dong dai nhat 146 ky tu -> ban cu
    # cat cut giua cau moi khi model chon cac dong nay (vd dong identity 109
    # ky tu bi cat o chu "helping"). Generate dung o '\n' nen nang tran khong
    # lam cau ngan dai ra.
    # Patch Debug dot 2 (D-06): ky tu ngoai vocab khong con bi nuot im lang -
    # canh bao ra stderr (stdout giu nguyen dinh dang).
    user_text = unicodedata.normalize("NFC", user_text)
    prompt = f"<user> {user_text}\n<bot> "
    dropped = sorted({c for c in prompt if c not in stoi})
    if dropped:
        print(f"[chat] canh bao: bo {len(dropped)} ky tu ngoai vocab: {dropped}", file=sys.stderr)
    ids = [stoi[c] for c in prompt if c in stoi]
    idx = torch.tensor([ids], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=max_new, temperature=temperature, top_k=20)
    gen = "".join(itos[int(i)] for i in out[0][len(ids):])
    # bot dung o cuoi dong / mo dau luot user moi
    for stop in ("\n", "<user>"):
        if stop in gen:
            gen = gen.split(stop)[0]
    return gen.strip()


TESTS = [
    "Chào bạn", "Chào buổi sáng", "Hello", "Good morning", "Bạn là ai?",
    "Bạn tên là gì?", "How are you?", "Bạn có khỏe không?", "Cảm ơn bạn",
    "Thank you", "Tạm biệt", "Goodbye", "2 cộng 3 bằng mấy?", "What is 5 plus 4?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()
    model, stoi, itos = load()
    if args.test:
        torch.manual_seed(7)
        for t in TESTS:
            print(f"USER: {t}\nBOT : {answer(model, stoi, itos, t, args.temp)}\n")
        return
    print("BonLLM mini - go 'thoat' de dung")
    while True:
        try:
            u = input("Bạn: ").strip()
        except EOFError:
            break
        if u.lower() in ("thoat", "exit", "quit"):
            break
        if u:
            print("BonLLM:", answer(model, stoi, itos, u, args.temp))


if __name__ == "__main__":
    main()
