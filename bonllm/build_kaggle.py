#!/usr/bin/env python3
"""Build bonllm-kaggle.ipynb - notebook 1 file tu chua:
corpus (nen zlib+base64) + model GPT from scratch + train (GPU Kaggle) + test generate.
Bon chi can: Kaggle > Create > Import Notebook > upload file nay > bat GPU > Run All.
"""
import base64, json, textwrap, zlib
from pathlib import Path

BASE = Path(__file__).resolve().parent
corpus_b64 = base64.b64encode(zlib.compress((BASE / "data/corpus.txt").read_bytes(), 9)).decode()
corpus_chunks = "\n".join(textwrap.wrap(corpus_b64, 100))
model_src = (BASE / "model.py").read_text(encoding="utf-8")

def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(True)}

cells = [
md("""# BonLLM-mini trên Kaggle (GPU free)
Train LLM **from scratch** (0.82M tham số, GPT kiểu nanoGPT) cho chào hỏi VN/EN.

**Trước khi Run All:**
1. Bên phải: **Settings -> Accelerator -> GPU T4** (hoặc P100)
2. **Settings -> Phone Verification** nếu Kaggle đòi (dùng GPU cần xác thực SĐT 1 lần)
3. Bấm **Run All** — train trên GPU nhanh hơn CPU ở nhà nhiều lần.

Kết quả nằm ở **Output**: `bonllm.pt` (weights) + `meta.json` (vocab). Tải về ở tab Output sau khi chạy xong.
"""),
code('''import torch, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHONG CO GPU - vao Settings > Accelerator > GPU T4 roi Run All lai")
'''),
md("## 1. Data — corpus 720 hội thoại VN/EN (giải nén từ chuỗi nhúng sẵn trong notebook, đúng bản đã train local)"),
code(f'''import base64, zlib, unicodedata
_B64 = """
{corpus_chunks}
"""
text = unicodedata.normalize("NFC", zlib.decompress(base64.b64decode(_B64)).decode("utf-8"))
print(f"corpus: {{len(text):,}} ky tu | so hoi thoai: {{text.count('<user>')}}")
print(text[:300])
'''),
md("## 2. Model — GPT Transformer from scratch (code giống hệt model.py ở bản local)"),
code(model_src),
code('''import json, math, os, random, time
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 5000          # tren GPU T4 chi mat vai phut - tang len 10000 neu muon hoc ky hon
BATCH = 64
LR = 3e-3
EVAL_EVERY = 500
torch.manual_seed(42); random.seed(42)

chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
print(f"vocab: {len(chars)} ky tu")

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data)); train_data, val_data = data[:n], data[n:]

cfg = Config(); cfg.vocab_size = len(chars)
model = BonGPT(cfg).to(DEVICE)
print(f"params: {model.num_params():,} | device: {DEVICE}")
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - cfg.block_size, (BATCH,))
    x = torch.stack([d[i:i+cfg.block_size] for i in ix]).to(DEVICE)
    y = torch.stack([d[i+1:i+cfg.block_size+1] for i in ix]).to(DEVICE)
    return x, y

@torch.no_grad()
def eval_loss():
    model.eval(); out = {}
    for split in ("train", "val"):
        ls = []
        for _ in range(20):
            x, y = get_batch(split); _, l = model(x, y); ls.append(l.item())
        out[split] = sum(ls)/len(ls)
    model.train(); return out

def lr_at(step):
    if step < 100: return LR * step / 100
    p = (step - 100) / max(1, STEPS - 100)
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(p * math.pi)))

t0 = time.time()
for step in range(STEPS + 1):
    for g in opt.param_groups: g["lr"] = lr_at(step)
    if step % EVAL_EVERY == 0:
        el = eval_loss()
        print(f"step {step:5d} | train {el['train']:.4f} | val {el['val']:.4f} | {time.time()-t0:.0f}s", flush=True)
    if step == STEPS: break
    x, y = get_batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
full_cfg = {k: getattr(cfg, k) for k in ("vocab_size","block_size","n_layer","n_head","n_embd","dropout","bias")}
torch.save({"model": model.state_dict(), "config": full_cfg}, f"{OUT}/bonllm.pt")
json.dump({"chars": chars, "stoi": stoi, "itos": {str(k): v for k, v in itos.items()}},
          open(f"{OUT}/meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("DA LUU:", f"{OUT}/bonllm.pt", os.path.getsize(f"{OUT}/bonllm.pt")//1024, "KB")
'''),
md("""## 3. Nạp model từ checkpoint + Test generate
Ô code dưới đây tự nạp lại model từ file `bonllm.pt` + `meta.json` đã lưu ở ô train.
**Nếu gặp lỗi `NameError: name 'stoi' is not defined`** (do Kaggle khởi động lại session,
hoặc bạn chạy thẳng ô test/chat mà ô train chưa chạy trong phiên này):
chỉ cần chạy lần lượt **ô Model (mục 2)** rồi **ô này**, KHÔNG phải train lại."""),
code('''# --- NAP LAI MODEL TU CHECKPOINT (chay duoc ca khi session bi restart) ---
import glob, json as _json, os as _os
import torch as _torch

_cands = ["/kaggle/working/bonllm.pt", "bonllm.pt", "./ckpt/bonllm.pt"] + glob.glob("/kaggle/working/**/bonllm.pt", recursive=True)
_pt = next((p for p in _cands if _os.path.exists(p)), None)
assert _pt, "Khong thay bonllm.pt - hay chay o TRAIN truoc (hoac Save & Run All cho train xong)"
_meta_path = _os.path.join(_os.path.dirname(_pt), "meta.json")
_meta = _json.load(open(_meta_path, encoding="utf-8"))
stoi = _meta["stoi"]
itos = {int(k): v for k, v in _meta["itos"].items()}
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
_ck = _torch.load(_pt, map_location=DEVICE, weights_only=False)
_cfg = Config()
for _k, _v in _ck["config"].items():
    if hasattr(_cfg, _k): setattr(_cfg, _k, _v)
model = BonGPT(_cfg).to(DEVICE)
model.load_state_dict(_ck["model"]); model.eval()
print(f"Da nap {_pt} | params {model.num_params():,} | device {DEVICE}")

def answer(user_text, temperature=0.7, max_new=160):
    import unicodedata as _u
    user_text = _u.normalize("NFC", user_text)
    prompt = f"<user> {user_text}\\n<bot> "
    ids = [stoi[c] for c in prompt if c in stoi]
    idx = _torch.tensor([ids], dtype=_torch.long, device=DEVICE)
    out = model.generate(idx, max_new_tokens=max_new, temperature=temperature, top_k=20)
    gen = "".join(itos[int(i)] for i in out[0][len(ids):])
    for stop in ("\\n", "<user>"):
        if stop in gen: gen = gen.split(stop)[0]
    return gen.strip()

torch.manual_seed(7)
for t in ["Chào bạn","Chào buổi sáng","Hello","Good morning","Bạn là ai?","Bạn tên là gì?",
          "How are you?","Bạn có khỏe không?","Cảm ơn bạn","Thank you","Tạm biệt","Goodbye",
          "2 cộng 3 bằng mấy?","What is 5 plus 4?"]:
    print(f"USER: {t}\\nBOT : {answer(t)}\\n")
print("Xong! Tai bonllm.pt + meta.json o tab Output ben phai.")
'''),
md("""## 4. Chat trực tiếp với BonLLM của bạn
Chạy ô dưới đây rồi gõ câu của bạn vào ô nhập hiện ra. Gõ `thoat` để dừng.
Lưu ý: ô chat chỉ chạy khi bạn đang mở notebook trực tiếp (Run từng ô). Nếu dùng Save & Run All (chạy nền) thì bỏ qua ô này - bản chạy nền sẽ tự dừng ở ô test."""),
code('''# CHAT TRUC TIEP - chi chay khi mo notebook truc tiep, khong chay o che do nen
try:
    print("=== Chat voi BonLLM (go 'thoat' de dung) ===")
    while True:
        u = input("Ban: ").strip()
        if u.lower() in ("thoat", "exit", "quit"):
            print("BonLLM: Tam biet ban!")
            break
        if u:
            print("BonLLM:", answer(u))
except EOFError:
    print("(Bo qua o chat: dang chay nen khong co ban phim - mo notebook truc tiep de chat.)")
'''),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.10"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 4}
out = BASE / "bonllm-kaggle.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB, {len(cells)} cells)")
