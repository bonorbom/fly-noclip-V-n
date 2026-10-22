# BonLLM-mini — LLM tự train from scratch (chơi / học)

LLM riêng train từ số 0 trên máy này (2 CPU, không GPU) — không dùng weights của ai.
Mục tiêu: chào hỏi + chat thường tiếng Việt (có dấu) và English.

## Học từ OpenAI / Anthropic thế nào? (chi tiết: team/research.md)
Hãng lớn: **Pretraining** (đoán token tiếp theo trên text khổng lồ) → **SFT**
(dạy format trợ lý bằng cặp user/bot mẫu) → **RLHF / RLAIF - Constitutional AI**
(chấm theo feedback người / nguyên tắc viết sẵn).

Bản mini này bắt chước thu nhỏ:
- Pretraining + SFT gộp 1 lần: train next-token prediction trên `data/corpus.txt`
  đã format sẵn `<user> ... / <bot> ...`
- Alignment: bỏ qua (không PPO) — thay bằng vòng Debug "Móc Lỗi" rà, lọc data, train lại.

## Kiến trúc (model.py — GPT decoder-only kiểu nanoGPT của Karpathy)
- Char-level tokenizer tự sinh từ corpus: **130 ký tự** (gọn, xử được tiếng Việt có dấu ở mức ký tự)
- 4 layer Transformer, 4 head, embd 128, context 128 ký tự, dropout 0.1, weight tying
- **824,832 tham số** — tham số này là số random lúc đầu, được train thành weights trong `ckpt/bonllm.pt`

## Chạy
```bash
./venv/bin/python make_data.py     # sinh lại corpus (seed 42)
./venv/bin/python train.py --steps 2500   # train from scratch (CPU, xem train.log)
./venv/bin/python chat.py          # chat thử
./venv/bin/python chat.py --test   # chạy bộ prompt test
```

## File
- `make_data.py` — Agent Data viết: sinh 720 hội thoại VN/EN
- `data/corpus.txt` — data train
- `model.py` / `train.py` / `chat.py` — Lead viết
- `ckpt/bonllm.pt` + `ckpt/meta.json` — weights + vocab sau train
- `team/research.md` — Agent Research
- `team/DEBUG_AGENT.md`, `team/debug-checklist.md`, `team/test_prompts.txt` — Agent Debug "Móc Lỗi"
- `team/data-report.md` — báo cáo data

## Giới hạn thật (không nói quá)
Model 0.82M tham số + 59k ký tự data chỉ học được mẫu chào hỏi/chat quen thuộc
trong data, không có kiến thức thế giới như GPT/Claude. Hỏi ngoài data sẽ nói nhảm —
đó là bản chất của train from scratch quy mô nhỏ, không phải bug.
Muốn mạnh hơn: tăng data (100x), tăng tham số, hoặc fine-tune base có sẵn.
