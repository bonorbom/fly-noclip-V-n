"""
Train BonLLM mini from scratch - giong pipeline OpenAI/Anthropic thu nho:
  Giai doan 1 (file nay): Pretraining - next-token prediction tren corpus VN/EN
  (Ban mini chi lam giai doan 1 + hoc format <user>/<bot> ngay trong corpus,
   tuong duong SFT nhe. RLHF bo qua - co ghi trong team/research.md)

Chay:  ./venv/bin/python train.py [--steps 3000]
Ket qua: ckpt/bonllm.pt (weights = tham so cua model, sinh ra tu train)
         + ckpt/meta.json (vocab char-level) + train.log
"""
import argparse, json, os, random, time, unicodedata
import torch

from model import BonGPT, Config

torch.manual_seed(42)
random.seed(42)
torch.set_num_threads(2)

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    args = ap.parse_args()

    # Chuan hoa NFC bat buoc (moc cua Debug): tron NFC/NFD lam "e + dau" thanh token khac
    text = unicodedata.normalize("NFC", open(os.path.join(HERE, "data/corpus.txt"), encoding="utf-8").read())
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    print(f"corpus: {len(text):,} chars | vocab: {len(chars)} chars")
    print(f"vocab sample: {''.join(chars[:60])}")

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    cfg = Config()
    cfg.vocab_size = len(chars)
    model = BonGPT(cfg)
    print(f"model params: {model.num_params():,} "
          f"(layers={cfg.n_layer} heads={cfg.n_head} embd={cfg.n_embd} block={cfg.block_size})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - cfg.block_size, (args.batch,))
        x = torch.stack([d[i:i + cfg.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + cfg.block_size + 1] for i in ix])
        return x, y

    @torch.no_grad()
    def eval_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = []
            for _ in range(20):
                x, y = get_batch(split)
                _, loss = model(x, y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    def lr_at(step):
        # warmup 100 + cosine decay xuong 10%
        if step < 100:
            return args.lr * step / 100
        p = (step - 100) / max(1, args.steps - 100)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(p * 3.14159)).item()))

    t0 = time.time()
    log_lines = []
    for step in range(args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        if step % args.eval_every == 0:
            el = eval_loss()
            line = f"step {step:4d} | train loss {el['train']:.4f} | val loss {el['val']:.4f} | {time.time()-t0:.0f}s"
            print(line, flush=True)
            log_lines.append(line)
        if step == args.steps:
            break
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    os.makedirs(os.path.join(HERE, "ckpt"), exist_ok=True)
    ckpt_path = os.path.join(HERE, "ckpt/bonllm.pt")
    # Luu DAY DU config (moc cua Debug): __dict__ cua instance chi co field da set tay,
    # thieu field la luc load lai lech kien truc -> output rac du checkpoint tot.
    full_cfg = {k: getattr(cfg, k) for k in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "dropout", "bias")}
    torch.save({"model": model.state_dict(), "config": full_cfg}, ckpt_path)
    with open(os.path.join(HERE, "ckpt/meta.json"), "w", encoding="utf-8") as f:
        json.dump({"chars": chars, "stoi": stoi, "itos": {str(k): v for k, v in itos.items()}},
                  f, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "train.log"), "w") as f:
        f.write("\n".join(log_lines) + f"\ntotal {time.time()-t0:.0f}s params {model.num_params()}\n")
    print(f"saved -> {ckpt_path} ({os.path.getsize(ckpt_path)/1e6:.2f} MB) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
