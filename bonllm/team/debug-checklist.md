# DEBUG CHECKLIST — BonLLM Mini

> Người giữ: Debug Agent "Móc Lỗi". Dùng khi code train được viết xong và ở mọi đợt train lại.
> Cách dùng: tick `[x]` chỉ khi đã **tự kiểm chứng bằng lệnh/log/sample**, không tick theo cảm giác.
> Mỗi mục fail → mở 1 lỗi theo định dạng trong `DEBUG_AGENT.md` mục 4.
> Giai đoạn hiện tại: code chưa có — checklist này là bản chờ, viết theo cấu trúc nanoGPT char-level chuẩn mà team sẽ dùng. Khi file thật xuất hiện, đối chiếu lại tên file/hàm và sửa checklist nếu lệch.

---

## 0. Môi trường & tính tái lập (kiểm trước tiên, sai ở đây là mọi thứ sau vô nghĩa)

- [ ] Chạy trong đúng venv của project (`~/workspace/bon-llm-mini/venv`), ghi version Python + PyTorch vào log train.
- [ ] `torch.cuda.is_available()` đã kiểm tra — kỳ vọng `False` trên VM này (CPU-only), code không được giả định có GPU ở bất cứ đâu (`.cuda()`, `device='cuda'` hard-code).
- [ ] Seed cố định cho `random`, `numpy`, `torch` ngay đầu script; chạy lại cùng seed + config → loss những step đầu khớp nhau.
- [ ] Config đầy đủ được in/lưu ra file cùng thư mục checkpoint: `vocab_size, block_size, n_layer, n_head, n_embd, batch_size, learning_rate, max_iters, eval_interval, dropout`.
- [ ] Số CPU thread đã biết (`torch.get_num_threads()`), không để mặc định gây giành CPU với tiến trình khác trên VM 2 core.

## 1. Theo từng file

### 1.1 File chuẩn bị data (`prepare.py` / phần load data trong `train.py`)

- [ ] Đọc corpus bằng encoding UTF-8 tường minh (`open(..., encoding='utf-8')`) — không để mặc định hệ thống.
- [ ] Corpus đã chuẩn hoá Unicode **NFC** (kiểm tra bằng `unicodedata.is_normalized('NFC', text)` trên toàn bộ text, hoặc chuẩn hoá cưỡng bức rồi ghi lại).
- [ ] Không còn ký tự rác: HTML tag, `&amp;`/`&lt;`, mojibake (`Ã`, `â€`), ký tự điều khiển lạ — đã quét bằng script/regex và ghi số lượng phát hiện.
- [ ] Đếm & ghi: tổng ký tự, tổng dòng, số dòng trùng nguyên văn, tỉ lệ VN/EN ước lượng, độ dài dòng dài nhất.
- [ ] Dòng dài nhất **ngắn hơn `block_size`** (hoặc đã có cách cắt/chunk rõ ràng) — nếu không, model không bao giờ học trọn câu dài.
- [ ] Tách train/val: val không rò — kiểm tra overlap: không dòng val nào trùng (hoặc gần trùng sau khi bỏ dấu câu/khoảng trắng) với dòng train.
- [ ] Tỉ lệ tách ghi rõ (vd 90/10) và tách bằng seed cố định.

### 1.2 Tokenizer / vocab (trong `prepare.py` hoặc `model.py`)

- [ ] Vocab xây từ **toàn bộ** text train+val sẽ dùng, và bộ ký tự bao phủ đủ: toàn bộ chữ VN có dấu (kể cả `đ Đ ă â ê ô ơ ư` + đủ 6 thanh trên mọi nguyên âm), chữ EN hoa/thường, số 0–9, dấu câu cơ bản, khoảng trắng, `\n`.
- [ ] Có test round-trip: `decode(encode(s)) == s` với mọi dòng corpus + các chuỗi test: `"Tiếng Việt có dấu đầy đủ: ăắằẳẵặ, ơớờởỡợ, ưứừửữự"`, `"Hello, World! 123"`, chuỗi có `\n\n`.
- [ ] Ký tự ngoài vocab: hành vi đã định rõ (raise / bỏ qua / `<unk>`) — và đã test thử bằng ký tự chắc chắn ngoài vocab (emoji 🙂, ký tự `€`).
- [ ] `stoi`/`itos` được **lưu ra file** (`meta.pkl` / `vocab.json` / trong checkpoint) — generate KHÔNG build lại vocab từ data.
- [ ] `vocab_size` trong config model lấy từ đúng vocab đã lưu, không hard-code tay ở 2 chỗ khác nhau.

### 1.3 File model (`model.py` — GPT char-level kiểu nanoGPT)

- [ ] Đếm tham số in ra khi khởi tạo (`sum(p.numel() for p in model.parameters())`) và con số hợp lý với config đã khai (sai lệch lớn = config không ăn vào model).
- [ ] Token embedding + positional embedding đúng kích thước: `vocab_size × n_embd` và `block_size × n_embd`.
- [ ] `n_embd` chia hết cho `n_head`; head_dim = n_embd / n_head nguyên.
- [ ] **Causal mask đúng:** attention đăng ký buffer tam giác dưới (`tril`), kiểm tra bằng test nhỏ: đổi token ở vị trí cuối KHÔNG làm thay đổi logits ở các vị trí trước (nếu thay đổi = hở mask, lỗi nặng).
- [ ] **Target shift đúng:** trong `forward(idx, targets)`, loss là so logits tại vị trí t với token t+1 — kiểm tra bằng mắt đoạn reshape/shift, đây là lỗi hay sai nhất khi tự viết lại.
- [ ] Loss dùng `F.cross_entropy` trên logits đã flatten đúng chiều `(B*T, vocab_size)` và targets flatten `(B*T)`.
- [ ] Weight tying (nếu có dùng chung embedding & lm_head) là chủ ý, không phải vô tình share tensor.
- [ ] Dropout chỉ bật khi train (`model.train()`), tắt khi eval/generate (`model.eval()`) — kiểm tra generate script có gọi `eval()` không.
- [ ] Hàm `generate()`: crop context về `block_size` khi prompt + phần sinh vượt giới hạn (không crash, không lặng lẽ sinh rác).
- [ ] Hàm `generate()`: có xử lý temperature (chia logits trước softmax), top_k (lọc đúng, `top_k <= vocab_size`), và lấy mẫu bằng `torch.multinomial` — thứ tự: temperature → top_k/top_p → softmax → multinomial.
- [ ] Không có `.cuda()` / `.half()` sót lại khiến chạy CPU crash hoặc chậm bất thường.

### 1.4 File train (`train.py`)

- [ ] `get_batch`: lấy đoạn ngẫu nhiên đúng độ dài `block_size`, `y` là `x` dịch 1 vị trí — kiểm tra bằng 1 batch in ra: `y[i] == x[i+1]` trên cùng đoạn gốc.
- [ ] Loss khởi điểm ở step 0 xấp xỉ `ln(vocab_size)` (±0.3). Lệch xa = nghi bug shift/loss/mask, dừng lại tìm trước khi train tiếp.
- [ ] Learning rate + schedule ghi rõ (warmup, decay); LR không để mặc định vô tình. AdamW với weight_decay hợp lý, betas mặc định (0.9, 0.95) kiểu nanoGPT hoặc đã ghi lý do khác.
- [ ] Gradient clipping có mặt (`clip_grad_norm_`, vd 1.0) — thiếu là dễ nổ loss (NaN) ở LR cao.
- [ ] Log train loss **và** val loss ở cùng các mốc eval; val tính bằng `estimate_loss` với `model.eval()` + `torch.no_grad()`.
- [ ] Có phát hiện NaN: nếu loss thành NaN thì dừng và báo, không train tiếp lặng lẽ cho hết iters.
- [ ] Checkpoint lưu định kỳ theo step (không chỉ cuối) gồm: `model.state_dict()`, `optimizer.state_dict()`, `iter_num`, config, và vocab/meta.
- [ ] Resume từ checkpoint chạy được thật (đã test: kill giữa chừng → chạy lại → tiếp tục đúng step, loss không nhảy loạn).
- [ ] Đo & ghi thời gian: giây/step trung bình sau warmup, tổng thời gian dự kiến cho `max_iters` — nếu > ngân sách CPU của VM thì báo Lead trước khi chạy full.
- [ ] Eval không quá dày: eval_interval × chi phí eval không ăn quá ~20% tổng thời gian train.
- [ ] Kết thúc train: in loss đầu/cuối (train + val), thời gian thật, đường dẫn checkpoint cuối.

### 1.5 File generate / chat (`sample.py` / `generate.py` / `chat.py`)

- [ ] Load checkpoint bằng `map_location='cpu'`, `strict=True` cho `load_state_dict` (nếu phải `strict=False` → ghi rõ key lệch).
- [ ] Config model đọc từ checkpoint/meta đã lưu — không hard-code lại n_layer/n_embd/block_size trong file này.
- [ ] Vocab load từ file kèm checkpoint; test nhanh: encode 1 câu VN có dấu bằng vocab đã load, decode lại phải y nguyên.
- [ ] Nhận đủ tham số dòng lệnh / tham số hàm: `--prompt, --max_new_tokens, --temperature, --top_k, --seed`.
- [ ] Có điều kiện dừng rõ: dừng ở token kết thúc hội thoại / `\n\n` (theo format corpus) hoặc hết max_new_tokens — và đã test prompt dễ "nói không dừng".
- [ ] Prompt rỗng và prompt 1 ký tự không crash (đã test thật, xem test_prompts [18][19]).
- [ ] Prompt dài hơn block_size không crash (crop có chủ ý ở phía trái).
- [ ] Chạy được trong venv sạch chỉ với checkpoint + code, không phụ thuộc file data train còn nằm đúng chỗ cũ.

## 2. Lỗi thường gặp của nanoGPT char-level (bảng tra nhanh)

| # | Triệu chứng | Nguyên nhân hay gặp nhất | Cách verify nhanh |
|---|---|---|---|
| N1 | Loss step 0 lệch xa `ln(vocab_size)` | Sai shift target, sai flatten loss, vocab_size khai lệch | In loss step 0 + vocab thực; kiểm tra tay 1 batch |
| N2 | Loss giảm đẹp, generate toàn ký tự lặp/rác | Học thuộc phân bố ký tự; train quá ít; chỉ nhìn train loss | Generate ở nhiều temp + xem val loss cùng checkpoint |
| N3 | Generate lặp 1 từ/ký tự vô hạn | Greedy decoding; thiếu điều kiện dừng; data có dòng lặp | Test greedy riêng, test có top_k + temp 0.7 |
| N4 | Load checkpoint xong generate rác 100% | Vocab build lại khác thứ tự; config lệch checkpoint | So `vocab_size` + vài cặp stoi/itos giữa lúc train và lúc load |
| N5 | Câu VN mất dấu / méo chữ | Trộn NFC/NFD; vocab thiếu ký tự dấu; data scrape mojibake | Round-trip test + in các ký tự vocab sắp xếp ra soi bằng mắt |
| N6 | Val loss tăng, train loss giảm | Overfit: model lớn so với data nhỏ, train quá lâu, dropout=0 | Vẽ/ghi chuỗi (train,val) theo step; dừng sớm ở val thấp nhất |
| N7 | Model chép nguyên văn corpus | Data nhỏ + train nhiều epoch = thuộc lòng; val rò từ train | Test nửa-đầu-dòng-corpus vs prompt biến thể (mục 3.8 DEBUG_AGENT) |
| N8 | Generate tự bịa cả câu hỏi lẫn trả lời | Corpus format hội thoại không có token/dấu dừng rõ ràng | Xem format 1 mẩu hội thoại trong corpus + test điều kiện dừng |
| N9 | Train chậm bất thường / treo | Eval quá dày; batch×block quá lớn cho CPU; in log mỗi step kèm generate dài | Đo giây/step; tắt eval/generate trong lúc đo |
| N10 | Loss → NaN giữa chừng | LR quá cao, không clip grad, log/softmax không ổn định số | Kiểm tra clip_grad_norm_, hạ LR, in grad norm khi NaN |
| N11 | Cùng seed, kết quả mỗi lần khác nhau | Quên set seed cho 1 trong random/numpy/torch; DataLoader shuffle không seed | Chạy 2 lần liên tiếp cùng lệnh, diff log 50 step đầu |
| N12 | Prompt dài bị cụt / crash | Không crop context về block_size trong generate | Test prompt dài > block_size ký tự |
| N13 | Trả lời sai ngôn ngữ (hỏi VN trả EN) | Data lệch tỉ lệ ngôn ngữ; prompt mở đầu mơ hồ | Đếm tỉ lệ VN/EN trong corpus; test prompt thuần từng ngôn ngữ |
| N14 | OOM / bị kill khi train | Adam state + batch lớn ăn RAM hơn dự tính trên VM ~8GB | Tính params×(4+8) byte + activation; giảm batch/block |
| N15 | Checkpoint không resume được | Không lưu optimizer/iter; đổi config giữa chừng | Test kill–resume thật 1 lần trước khi train dài |

## 3. Verify generate sau train (quy trình chuẩn, làm đủ mới được kết luận)

### 3.1 Chuẩn bị
- [ ] Dùng checkpoint **cuối** và checkpoint **val loss thấp nhất** (nếu khác nhau thì test cả hai, ghi rõ đang chấm checkpoint nào).
- [ ] Ghi kèm mọi kết quả: đường dẫn checkpoint, seed, temperature, top_k, max_new_tokens, thời gian generate/prompt.
- [ ] Máy/venv ghi rõ — kết quả phải tái hiện được trên VM này.

### 3.2 Chạy bộ prompt cố định
- [ ] Chạy toàn bộ 40 prompt trong `team/test_prompts.txt` ở mức (A) `temperature=0.7, top_k=40`.
- [ ] Chạy lại ít nhất các prompt [18][19][20][34] (rỗng/ngắn/lặp) ở mức (B) greedy — đây là nhóm dễ lộ vòng lặp nhất.
- [ ] Chạy prompt [01] và [37] với 3 seed khác nhau — kiểm tra độ ổn định: cùng prompt, output có giữ cùng ý/ngôn ngữ không.
- [ ] Không cherry-pick: lưu **toàn bộ** output thô vào 1 file kết quả của đợt (vd `team/results-<ngày>-<checkpoint>.txt`), báo cáo trích từ file đó.

### 3.3 Chấm điểm từng output (thang cố định)
Mỗi output chấm đúng 1 mức:
- **ĐẠT** — đúng ngôn ngữ của prompt, đúng nhóm ý (chào/cảm ơn/tạm biệt/hỏi), câu hoàn chỉnh, không rác ký tự.
- **ĐẠT MỘT NỬA** — đúng ngôn ngữ nhưng sai/thiếu ý, hoặc đúng ý nhưng đứt câu, hoặc có đoạn lặp ngắn.
- **SẬP** — rác ký tự, sai ngôn ngữ, sai nhóm ý, vòng lặp dài, bịa tự tin ở prompt ngoài data, hoặc tự sinh cả hội thoại 2 vai.

Ghi thêm cho từng output:
- [ ] Có trùng nguyên văn ≥ 30 ký tự liên tiếp với một dòng corpus không (đánh dấu thuộc lòng, ghi số dòng).
- [ ] Độ dài output có hợp lý không (quá ngắn cụt / quá dài không dừng).

### 3.4 Tổng hợp bắt buộc trong báo cáo đợt
- [ ] Tỉ lệ ĐẠT / MỘT NỬA / SẬP trên 40 prompt, tách riêng theo nhóm A–H của test_prompts.
- [ ] Cặp (train loss, val loss) của checkpoint đã chấm + kết luận overfit/chưa kèm số.
- [ ] Tỉ lệ thuộc lòng: số output bị đánh dấu thuộc lòng / tổng số output.
- [ ] 3 output tệ nhất + 3 output tốt nhất, trích nguyên văn (không sửa chính tả).
- [ ] So với đợt trước (nếu có): tỉ lệ ĐẠT thay đổi thế nào, lỗi cũ nào hết, lỗi mới nào xuất hiện.
- [ ] Verdict 1 câu theo 3 mức ở `DEBUG_AGENT.md` mục 6: CHƠI ĐƯỢC / CHƠI DỞ / CHƯA CHƠI ĐƯỢC + lý do chính.

### 3.5 Cờ đỏ — thấy 1 trong các điều sau là KHÔNG được kết luận "ổn"
- [ ] Chỉ đưa sample đẹp, không đưa file output thô toàn bộ.
- [ ] Chỉ có train loss, không có val loss.
- [ ] Generate test bằng prompt lấy nguyên văn từ corpus mà không nói rõ.
- [ ] Không ghi seed/temperature — kết quả không tái hiện được.
- [ ] Checkpoint chấm không phải checkpoint sẽ bàn giao.
