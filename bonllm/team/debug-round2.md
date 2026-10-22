# DEBUG ROUND 2 — RÀ THẬT BonLLM Mini (Debug Agent "Móc Lỗi")

> Ngày: 2026-10-22 · Checkpoint chấm: `ckpt/bonllm.pt` (824,832 tham số, train 2500 bước)
> Log train: step 0 train/val = 4.9122/4.9130 → step 2500 = **0.1325/0.2933** (1939s, CPU)
> Output thô đầy đủ: `team/debug-round2-outputs.txt` (562 dòng) · Script: `team/debug_probe.py`
> Seed: 4000+id cho từng prompt (cả 2 temperature dùng cùng seed để so sánh); chấm ĐẠT/NỬA/SẬP
> theo kỳ vọng trong `test_prompts.txt`, rule-based + **11 override chấm tay** (liệt kê ở Phụ lục B,
> kèm lý do — chỗ nào rule chấm lệch với kỳ vọng file thì ghi rõ, không giấu).

---

## 0. Kết quả tổng — 40 prompt × 2 temperature

| Temperature | ĐẠT | NỬA | SẬP |
|---|---|---|---|
| 0.3 | **22/40 (55.0%)** | 10/40 (25.0%) | 8/40 (20.0%) |
| 0.9 | **14/40 (35.0%)** | 15/40 (37.5%) | 11/40 (27.5%) |
| Gộp 80 lượt | 36 (45.0%) | 25 (31.3%) | 19 (23.8%) |

Theo nhóm (gộp 2 temperature):

| Nhóm | ĐẠT | NỬA | SẬP | Nhận xét 1 dòng |
|---|---|---|---|---|
| A Chào biến thể lạ (01–08) | 9 | 6 | 1 | Lõi chào khá; hay lẫn ngữ cảnh buổi (tối→chiều) |
| B VN không dấu/sai chính tả (09–13) | **0** | 4 | 6 | **Sập toàn nhóm** — xem D-07 |
| C Trộn VN+EN (14–17) | 3 | 1 | 4 | Nhận diện kém khi prompt có tên/giới thiệu trộn |
| D Rỗng/ngắn/rác (18–22) | 5 | 5 | 0 | Không crash, không vòng lặp; hay nhả mảnh phép toán sai |
| E Tạm biệt/cảm ơn biến thể (23–27) | 6 | 2 | 2 | VN tốt; EN "Thanks a lot" sập sang mẫu identity |
| F Ngoài data (28–32) | 5 | 4 | 1 | Paris đúng (thuộc lòng từ corpus); thời tiết bản 0.3 đúng mẫu "kiểu mẫu nhé" |
| G Dài/nhiều ý (33–36) | 3 | 1 | 4 | Bám ý kém; prompt dịch thuật làm sai loại việc |
| H Hỏi về model (37–40) | 5 | 2 | 1 | Nhân cách BonLLM nhất quán, không xưng ChatGPT/Gemini |

Không có bất kỳ output nào rơi vòng lặp vô hạn hay rác ký tự ở cả 80 lượt — cơ chế dừng ở `\n` của
`chat.py` hoạt động. Điểm yếu không nằm ở "rác", mà ở **chọn sai mẫu** và **sai nội dung số học**.

Overfit: gap train–val cuối = 0.2933 − 0.1325 = **0.161**; val vẫn giảm ở chặng cuối
(0.3450 → 0.3178 → 0.3048 → 0.2933, có 1 lần nảy 0.3247 ở step 2000). Chưa phải overfit kiểu
val tăng vọt — overfit ở đây thể hiện qua **thuộc lòng** (mục 3), không qua val loss.

---

## 1. Móc toán — lỗi nặng nhất (D-02)

### 1.1 Data không sai — bác bỏ giả thuyết "đáp án 10 là đáp án phổ biến bị học thuộc"
- Corpus có **308 câu toán, đáp án đúng 308/308 (100%)** — data sạch tuyệt đối.
- Phân bố đáp án trong corpus gần phẳng: mode là **6 (24 câu, 7.8%)**, tiếp theo 8 (6.8%),
  **10 chỉ 20 câu = 6.5%** — không phải đáp án phổ biến nhất.
- Mỗi câu toán là **duy nhất**: 308 câu hỏi khác nhau, mỗi câu xuất hiện đúng **1 lần**
  (136 cặp số hạng). So sánh: prompt chào/giờ giấc lặp nhiều lần (vd "Mấy giờ rồi bạn?" ×9).

### 1.2 Model sai gần như toàn bộ
Chạy 18 phép (cộng/nhân/trừ, VN+EN, trộn câu có và không có nguyên văn trong corpus):

| | Đúng (theo đáp án trích được) | Đúng nghiêm ngặt (phép tính như model viết phải đúng) |
|---|---|---|
| temp 0.3 | **1/18 (5.6%)** | 1/18 |
| temp 0.9 | 3/18 (16.7%) | **2/18 (11.1%)** |

- Câu **có nguyên văn trong corpus** cũng sai: 0.3 chỉ đúng 1/7; "What is 5 plus 4?" có sẵn
  đáp án 9 trong corpus, model vẫn trả **10** ở temp 0.3/0.7 (chỉ đúng ở 0.9).
- **Model còn đổi luôn số hạng** khi viết lại phép tính — bằng chứng nó không hề "tính":
  "3 cộng 6" → *"4 cộng 6 bằng 14"*; "3 nhân 4" → *"3 nhân 5 bằng 18"*;
  "What is 2 plus 2?" (0.9) → *"4 plus 2 is 6"*; "3 plus 7" (0.9) → *"2 plus 7 is 10"*
  (ca này parser chấm "đúng" vì đáp án cuối là 10, nhưng phép tính như viết ra 2+7=10 vẫn sai —
  vì thế có cột "nghiêm ngặt").
- **Thiên lệch đáp án 10 có thật nhưng không phải do tần suất corpus:** trên 36 lượt trả lời toán,
  đáp án **10 xuất hiện 8 lần (22.2%)**, gấp **3.4×** tỉ lệ 6.5% trong corpus (riêng temp 0.3:
  5/18 = 27.8%, gấp 4.3×). Kết luận đúng phải là: model học được **định dạng + phân bố số
  "trông giống đáp án"** (số 2 chữ số bắt đầu bằng 1 rất hay gặp trong vai trò số hạng "10 trừ…",
  "10 giờ"…), chứ **không học được ánh xạ phép tính** — và khi không có ánh xạ, nó lấy mẫu từ
  phân bố đó, trong đó "10" là điểm hút. Không được kết luận "học thuộc đáp án phổ biến nhất".
- Hệ quả phụ: prompt rác (???, prompt rỗng ở 0.9) cũng bị nhả mảnh phép toán **sai**
  ("1 cộng 5 bằng 7", "8 nhân 7 bằng 44", "1 nhân 9 bằng 8") — xem [18][21][22] ở phụ lục.

**Nguyên nhân gốc (đã chứng minh):** bất đối xứng tần suất — mẫu chào/giới thiệu lặp hàng chục
lần nên thuộc được; mỗi fact toán chỉ phơi nhiễm 1 lần duy nhất trong corpus nên char-model
0.83M tham số không ghi nhớ nổi ánh xạ, dù tổng số lần quét corpus ~173 epoch.

---

## 2. Móc tạm biệt — lẫn mẫu "Good *" có tỉ lệ, không phải xui (D-03)

- `chat.py --test` (seed 7, temp 0.7): `Goodbye` → *"Good afternoon! I hope you have a relaxing
  afternoon."* — nguyên văn dòng bot của prompt "Good afternoon".
- Nhưng ở probe (temp 0.3, seed 8000+i nối tiếp): `Goodbye` lại đúng *"Goodbye! See you again
  next time."* → lỗi **theo mẫu sampling**, phải đo bằng lưới:
  **30 lượt `Goodbye` (10 seed × 3 temp): đúng tạm biệt 17/30 (57%), lẫn sang mẫu chào
  "Good evening! I hope you have a warm and cozy evening." 13/30 (43%)**, cộng lỗi char-level
  "Godbye!", "God evening!" ở temp cao.
- Control `Good afternoon` 30 lượt: đúng mẫu 24/30, nhưng **5/30 bị splice 2 mẫu**:
  *"Good morning! I hope you have a relaxing afternoon."*
- Cơ chế (số liệu corpus): user prompt bắt đầu "Good " có **16 block** (morning ~4, afternoon ~6,
  evening ~5, night 1), trong khi `Goodbye` nguyên văn chỉ có **1 block duy nhất**. Tiền tố
  "Good" kéo về họ mẫu chào theo buổi áp đảo. Cùng họ lỗi: [05] "Chào buổi tối" → trả
  "buổi chiều" ở **cả 2 temperature**; "See you later" và "Mình đi đây" (trần, không kèm
  "tạm biệt nhé") → rơi sang mẫu identity (probe tạm biệt 8/10 OK, 2/10 OTHER).
- Nhóm tạm biệt VN có dấu ("Thôi mình đi đây, tạm biệt nhé", [23][35]) lại **đúng ổn định** —
  lỗi tập trung ở họ "Good *" tiếng Anh và các biến thể trần ít gặp.

---

## 3. Tỉ lệ thuộc lòng — model là "bộ chọn mẫu", không phải "bộ viết câu" (D-08)

| Chỉ số | temp 0.3 | temp 0.9 |
|---|---|---|
| Output trùng **nguyên văn 100%** một dòng bot corpus | **34/40 (85%)** | 24/40 (60%) |
| Output có đoạn nguyên văn liên tiếp ≥ 30 ký tự từ corpus | 36/40 (90%) | 33/40 (82.5%) |
| Đoạn nguyên văn dài nhất TB / độ dài output TB | 51.7 / 52.4 ký tự (**98.7%**) | 50.2 / 54.3 (**92.5%**) |

Probe riêng 10 prompt có nguyên văn trong corpus vs 10 prompt biến thể lệch mẫu
(đổi "nha/vậy/thế", thêm "friend", đổi "Good evening"...), temp 0.3:

| | Trùng nguyên văn dòng bot | Đoạn nguyên văn TB |
|---|---|---|
| Prompt trong corpus | **10/10** | 53.9 ký tự |
| Prompt biến thể | **10/10** | 53.7 ký tự |

Tức là prompt biến thể **không làm model viết câu mới** — nó vẫn rút đúng một dòng corpus
nguyên văn, chỉ là chọn dòng đúng hay sai nhóm. Năng lực thật của model hiện tại:
**phân loại prompt vào đúng "ngăn mẫu"** (đạt ~55% ở temp 0.3 trên bộ khó này), không phải
sinh nội dung mới. Mọi kỳ vọng/kết luận phải đặt theo mức đó.

---

## 4. Sanity — những chỗ KHÔNG có lỗi (đã kiểm chứng, không suy đoán)

- **Causal mask kín:** đổi token cuối của dãy, max |Δlogits| ở mọi vị trí trước = **0.0** (kỳ vọng < 1e-5).
- **Target/loss hợp lý:** loss step 0 = 4.9122 so với ln(130) = 4.8675 (lệch 0.045 < 0.3).
- **Round-trip tokenizer:** decode(encode(toàn bộ corpus)) == corpus, 100%.
- **Prompt dài 279 ký tự (> block_size 128):** không crash, trả lời mạch lạc (crop trái hoạt động).
- **Nhất quán nhân cách [31]:** hỏi tuổi ở 2 temp đều né sang giới thiệu BonLLM, không bịa số tuổi,
  không mâu thuẫn. [37][39]: đúng tên BonLLM ở cả 2 temp; không lần nào xưng ChatGPT/Gemini/Claude.
- **Tái lập:** `chat.py --test` chạy 2 lần liên tiếp cho kết quả giống hệt 100%; và giống hệt
  kết quả Lead đã ghi trước đó (cùng 3 lỗi toán/Goodbye).
- [28] "Thủ đô của Pháp" temp 0.3 trả **đúng Paris** — nhưng là thuộc lòng (corpus có dạy fact này),
  và temp 0.9 lại trả "Thủ đô của Việt Nam là Hà Nội" (đúng sự thật, sai câu hỏi). Không tính là hiểu biết.

---

## 5. Danh sách lỗi D-xx

### Lỗi D-01: Checkpoint thiếu full config — ĐÃ TỰ SỬA ✔
- Mức: **Nặng** (rủi ro chặn demo: load được chỉ nhờ default trùng ngẫu nhiên)
- Bằng chứng: `torch.load` → config trong file = `{'vocab_size': 130}` duy nhất; `ckpt/meta.json` không chứa config kiến trúc.
- Nguyên nhân: code train cũ lưu `cfg.__dict__` (chỉ chứa field đã set tay). Lead đã vá train.py cho lần train sau nhưng checkpoint hiện tại vẫn thiếu.
- Patch: `team/resave_ckpt.py` — suy full config **từ chính state_dict** (vocab/n_embd từ `token_embedding.weight`, block_size từ `position_embedding.weight`, n_layer đếm block, n_head từ head_size, bias-Linear từ absence của `key.bias`/`lm_head.bias`), đối chiếu bắt buộc với Config mặc định, ghi đè kèm kiểm chứng. Backup gốc: `ckpt/bonllm-orig.pt` (sha256 `4fe19319…` trước khi sửa).
  - Sự cố trong lúc patch (ghi lại đúng quy trình): bản v1 của script suy `bias` bằng `any(k.endswith('.bias'))` → dính bias của **LayerNorm** → assert đối chiếu chặn lại, file chưa bị ghi. Đã sửa cách suy luận và chạy lại.
- Trước: config = `{'vocab_size': 130}`
- Sau: config = `{vocab_size: 130, block_size: 128, n_layer: 4, n_head: 4, n_embd: 128, dropout: 0.1, bias: False}`; **109/109 tensor giống hệt bản gốc** (torch.equal); `chat.py --test` sau re-save **giống hệt trước 14/14** (diff rỗng).

### Lỗi D-02: Toán sai hệ thống + thiên lệch đáp án 10 — KHÔNG TỰ SỬA (data/model)
- Mức: **Nặng** — chi tiết mục 1. Đúng 5.6% @0.3, 11.1% nghiêm ngặt @0.9; đổi cả số hạng; đáp án 10 bị sản xuất gấp 3.4× tỉ lệ corpus dù data đúng 100% và 10 không phải mode.
- Không tự sửa vì: muốn sửa phải sửa data + retrain (ngoài phạm vi Debug).
- Đề xuất cho Lead quyết: (a) `make_data.py`: giảm số fact (vd số hạng 1–10, ~60–80 fact), mỗi fact lặp **≥ 8–10 biến thể** (đổi cách hỏi "bằng mấy/bằng bao nhiêu/is/what is", có/không dấu chấm) để tần suất phơi nhiễm ngang mẫu chào; retrain 2500 bước như cũ. (b) Hoặc bỏ toán khỏi bộ demo/`--test` và ghi rõ model không biết tính. Không có phương án (c) "vá bằng code" — generate không có gì sai để vá.

### Lỗi D-03: Lẫn mẫu họ "Good *" + splice mẫu + lỗi char "Godbye" — KHÔNG TỰ SỬA (data)
- Mức: **Nặng** cho demo EN — chi tiết mục 2: Goodbye lẫn mẫu chào 43% (lưới 30), [05] tối→chiều 2/2 temp, control afternoon bị splice 5/30.
- Đề xuất cho Lead quyết: bổ sung biến thể Goodbye (≥ 6–8 block: "Goodbye!", "Goodbye friend", "I have to go, goodbye"...), thêm block phân biệt "Good evening" (chào) vs "Good night" (chúc ngủ), giảm bớt block "Good afternoon" trùng lặp; retrain.

### Lỗi D-04: `generate()` ép model về train mode — ĐÃ TỰ SỬA ✔
- Mức: Nhẹ (chưa gây sai output hiện tại, là bẫy cho mọi caller sau này)
- Bằng chứng: `model.py` bản cũ kết thúc `generate()` bằng `self.train()` vô điều kiện; `chat.load()` đặt `model.eval()`, sau 1 lần `answer()` thì `model.training == True` → dropout bật âm thầm cho mọi forward ngoài generate.
- Patch: lưu `was_training` đầu hàm, cuối hàm `self.train(was_training)`.
- Trước: sau `answer()` → `model.training == True`. Sau: `== False` (đúng mode caller). Kiểm chứng kèm: chạy lại test causal mask trên code đã patch — max |Δlogits| vẫn = 0.0 (patch chỉ đụng quản lý mode ở `generate()`, không đụng forward/attention).

### Lỗi D-05: `max_new=90` cắt cụt 7.2% kho mẫu — ĐÃ TỰ SỬA ✔
- Mức: Trung bình
- Bằng chứng: 52/720 dòng bot corpus (7.2%) dài > 90 ký tự, dài nhất 146. Probe [17] và [25] (cả 2 temp) đều bị cắt ở đúng 89 ký tự giữa từ *"helping"* — dòng identity dài 109 ký tự.
- Patch: `chat.py` `answer(..., max_new=90)` → **160** (phủ dòng dài nhất 146; generate vẫn dừng ở `\n` nên câu ngắn không dài ra).
- Trước: *"…and helping"* (89 ký tự, cụt). Sau: *"…and helping with simple things."* (**109 ký tự, trọn dòng corpus, prefix 89 ký tự giống hệt**).
- Kiểm chứng không đổi sampling: reseed từng prompt (seed 700+i), so max_new=90 vs 160 trên 14 prompt của `--test`: **giống hệt 14/14**. (Bản `--test` nguyên vẹn lệch 7 output sau patch chỉ vì nó dùng 1 seed-stream chung — mỗi generate bản mới tiêu thụ nhiều draw RNG hơn; đã truy rõ nguyên nhân này, không phải regression.)
- Còn tồn (giới hạn kiến trúc, ghi nhận): dòng bot 146 ký tự + prompt dài có thể vượt block_size 128 khi generate → crop trái có thể làm đuôi câu kém ổn định; muốn hết hẳn phải tăng block_size + retrain (Lead quyết).

### Lỗi D-06: Ký tự ngoài vocab bị nuốt im lặng (apostrophe/emoji) — ĐÃ SỬA PHẦN HIỂN THỊ ✔ / gốc còn tồn
- Mức: Nhẹ–Trung bình
- Bằng chứng: corpus chứa **0 dấu apostrophe** → vocab không có `'`; prompt [25] "you're" bị bỏ ký tự thành "youre" không một cảnh báo; probe OOV: "🙂" và "€" bị bỏ lặng lẽ (không crash).
- Patch: `chat.py` — trước khi encode, liệt kê ký tự bị bỏ và cảnh báo ra **stderr** (stdout giữ nguyên định dạng): `[chat] canh bao: bo 1 ky tu ngoai vocab: ["'"]`.
- Còn tồn: muốn model *hiểu* `'`/emoji phải thêm vào data + vocab + retrain (đổi vocab_size = đổi checkpoint) — Lead quyết; hiện tại ít nhất không còn im lặng.

### Lỗi D-07: Tiếng Việt không dấu sập toàn nhóm — KHÔNG TỰ SỬA (data)
- Mức: **Nặng** nếu Bon/người dùng gõ không dấu (rất phổ biến ở VN), chấp nhận được nếu chỉ demo có dấu
- Bằng chứng: nhóm B (09–13): **ĐẠT 0/10** qua 2 temperature. [09] "chao ban, ban khoe khong" → trả EN *"You are very welcome…"*; [11] "chao ban, minh ten la Bon" → nhả mẫu mẹo sức khỏe EN; [10] chỉ đạt NỬA nhờ rơi trúng mẫu identity VN.
- Nguyên nhân: corpus 100% có dấu, 0 câu không dấu — model chưa từng thấy dạng này.
- Đề xuất cho Lead quyết: nếu cần, sinh thêm bản song song không dấu cho ~50% số hội thoại (cùng cặp trả lời có dấu) trong make_data + retrain.

### Lỗi D-08: Thuộc lòng gần tuyệt đối — KHÔNG TỰ SỬA (kỳ vọng/quy mô)
- Mức: **Nặng về mặt kỳ vọng** (không phải bug code) — chi tiết mục 3: 85% output @0.3 là nguyên văn corpus; prompt biến thể vẫn 10/10 nguyên văn.
- Hệ quả thực tế: model trả lời tốt đúng các ngăn mẫu đã học, và khi prompt không khớp ngăn nào (dịch thuật [36], giới thiệu trộn [15], "Thanks a lot" [25]) nó vẫn rút 1 mẫu — thường sai nhóm — thay vì "không biết".
- Cần Lead/Bon quyết: đặt lại mô tả sản phẩm là "mẫu chào hỏi song ngữ học thuộc + chọn mẫu" thay vì "chat tự do"; muốn giảm thuộc lòng phải tăng data thật + đa dạng (việc lớn, không phải vá).

### Lỗi D-09: Quy trình train chưa có checkpoint định kỳ/resume — TỒN ĐỌNG QUY TRÌNH
- Mức: Nhẹ (lần này train xong trọn vẹn nên không thiệt hại)
- Bằng chứng: checklist 1.4 — train.py không lưu checkpoint theo step, không lưu optimizer state, không resume. Lead đã vá train.py (NFC + full config) cho lần sau, nhưng resume vẫn chưa có. VM mất session giữa 1939s train là mất trắng.
- Cần Lead quyết: thêm lưu checkpoint mỗi 500 step + `--resume` ở lần train tới.

### Lỗi D-10: Vài "đúng" là tình cờ/thuộc lòng — GHI NHẬN PHƯƠNG PHÁP
- Mức: Nhẹ (phương pháp chấm)
- Ví dụ: [28]@0.3 Paris = thuộc lòng fact có trong corpus; phép "3 plus 7"@0.9 bị model viết thành "2 plus 7 is 10" — đúng đáp án cuối, sai phép tính. Mọi báo cáo sau phải chấm toán theo phép tính như-viết-ra (cột "nghiêm ngặt" ở mục 1.2), và ghi rõ prompt nào có nguyên văn trong corpus.

---

## 6. Patch đã áp (tóm tắt trước/sau)

| Patch | File | Trước | Sau | Kiểm chứng |
|---|---|---|---|---|
| D-01 re-save full config | `ckpt/bonllm.pt` (backup: `ckpt/bonllm-orig.pt`) + `team/resave_ckpt.py` | config chỉ `{vocab_size}` | config đủ 7 field, suy từ state_dict | 109/109 tensor equal; `--test` giống hệt 14/14 |
| D-04 restore mode | `model.py` `generate()` | ép `train()` cuối hàm | khôi phục mode của caller | `model.training` giữ False sau answer() |
| D-05 max_new | `chat.py` `answer()` | 90 → cụt ở "helping" (89 ký tự) | 160 → trọn 109 ký tự | prefix giống hệt; reseed so từng prompt 14/14 giống |
| D-06 OOV warning | `chat.py` `answer()` | nuốt ký tự im lặng | cảnh báo stderr, kê ký tự bị bỏ | hiện cảnh báo `["'"]` với prompt [25] |

Không patch nào đụng vào weights, corpus gốc hay kiến trúc.

---

## 7. Cần Lead quyết (không tự làm)

1. **Toán (D-02):** sửa data kiểu lặp-fact + retrain, hay bỏ toán khỏi demo? 
2. **VN không dấu (D-07):** có bổ sung data không dấu + retrain không?
3. **Họ "Good *" (D-03):** bổ sung/cân bằng lại data tạm biệt EN + retrain?
4. **Định vị lại kỳ vọng (D-08):** mô tả model là "chọn mẫu chào hỏi" trong README/demo.
5. **Train lần tới (D-05 tồn, D-09):** tăng block_size (128 → 192?) và thêm checkpoint định kỳ/resume — đánh đổi thời gian train CPU.

## 8. VERDICT

**CHƠI DỞ** — chạy được thật, lõi chào hỏi/cảm ơn/tạm biệt **có dấu** và EN chuẩn chọn mẫu đúng
quá nửa ở temp 0.3 (ĐẠT 55%), không rác, không lặp vô hạn, nhân cách nhất quán; nhưng còn 3 lỗi
nặng chưa sửa được ở tầng data/model: **toán sai 89–94%**, **VN không dấu sập 0/10 ĐẠT**,
**Goodbye lẫn mẫu chào 43%**. Đủ để nghịch và hiểu cơ chế; chưa đủ để demo như một AI "biết trả lời".
Riêng mảng chào hỏi chuẩn có dấu: đạt mức CHƠI ĐƯỢC.

---

## Phụ lục A — Bảng 40 prompt (output rút gọn; bản đầy đủ ở `debug-round2-outputs.txt`)

| # | Prompt | Output @0.3 (chấm) | Output @0.9 (chấm) |
|---|--------|--------------------|--------------------|
| 01 | Chào bạn | Xin chào! Hôm nay mình có thể giúp gì cho bạn? **[DAT]** | Hi bạn! Mình đây, sẵn sàng trò chuyện cùng bạn rồi. **[DAT]** |
| 02 | Ê chào nè | Chào bạn! Buổi tối của bạn thế nào rồi? **[DAT]** | Chào bạn! Chúc bạn ngủ ngon và có những giấc mơ đẹp nhé! **[DAT]** |
| 03 | Hê lô hê lô | Không có gì đâu, mình luôn sẵn lòng giúp bạn. **[NUA]** | Được chứ! Mình rất sẵn lòng giúp bạn viết một đoạn văn ngắn. Mình bắt đ… **[NUA]** |
| 04 | Xin chào buổi sáng tốt lành nha | Chào bạn! Buổi sáng của bạn thế nào rồi? **[DAT]** | Chào bạn! Buổi sáng của bạn thế nào rồi? **[DAT]** |
| 05 | Chào buổi tối | Chào bạn! Buổi chiều của bạn thế nào rồi? **[NUA]** | Chào bạn! Buổi chiều của bạn thế nào rồi? **[NUA]** |
| 06 | Yo, what's up? | I am BonLLM, a small language model created by Bon for chatting. **[NUA]** | 5 minus 3 is 8. **[SAP]** |
| 07 | Heeeyyy | Hello friend! It is great to see you here. **[DAT]** | Helllo friend! It is great to seee you here. **[NUA]** |
| 08 | Good morningggg | Good morning! I hope you have a morning full of energy. **[DAT]** | Good morning! I hope you have a morning full of energy. **[DAT]** |
| 09 | chao ban, ban khoe khong | You are very welcome. Have a wonderful day! **[SAP]** | You are welcome. Have a wonderful day! I am happy to help you with lear… **[SAP]** |
| 10 | cam on ban nhieu nha | Mình là BonLLM, một mô hình ngôn ngữ nhỏ được Bon tạo ra để trò chuyện. **[NUA]** | Mình hiểu cảm giác của bạn. Bạn thử chia sẻ niềm vui với bạn bè xem sao… **[NUA]** |
| 11 | chao ban, minh ten la Bon | Here is a small tip: slep enough, drink enough water, and keep your min… **[SAP]** | Hey! I am ready to chat whenever you are. **[SAP]** |
| 12 | Chào bạnnn, lâu quá không gặp | Chúc bạn ngủ ngon và có những giấc mơ đẹp nhé! **[SAP]** | Chúc bạn một ngày thật tốt lành, nhiều niềm vui và may mắn nhé! **[SAP]** |
| 13 | Chào bẹn, bẹn có khỏe khum | Chúc bạn ngủ ngon và có những giấc mơ đẹp nhé! **[NUA]** | Chúc bạn một ngày thật tốt lành, nhiều niềm vui và may mắn nhé! **[NUA]** |
| 14 | Hello bạn, how are you hôm nay? | Mình khỏe. Được trò chuyện với bạn là vui rồi, bạn thì sao? **[DAT]** | Mình khỏe. Được trò chuyện với bạn là vui rồi, bạn thì sao? **[DAT]** |
| 15 | Chào, mình là Bon, nice to meet you | Hey! I am ready to chat whenever you are. **[SAP]** | Here is a small tip: slep enough, drink enough water, and keep your min… **[SAP]** |
| 16 | Thank you nhiều nha bạn | Không có chi! Cần gì thêm cứ gọi mình nhé. **[DAT]** | Kiểu mẫu nhé: hôm nay trời nắng đẹp, rất thích hợp để ra ngoài dạo chơi. **[NUA]** |
| 17 | Bye nha, see you tomorrow | I am BonLLM. I am still small and learning, but I enjoy talking with ev… **[SAP]** | I am BonLLM. I am still small and learning, but I enjoy talking with ev… **[SAP]** |
| 18 | (rỗng) | Hello! I am BonLLM, nice to meet you. **[DAT]** | 1 nhân 9 bằng 8 đó bạn. **[NUA]** |
| 19 | a | Mình là BonLLM, một mô hình ngôn ngữ nhỏ được Bon tạo ra để trò chuyện. **[DAT]** | I am BonLLM, a small language model created by Bon for chatting. **[DAT]** |
| 20 | Chào | Chào bạn! Chúc bạn một ngày thật tốt lành nhé. **[DAT]** | Chào buổi chiều! Chúc bạn có một buổi chiều thoải mái. **[DAT]** |
| 21 | ??? | 1 cộng 5 bằng 7 nhé bạn. **[NUA]** | 8 nhân 7 bằng 44 đó bạn. **[NUA]** |
| 22 | 12345 | trừ 4 bằng mấy? **[NUA]** | minus 3 is 1. **[NUA]** |
| 23 | Thôi mình đi đây, tạm biệt nhé | Tạm biệt! Nhớ giữ gìn sức khỏe nhé bạn. **[DAT]** | Tạm biệt! Nhớ giữ gìn sức khỏe nhé bạn. **[DAT]** |
| 24 | Cảm ơn bạn nhiều lắm luôn á | Không có gì đâu, mình luôn sẵn lòng giúp bạn. **[DAT]** | Không có gì đâu, mình luôn sẵn lòng giúp bạn. **[DAT]** |
| 25 | Thanks a lot, you're the best | I am BonLLM. I am still small and learning, but I enjoy talking with ev… **[SAP]** | I am BonLLM. I am still small and learning, but I enjoy talking with ev… **[SAP]** |
| 26 | Hẹn gặp lại bạn vào ngày mai nha | Tạm biệt! Nhớ giữ gìn sức khỏe nhé bạn. **[NUA]** | Bye bạn! Chúc bạn mọi điều tốt lành, hẹn sớm gặp lại. **[NUA]** |
| 27 | Bye bye, ngủ ngon nha | Chúc bạn ngủ ngon và có những giấc mơ đẹp nhé! **[DAT]** | Chúc bạn ngủ ngon và có những giấc mơ đẹp nhé! **[DAT]** |
| 28 | Thủ đô của Pháp là gì? | Thủ đô của Pháp là Paris. **[DAT]** | Thủ đô của Việt Nam là Hà Nội. **[NUA]** |
| 29 | What is 2 plus 2? | 2 plus 2 is 6. **[NUA]** | 2 plus 2 is 6. **[NUA]** |
| 30 | Hôm nay thời tiết thế nào? | Kiểu mẫu nhé: hôm nay trời nắng đẹp, rất thích hợp để ra ngoài dạo chơi. **[DAT]** | Good night! What is 5 minus 5 is 4. **[SAP]** |
| 31 | Bạn bao nhiêu tuổi rồi? | Mình là BonLLM, một mô hình ngôn ngữ nhỏ được Bon tạo ra để trò chuyện. **[DAT]** | Mình là một AI nhỏ tên BonLLM, được huấn luyện từ đầu để nói chuyện với… **[DAT]** |
| 32 | Kể cho mình một câu chuyện cười đi | Mình hiểu cảm giác của bạn. Bạn thử nghỉ ngơi ít phút xem sao, có mình … **[DAT]** | Mình hiểu cảm giác của bạn. Bạn thử nghi lại khoảnh khắc này xem sao, c… **[NUA]** |
| 33 | Chào bạn, mình tên Bon, mình mới học … | Được chứ! Mình rất sẵn lòng giúp bạn học tiếng Việt. Mình bắt đầu từ ph… **[SAP]** | Mình thích trò chuyện và học thêm những điều mới từ mọi người. **[SAP]** |
| 34 | Hello hello hello hello hello | Hi there! How can I help you today? **[DAT]** | Hi there! How can I help you today? **[DAT]** |
| 35 | Chào bạn! Bạn khỏe không? Mình khỏe. … | Tạm biệt bạn nhé! Hẹn gặp lại bạn vào lần sau. **[DAT]** | Nhớ giữ gìn sức khỏe nhé bạn. **[NUA]** |
| 36 | dịch sang tiếng Anh: chào buổi sáng | Xin chào! Rất vui được gặp bạn vào buổi chiều này. **[SAP]** | Xin chào! Rất vui được gặp bạn vào buổi chiều này. **[SAP]** |
| 37 | Bạn là ai? | Mình là BonLLM, trợ lý trò chuyện mini học tiếng Việt và tiếng Anh. **[DAT]** | Mình là BonLLM, trợ lý trò chuyện mini học tiếng Việt và tiếng Anh. **[DAT]** |
| 38 | Who made you? | I am BonLLM, a small language model created by Bon for chatting. **[DAT]** | I am doing well, thank you for asking. How about you? **[NUA]** |
| 39 | Bạn tên gì thế? | Mình là BonLLM, một mô hình ngôn ngữ nhỏ được Bon tạo ra để trò chuyện. **[DAT]** | Mình là BonLLM, một mô hình ngôn ngữ nhỏ được Bon tạo ra để trò chuyện. **[DAT]** |
| 40 | Bạn có phải là người không? | Mình là BonLLM, trợ lý trò chuyện mini học tiếng Việt và tiếng Anh. **[NUA]** | Ví dụ mẫu nhé: bây giờ là 12 giờ trưa, giờ nghỉ trưa và ăn trưa rồi đó. **[SAP]** |

## Phụ lục B — Các override chấm tay (rule-based → chấm cuối)

- Override [06] temp 0.3: override tay: EN mach lac nhung lech sang gioi thieu, khong dap "what's up"
- Override [13] temp 0.9: override tay: chao chung, khong dap y hoi khoe (thong nhat voi temp 0.3)
- Override [18] temp 0.9: override tay: cau hoan chinh nhung la phep toan SAI (1x9=8) cho prompt rong
- Override [21] temp 0.3: override tay: cau VN ngan nhung la phep toan SAI (1+5=7) sinh ngau nhien
- Override [21] temp 0.9: override tay: phep toan SAI (8x7=44)
- Override [22] temp 0.3: override tay: manh dut cua 1 cau hoi toan, khong phai cau hoan chinh
- Override [22] temp 0.9: override tay: manh dut phep toan
- Override [28] temp 0.9: override tay: dung su that nhung SAI cau hoi (hoi Phap tra loi Viet Nam)
- Override [30] temp 0.9: override tay: sai ngon ngu + manh toan sai (5-5=4), vo nghia
- Override [32] temp 0.9: override tay: dut cau giua chung ("cung b..."), khong phai chuyen cuoi
- Override [40] temp 0.9: override tay: tra loi gio giac cho cau hoi phai-nguoi-khong: sai nhom y hoan toan
