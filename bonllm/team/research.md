# BonLLM-mini — Research: OpenAI / Anthropic train AI thế nào, và bản thu nhỏ cho máy 2 CPU / ~8 GB RAM

> Agent RESEARCH — 2026-10-22. Phạm vi: chỉ research, không code train, không data/corpus.
> Nguồn: 16 nguồn web đã đọc bằng browser.search (URL đầy đủ ở mục 6). Chỗ nào là **ước tính/suy luận cho máy của Bon** (chưa benchmark thật) đều ghi rõ "chưa kiểm chứng".

---

## 0. Kết luận 1 dòng

OpenAI và Anthropic đều dùng cùng 1 xương sống: **(1) Pretraining** dự đoán token tiếp theo trên text thô khổng lồ → **(2) SFT** dạy format trợ lý bằng cặp prompt–response mẫu → **(3) Alignment** bằng feedback (người: RLHF, hoặc AI theo nguyên tắc viết sẵn: RLAIF / Constitutional AI; bản rút gọn hiện đại: DPO).
BonLLM-mini bắt chước đúng thứ tự đó nhưng thu nhỏ ~6 bậc độ lớn: gộp (1)+(2) thành 1 lần train trên corpus hội thoại đã format sẵn, và thay (3) bằng vòng "tự phê bình – sửa – train lại bằng data đã lọc" (RLAIF-lite), **không làm PPO**.

---

## 1. Pipeline gốc của hãng lớn

### Giai đoạn 0 — Data pipeline + Tokenizer (trước khi train)

- Thu text thô (web, sách, code, hội thoại) → lọc chất lượng, khử trùng lặp (dedup), lọc nội dung độc hại/riêng tư, rồi **tokenize** và chia shard.
- Tokenizer của hãng lớn là **BPE / byte-level BPE (subword)**, vocab ~32k–150k+ token: từ thường gặp = 1 token, từ hiếm tách thành mảnh nhỏ, có byte-fallback nên không bao giờ gặp ký tự "không mã hoá được" (OOV). Đánh đổi: tokenizer huấn luyện chủ yếu trên tiếng Anh khiến ngôn ngữ khác chịu "token tax" — cùng 1 nội dung tốn nhiều token hơn.
- Nguồn: pipeline data→pretrain→SFT→RLHF ở Comet/O'Reilly và bài tổng pipeline ở q4bits; token tax + so sánh word/char/subword ở LLM Book và bài Tokenization Deep Dive (mục 6).

### Giai đoạn 1 — Pretraining (base model)

- Kiến trúc: **Transformer decoder-only**, causal / masked self-attention — mỗi vị trí chỉ nhìn về quá khứ.
- Bài toán duy nhất: **next-token prediction**. Model đoán token tiếp theo, so với token thật bằng cross-entropy loss, sửa weights bằng backpropagation + gradient descent (Adam/AdamW). Lặp trên hàng nghìn tỷ token.
- Sản phẩm: *base model* — "biết" ngữ pháp, sự kiện, pattern ngôn ngữ dưới dạng xác suất thống kê (không phải "hiểu" theo nghĩa người), nhưng **chưa biết làm trợ lý**: hỏi 1 câu nó có thể sinh tiếp thêm 3 câu hỏi tương tự thay vì trả lời.
- Quy luật tỉ lệ (Chinchilla, DeepMind): compute-optimal ≈ **20 token train / 1 tham số**, `C ≈ 6·N·D` FLOPs. GPT-3 (175B tham số, 300B token, tỉ lệ ~2:1) bị coi là under-trained theo luật này. Hệ quả thực tế cho Bon: **model nhỏ train kỹ trên data vừa sức thắng model to train nửa vời cùng compute**.
- Nguồn: q4bits (sơ đồ pipeline đầy đủ), Comet (InstructGPT formal hoá 3 giai đoạn), Chinchilla DeepWiki + bản tóm tắt scaling laws (mục 6).

### Giai đoạn 2 — SFT (Supervised Fine-Tuning / instruction tuning)

- Train tiếp base model trên tập **(prompt, response lý tưởng)** do người viết/chọn — số lượng ít hơn pretraining rất nhiều nhưng chất lượng cao (tư liệu tổng hợp nhắc ví dụ LIMA: ~1.000 ví dụ tốt đã cạnh tranh được đáng kể).
- Mục đích: dạy **format lượt thoại của trợ lý** (user hỏi → assistant trả lời, biết dừng đúng chỗ), văn phong, cách tuân lệnh.
- Sau SFT: model đã "biết trả lời", nhưng chưa có *phán đoán* — chưa biết trong 2 câu trả lời đúng-format thì câu nào hay/an toàn hơn.
- Nguồn: q4bits, Comet, Weaviate/RLHF Book qua kết quả search của team (SFT = low quantity, high quality prompt–response).

### Giai đoạn 3 — Alignment: RLHF (OpenAI) / Constitutional AI + RLAIF (Anthropic) / DPO (bản rút gọn)

**RLHF kiểu OpenAI (InstructGPT → ChatGPT):**
1. Người xếp hạng nhiều response cho cùng 1 prompt (response nào tốt hơn).
2. Train 1 **Reward Model** dự đoán sở thích người từ các xếp hạng đó.
3. Tối ưu model (policy) bằng **PPO** để maximize điểm reward, có **KL penalty** giữ model không trôi quá xa khỏi bản SFT. PPO cổ điển phải giữ cùng lúc tới 4 model (policy, reference, reward, value/critic) → rất tốn VRAM/RAM và nhạy hyperparameter, dễ bất ổn.

**Constitutional AI kiểu Anthropic (Claude):**
- Điểm khác cốt lõi: thay phần lớn nhãn "harmful" của người bằng **1 bộ nguyên tắc viết sẵn (constitution)** — đích đến là *helpful, honest, harmless (HHH)*.
- Pha SL: model tự sinh response cho prompt "bẫy", rồi **tự phê bình theo 1 nguyên tắc được chọn, tự sửa lại**, lấy bản đã sửa làm data fine-tune.
- Pha RL = **RLAIF**: model chấm xem trong 2 response, cái nào hợp nguyên tắc hơn → tạo preference data bằng AI → train preference/reward model → RL trên tín hiệu đó. Anthropic báo cáo bản RL-CAI vừa harmless hơn vừa **ít né tránh (non-evasive)** hơn model train bằng nhãn người truyền thống, và *không dùng nhãn người cho phần harmlessness*.
- Ý nghĩa: alignment có thể scale bằng nguyên tắc + AI feedback thay vì thuê người gán nhãn mọi thứ.

**DPO (Direct Preference Optimization, 2023 — hướng rút gọn sau này):**
- Bỏ reward model riêng và bỏ vòng RL/PPO: train trực tiếp trên cặp **chosen / rejected** bằng loss dạng supervised (binary cross-entropy trên chênh lệch log-prob so với reference model).
- Chỉ cần giữ 2 bản model (policy + reference) thay vì 4, ổn định và rẻ compute hơn RLHF rõ rệt; đánh đổi: phụ thuộc hoàn toàn vào chất lượng cặp preference có sẵn (offline), kém linh hoạt hơn RL online khi cần khám phá/thưởng động. DPO **không** xoá nhu cầu preference data — cặp chosen/rejected vẫn phải đến từ người, AI chấm, hoặc luật tổng hợp.
- Nguồn: Anthropic (2 bài chính chủ ở mục 6), Turing Post (RLHF vs RLAIF vs DPO), AIML.com + dev.to (bảng so sánh RLHF/DPO).

---

## 2. BonLLM-mini bắt chước thế nào — từng giai đoạn

Mục tiêu của Bon (từ brief gốc): AI **chào hỏi + chat thường, song ngữ EN/VN**, chạy local để chơi/học. Không phải trợ lý thông minh thật.

| Giai đoạn hãng lớn | BonLLM-mini thu nhỏ | Giữ nguyên nguyên lý gì | Bỏ / thay gì, vì sao |
|---|---|---|---|
| 0. Data + Tokenizer BPE 50k–150k | Corpus hội thoại VN+EN nhỏ (mục tiêu: hàng chục–vài trăm KB text); **tokenizer char-level** tự suy từ chính corpus (mục 3) | Text vẫn phải làm sạch, dedup câu trùng, chuẩn hoá Unicode trước khi train; tokenizer train/inference phải là 1 | Bỏ BPE: corpus quá nhỏ, train BPE ra mảnh vụn vô nghĩa; char-level đơn giản, minh bạch, đúng tinh thần nanoGPT char |
| 1. Pretraining nghìn tỷ token | Train causal LM next-char prediction trên toàn corpus đã format hội thoại, nhiều epoch | Đúng 100% cơ chế: đoán ký tự tiếp theo, cross-entropy, AdamW, LR warmup + decay, eval loss trên tập val tách riêng | Gộp chung với giai đoạn 2 (xem dưới); quy mô token ít hơn Chinchilla-optimal hàng nghìn lần → **cố ý overfit có kiểm soát**, giống config char của Karpathy vốn "deliberately overfits" |
| 2. SFT bằng cặp prompt–response người viết | Corpus ngay từ đầu đã viết dạng lượt thoại nhất quán (ví dụ dòng `User: ...` / dòng trả lời của assistant / dòng trống ngăn cách — format cụ thể do Agent Data chốt, Research không đụng corpus) | Model học format trợ lý **ngay trong pretraining** vì mọi token nó thấy đều đã là format đích | Không có vòng SFT riêng sau pretrain: với corpus tí hon, train 2 vòng tốn thời gian CPU gấp đôi mà vòng 2 gần như học lại cùng data. Đánh đổi: không có "base model thuần" để so sánh trước/sau SFT như hãng lớn |
| 3a. RLHF: reward model + PPO | **Không làm** | — | PPO cần 4 model + RL loop: độ phức tạp/sập train không tương xứng cho model ~1M tham số học chào hỏi; reward model train trên preference tí hon sẽ chấm nhiễu |
| 3b. Constitutional AI / RLAIF: tự phê bình–sửa theo constitution, AI chấm preference | **RLAIF-lite**: (i) viết 1 "constitution mini" ~5–8 nguyên tắc dạng văn bản (trả lời đúng ngôn ngữ của user; ngắn, thân thiện; không bịa khi không biết; giữ format lượt thoại); (ii) sau khi model sinh thử, đối chiếu output với constitution bằng luật đơn giản + người (Bon) duyệt; (iii) các output đạt chuẩn / bản đã sửa được đưa **trở lại corpus làm thêm 1 vòng train ngắn (SFT vòng 2, LR thấp)** | Giữ vòng lặp critique → revise → train trên bản revised của Anthropic (pha SL của CAI), và giữ ý "nguyên tắc viết sẵn thay nhãn người hàng loạt" | Bỏ pha RL thật và preference model: "AI chấm" ở đây là luật + Bon duyệt, không phải model chấm model. Đây là **mô phỏng quy trình, không phải RLAIF đúng nghĩa** — phải ghi rõ vậy khi báo cáo, không gọi là RLHF/RLAIF |
| 3c. DPO trên cặp chosen/rejected | Tuỳ chọn nâng cấp sau, chỉ khi đã có ≥ vài trăm cặp chosen/rejected thật từ vòng 3b | Nếu làm, DPO là hướng khả thi duy nhất cho CPU vì chỉ giữ 2 model và loss dạng supervised | Hoãn: chưa có preference data thì DPO không có gì để ăn. Không hứa trong bản đầu |

**Bài học rút ra cho cả team:** ở quy mô mini, *chất lượng + tính nhất quán của corpus quyết định gần như toàn bộ chất lượng output* — model nhỏ không có sức tự suy ra cấu trúc mà data không dạy tường minh (khớp nhận định trong các nguồn về small LLM và về SFT "quality > quantity").

---

## 3. Tokenizer char-level cho tiếng Việt có dấu — điểm dễ gãy nhất

1. **Chuẩn hoá Unicode NFC (precomposed) là bắt buộc** trước khi xây vocab: cùng chữ "ế" có thể là 1 ký tự ghép sẵn (NFC) hoặc 3 ký tự rời (e + dấu mũ + dấu sắc, NFD). Trộn 2 dạng = vocab phình, cùng 1 chữ thành 2 cách mã hoá, model học loạn. Toàn corpus và mọi prompt lúc inference phải cùng NFC.
2. **Không dùng byte-level cho bản này:** 1 chữ tiếng Việt có dấu tốn 2–3 byte UTF-8 → sequence dài gấp ~2–3 lần so với char-level NFC, đốt block_size vô ích. Char-level NFC: 1 chữ có dấu = đúng 1 token.
3. **Vocab suy từ corpus:** `vocab_size = số ký tự khác nhau trong corpus`. Ước tính cho corpus chào hỏi/chat thường VN+EN (chưa kiểm chứng — phải đếm thật từ corpus của Agent Data):
   - Chữ thường + hoa tiếng Anh: ~52
   - Nguyên âm tiếng Việt có dấu (a/ă/â, e/ê, i, o/ô/ơ, u/ư, y × 6 thanh điệu × thường/hoa) + `đ/Đ`: ~130–145 ký tự phân biệt
   - Số 0–9, dấu câu, khoảng trắng, xuống dòng, ký tự đánh dấu lượt thoại: ~25–35
   - **Tổng dự kiến: ~200–230 ký tự.** Khác xa con số 65 của Tiny Shakespeare (chỉ tiếng Anh) — đừng copy vocab_size=65.
4. **Hệ quả cứng của char-level theo corpus:** ký tự **không xuất hiện trong corpus thì không encode được và model không bao giờ sinh ra được**. Thiếu 1 tổ hợp dấu (ví dụ không có chữ "ữ" trong corpus) là model câm đúng chữ đó. Agent Data phải đảm bảo corpus phủ đủ bảng chữ có dấu VN (cả thường lẫn hoa nếu muốn model viết hoa đúng), Research chỉ nêu yêu cầu này, không tự sửa corpus.
5. Đánh đổi đã biết của char-level (theo nguồn tokenizer): sequence dài hơn subword ~3–4 lần, attention tốn O(n²) theo độ dài — đây là lý do block_size và n_embd ở mục 4 phải giữ nhỏ.

---

## 4. Kiến trúc đề xuất cho máy 2 CPU / ~8 GB RAM

Tham chiếu gốc: nanoGPT `train_shakespeare_char.py` của Karpathy = `n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=64, dropout=0.2, lr=1e-3, max_iters=5000` → **~10,7M tham số**, train ~3 phút trên A100 nhưng **~12 giờ trên CPU** theo các bản port/báo cáo CPU (200–400 ms/iter ở bản .NET với cấu hình tương đương; bản Python CPU qua đêm mới đạt ~143k iter ở cấu hình nhỏ hơn). Kết luận: **không copy nguyên config Karpathy cho máy này.**

### 4.1. Cấu hình chính (đề xuất chốt)

| Tham số | Giá trị | Lý do |
|---|---|---|
| `vocab_size` | Theo corpus thật, dự kiến ~200–230 (mục 3) | Char-level VN+EN |
| `n_layer` | **4** | Đủ sâu để học pattern hội thoại ngắn; mỗi layer thêm ~0,20M tham số + thời gian/iter gần tuyến tính |
| `n_head` | **4** (head_dim = 128/4 = 32) | n_embd phải chia hết cho n_head; head_dim 32 là mức tối thiểu còn tử tế |
| `n_embd` | **128** | Điểm ngọt cho CPU: 384 (Karpathy) làm mỗi iter nặng gấp ~9 lần ở các phép nhân attention/MLP |
| `block_size` | **128 ký tự** | Đủ cho 1 lượt chào + trả lời ngắn 2 ngôn ngữ; 256 sẽ gấp ~4 lần chi phí attention. Đây cũng là giới hạn cứng độ dài ngữ cảnh — câu dài hơn bị cắt, phải chấp nhận |
| `batch_size` | **32** | 32 × 128 = 4.096 ký tự/bước — vừa RAM, loss đỡ nhiễu hơn batch 8–16 |
| `dropout` | **0.2** | Theo đúng config char của Karpathy; corpus nhỏ + train nhiều epoch = overfit là rủi ro chính, cần dropout giữ lại dù mục tiêu có phần "học thuộc format" |
| `learning_rate` | **1e-3** (min ~1e-4, warmup + cosine decay) | Theo config char gốc cho model nhỏ |
| `max_iters` | **3.000–5.000** (có eval + checkpoint theo val loss) | Xem ước tính thời gian ở 4.3 |
| Optimizer | AdamW | Chuẩn nanoGPT/GPT |

**Số tham số tính tay (vocab lấy 210, lm_head tied với token embedding):**
- Token embedding: 210 × 128 = 26.880; Position embedding: 128 × 128 = 16.384
- Mỗi layer ≈ attention QKV 128×384 + proj 128×128 + MLP 128×512 + 512×128 (≈197.000) + bias/LayerNorm ≈ **~198.000** → 4 layer ≈ 792.000
- **Tổng ≈ 0,84–0,90 triệu tham số** (untied lm_head cộng ~27.000). Tức ~1/12 model char của Karpathy.
- File weights: ~3,5 MB ở float32 (~0,9 MB nếu sau này lượng tử hoá int8 — ngoài phạm vi bản đầu). RAM train (weights + gradients + Adam states ≈ 16 byte/tham số cho AdamW fp32) ≈ **~15 MB cho phần model** — RAM 8 GB không phải nút thắt; **CPU mới là nút thắt**.

### 4.2. Cấu hình khói (smoke test — chạy trước cấu hình chính)

`n_layer=2, n_head=4, n_embd=64, block_size=64, batch_size=16` → ~0,2M tham số. Mục đích duy nhất: chứng minh pipeline end-to-end (data → train → sinh chữ) sống trong vài phút, trước khi đốt thời gian vào cấu hình chính. Đây là đúng mẹo cộng đồng nanoGPT vẫn dùng khi test trên laptop không GPU: hạ xuống "ridiculously small" để check vòng lặp trước.

### 4.3. Ước tính thời gian train (CHƯA kiểm chứng — phải benchmark 100 iter đầu rồi ngoại suy)

- Ngoại suy từ số liệu CPU đã công bố (200–400 ms/iter cho model 10,7M, block 256, batch lớn hơn): cấu hình chính nhỏ hơn ~12 lần tham số, block/bátch nhỏ hơn → dự đoán **~0,2–0,6 giây/iter trên 2 CPU** → 5.000 iter ≈ **20–50 phút**. Nếu benchmark 100 iter đầu cho >1 giây/iter thì cắt max_iters xuống 3.000 hoặc hạ n_embd còn 96/64 thay vì cố chạy.
- Chinchilla soi chiếu: 0,9M tham số "đáng lẽ" cần ~18M token; corpus mini chỉ có ~10⁴–10⁵ ký tự. Model sẽ **học thuộc pattern chào hỏi là chính, khái quát hoá rất hẹp** — đúng kỳ vọng "chơi/học", và là lý do không được mô tả output như "AI biết nói chuyện" khi bàn giao.

### 4.4. Những thứ cố ý KHÔNG đưa vào

- Không `torch.compile` (trên CPU/Windows-WSL hay lỗi, cộng đồng nanoGPT cũng khuyên tắt khi chạy CPU), không mixed precision fp16 trên CPU (không nhanh hơn, dễ NaN), không DDP/multi-GPU (không có GPU).
- Không BPE, không pretrained weights, không lấy checkpoint GPT-2: brief là train **from scratch**, lấy weights có sẵn là sai đề.

---

## 5. Ba rủi ro lớn nhất khi train mini trên CPU (chi tiết ở phần trả parent)

1. **Thời gian train phình / chọn cấu hình quá tay** — copy config Karpathy (10,7M) là ~12 giờ CPU; mọi quyết định size phải qua smoke test + benchmark 100 iter trước.
2. **Overfit / học vẹt trên corpus tí hon** — train loss rất thấp nhưng val loss tách xa (bản port nanoGPT ghi nhận gap 0,83), model lặp nguyên văn câu mẫu và vỡ format ngay khi prompt lệch mẫu; dropout 0.2 + val split + checkpoint theo val là bắt buộc, và kỳ vọng output phải đặt ở mức "bắt chước format chào hỏi".
3. **Tokenizer tiếng Việt gãy âm thầm** — trộn NFC/NFD hoặc corpus thiếu tổ hợp dấu → ký tự không encode/sinh được, lỗi này không báo crash mà lặng lẽ cho output sai dấu/thiếu chữ; phải đếm vocab và kiểm tra phủ bảng chữ VN từ corpus thật trước khi train.

---

## 6. Nguồn đã đọc (16)

**OpenAI / pipeline chung:**
1. http://blog.q4bits.com/en/articles/how-llms-are-created/
2. https://www.comet.com/site/blog/pretraining/
3. https://www.oreilly.com/radar/?p=19751
4. https://realpython.com/ref/ai-coding-glossary/gpt/

**Anthropic — Constitutional AI / RLAIF:**
5. https://www.anthropic.com/news/claudes-constitution
6. https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback
7. https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input

**DPO / RLHF so sánh:**
8. https://www.turingpost.com/p/rlhfvariants
9. https://aiml.com/reinforcement-learning-based-fine-tuning-rlhf-vs-dpo/
10. https://dev.to/shrsv/dpo-vs-ppo-vs-rlhf-when-should-you-use-each-for-llms-1677

**nanoGPT / Karpathy char-level + CPU:**
11. https://deepwiki.com/karpathy/nanoGPT/6.1-training-a-character-level-model
12. https://martincapodici.com/2023/07/15/no-local-gpu-no-problem-running-andrej-karpathys-nanogpt-on-modal-com/
13. https://til.simonwillison.net/tils/til/llms_nanogpt-shakespeare-m2~2Emd
14. https://deepwiki.com/biegehydra/NanoGptDotnet/4.4-performance-and-results

**Tokenizer:**
15. https://llmbook.icsgen-ai.org/part-1-llm-building-blocks/module-01-foundations-nlp-text-representation/section-1.5.html
16. https://hub.cissychen.com/ai-ml/tokenization-day11.en.html

**Scaling laws (đã đọc thêm, dùng ở mục 1 & 4):**
- https://deepwiki.com/datawhalechina/diy-llm/4.1.2-chinchilla-scaling-and-compute-optimal-training
- https://deepwiki.com/karpathy/nanoGPT/7.1-scaling-laws-analysis

> Đếm nguồn: 16 nguồn chính ở danh sách đánh số + 2 nguồn scaling laws bổ sung = **18 URL đã đọc qua kết quả search**.

---

## 7. Việc bàn giao cho các agent sau

- **Agent Data:** corpus format lượt thoại nhất quán, chuẩn hoá NFC, phủ đủ bảng chữ có dấu VN thường+hoa, báo lại `vocab_size` đếm thật + tổng số ký tự.
- **Agent Logic/Code:** cài cấu hình mục 4.1 (+ smoke 4.2 trước), log train/val loss, lưu checkpoint tốt nhất theo val.
- **Agent QA/Debug:** benchmark 100 iter đầu đối chiếu ước tính 4.3; kiểm tra output có đúng ngôn ngữ theo prompt, đúng format, không mất dấu tiếng Việt.
