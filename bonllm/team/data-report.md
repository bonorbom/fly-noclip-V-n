# Data Report — BonLLM mini (Agent DATA)

Ngày: 2026-10-22 · Script: `make_data.py` (seed=42, tái lập được) · Output: `data/corpus.txt`

## Thống kê corpus (đã kiểm chứng độc lập sau khi sinh)

| Hạng mục | Giá trị |
|---|---|
| Số hội thoại | **720** (mục tiêu ≥400) |
| Tổng ký tự | **58,997 ký tự** (mục tiêu 40k–120k) — file 68,452 byte UTF-8 |
| Tổng từ (xấp xỉ) | ~12,683 từ |
| Tiếng Việt | **430 hội thoại — 59.7%** |
| English | **290 hội thoại — 40.3%** |
| Vocab ký tự (unique chars) | **130 ký tự** |
| Hội thoại trùng lặp | 0 |
| Lỗi định dạng | 0 (720/720 block đúng 2 dòng `<user>`/`<bot>`) |
| Ký tự lạ / control char | 0 — không ký tự nào ngoài bảng chữ Latin + dấu tiếng Việt, dấu câu cơ bản và `\n` |

Ghi chú kiểm chứng: đúng 430 block chứa dấu tiếng Việt — khớp tuyệt đối với số hội thoại VN, tức là mọi câu VN đều có dấu đầy đủ, câu EN không lẫn dấu.

## Cách sinh data

- Tự viết sạch bằng template + slot-filling trong `make_data.py`, **không copy dài từ web** (chỉ tham khảo dạng chào hỏi phổ biến VN/EN).
- Mỗi chủ đề: nhiều mẫu câu user × nhiều mẫu câu bot cùng ý định, cộng slot theo ngữ cảnh (buổi trong ngày, trạng thái/cảm xúc, thời tiết, giờ mẫu, việc cần giúp…) → đa dạng, không lặp nguyên xi; dedupe bằng tập hợp (user, bot).
- Phủ các chủ đề: chào theo buổi + chào chung, hỏi thăm sức khỏe / hôm nay thế nào, bạn là ai / giới thiệu bản thân (tên **BonLLM**) / bạn làm được gì, cảm ơn, tạm biệt + chúc ngủ ngon, hỏi giờ giấc và thứ ngày (trả lời *kiểu mẫu*, có nói rõ là ví dụ mẫu — không bịa giờ thật), thời tiết kiểu mẫu, cảm xúc của user → bot đồng cảm + gợi ý, nhờ giúp đỡ, sở thích/small talk, Q&A sự thật đơn giản (thủ đô, mặt trời, số ngày/tháng…), toán cộng/nhân/trừ nhỏ **sinh bằng chương trình nên đáp án luôn đúng**, lời chúc/lời khuyên.

## 5 hội thoại ví dụ (trích nguyên văn từ corpus)

```text
<user> Chào buổi sáng
<bot> Chào bạn! Buổi sáng của bạn thế nào rồi?

<user> Bạn là ai?
<bot> Mình là một AI nhỏ tên BonLLM, được huấn luyện từ đầu để nói chuyện với bạn.

<user> Hôm nay thời tiết thế nào?
<bot> Kiểu mẫu nhé: hôm nay trời nắng đẹp, rất thích hợp để ra ngoài dạo chơi.

<user> Good morning
<bot> Hello! Wishing you a morning full of energy today.

<user> Who are you?
<bot> I am BonLLM, a tiny AI trained from scratch to have simple conversations.
```

## Nhận xét cho Agent Train

- Vocab ký tự chỉ 130 → char-level tokenizer rất gọn, embedding nhỏ, hợp model mini train CPU.
- Corpus ~59k ký tự là cỡ "học thuộc phong cách chào hỏi" cho model from-scratch mini: đủ để học format `<user>`/`<bot>` và các mẫu câu ngắn, chưa đủ cho kiến thức rộng — đúng mục tiêu chơi/học đã chốt.
- Muốn tăng độ khó: chạy lại `make_data.py` sau khi tăng `TARGET_VN`/`TARGET_EN` (pool hiện còn dư: VN 642, EN 594 cặp duy nhất).
