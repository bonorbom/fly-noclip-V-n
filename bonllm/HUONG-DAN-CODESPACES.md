# CHẠY BonLLM TRÊN GITHUB CODESPACES (để CHAT là chính)

Repo này chứa BonLLM-mini trong thư mục `bonllm/` (model đã train sẵn, không cần train lại).
Codespace mở lên sẽ tự: cài PyTorch CPU + giải mã weights (`bonllm.pt.b64` -> `bonllm.pt`).

## Các bước

1. Ở trang repo này, bấm nút xanh **Code** -> tab **Codespaces** -> **Create codespace on main**.
   - Tạo lần đầu mất vài phút (tải máy ảo + cài PyTorch).
   - Chọn máy **2 core** là đủ cho chat (máy to hơn cháy quota nhanh gấp bội).
2. VS Code hiện ra trong trình duyệt. Mở **Terminal**: menu Terminal -> New Terminal
   (hoặc kéo panel ở đáy màn hình lên).
3. Gõ 2 lệnh:
   ```
   cd bonllm
   python chat.py
   ```
4. Chat luôn trong Terminal: gõ câu (tiếng Việt CÓ DẤU hoặc English) -> Enter.
   Gõ `thoat` để thoát chat.
   Chạy bộ test mẫu: `python chat.py --test`
5. **Dùng xong PHẢI Stop codespace**: vào https://github.com/codespaces -> dấu ... ở codespace của bạn -> **Stop codespace**.
   Chỉ tắt tab trình duyệt là máy vẫn chạy ngầm và vẫn trừ giờ free.

## Quota free (quan trọng)

- GitHub Free: 120 core-giờ/tháng = máy 2 core chạy ~60 giờ thật + 15 GB storage/tháng.
- Hết quota mà không gắn thẻ thanh toán là bị chặn mở codespace tới tháng sau (quota hồi đầu tháng).
- Chat nhẹ nhưng Codespaces trừ giờ theo THỜI GIAN MÁY MỞ, không theo độ nặng — Stop là tiết kiệm nhất.

## Nếu codespace không tự cài xong / thiếu file model

Trong Terminal, chạy tay:
```
pip install -r bonllm/requirements.txt
python3 bonllm/restore_model.py
cd bonllm && python chat.py
```

## Train lại (khi nào muốn)

```
cd bonllm
python train.py --steps 2500
python chat.py --test
```
(CPU codespace train 2.500 bước mất cỡ nửa tiếng như máy local. Model lớn cần GPU thì dùng file `bonllm/bonllm-kaggle.ipynb` trên Kaggle.)
