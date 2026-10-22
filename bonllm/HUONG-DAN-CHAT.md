# HƯỚNG DẪN CHAT VỚI BonLLM-mini — các thao tác cần chọn

## Cách A (dễ nhất, trên điện thoại): chat ngay trên Kaggle bằng GPU free

1. Tải file `bonllm-kaggle.ipynb` (đi kèm gói này, cũng đã gửi riêng trong chat).
2. Vào kaggle.com -> đăng nhập tài khoản bạn đã tạo.
3. **Create -> Notebook -> File -> Import Notebook** -> chọn file `bonllm-kaggle.ipynb`.
4. Nếu Kaggle hỏi **framework**: chọn **PyTorch** (toàn bộ code + weights của mình là PyTorch).
5. Panel bên phải -> **Settings -> Accelerator -> GPU T4**.
   - Nếu GPU bị khoá: vào Profile -> Settings -> **Phone Verification**, nhập SĐT nhận mã 1 lần.
6. Bấm **Run All** (hoặc **Save & Run All** nếu muốn đóng điện thoại cho nó chạy nền).
   - Notebook tự: giải nén data -> build model -> train 5.000 bước trên GPU -> chạy 14 câu test.
7. Khi ô test in xong, chạy ô cuối **"4. Chat trực tiếp"**:
   - Ô nhập hiện ra -> gõ câu của bạn (VN có dấu hoặc EN) -> Enter -> nó trả lời ngay trong ô.
   - Gõ `thoat` để dừng chat.
8. Muốn giữ weights bản Kaggle: tab **Output** -> tải `bonllm.pt` + `meta.json`.

Chat hay nhất với các câu kiểu: chào hỏi (Chào bạn, Chào buổi sáng, Hello, Good morning),
Bạn là ai?, How are you?, Cảm ơn bạn, Tạm biệt...
Tránh: tiếng Việt không dấu (nó sập mảng này), toán (nó sai ~90%) — xem team/debug-round2.md.

## Cách B (có PC sau này): chat local bằng bản đã train sẵn trong gói này

1. Giải nén gói `bonllm-mini.zip`.
2. Cài Python 3.10+ rồi cài PyTorch CPU:
   `pip install torch --index-url https://download.pytorch.org/whl/cpu`
3. Trong thư mục đã giải nén: `python chat.py`
4. Gõ câu chat, gõ `thoat` để dừng. Chạy bộ test: `python chat.py --test`

## Cách C (hợp với Bon nhất khi code/test): GitHub Codespaces — VS Code + Terminal trên web

Project đã có sẵn `.devcontainer/devcontainer.json` + `requirements.txt`:
Codespace mở lên là tự cài PyTorch CPU, khỏi cài tay.

1. Đẩy thư mục này lên 1 repo GitHub của bạn (hoặc tạo repo mới rồi upload file trong gói zip).
2. Trong repo: nút **Code -> Codespaces -> Create codespace on main**.
3. Chọn máy **2 core** thôi (máy càng nhiều core càng cháy quota nhanh — xem cảnh báo dưới).
4. VS Code web mở ra, mở Terminal (dấu `>` dưới cùng) và gõ như máy mình:
   `python train.py --steps 2500` rồi `python chat.py` để chat, `python chat.py --test` để chạy test.
5. **Dùng xong phải STOP**: vào github.com/codespaces -> ... -> **Stop codespace**.
   Treo ngầm là nó vẫn trừ giờ (đúng vụ cháy quota tháng 10 của bạn).

Cảnh báo quota: GitHub Free cho 120 core-giờ/tháng = máy 2 core chạy được ~60 giờ thật,
kèm 15GB storage/tháng. Quota của bạn tháng 10/2026 ĐÃ CHÁY HẾT (bạn báo hôm 20/10),
nó hồi vào đầu tháng 11 — trước hôm đó mở Codespace sẽ bị chặn/trừ phí.
Codespaces KHÔNG có GPU: train mini CPU thì ngon, model lớn vẫn phải qua Kaggle.

## Cách D: train lại từ đầu (local CPU)
`python make_data.py` (sinh corpus) -> `python train.py --steps 2500` -> `python chat.py --test`
