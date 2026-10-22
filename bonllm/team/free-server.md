# Free Server cho AI — Ném script lên web rồi chạy ở đâu còn FREE thật?

> Agent FREE-SERVER — team BonLLM-mini — khảo sát **22/10/2026**
> Nguồn: search web thật, ưu tiên trang chính chủ (Google, Kaggle, AWS, Modal, Render, GitHub, Hugging Face, Lightning, DigitalOcean) + các bài tổng hợp kiểm chứng Sep–Oct 2026. URL đầy đủ ở mục 6.
> ⚠️ Quy tắc chung của mọi free tier: **hạn mức thay đổi không báo trước**. Số ghi ở đây là trạng thái lúc khảo sát — trước khi train dài, mở lại trang pricing chính chủ của nhà đó 1 phút để xác nhận.

---

## 0. Kết luận nhanh cho Bon (chỉ có điện thoại Android, PC hỏng)

- **Train LLM nhỏ: Kaggle số 1** (30 giờ GPU/tuần cố định, chạy nền kể cả khi đóng trình duyệt), **Colab số 2** (dễ bắt đầu nhất, nhưng GPU "hên xui").
- **Host bot/script chạy lâu dài: Oracle Cloud Always Free là máy chủ free 24/7 thật duy nhất** trong danh sách — đổi lại phải có **thẻ để xác minh** và hay hết capacity. Phương án dễ hơn: **Render Free** (nhà Bon đang dùng cho bot Discord rồi) nhưng ngủ sau 15 phút.
- **Đã chết, đừng đăng ký phí công: Amazon SageMaker Studio Lab** (đóng cửa người mới từ 30/07/2026, chính chủ AWS xác nhận).
- **Tưởng free mà không hẳn:** HF Spaces CPU cho tài khoản mới (nhiều báo cáo quota = 0 từ giữa 2026), Lightning "80 GPU hours" (là quà khởi đầu 1 lần, không phải mỗi tháng), Replit "always-on free" (đã bỏ từ lâu).

---

## 1. Bảng so sánh đầy đủ

| Nền tảng | Còn free thật? | GPU / VRAM | Giới hạn phiên / tuần | Cần thẻ tín dụng? | Lưu file bền? | Train PyTorch tuỳ ý? | Đăng ký bằng điện thoại? |
|---|---|---|---|---|---|---|---|
| **Google Colab Free** | ✅ Có, chính chủ FAQ vẫn ghi free; GPU bị "heavily restricted", không bảo đảm có GPU | T4 16GB (thường gặp, **không cam kết**; có lúc K80/không có) | Tối đa **12 giờ/phiên**; quota tuần **không công bố**, dao động theo tải; idle ~90' bị ngắt (cộng đồng đo) | ❌ Không | ❌ VM xoá khi idle/hết phiên — lưu checkpoint vào **Google Drive** (15GB free) | ✅ Notebook, cài sẵn PyTorch; chạy `train.py` qua cell `!python` được | ✅ Dễ nhất: chỉ cần tài khoản Google; mở web trên Chrome Android. Sửa notebook trên điện thoại hơi chật nhưng upload + Run all thì ok |
| **Kaggle Notebooks** | ✅ Có, và **ổn định nhất** nhóm notebook | P100 16GB **hoặc T4 x2** (mỗi card 16GB), có cả TPU | **30 giờ GPU/tuần** (trang chính chủ Kaggle xác nhận), phiên đơn tối đa **9 giờ** theo trang tips chính chủ (một số docs ghi 12 giờ thực thi — kiểm tra trong tài khoản) | ❌ Không — nhưng **bắt buộc xác thực số điện thoại** mới mở GPU + Internet | ⚠️ Nửa vời: ổ làm việc mất theo phiên, nhưng **Save Version / lưu output thành Dataset–Model thì bền** trên Kaggle | ✅ Cài sẵn PyTorch/TF; có **chạy nền "Save & Run All"** — đóng tab train vẫn chạy | ✅ Đăng ký web trên điện thoại được; bước xác thực SMS là bắt buộc (số VN thường nhận được, nếu kẹt thì thử lại giờ khác) |
| **HF Spaces — ZeroGPU** | ⚠️ Còn nhưng **chỉ để demo inference**: free **5 phút GPU/ngày** (chưa đăng nhập 2 phút) | GPU chia sẻ RTX PRO 6000 Blackwell (~nửa GPU, ~48GB; đời trước là H200 70GB) | Quota tính **theo ngày**, reset 24h sau lần dùng đầu; hàm GPU mặc định tối đa ~60 giây/lượt | ❌ Không (cần email đã xác minh + tài khoản >30 ngày; tối đa 2 ZeroGPU Spaces) | ❌ Mặc định không bền — lưu model ra **Hub repo** | ❌ **Không train được**: chỉ Gradio + PyTorch, cấm `torch.compile`, GPU chỉ cấp theo từng lần gọi hàm | ✅ Đăng ký bằng email trên điện thoại, rất dễ |
| **HF Spaces — CPU Basic** | ⚠️ **Đang bị siết thực tế**: docs vẫn ghi free (2 vCPU / 16GB RAM), nhưng từ ~06–09/2026 nhiều tài khoản mới báo **quota cpu-basic = 0**, nhân viên HF trả lời trên forum là tạo Space chạy compute cần gói trả phí — docs và thực tế đang lệch nhau | Không GPU | Space free bị **ngủ sau ~48h** không dùng | ❌ Không | ❌ Không mặc định | ❌ Không hợp train (CPU yếu + nguy cơ không tạo được Space) | ✅ Dễ, nhưng đừng đặt kỳ vọng cho tài khoản mới |
| **Lightning AI Studio** | ✅ Còn free plan, nhưng số giờ **mơ hồ nhất nhóm** (xem mục 5) | T4 / L4 / A10… tuỳ credit; máy free mặc định là **Studio CPU 4 nhân** | 1 Studio free chạy liên tục dạng CPU với **restart mỗi 4 giờ**; GPU ăn credit: tặng **5 credit lúc đăng ký** (+25 nếu gắn thẻ), các nguồn thứ cấp ghi ~15 credit/tháng (~22 giờ T4) — **chính chủ không cam kết con số tháng cố định** | ❌ Không bắt buộc (gắn thẻ chỉ để nhận thêm 25 credit) | ✅ **Có** — workspace bền như VS Code cloud, 10GB đầu free (docs billing chính chủ) | ✅ Terminal thật, train script tuỳ ý khi Studio gắn GPU | ⚠️ Được trên trình duyệt điện thoại (IDE hơi chật); **bắt buộc xác thực điện thoại** để nhận gói giờ GPU |
| **Modal (Starter)** | ✅ **$30 credit compute MỖI THÁNG** — trang pricing chính chủ xác nhận, hiếm hoi là free "tiền thật" làm mới hàng tháng | T4, A10G, L4, L40S, A100 40/80GB, H100, H200 — chọn tuỳ ý | Không giới hạn phiên kiểu notebook: **trả theo giây chạy thật**, không chạy = $0. $30 ≈ **~50 giờ T4 hoặc ~37 giờ L4** (ước tính theo bảng giá) | ⚠️ Pricing chính chủ không ghi bắt buộc; chưa xác nhận được lúc đăng ký có đòi thẻ/xác minh để mở GPU không — **kiểm tra tại bước signup** | ✅ Có **Volumes** lưu bền; nhưng bản chất là serverless (code-first), không phải VM/notebook | ✅ Rất hợp **train dạng job**: bọc `train.py` thành 1 hàm Modal, bấm chạy là xong, không phải giữ phiên. Không hợp gõ interactive trên điện thoại | ⚠️ Đăng ký web bằng Google/GitHub trên điện thoại được, nhưng luồng làm việc code-first (SDK/CLI) **khó thao tác thuần điện thoại** — hợp để Lead/Muse ném job hộ hơn là Bon tự bấm |
| **Amazon SageMaker Studio Lab** | ❌ **ĐÃ ĐÓNG với người mới từ 30/07/2026** (AWS đưa vào "maintenance", docs chính chủ ghi "no longer open to new customers") | (Người cũ) T4 16GB | (Người cũ) GPU 4 giờ/ngày, 15GB lưu bền | Trước đây không cần thẻ/AWS account | (Người cũ) Có | (Người cũ) Có | ❌ Bon là người mới → **không đăng ký được nữa**, bỏ qua |
| **Paperspace Gradient (DigitalOcean)** | ⚠️ Docs DigitalOcean (kiểm chứng 07–09/2026) **vẫn còn máy free**, nhưng là hàng yếu/cũ và chưa ai kiểm chứng lấy được capacity thật | "Free GPU" **M4000** (Quadro đời cũ, ~8GB) | **Auto-shutdown tối đa 6 giờ**, idle 1 giờ tự tắt, **1 notebook free tại 1 thời điểm**, máy free trên Free plan **luôn ở chế độ public** | ⚠️ Chưa xác nhận chính thức | ⚠️ Storage free ít (~5GB theo tổng hợp) | ✅ Notebook PyTorch được, nhưng GPU cũ yếu — train mini thì được, LLM lớn thì không | ⚠️ Đăng ký web được; giao diện nặng trên điện thoại |
| **Oracle Cloud Always Free** | ✅ **Free vĩnh viễn thật** — nhưng đã bị **cắt một nửa 06/2026**: từ 4 OCPU/24GB xuống **2 OCPU / 12GB ARM** (1.500 OCPU-giờ + 9.000 GB-giờ/tháng), cộng 2 VM AMD micro 1GB, **200GB disk, 10TB egress/tháng** | ❌ **Không GPU** — CPU ARM Ampere A1 | **Không giới hạn phiên: VM thật chạy 24/7.** ⚠️ Bị thu hồi nếu 7 ngày liền CPU/mạng/RAM đều <20% (95th percentile); tài khoản bỏ không 30 ngày có thể bị khoá | ✅ **CÓ — bắt buộc thẻ để xác minh danh tính** (không trừ tiền nếu ở trong hạn free) — rào cản lớn nhất cho Bon | ✅ **Bền hoàn toàn** — ổ đĩa VM thật 200GB | ⚠️ Train được PyTorch bản ARM ở mức **CPU chậm** — hợp model mini kiểu BonLLM (0,8M tham số train CPU) chạy nhiều ngày không ai trông; không hợp model cần GPU | ⚠️ Đăng ký web trên điện thoại được nhưng console OCI rất rối trên màn hình nhỏ + hay gặp **"out of host capacity"** phải thử lại nhiều lần/đổi availability domain; cần app SSH kiểu Termux để quản trị |
| **GitHub Codespaces** | ✅ Free cho tài khoản cá nhân: **120 core-giờ/tháng** (= 60 giờ máy 2-core) + **15GB storage/tháng** (trang plans chính chủ GitHub) | ❌ Không GPU — CPU 2–32 core (máy càng to đốt quota càng nhanh) | Tự dừng sau **30 phút idle** (mặc định); không phải server chạy nền | ❌ Không (trong hạn quota cá nhân) | ✅ File bền trong codespace (tính vào 15GB), codespace lâu không dùng có thể bị xoá theo retention | ✅ Terminal Linux thật, train CPU PyTorch tuỳ ý được (chậm như VM 2 nhân hiện tại của team) | ✅ Bon **đã có GitHub** nên rào thấp nhất nhóm này; mở codespace từ trình duyệt điện thoại được (VS Code web hơi chật) |
| **Replit (Starter)** | ⚠️ Free-forever nhưng **rất mỏng**: ~10 app dev (chỉ public), **1 app published** (link của app free **sập sau 30 ngày** theo docs/tổng hợp 2026), ~1.200 phút dev/tháng, 2GiB storage | ❌ Không GPU, CPU/RAM nhỏ | Không còn always-on free: app chạy nền/always-on (Reserved VM) là **tính năng trả phí** | ❌ Không | ✅ File trong workspace bền (2GiB) | ❌ Chạy script ngắn để học thì được; **không hợp train** (phút dev ít, máy yếu, ngủ) | ✅ **Thân thiện điện thoại nhất**: có app + web editor, đăng ký 1 phút |
| **Render Free** | ✅ Còn: workspace Hobby $0, **750 giờ instance/tháng**, web service **512MB RAM**, CPU lẻ | ❌ Không GPU | **Ngủ sau 15 phút** không có traffic, dậy mất ~1 phút; filesystem **mất hết mỗi lần ngủ/redeploy**; free Postgres 1GB **hết hạn sau 30 ngày** (+14 ngày ân hạn rồi xoá) — docs chính chủ | ❌ Không | ❌ **Không** (đĩa bền là tính năng trả phí) — data phải để ở DB ngoài | ❌ Không train (512MB RAM). Chỉ host **web service/bot nhẹ kiểu HTTP–webhook** | ✅ Rất dễ trên điện thoại: nối GitHub repo là deploy (đây chính là nhà Bon đang chạy `bon chan` + `Bonbanai` — đã nếm mùi ngủ/restart của nó rồi) |

---

## 2. Xếp hạng theo nhu cầu

### (A) Train BonLLM-mini / LLM nhỏ bằng PyTorch

| Hạng | Nền tảng | Vì sao |
|---|---|---|
| 🥇 1 | **Kaggle** | 30 giờ GPU/tuần **cố định, đoán trước được**; P100/T4 thừa sức cho model nhỏ + cả LoRA 7B; chạy nền "Save & Run All" không cần giữ máy — hợp nhất với người chỉ có điện thoại |
| 🥇 2 | **Google Colab Free** | Bắt đầu nhanh nhất (0 bước cài đặt), T4 16GB; trừ điểm: quota giấu kín, idle là ngắt, phải lưu checkpoint vào Drive liên tục |
| 🥇 3 | **Modal** | $30/tháng credit thật, chọn GPU mạnh (kể cả A100/H100), tính tiền theo giây nên train job ngắn rất hời; trừ điểm: phải bọc script theo SDK, khó tự làm trên điện thoại |
| 4 | **Lightning AI** | Workspace bền + terminal thật rất sướng khi sửa code qua lại; trừ điểm: chính sách credit không rõ ràng, dễ hiểu nhầm "80 giờ/tháng" |
| 5 | **Oracle ARM / Codespaces (CPU)** | Không GPU nhưng **không giới hạn phiên** (Oracle) — riêng BonLLM-mini 0,8M tham số train CPU được, hợp kiểu "ném lên chạy 2 ngày kệ nó"; Codespaces 60 giờ/tháng tương đương VM hiện tại của team |
| 6 | **Paperspace Gradient Free** | Có GPU free trên giấy nhưng M4000 quá cũ + máy free luôn public (lộ code/data) — chỉ dùng khi các nhà trên đều kẹt |
| 🚫 | HF ZeroGPU, Replit, Render | Không phải chỗ train: ZeroGPU là demo inference 5 phút/ngày; Replit/Render không GPU + ngủ |

### (B) Host script / bot AI chạy lâu dài

| Hạng | Nền tảng | Vì sao |
|---|---|---|
| 🥇 1 | **Oracle Cloud Always Free** | Máy chủ free 24/7 **duy nhất đúng nghĩa**: IP thật, SSH, ổ bền 200GB, 12GB RAM chạy bot + inference CPU model nhỏ thoải mái. Đổi lại: cần thẻ xác minh + hên xui capacity + nguy cơ thu hồi nếu để idle |
| 🥇 2 | **Render Free** | Dễ nhất cho bot kiểu web/webhook, deploy từ GitHub ngay trên điện thoại, 750 giờ ≈ 1 service chạy cả tháng. Trừ điểm: ngủ 15 phút (bot Discord giữ gateway 24/7 sẽ bị đá — phải có ping giữ ấm và chấp nhận reconnect), mất file khi ngủ, DB free tự huỷ sau 30 ngày |
| 🥇 3 | **Modal** | Hợp **API inference theo lượt gọi** (scale về 0, không tốn khi không ai dùng) trong $30 credit/tháng; không hợp bot cần kết nối thường trực |
| 4 | **HF Spaces** | Chỉ hợp **demo công khai** (Gradio) cho người ta vào thử model; kèm cảnh báo quota tài khoản mới 2026; ngủ sau ~48h |
| 5 | **Replit Starter** | Dựng + khoe app nhỏ nhanh trên điện thoại; free published app chỉ sống 30 ngày nên không tính là host lâu dài |
| 🚫 | Colab, Kaggle, Codespaces | Là môi trường **phiên/notebook/dev** — Colab free còn **cấm chính thức** việc file hosting/web service/SSH trên runtime free (FAQ chính chủ). Đừng host bot ở đây, ăn ban/gãy phiên là chuyện sớm muộn |

---

## 3. Top 3 khuyến nghị cho Bon + cách bắt đầu trên điện thoại

### 🥇 #1 — Kaggle (train chính)
**Lý do:** GPU free ổn định nhất, chạy nền được, không cần thẻ — chỉ cần số điện thoại Bon đang dùng để xác thực 1 lần.

1. Mở Chrome Android → `kaggle.com` → đăng ký (dùng Google cho nhanh).
2. Vào **Settings → Phone verification** → nhập SĐT, nhập mã SMS. *(Bỏ qua bước này là không bật được GPU/Internet.)*
3. **Code → New Notebook** → panel phải: **Accelerator = GPU (P100 hoặc T4 x2)**, **Internet = On**.
4. Ném dữ liệu lên: tạo **Dataset** riêng chứa `corpus.txt` + các file script của BonLLM, rồi trong notebook chọn **Add Data** gắn dataset đó vào (file hiện ở `/kaggle/input/...`).
5. Dán/chạy cell train; khi chạy dài chọn **Save Version → Save & Run All** để train chạy nền — tắt điện thoại đi ngủ, sáng mai vào **Output** của version đó tải checkpoint về.
6. Canh quota ở trang profile: 30 giờ/tuần, reset đầu tuần — đừng bật GPU khi chỉ sửa code (sửa trên CPU, lúc train mới bật GPU).

### 🥈 #2 — Google Colab Free (train phụ / chạy thử nhanh)
**Lý do:** 0 cài đặt, quen thuộc nhất, hợp test script 5–30 phút trước khi ném job dài sang Kaggle.

1. Chrome Android → `colab.research.google.com` → đăng nhập Google → **File → Upload notebook** (hoặc Open from GitHub, dán repo của Bon).
2. **Runtime → Change runtime type → T4 GPU** → Run từng cell; cell đầu luôn có `!pip install ...` vì mỗi phiên là máy mới.
3. Mount Drive ngay từ đầu (`drive.mount`) và chỉnh script **lưu checkpoint vào Drive sau mỗi N step** — đây là bảo hiểm duy nhất khi phiên bị ngắt giữa chừng.
4. Đừng để màn hình tắt/idle lâu (~90 phút không động là ngắt); train > vài giờ thì chuyển sang Kaggle chạy nền.

### 🥉 #3 — Oracle Cloud Always Free (host bot 24/7 — nếu qua được cổng đăng ký)
**Lý do:** Đây là đáp án duy nhất cho "server free chạy mãi không ngủ". 2 OCPU ARM + 12GB RAM + 200GB disk thừa cho bot Discord + 1 model mini inference CPU.

1. ⚠️ Chuẩn bị trước: **1 thẻ ngân hàng dùng được cho xác minh quốc tế** (Oracle chỉ verify, không trừ tiền trong hạn free) — không có thẻ thì bỏ qua nhà này, dùng Render.
2. Trên điện thoại: `oracle.com/cloud/free` → đăng ký → **chọn Home Region cẩn thận (không đổi được sau này; chọn region gần như Singapore nếu còn)** → xác minh thẻ.
3. Tạo **VM Ampere A1: 2 OCPU / 12GB, Ubuntu ARM**, lưu kỹ **SSH private key** tải về máy.
4. Cài app **Termux** trên Android để SSH quản trị (console web của OCI trên điện thoại rất khó dùng).
5. Nếu báo **"Out of host capacity"**: thử availability domain khác / thử lại vào giờ thấp điểm — đây là bệnh kinh niên, không phải lỗi của Bon.
6. Sau khi có máy: chạy 1 tác vụ nền nhẹ thường xuyên (bot tự nó đã đủ) để tránh bị thu hồi vì idle <20% suốt 7 ngày; **tuyệt đối không nâng cấp gói** nếu không muốn phát sinh tiền.

> **Phương án thay thế #3 nếu không có thẻ:** dùng **Render Free** (đang dùng sẵn cho bot) cho bot dạng webhook + **Modal** ($30 credit/tháng) cho các job train/inference ngắn do Lead ném hộ. Cặp này không cần thẻ ở bước đầu và không cần SSH.

---

## 4. Mục "đã chết / không còn free như lời đồn" — tránh mất công

| Cái tên | Sự thật 2026 | Bằng chứng |
|---|---|---|
| **Amazon SageMaker Studio Lab** | ❌ **Đóng cửa với khách mới từ 30/07/2026**, vào diện maintenance, không thêm tính năng; chỉ khách cũ dùng tiếp (GPU T4, 4 giờ/ngày). Mọi bài blog "free GPU không cần AWS account" đều đã lỗi thời | Docs chính chủ AWS: *"SageMaker Studio Lab is no longer open to new customers"* + trang AWS Service Availability Updates 06/2026 |
| **Oracle "4 OCPU / 24GB free vĩnh viễn"** | ⚠️ **Đã cắt còn 2 OCPU / 12GB** từ ~15/06/2026, **không thông báo công khai** — mọi guide cũ ghi 24GB đều sai | Docs Oracle hiện hành (1.500 OCPU-giờ / 9.000 GB-giờ) + nhiều nguồn độc lập đối chiếu 09–10/2026 |
| **HF Spaces CPU Basic "free không giới hạn"** | ⚠️ Từ ~06–09/2026 nhiều tài khoản mới tạo/restart Space bị báo **quota = 0**; nhân viên HF trả lời trên forum là Space chạy compute (Gradio/Docker) cần gói trả phí — **docs chưa sửa theo thực tế thi hành** | Thread forum chính chủ HF 27/09/2026 |
| **Lightning "80 free GPU hours mỗi tháng"** | ⚠️ Câu marketing thật nhưng là **gói khởi đầu 1 lần**: 5 credit khi đăng ký (+25 khi gắn thẻ), credit khuyến mãi **hết hạn sau 12 tháng** và không tích luỹ; con số credit/tháng cố định **không được chính chủ cam kết** (các nguồn thứ cấp tự ghi 15 credit/tháng ≈ 22 giờ T4) | Trang pricing qua các nguồn trích dẫn điều khoản chính chủ + docs billing Lightning |
| **Replit "Always On" free** | ❌ Đã bỏ từ lâu: free Starter chỉ có **1 published app và link sập sau 30 ngày**; chạy thường trực (Reserved VM) từ ~$20/tháng | Pricing/docs Replit qua tổng hợp 05–10/2026 |
| **Render free Postgres "90 ngày"** | ⚠️ Giảm còn **30 ngày** từ 20/05/2024 (+14 ngày ân hạn rồi **xoá hẳn**) — ai lưu data quan trọng vào đây là mất | Docs chính chủ Render (Deploy for Free) |
| **Colab "GPU free không giới hạn"** | ❌ Chưa bao giờ đúng: chính chủ ghi rõ free tier **không công bố hạn mức**, GPU "heavily restricted", tối đa 12 giờ/phiên, và **cấm** dùng runtime free để host web service / SSH / đào / chạy worker phân tán | FAQ chính chủ Colab (mục Resource Limits + Restricted activities) |
| **SambaNova Cloud free tier / Hyperbolic serverless free** *(ngoài danh sách, ghi thêm vì hay được blog cũ nhắc)* | ❌ SambaNova bỏ free từ 10/2026 (phải nạp tiền); Hyperbolic đã nghỉ mảng serverless inference năm 2026 | Tổng hợp free-credits 10/2026 |

---

## 5. Ghi chú riêng cho case BonLLM-mini

- BonLLM-mini hiện tại (0,82M tham số, char-level) **đã train được trên CPU 2 nhân** — vì vậy gần như mọi nhà ở bảng trên đều chạy nổi nó; GPU của Kaggle/Colab chỉ giúp rút từ hàng chục phút xuống vài phút, và mở đường cho bản lớn hơn sau này (1–7B + LoRA thì Kaggle/Colab/Modal là bắt buộc).
- Quy trình hợp nhất cho team: **viết/sửa script ở VM này → train thử nhỏ ở Colab → train thật ở Kaggle (chạy nền) → tải checkpoint về → host demo ở HF Spaces (nếu quota cho) hoặc host bot ở Oracle/Render**.
- Không nhà free nào cho phép "train vô hạn + host vô hạn" cùng lúc. Đừng tin bất kỳ bài nào hứa vậy mà không có ngày kiểm chứng trong 2026.

---

## 6. Nguồn (URL nguyên văn)

**Chính chủ:**
- Colab FAQ — https://research.google.com/colaboratory/faq.html
- Kaggle GPU tips (30 giờ/tuần, phiên 9 giờ) — https://www.kaggle.com/page/GPU-tips-and-tricks
- AWS docs: Studio Lab no longer open to new customers — https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-use-shutdown.html
- AWS Service Availability Updates (30/07/2026) — https://aws.amazon.com/about-aws/whats-new/2026/06/aws-service-availability/
- Modal Pricing ($30/tháng Starter) — https://modal.com/pricing
- Render — Deploy for Free (spin down 15', 750 giờ, Postgres 30 ngày) — https://render.com/docs/free
- GitHub Plans (Codespaces 120 core-giờ / 15GB) — https://docs.github.com/en/enterprise-cloud@latest/get-started/learning-about-github/githubs-plans
- HF ZeroGPU docs (free 5 phút/ngày) — https://huggingface.co/docs/hub/en/spaces-zerogpu
- HF Spaces GPU docs — https://huggingface.co/docs/hub/en/spaces-gpus
- HF Forum: cpu-basic quota = 0 cho tài khoản mới (27/09/2026) — https://discuss.huggingface.co/t/free-cpu-basic-quota-is-0-on-new-account/180764
- Lightning Billing docs (Studio CPU free, 10GB storage free) — https://lightning.ai/docs/platform/overview/faq/billing/
- DigitalOcean/Paperspace Machines features (free machine, auto-shutdown 6 giờ, public) — https://docs.digitalocean.com/products/paperspace/machines/details/features/
- DigitalOcean/Paperspace Notebooks features — https://docs.digitalocean.com/products/paperspace/notebooks/details/features/

**Tổng hợp / đối chiếu độc lập (Sep–Oct 2026):**
- Free Cloud GPUs in 2026: What Works and What Closed (21/09/2026) — https://gpuperhour.com/blog/free-cloud-gpus-and-credits
- Free GPU credits for students/researchers (2026) — https://www.aquanode.io/blog/free-gpu-credits-for-students-and-researchers
- Free GPU Cloud Credits: 9 Programs That Changed in 2026 — https://www.spheron.network/blog/free-gpu-cloud-credits-2026/
- Free AI credits that reset daily/monthly (Modal ≈ giờ GPU, ZeroGPU) — https://inferencehub.org/blog/free-ai-credits-that-reset-2026
- Best Free GPU Cloud Options 2026 (Colab/Kaggle/Lightning/Studio Lab) — https://www.gmicloud.ai/en/blog/best-free-gpu-cloud-options-for-ai-startups-and-researchers
- Lightning AI free tier terms (5 + 25 credit, hạn 12 tháng) — https://freetokens.custats.info/offers/lightning-ai-free-tier.html
- Lightning AI pricing calculator (kiểm chứng 09/2026) — https://www.usagepricing.com/tools/pricing-calculator/lightning-ai
- Oracle Cloud Free Tier 2026: Limits and Catches (cắt 15/06/2026, thu hồi idle) — https://www.toolbistro.com/radar/oracle-cloud-free-tier
- Oracle Always Free (09/2026: 2 OCPU/12GB, 200GB, 10TB) — https://site2us.com/en/servers/oracle-cloud-free-tier
- Free Cloud Credits 2026 (Oracle reclaim <20% × 7 ngày) — https://reconscribe.com/free-credits/cloud/
- Is Render Free? (2026) — https://justinmckelvey.com/blog/is-render-free
- Replit pricing 05/2026 (Starter: 1.200 phút, 2GiB, 1 published app) — https://flowstep.ai/blog/replit-review/
- Kaggle phone verification để mở GPU — https://machinelearningmastery.com/using-kaggle-in-machine-learning-projects/
