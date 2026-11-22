# Báo Cáo Đối Chứng Số Liệu Máy Học: Quét Toàn Trang (Full-Scan) vs Dừng Sớm (Early Exit), Ablation M1–M4 & So Sánh MiT-B2 vs MiT-B0

> **Mục tiêu tài liệu:** Cung cấp đầy đủ bằng chứng định lượng máy học (Machine Learning metrics) được đo đạc trực tiếp trên tập dữ liệu chuẩn **RealTextMan (RTM)** và các mô hình triển khai thực tế để chứng minh:
> 1. **Tại sao không được sử dụng Early Exit** trong bài toán giám định tài liệu (bắt buộc **Full-Scan** để phát hiện đầy đủ mọi điểm giả mạo trên trang).
> 2. **Đóng góp tuần tự của 3 cải tiến (Ablation M1–M4)** trên cùng giao thức Full-Scan.
> 3. **Sự đánh đổi toàn diện giữa `MiT-B2` và `MiT-B0` (cũng như `FP32` vs `INT8`)** về độ chính xác, độ phủ đa vùng, số tham số, GFLOPs, sai số lượng tử hóa, RAM và thời gian xử lý khi bắt buộc chạy **Full-Scan (không Early Exit)**.

---

## 1. Bằng chứng từ dữ liệu gốc (`RealTextMan/val.txt` — 1,203 tài liệu): Vì sao Early Exit gây bỏ sót có hệ thống?

Thống kê trực tiếp trên toàn bộ **1,203 ảnh giả mạo** trong tập kiểm định chuẩn `RealTextMan/val.txt` (bao gồm đầy đủ 6 dạng tấn công: `cpmv: 596`, `cover: 207`, `inpaint: 156`, `splice: 123`, `insert: 73`, `edit: 48`) với lưới cửa sổ trượt `512×512` (stride `384`, hoặc `480` nếu cạnh `> 1500` px):

| Chỉ số thống kê dữ liệu (`RealTextMan/val.txt`) | Giá trị thực đo | Ý nghĩa đối với cơ chế quét |
| :--- | :---: | :--- |
| Tổng số ảnh giả mạo (`val.txt`) | **1,203 ảnh** | Tập chuẩn đánh giá SOTA của bài báo gốc ASC-Former |
| Độ phân giải trung bình / trung vị | **988 × 1202 px** / **928 × 1162 px** | Phù hợp kích thước tài liệu chụp / scan phổ thông |
| Số tile `512×512` trên mỗi ảnh (TB / Trung vị / Max) | **8.31** / **8.0** / **20 tiles** | Cần quét nhiều ô mới bao phủ toàn bộ trang giấy |
| Tỷ lệ diện tích pixel bị giả mạo trên toàn tập | **1.1310%** | Bài toán mất cân bằng cực độ (extreme class imbalance) |
| Tổng số vùng giả mạo độc lập *(dilate 15×15 CC, area ≥ 50 px)* | **2,138 vùng** (**1.78 vùng/ảnh**) | Trung bình mỗi tài liệu giả mạo bị sửa gần 2 vị trí |
| **Số ảnh có ≥ 2 vùng giả mạo độc lập** | **582 / 1,203 (48.4%)** | **Gần một nửa tài liệu bị sửa nhiều vị trí tách biệt** |
| **Số ảnh có vết giả mạo nằm trên ≥ 2 tile `512×512`** | **1,058 / 1,203 (87.9%)** | **87.9% tài liệu sẽ bị bỏ sót phần giả mạo còn lại nếu dừng sớm sau tile đầu tiên!** |

**Kết luận phần 1:** Về mặt hình học dữ liệu, cơ chế **Early Exit** (dừng ngay khi tile đầu tiên vượt ngưỡng) chỉ trả lời được bài toán phân loại nhị phân cấp ảnh (*"ảnh có giả hay không"*), hoàn toàn **không đáp ứng** bài toán khoanh vùng đầy đủ (tamper localization), vì có tới **48.4%** tài liệu chứa từ 2 vùng sửa trở lên và **87.9%** tài liệu có vết sửa trải qua từ 2 tile trở lên.

---

## 2. Thực nghiệm đối chứng trực tiếp: `Early Exit` vs `Full-Scan (Không Early Exit)`

### Giao thức thực nghiệm (tái lập 100%, lưu tại [`results/fullscan_vs_earlyexit_benchmark.json`](file:///home/duy/DACN/RTM/results/fullscan_vs_earlyexit_benchmark.json))
- **Tập dữ liệu kiểm định phân tầng (`seed = 3407`):** **30 tài liệu thật** từ `RealTextMan/test.txt`:
  - **18 ảnh giả mạo đa vùng** (chọn đúng 3 ảnh cho mỗi dạng trong 6 dạng tấn công `edit`, `cpmv`, `splice`, `insert`, `inpaint`, `cover`, đều có $\ge 2$ vùng sửa độc lập trải trên $\ge 2$ tile, tổng cộng **40 vùng Ground-Truth**).
  - **12 ảnh nguyên bản** (`good_*`) để đo đầy đủ các chỉ số phân loại cấp ảnh (`Accuracy`, `Specificity`, `FPR`, `FNR`, `ROC-AUC`).
  - Tổng cộng **158 tiles `512×512`** (trung bình **5.20 tiles/ảnh**).
- **Môi trường đo thực tế:** CPU `Intel Core i5-12400F`, ONNX Runtime `1.19.2` (`CPUExecutionProvider`, 8 luồng), cùng thứ tự quét `Smart Sector-Priority Wave Scheduling` và tiền xử lý `ELA (Q=80)` như trong ứng dụng Android.
- **Điều kiện đánh giá Region Hit:** Một vùng GT được tính là phát hiện trúng nếu mask dự đoán giao với vùng GT tối thiểu 10 pixel hoặc $\ge 5\%$ diện tích vùng GT. **Complete-Doc Rate** là tỷ lệ ảnh giả mạo tìm trúng **đủ 100%** các vùng GT trên cùng trang đó.

### Bảng 1: So sánh `Early Exit` vs `Full-Scan (Không Early Exit)` trên `MiT-B2 INT8` và `MiT-B0 INT8`

| Mô hình (ONNX INT8) | Ngưỡng $\tau$ | Chế độ quét | Tamper IoU (%) | mIoU (%) | Pixel Prec (%) | Pixel Recall (%) | Pixel F1 (%) | Pixel FNR (%) ↓ | Region Recall (`Hit/40`) | Đủ 100% vùng/ảnh (`Complete`) | Img Acc (%) | Img F1 (%) | Img FPR (%) ↓ | ROC-AUC (%) | TB Tiles | Độ trễ TB (s) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MiT-B2 INT8** | `0.50` | Early Exit | 6.16 | 52.83 | 29.15 | 7.24 | 11.60 | 92.76 | 10.0% (`4/40`) | 0.00% (`0/18`) | 56.67 | 51.85 | 16.67 | 65.28 | 4.00 | 7.78s |
| **MiT-B2 INT8** | `0.50` | **Full-Scan (No EE)** | **11.17** *(+81.3%)* | **55.32** | **31.28** | **14.80** *(+104.4%)* | **20.10** *(+73.3%)* | **85.20** | **15.0%** (`6/40`) | **5.56%** (`1/18`) | 56.67 | 51.85 | 16.67 | **67.13** | 5.20 | 10.14s |
| **MiT-B2 INT8** | `0.35` | Early Exit | 6.77 | 53.09 | 19.52 | 9.39 | 12.68 | 90.61 | 15.0% (`6/40`) | 5.56% (`1/18`) | 66.67 | 68.75 | 25.00 | 64.35 | 3.60 | 7.01s |
| **MiT-B2 INT8** | `0.35` | **Full-Scan (No EE)** | **10.63** *(+57.0%)* | **54.99** | **21.71** | **17.24** *(+83.6%)* | **19.22** *(+51.6%)* | **82.76** | **17.5%** (`7/40`) | **5.56%** (`1/18`) | 66.67 | 68.75 | 25.00 | **67.13** | 5.20 | 10.14s |
| **MiT-B2 INT8** | `0.20` | Early Exit | 7.87 | 53.60 | 17.30 | 12.63 | 14.60 | 87.37 | 20.0% (`8/40`) | 11.11% (`2/18`) | 70.00 | 75.68 | 41.67 | 67.59 | 3.00 | 5.85s |
| **MiT-B2 INT8** | `0.20` | **Full-Scan (No EE)** | **8.79** *(+11.7%)* | **53.94** | 13.91 | **19.26** *(+52.5%)* | **16.16** *(+10.7%)* | **80.74** | **25.0%** (`10/40`) | **16.67%** (`3/18`) | **70.00** | **75.68** | 41.67 | 67.13 | 5.20 | 10.14s |
| **MiT-B0 INT8** | `0.50` | Early Exit | 0.00 | 49.70 | 0.00 | 0.00 | 0.00 | 100.00 | 0.0% (`0/40`) | 0.00% (`0/18`) | 43.33 | 26.09 | 16.67 | 50.00 | 4.80 | 5.11s |
| **MiT-B0 INT8** | `0.50` | **Full-Scan (No EE)** | **0.25** | **49.80** | **0.84** | **0.36** | **0.50** | **99.64** | **2.5%** (`1/40`) | 0.00% (`0/18`) | 43.33 | 26.09 | 16.67 | **50.46** | 5.20 | 5.54s |
| **MiT-B0 INT8** | `0.20` | Early Exit | 2.85 | 50.98 | 5.35 | 5.75 | 5.54 | 94.25 | 2.5% (`1/40`) | 0.00% (`0/18`) | 46.67 | 42.86 | 33.33 | 48.61 | 4.20 | 4.48s |
| **MiT-B0 INT8** | `0.20` | **Full-Scan (No EE)** | 2.29 | 50.50 | 3.39 | **6.61** *(+15.0%)* | 4.48 | **93.39** | **5.0%** (`2/40`, **+100%**) | 0.00% (`0/18`) | 46.67 | 42.86 | 33.33 | **50.46** | 5.20 | 5.54s |

### Nhận xét định lượng rút ra từ Bảng 1:
1. **Chứng minh yêu cầu bỏ Early Exit của giảng viên là hoàn toàn chính xác:**
   - Trên **`MiT-B2 INT8` tại ngưỡng chuẩn $\tau = 0.50$**, việc tắt Early Exit sang **Full-Scan** giúp **Pixel Recall tăng gấp đôi (+104.4%, từ `7.24%` lên `14.80%`)**, **Tamper IoU tăng +81.3% (`6.16% → 11.17%`)**, **Pixel F1 tăng +73.3% (`11.60% → 20.10%`)**, đồng thời tăng cả **Precision (`29.15% → 31.28%`)** lẫn **Region Recall (`10.0% → 15.0%`)** và **ROC-AUC (`65.28% → 67.13%`)**.
   - Tại ngưỡng $\tau = 0.20$, Full-Scan nâng **Region Recall từ `20.0%` (`8/40`) lên `25.0%` (`10/40`)** và tăng số tài liệu tìm đủ 100% vùng giả mạo (**Complete-Doc Rate**) từ `11.11%` lên **`16.67%` (+50%)**.
   - Điều đáng chú ý: với `Early Exit`, tỷ lệ nhận diện cấp ảnh (`Img Acc`, `Img Recall`, `Img F1`) gần như không đổi, nhưng **chất lượng khoanh vùng pixel và độ phủ đa vùng sụt giảm nghiêm trọng** (Pixel Recall mất từ 34% đến 51%).
2. **Chi phí đánh đổi khi bỏ Early Exit:**
   - Thời gian quét trung bình tăng từ `5.85–7.78s` lên **`10.14s`** (với `MiT-B2 INT8`) và từ `4.48–5.11s` lên **`5.54s`** (với `MiT-B0 INT8`) trên ảnh trung bình `5.2` tiles — đây chính là lý do phải phân tích kỹ bài toán đánh đổi **MiT-B2 vs MiT-B0** ở Mục 4 bên dưới.

---

## 3. Bảng tổng kết Ablation Study M1–M4 (Full-Scan trên 1,203 ảnh `RealTextMan/val.txt`)

Tất cả 4 mô hình Ablation đều được đánh giá theo **Full-Scan Sliding-Window (không Early Exit)** trên toàn bộ **1,203 ảnh** `val.txt` (`MiT-B0`, `16,000 iters`, `batch_size = 2`, 1× NVIDIA A100-40GB). Dữ liệu gốc tại [`results/ablation16k_b0/`](file:///home/duy/DACN/RTM/results/ablation16k_b0/).

### Bảng 2: Ablation Study M1–M4 tại ngưỡng chuẩn ($\tau = 0.500$) và ngưỡng tối ưu ($\tau^*$)

| Phiên bản | OCR | Focus Crop | Prototype | @ $\tau = 0.500$ Tamper IoU (%) | @ $\tau = 0.500$ Precision (%) | @ $\tau = 0.500$ F1 (%) | @ $\tau = 0.500$ mIoU (%) | Peak $\tau^*$ | @ $\tau^*$ Tamper IoU (%) | @ $\tau^*$ Precision (%) | @ $\tau^*$ Recall (%) | @ $\tau^*$ F1 (%) | @ $\tau^*$ mIoU (%) | Train Time | VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** (Baseline 2-Stream) | | | | 0.00 | 0.00 | 0.00 | 49.43 | 0.050 | 4.20 | 6.24 | 11.40 | 8.06 | 50.63 | 41m00s | 2,288 |
| **M2** (+Tri-Stream OCR) | ✓ | | | 0.00 | 0.00 | 0.00 | 49.43 | 0.040 | 3.77 | 5.24 | **11.84** | 7.27 | 50.18 | 56m06s | 3,040 |
| **M3** (+Focused Crop) | ✓ | ✓ | | 0.00 | 0.00 | 0.00 | 49.43 | 0.100 | 4.30 | 6.37 | 11.72 | 8.25 | 50.68 | 56m12s | 3,088 |
| **M4** (+Dual Prototype) | ✓ | ✓ | ✓ | **0.84** | **53.49** | **1.66** | **49.85** | **0.150** | **4.43** | **7.37** | 10.01 | **8.49** | **50.99** | 56m30s | **2,984** |

- **M2 (+OCR):** Nâng độ phủ Recall lên cao nhất bảng (`11.84% > 11.40%`), bổ sung dẫn đường không gian văn bản.
- **M3 (+Focus Crop):** Đẩy ngưỡng tự tin cực đại $\tau^*$ tăng **2.5×** (`0.040 → 0.100`), vượt Baseline trên toàn bộ IoU/Precision/Recall/F1.
- **M4 (+Dual Prototype):** Là cấu hình **duy nhất** phá vỡ trạng thái sụp đổ dự đoán tại ngưỡng chuẩn $\tau = 0.500$ (`Precision = 53.49%`, `Tamper IoU = 0.84%`), đạt đỉnh toàn bảng 16k ở $\tau^* = 0.150$ (`IoU = 4.43%`, `F1 = 8.49%`, `mIoU = 50.99%`) và **tiết kiệm 104 MB VRAM** (`3,088 → 2,984 MB`) so với Memory Bank.

---

## 4. So sánh toàn diện `MiT-B2` vs `MiT-B0` (và `FP32` vs `INT8`) ở chế độ Bắt Buộc Full-Scan

### Bảng 3: Độ chính xác chuẩn trên toàn tập `RealTextMan/val.txt` (1,203 ảnh, Full-Scan Sliding-Window, PyTorch FP32 + OCR)

| Mô hình / Phương pháp | Backbone | Mẫu train (`iters × bs`) | Ngưỡng $\tau$ | Tamper IoU (%) | Bg IoU (%) | mIoU (%) | Precision (%) | Recall (%) | F1 / Dice (%) | Train VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ASC-Former *(Paper gốc, PR 2025)* | MiT-B2 (2 luồng) | 960,000 (`80k × 12`) | `0.500` | 19.71 | ~98.5 | ~59.10 | **50.44** | 24.44 | 32.93 | Multi-GPU |
| **Ours — MiT-B0 30k Mobile** *(Best Iter 20k)* | **MiT-B0 (3 luồng)** | 120,000 (`30k × 4`) | `0.500` | 4.11 | 98.76 | 51.44 | 25.06 | 4.69 | 7.90 | **5,726 MB** |
| **Ours — MiT-B2 30k SOTA** *(Best Iter 30k)* | **MiT-B2 (3 luồng)** | 240,000 (`30k × 8`) | `0.500` | 19.13 | **98.87** | 59.00 | 50.27 | 23.60 | 32.12 | 22,950 MB |
| **Ours — MiT-B2 30k SOTA** *(Best Iter 30k)* | **MiT-B2 (3 luồng)** | 240,000 (`30k × 8`) | `0.400` | 19.84 | 98.78 | **59.31** | 43.55 | 26.71 | 33.11 | 22,950 MB |
| **Ours — MiT-B2 30k SOTA** *(Peak $\tau^*$)* | **MiT-B2 (3 luồng)** | **240,000 (`30k × 8`)** | **`0.350`** | **19.91** | 98.70 | 59.30 | 39.85 | **28.46** | **33.20** | 22,950 MB |

*(Nguồn log gốc: [`results/full30k_b2_sota/Best_mIoU_Ckpt_sweep.json`](file:///home/duy/DACN/RTM/results/full30k_b2_sota/Best_mIoU_Ckpt_sweep.json) và [`ASCFormer/work_dirs/b0_30k_sota_mobile/20261026_134024/20261026_134024.log`](file:///home/duy/DACN/RTM/ASCFormer/work_dirs/b0_30k_sota_mobile/20261026_134024/20261026_134024.log).)*

---

### Bảng 4: Độ phức tạp tính toán, Dung lượng, Sai số lượng tử hóa & Thời gian `Full-Scan` (Đo độc lập trên CPU)

Đo trực tiếp trên `Intel Core i5-12400F`, ONNX Runtime `1.19.2` (`CPUExecutionProvider`, từng mô hình chạy trong tiến trình cô lập để đo chính xác `Peak RSS RAM`; thời gian tile trung bình trên 5 lần chạy sau warmup; `Params` & `GFLOPs` đo bằng PyTorch `FlopCounterMode` trên `OnDeviceRTMModel` đầu vào `512×512`).

| Chỉ tiêu đo đạc (On-Device Full-Scan) | `MiT-B2 FP32` | `MiT-B2 INT8` | `MiT-B0 FP32` | `MiT-B0 INT8` | Chênh lệch `B0 INT8` vs `B2 INT8` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tổng tham số PyTorch (`Params`)** | 123.07M | 123.07M | 25.84M | **25.84M** | **Giảm 4.76× (-79.0%)** |
| **Tham số trong graph ONNX (`Initializers`)** | 121.92M | 121.92M | 25.55M | **25.55M** | **Giảm 4.77× (-79.0%)** |
| **Độ phức tạp tính toán (`GFLOPs / tile 512×512`)** | 384.75 | 384.75 | 231.48 | **231.48** | **Giảm 1.66× (-39.8%)** |
| **Dung lượng file ONNX trên đĩa (`MB`)** | 524.44 MB | 192.92 MB | 129.22 MB | **60.78 MB** | **Nhẹ hơn 3.17× (-68.5%)** *(8.63× vs B2 FP32)* |
| **Số node toán tử ONNX** | 28,514 | 31,097 | 25,525 | 27,304 | Giảm 12.2% số node |
| **Thời gian nạp model (`Load Time`, 4T)** | 2,857.7 ms | 3,301.9 ms | **2,353.2 ms** | 3,373.7 ms | Tương đương (~3.3s lúc khởi động 1 lần) |
| **Độ trễ 1 tile `512×512` — CPU 4 luồng (ms)** | 2,036.0 ± 13.3 | 1,904.8 ± 58.7 | **1,097.8 ± 57.0** | 1,140.8 ± 73.8 | **Nhanh hơn 1.67×** *(4 luồng)* |
| **Độ trễ 1 tile `512×512` — CPU 8 luồng (ms)** | 2,014.9 ± 9.0 | 1,864.2 ± 62.7 | 1,188.5 ± 42.3 | **955.1 ± 59.7** | **Nhanh hơn 1.95×** *(8 luồng)* |
| **RAM đỉnh khi suy luận (`Peak RSS`, 4T)** | 1,321.0 MB | 1,181.9 MB | **701.4 MB** | 808.6 MB | **Tiết kiệm 31.6% RAM** *(38.8% vs B2 FP32)* |
| **Sai số lượng tử hóa `FP32→INT8` (`MAE`)** *(5 tiles = 1.31M px)* | 0 *(chuẩn)* | 0.00407 | 0 *(chuẩn)* | **0.00127** | B0 giữ ổn định phân bố xác suất tốt hơn |
| **Sai số lượng tử hóa `FP32→INT8` (`RMSE`)** | 0 *(chuẩn)* | 0.01097 | 0 *(chuẩn)* | **0.00334** | Sai số bậc 2 nhỏ hơn **3.28×** |
| **Tương quan xác suất `FP32→INT8` (`Pearson r`)** | 1.00000 | 0.97830 | 1.00000 | **0.99598** | **Bảo toàn 99.60% bản đồ xác suất** |
| **Độ lệch lớn nhất `FP32→INT8` (`MaxDiff`)** | 0.0000 | 0.1791 | 0.0000 | **0.0416** | Ít dao động điểm cực trị hơn |

> **Lưu ý kỹ thuật về `rtm_b0_fp16.onnx`:** Qua kiểm định trực tiếp bằng ONNX Runtime CPU `1.19.2`, file `rtm_b0_fp16.onnx` (`68.83 MB`) bị lỗi không tương thích kiểu dữ liệu tại nút `/extra_score_predictor.0/Cast` (`float16` vs `float32`), nên **không khuyến nghị đưa vào bảng triển khai CPU**; các phiên bản hoạt động ổn định đã kiểm chứng là `FP32` và `INT8`.

---

### Bảng 5: Thời gian quét `Full-Scan` (KHÔNG Early Exit) theo kích thước tài liệu thực tế

*(Quy đổi trực tiếp từ độ trễ thực đo 1 tile ở Bảng 4 — không dùng Early Exit, quét 100% số tile trên trang)*:

| Kích thước tài liệu (ước lượng) | Số tile `512×512` | `MiT-B2 INT8` Full-Scan (8T / 4T) | `MiT-B0 INT8` Full-Scan (8T / 4T) | Thời gian tiết kiệm được (8T) |
| :--- | :---: | :---: | :---: | :---: |
| Ảnh nhỏ (~700–900 px) | **4 tiles** | 7.46s / 7.62s | **3.82s / 4.56s** | **-3.64s (nhanh hơn 1.95×)** |
| Tập benchmark 30 ảnh *(thực đo trực tiếp)* | **5.2 tiles (TB)** | **10.14s** *(Median 11.62s, P95 11.95s)* | **5.54s** *(Median 6.31s, P95 6.50s)* | **-4.60s (nhanh hơn 1.83×)** |
| Trang trung vị tập RTM Val (~928×1162 px) | **8 tiles** | 14.91s / 15.24s | **7.64s / 9.13s** | **-7.27s (nhanh hơn 1.95×)** |
| Trang A4 sắc nét (~1500 px) | **12 tiles** | 22.37s / 22.86s | **11.46s / 13.69s** | **-10.91s (nhanh hơn 1.95×)** |
| Trang scan lớn (~2000 px) | **16 tiles** | 29.83s / 30.48s | **15.28s / 18.25s** | **-14.55s (nhanh hơn 1.95×)** |

---

## 5. Tổng hợp lập luận khoa học & Hướng giải quyết tối ưu khi bỏ Early Exit

Dựa trên toàn bộ số liệu đo đạc thực nghiệm ở trên, có **3 kết luận then chốt** để trình bày và bảo vệ trước hội đồng/giảng viên hướng dẫn:

1. **Bắt buộc bỏ Early Exit — dùng 100% Full-Scan là đúng về mặt nghiệp vụ và máy học:**
   - Trong tập RTM Val, **48.4%** tài liệu có $\ge 2$ vùng giả mạo và **87.9%** tài liệu có vết sửa trải trên $\ge 2$ tile.
   - Thực nghiệm đối chứng chứng minh tắt Early Exit sang Full-Scan giúp **Pixel Recall tăng từ +52.5% đến +104.4%** và **Tamper IoU tăng tới +81.3%** trên `MiT-B2 INT8`, không để lọt các vị trí sửa đổi thứ 2, thứ 3 trên cùng văn bản.

2. **Đánh đổi khách quan giữa `MiT-B2` và `MiT-B0` dưới chế độ Full-Scan:**
   - **Về độ chính xác định vị đa vùng:** **`MiT-B2` vượt trội hoàn toàn** so với `MiT-B0` (trên 1,203 ảnh Val Full-Scan + OCR: `Tamper IoU = 19.13–19.91%` vs `4.11%` — **cao gấp 4.65×**; trên 30 ảnh ONNX INT8 Full-Scan: `Tamper IoU = 11.17%` vs `2.29%` — **cao gấp 4.88×**, `Region Recall = 25.0%` vs `5.0%` @ $\tau=0.20$, `ROC-AUC = 67.13%` vs `50.46%`). Lý do là kiến trúc `MiT-B0` chỉ có `25.84M` tham số (bằng 1/4.76 của `MiT-B2` `123.07M`), đồng thời bản ONNX On-Device không dùng PaddleOCR ngoài mà dùng bộ lọc hình thái nội bộ `SelfGuidedOCRPriors`.
   - **Về tài nguyên và độ trễ Full-Scan trên thiết bị biên:** **`MiT-B0 INT8` nhẹ hơn 3.17× dung lượng** (`60.78 MB` vs `192.92 MB`), **giảm 31.6% RAM** (`809 MB` vs `1,182 MB`), **lượng tử hóa gần như không mất thông tin** (`Pearson r = 0.996`, `MAE = 0.00127`) và **quét Full-Scan nhanh gần gấp đôi (~1.83–1.95×)** (`5.54s` vs `10.14s` cho 5.2 tiles; `7.64s` vs `14.91s` cho 8 tiles).

3. **Chiến lược triển khai không dùng Early Exit nhưng vẫn giảm thời gian xử lý (đúng yêu cầu thầy):**
   - **Tắt hoàn toàn Early Exit** trong ứng dụng di động (quét đủ 100% các tile chứa văn bản để tìm mọi điểm giả mạo).
   - **Kiến trúc 2 tầng rõ ràng:**
     - **Chế độ Chính xác cao (Server/GPU hoặc Mobile High-Precision Full-Scan):** Dùng **`MiT-B2`** (đạt SOTA `19.91% Tamper IoU / 33.20% F1` với chỉ 25% ngân sách train so với bài báo gốc ASC-Former).
     - **Chế độ Siêu nhẹ Offline (Mobile Fast Full-Scan):** Dùng **`MiT-B0 INT8`** (`60.78 MB`) cho thiết bị cấu hình thấp.
   - **Các kỹ thuật tăng tốc Full-Scan không làm giảm Recall (thay thế Early Exit):**
     1. **Bỏ qua ô giấy trắng tuyệt đối (Blank-Tile Skipping):** Đã tích hợp sẵn ngưỡng năng lượng biên / tỷ lệ mực (`energy < 0.8 && ink_ratio < 0.002`) — chỉ bỏ qua lề trắng không có chữ, giữ nguyên 100% các ô chứa văn bản.
     2. **Xử lý song song nhiều tile (Multi-Worker / Batch Parallel Inference):** Thay vì 1 luồng chạy tuần tự từng tile, phân chia tập tile cho nhiều worker/session chạy song song hoặc tận dụng **NNAPI / GPU Delegate** trên di động để giảm thời gian Full-Scan mà không cắt ngang quá trình tìm kiếm.
