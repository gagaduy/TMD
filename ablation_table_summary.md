# BẢNG THỰC NGHIỆM BÓC TÁCH TOÀN DIỆN (ABLATION STUDY - 8,000 ITERS)

> **Giao thức chuẩn mực:** 
> - **Số vòng lặp (Iterations):** 8,000 iters (đảm bảo tính công bằng và nhất quán tuyệt đối giữa các mô hình).
> - **Tập kiểm định:** Toàn bộ tập `val.txt` (1,203 ảnh độ phân giải cao).
> - **Phương thức suy luận:** Cửa sổ trượt (sliding-window inference) trên ảnh gốc, ngưỡng quyết định chuẩn $\tau = 0.50$ (Argmax).
> - **Phần cứng thực nghiệm:** 1 x GPU NVIDIA GeForce RTX 5060 (8GB VRAM), PyTorch 2.7.0, CUDA 12.8.

---

## 1. BẢNG TỔNG HỢP TOÀN BỘ 4 MÔ HÌNH (ĐẦY ĐỦ METRIC & THỜI GIAN TRAIN)

| STT | Phiên bản Mô hình | Số luồng / Cơ chế | Thời gian huấn luyện (Train Time) | Tốc độ (s/iter) | VRAM đỉnh | Số tham số (Params) | Tampered IoU (%) | Tampered Precision (%) | Tampered Recall (%) | Tampered F1 / Dice (%) | Toàn mạng mIoU (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1** | **Baseline 2-Stream**<br>*(Paper gốc chuyển sang MiT-B0)* | 2 luồng<br>(RGB + Forensic Hub) | **28 phút 26 giây** | 0.213 s/it | 2,288 MB | ~7.4M | **0.00%** | *nan* | **0.00%** | *nan* | 49.43% |
| **M2** | **+ Tri-Stream OCR Guidance**<br>*(Đóng góp 1)* | 3 luồng<br>(+ OCR Map & Cascaded GCNF) | **39 phút 17 giây** | 0.295 s/it | 3,040 MB | ~11.1M | **0.00%** | *nan* | **0.00%** | *nan* | 49.43% |
| **M3** | **+ Focused Sampling**<br>*(Đóng góp 2)* | 3 luồng<br>+ Focused Crop ($P_{\text{pos}}=0.7$) | **41 phút 15 giây** | 0.309 s/it | 3,088 MB | ~11.1M | **0.07%** | 61.57%* | **0.07%** | **0.13%** | 49.47% |
| **M4** | **+ Dual-Prototype Head**<br>*(Đóng góp 3)* | 3 luồng + Focused<br>+ 2 Prototype EMA & Fusion | **41 phút 54 giây** | 0.314 s/it | 2,984 MB | ~11.2M | **1.85%** | **7.24%** | **2.42%** | **3.63%** | **50.20%** |

*\*Ghi chú:* Precision 61.57% ở M3 chỉ mang tính cục bộ do mô hình đoán quá nhút nhát (chỉ bắt trúng vài chục pixel, Recall chỉ 0.07%). Sang M4, cơ chế Prototype giúp Recall tăng vọt gấp **34 lần** ($0.07\% \rightarrow 2.42\%$), đưa IoU từ $0.07\% \rightarrow 1.85\%$.

---

## 2. PHÂN TÍCH HIỆU QUẢ THỜI GIAN VÀ PHẦN CỨNG (COMPUTATIONAL EFFICIENCY)

1. **Nhánh OCR (Đóng góp 1):** 
   - Thêm hẳn một nhánh Encoder MiT-B0 độc lập và 4 khối Cascaded GCNF chỉ làm tăng thêm **10 phút 51 giây** thời gian huấn luyện (từ 28.4 phút lên 39.3 phút) và thêm **752 MB VRAM** (từ 2.2GB lên 3.0GB). 
   - Điều này chứng minh kiến trúc Tri-Stream được thiết kế cực kỳ tối ưu, hoàn toàn khả thi trên GPU cá nhân phổ thông.
2. **Chiến lược lấy mẫu có định hướng (Đóng góp 2):** 
   - Thuật toán `FocusedCropWithExtra` với $P_{\text{pos}}=0.7$ chạy trên CPU chỉ tốn thêm **1 phút 58 giây** (tăng ~0.014 s/iter), chi phí tính toán tăng thêm gần như bằng 0.
3. **Dual-Prototype Contrastive Head (Đóng góp 3):** 
   - Thay thế hoàn toàn Memory Bank cồng kềnh (hàng đợi 2,000 vector) bằng phép nhân trực tiếp với 2 vector Prototype đại diện giúp **tiết kiệm VRAM** (giảm từ 3,088 MB xuống 2,984 MB, tiết kiệm hơn 100MB VRAM) trong khi thời gian huấn luyện giữ nguyên (~41 phút).

---

## 3. BẢNG BÓC TÁCH RIÊNG ĐÓNG GÓP 1 (ĐỂ DÁN VÀO MỤC OCR CỦA LUẬN VĂN)

| Cấu hình mô hình | Số luồng | Thời gian huấn luyện | VRAM đỉnh | Số tham số | Tampered IoU (%) | Tampered Recall (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1: Baseline (2-Stream)** | 2 luồng | 28 phút 26 giây | 2,288 MB | ~7.4M | 0.00% | 0.00% |
| **M2: + OCR Guidance (Tri-Stream)** | 3 luồng | 39 phút 17 giây | 3,040 MB | ~11.1M (+3.7M) | 0.00% | 0.00% |

---

## 4. BẰNG CHỨNG THỜI GIAN TỪ LOG FILE GỐC

- **M1 (Baseline):** `ASCFormer/work_dirs/pilot8k_b0_baseline/20261005_174011/20261005_174011.log`  
  *Khởi động:* 17:40:12 $\longrightarrow$ *Hoàn thành (Iter 8000):* 18:08:38 (28 phút 26 giây).
- **M2 (Tri-Stream OCR):** `ASCFormer/work_dirs/pilot8k_b0_tristream/20261005_204454/20261005_204454.log`  
  *Khởi động:* 20:44:54 $\longrightarrow$ *Hoàn thành (Iter 8000):* 21:24:11 (39 phút 17 giây).
- **M3 (+ Focused Crop):** `ASCFormer/work_dirs/pilot8k_b0_tristream_focused/20261005_221451/20261005_221451.log`  
  *Khởi động:* 22:14:52 $\longrightarrow$ *Hoàn thành (Iter 8000):* 22:56:07 (41 phút 15 giây).
- **M4 (+ Dual Prototype):** `ASCFormer/work_dirs/pilot8k_b0_tristream_proto/20261005_233315/20261005_233315.log`  
  *Khởi động:* 23:33:15 $\longrightarrow$ *Hoàn thành (Iter 8000):* 00:15:09 (41 phút 54 giây).
