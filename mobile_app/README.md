# RTM Text Forensic - Mobile App & Web Simulator

Hệ thống giám định và phát hiện giả mạo văn bản tài liệu trên thiết bị di động (On-Device Text Tamper Detection) dựa trên mô hình **MiT-B0 SOTA (ONNX INT8)**.

---

## 🌟 Điểm nổi bật & Đóng góp kỹ thuật

1. **Kiến trúc On-Device siêu nhẹ:**
   - Sử dụng backbone Transformer MiT-B0 kết hợp Tri-Stream Feature Fusion (RGB + ELA + OCR Priors).
   - Mô hình được lượng tử hóa động INT8 chỉ **60.8 MB**, chạy hoàn toàn offline không cần kết nối Internet hay máy chủ ngoài.

2. **Thuật toán định tuyến ô thông minh (Smart Sector-Priority Wave Scheduling):**
   - Phân chia tài liệu thành lưới ngữ cảnh 9 vùng không gian ($3 \times 3$ Sectors).
   - Thay vì quét tuần tự từ trên xuống dưới hoặc ưu tiên nhầm các vùng tiêu đề/logo in đậm, hệ thống ưu tiên quét làn sóng (Wave Priority) tại các khu vực dễ bị làm giả nhất:
     - **Sector 4 (Thân văn bản):** Đoạn văn, điều khoản hợp đồng.
     - **Sector 2 & 5 (Cột phải & phía trên):** Số tiền thanh toán, số hợp đồng, ngày tháng năm, tên người nhận.
     - **Sector 8 (Góc phải dưới):** Chữ ký, con dấu, cơ quan thẩm quyền.
   - **Early Exit thông minh:** Dừng ngay lập tức khi phát hiện điểm sửa đổi đạt độ tin cậy cao ($\ge 20\%$, diện tích $\ge 48$ px), rút ngắn thời gian phát hiện từ 7–16s xuống chỉ còn **1–3 giây**.

3. **Tăng tốc phần cứng di động (Hardware Acceleration):**
   - **NNAPI Execution Provider:** Tự động điều phối các phép toán sang NPU/GPU/DSP chuyên dụng trên các dòng SoC di động (Snapdragon, Dimensity, Tensor).
   - **XNNPACK Execution Provider:** Tối ưu hóa tập lệnh vi kiến trúc đa luồng ARM NEON cho các toán tử Conv/GEMM trên CPU di động.

---

## 📂 Cấu trúc thư mục `mobile_app/`

```
mobile_app/
├── android_app/               # Mã nguồn ứng dụng Android (Java + WebView UI)
│   ├── app/
│   │   ├── build.gradle       # Cấu hình ONNX Runtime Android & packaging
│   │   └── src/main/
│   │       ├── AndroidManifest.xml
│   │       ├── assets/
│   │       │   ├── index.html # Giao diện người dùng Mobile hiện đại
│   │       │   └── samples/   # Ảnh mẫu kiểm tra tích hợp
│   │       └── java/com/rtm/tamperdetector/
│   │           ├── MainActivity.java      # Pipeline xử lý ảnh, ELA, ONNX & Early Exit
│   │           └── LocalFileProvider.java # Provider mở file camera/gallery
│   ├── build.gradle
│   └── settings.gradle
├── run_simulator.py           # Web Simulator chạy mô phỏng on-device cục bộ
├── export_ondevice_model.py   # Script trích xuất và tối ưu hóa mô hình sang ONNX INT8
├── realtext_v2_samples/       # Bộ mẫu kiểm thử độ phân giải cao và mask ground-truth
├── test_images_pack/          # Tập ảnh văn bản giả mạo thực tế
└── demo_outputs/              # Ảnh minh họa kết quả Bounding Box & Forensic Heatmap
```

---

## 🚀 Hướng dẫn sử dụng

### 1. Khởi chạy Web Simulator
Simulator sử dụng đúng pipeline tiền xử lý, thuật toán định tuyến ô và giao diện như trên ứng dụng Android:
```bash
python mobile_app/run_simulator.py --port 8080
```
Truy cập qua trình duyệt: `http://localhost:8080`

### 2. Biên dịch Android APK
Yêu cầu JDK 17 và Android SDK (Compile SDK 34):
```bash
cd mobile_app/android_app
./gradlew assembleDebug
```
File APK thành phẩm sẽ nằm tại `app/build/outputs/apk/debug/app-debug.apk`.
