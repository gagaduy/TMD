# BỘ MẪU KIỂM THỬ REALTEXT-V2 (ZERO-SHOT CROSS-DATASET EVALUATION)

Tập dữ liệu **RealText-V2** (ACM MM 2026 MGC GenText-Forensics Challenge) là tập dữ liệu văn bản thực tế độc lập hoàn toàn mà mô hình RTM **chưa từng được huấn luyện qua** (Zero-shot generalization).

---

## 📁 Cấu trúc thư mục

* **`anh_test/`**: 4 ảnh kích thước chuẩn $512 \times 512$ px (được cắt tập trung vào khu vực quan trọng).
  * Nạp vào điện thoại quét cực nhanh chỉ mất **~1.5 - 2 giây** / ảnh.
* **`dap_an_mask/`**: Mặt nạ nhị phân (Ground-Truth Mask) tương ứng trích xuất từ dataset chuẩn để đối chiếu.
* **`anh_goc_full/`**: Ảnh scan văn bản gốc nguyên khổ A4 ($2550 \times 3300$ px) kèm mask gốc.

---

## 🧪 Chi tiết từng mẫu & Kết quả kiểm thử thực tế của mô hình

| Tên file test | Loại tài liệu | Hành vi giả mạo trong tài liệu | Kết quả mô hình RTM On-Device |
| :--- | :--- | :--- | :--- |
| **`sample_1_GenText_Forensic_00000002.png`** | Hợp đồng bảo hiểm / Văn bản pháp lý (*Insurance Policy & Legal Citation*) | Sửa đổi số hợp đồng (Header Policy Number) và ngày tháng có hiệu lực (Date manipulation). | **PHÁT HIỆN GIẢ MẠO (Độ tin cậy 100.0%)**<br>Khoanh trúng 1 vùng giả mạo lớn. |
| **`sample_2_GenText_Forensic_00000008.png`** | Đơn đăng ký học phần & Thanh toán học phí (*Course Enrollment & Payment Form*) | Xóa tên giảng viên (Instructor name redaction) và đổi hình thức thanh toán (Payment type change). | **PHÁT HIỆN GIẢ MẠO (Độ tin cậy 100.0%)**<br>Bắt trúng 45.157 pixels giả mạo. |
| **`sample_3_GenText_Forensic_00000010.png`** | Biểu mẫu hồ sơ tài chính / Thời hạn vay (*Financial Application / Terms Table*) | Sửa các ô thời hạn vay (Term), số năm (Years) trong bảng biểu. | **PHÁT HIỆN GIẢ MẠO (Độ tin cậy 98.3%)**<br>Khoanh trúng 3 Bounding Boxes. |
| **`sample_4_GenText_Forensic_00000006_authentic.png`** | Phần văn bản gốc không bị can thiệp (*Authentic Control Sample*) | Vùng văn bản nguyên bản 100% không có pixel nào bị chỉnh sửa. | **ẢNH NGUYÊN BẢN (Độ tin cậy 98.6%)**<br>Tỷ lệ báo động giả = 0%. |

---

## 📱 Cách test trên điện thoại

1. Chép thư mục `anh_test/` vào điện thoại (hoặc gửi qua Telegram/Zalo ở chế độ **Gửi file tài liệu / Không nén**).
2. Mở ứng dụng **RTM Tamper Detector**, bấm **"Chọn ảnh từ máy"** và chọn 1 trong 4 ảnh trên.
3. Xem Tab 1 (Khoanh vùng Bounding Box đỏ) và Tab 2 (Bản đồ nhiệt nóng rực tại vùng can thiệp).
