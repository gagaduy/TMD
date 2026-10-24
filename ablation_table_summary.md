# Tổng hợp Kết quả Thực nghiệm Ablation Study (16k MiT-B0) & Full SOTA Model (30k MiT-B2) trên Tập RTM

## 1. Giao thức Huấn luyện Ablation Study (Table 2 & Table 3)
- **Số bước huấn luyện:** 16,000 iterations trên tập RTM (`train.txt` - 5,801 ảnh)
- **Tập đánh giá:** Tập Validation (`val.txt` - 1,203 ảnh)
- **Backbone:** SegFormer MiT-B0
- **Batch size:** 2
- **Phần cứng:** 1× GPU NVIDIA A100-SXM4-40GB
- **Chế độ đánh giá:** Sliding-window trên ảnh gốc khi valid

### Table 2: Ablation study for each contribution at standard threshold ($\tau = 0.500$)

| OCR | Focus Crop | Prototype | Tamper IoU (%) | mIoU (%) | Precision (%) | Recall (%) | F1 (%) | Train time | VRAM (MB) |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| | | | 0.00 | 49.43 | nan | 0.00 | nan | 41m00s | 2288 |
| ✓ | | | 0.00 | 49.43 | nan | 0.00 | nan | 56m06s | 3040 |
| ✓ | ✓ | | 0.00 | 49.43 | nan | 0.00 | nan | 56m12s | 3088 |
| ✓ | ✓ | ✓ | **0.84** | **49.85** | **53.49** | **0.84** | **1.66** | 56m30s | **2984** |

### Table 3: Ablation study at optimal calibrated threshold ($\tau^*$ peak of each model)

| OCR | Focus Crop | Prototype | Peak $\tau^*$ | Tamper IoU (%) | mIoU (%) | Precision (%) | Recall (%) | F1 (%) |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| | | | 0.050 | 4.20 | 50.63 | 6.24 | 11.40 | 8.06 |
| ✓ | | | 0.040 | 3.77 | 50.18 | 5.24 | **11.84** | 7.27 |
| ✓ | ✓ | | **0.100** | **4.30** | **50.68** | **6.37** | **11.72** | **8.25** |
| ✓ | ✓ | ✓ | **0.150** | **4.43** | **50.99** | **7.37** | 10.01 | **8.49** |

---

## 2. Kết quả Mô hình Hoàn chỉnh Full SOTA (`MiT-B2, 30,000 iters, batch_size = 8`) & So sánh với Paper Gốc

- **File cấu hình:** [`ASCFormer/configs/ascformer/ascformer_rtm_b2_tristream_proto_60k_a100.py`](file:///home/duy/DACN/RTM/ASCFormer/configs/ascformer/ascformer_rtm_b2_tristream_proto_60k_a100.py)
- **Backbone:** SegFormer MiT-B2 (`mit_b2_20220624-66e8bf70.pth`) cho cả 3 nhánh (`RGB + Forensic + OCR`)
- **Quy mô mẫu huấn luyện:** `30,000 iterations × batch_size 8 = 240,000 crops` (~41.4 epochs, bằng **1/4** lượng mẫu `80,000 × 12 = 960,000 crops` của bài báo ASC-Former gốc)
- **Sliding-window inference:** `crop_size = (512, 512), stride = (384, 384)` (bội số của 8 và 32, bảo toàn 100% lưới tần số DCT `8×8`)

### Table 4: Comparison with state-of-the-art methods and original ASC-Former paper on RTM

| Method | Venue | Train Samples | Tamper IoU (%) | Precision (%) | Recall (%) | F1 (%) |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| PSCC-Net | TCSVT 2022 | 960k (80k×12) | 3.31 | 3.59 | **30.28** | 6.41 |
| RRU-Net | CVPRW 2019 | 960k (80k×12) | 3.73 | 15.20 | 4.72 | 7.20 |
| Liang et al. | TrustCom 2022 | 960k (80k×12) | 4.63 | 5.37 | 24.99 | 8.84 |
| MVSS-Net++ | TPAMI 2023 | 960k (80k×12) | 5.11 | 7.34 | 14.41 | 9.73 |
| DTD (DocTamper) | CVPR 2023 | 960k (80k×12) | 6.51 | 11.94 | 12.52 | 12.22 |
| HRNet-OCR | TPAMI 2021 | 960k (80k×12) | 6.81 | 24.15 | 8.67 | 12.76 |
| UperNet | ECCV 2018 | 960k (80k×12) | 8.26 | 32.49 | 9.98 | 15.27 |
| DeepLabV3+ | ECCV 2018 | 960k (80k×12) | 8.56 | 32.24 | 10.43 | 15.76 |
| CAT-Net v2 | IJCV 2022 | 960k (80k×12) | 11.30 | 30.18 | 15.30 | 20.31 |
| Mask2Former | CVPR 2022 | 960k (80k×12) | 12.35 | 19.10 | 25.89 | 21.99 |
| MaskFormer | NeurIPS 2021 | 960k (80k×12) | 13.72 | 26.00 | 22.50 | 24.13 |
| SegFormer (MiT-B2) | NeurIPS 2021 | 960k (80k×12) | 15.72 | 38.45 | 21.01 | 27.17 |
| **ASC-Former (Original Paper)** | **Pattern Recognition 2025** | **960k (80k×12)** | **19.71** | **50.44** | 24.44 | 32.93 |
| **Ours (Full MiT-B2, $\tau=0.500$)** | **Thesis (1× A100)** | **240k (30k×8)** | **19.13** | **50.27** | **23.60** | **32.12** |
| **Ours (Full MiT-B2, $\tau=0.400$)** | **Thesis (1× A100)** | **240k (30k×8)** | **19.84** | 43.55 | **26.71** | **33.11** |
| **Ours (Full MiT-B2, $\tau^*=0.350$)** | **Thesis (1× A100)** | **240k (30k×8)** | **19.91** | 39.85 | **28.46** | **33.20** |
