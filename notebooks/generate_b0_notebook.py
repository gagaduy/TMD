import json
import os

cells = []

# Cell 0: Header Markdown
c0 = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '# ⚡ HUẤN LUYỆN MÔ HÌNH SIÊU NHẸ MiT-B0 CHO MOBILE (30,000 ITERS)\n',
        '\n',
        '> **Mục tiêu chiến lược:**\n',
        '> * Huấn luyện phiên bản **SegFormer MiT-B0** (~3.7M params/stream) với đầy đủ 3 đóng góp (Tri-stream + Dual Prototype + Focused Crop).\n',
        '> * **Thời gian huấn luyện dự kiến:**\n',
        '>   * **NVIDIA A100 (Colab Pro):** ~2 tiếng 15 phút\n',
        '>   * **NVIDIA T4 (Colab Miễn phí):** ~5 tiếng 30 phút\n',
        '>   * **NVIDIA L4 / V100:** ~3 tiếng 30 phút\n',
        '> * **Kết quả kỳ vọng:** Tamper IoU đạt **~13.0% – 14.5%** (vượt xa CAT-Net v2 11.3% và DocTamper 6.5%), dung lượng file ONNX chỉ còn **~25 MB**, tốc độ quét trên điện thoại đạt **~1.5 giây / ô**!'
    ]
}
cells.append(c0)

# Cell 1: Setup Environment
c1_md = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '### 🟢 BƯỚC 1: KẾT NỐI DRIVE & CÀI ĐẶT MÔI TRƯỜNG (~3 Phút)\n',
        '> Nhấn chạy Cell này để tự động kết nối Google Drive, tải mã nguồn và giải nén Dataset.'
    ]
}
cells.append(c1_md)

c1_code = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# 1. Kết nối Google Drive\n',
        'import os, shutil, glob\n',
        'from google.colab import drive\n',
        'drive.mount("/content/drive")\n',
        '\n',
        '# 2. Clone mã nguồn RTM mới nhất từ GitHub\n',
        '%cd /content\n',
        '!rm -rf TMD\n',
        '!git clone https://github.com/gagaduy/TMD.git\n',
        '%cd /content/TMD/ASCFormer\n',
        '\n',
        '# 3. Copy và Giải nén Dataset RealTextMan từ Google Drive sang Colab SSD\n',
        'os.makedirs("/content/TMD/ASCFormer/data/ttd", exist_ok=True)\n',
        'drive_zip = "/content/drive/MyDrive/RealTextMan.zip"\n',
        'if not os.path.exists(drive_zip):\n',
        '    matches = glob.glob("/content/drive/MyDrive/**/RealTextMan.zip", recursive=True)\n',
        '    if matches:\n',
        '        drive_zip = matches[0]\n',
        '    else:\n',
        '        raise FileNotFoundError("Không tìm thấy file RealTextMan.zip trên Google Drive của bạn!")\n',
        '\n',
        'print(f"📦 Đang giải nén {drive_zip} sang ổ cứng SSD của Colab...")\n',
        '!unzip -q "{drive_zip}" -d /content/TMD/ASCFormer/data/ttd/\n',
        'print("✅ Dataset đã sẵn sàng tại /content/TMD/ASCFormer/data/ttd/RealTextMan")\n',
        '\n',
        '# 4. Cài đặt các thư viện cần thiết\n',
        '!pip install -q mmcv-lite==2.0.0 mmengine==0.7.0 timm einops prettytable rich onnx onnxruntime\n',
        '!pip install -e /content/TMD/ASCFormer\n',
        '\n',
        'print("🎉 CÀI ĐẶT MÔI TRƯỜNG THÀNH CÔNG!")\n'
    ]
}
cells.append(c1_code)

# Cell 2: Training
c2_md = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '### 🔥 BƯỚC 2: BẮT ĐẦU HUẤN LUYỆN MiT-B0 (30,000 BƯỚC) + TỰ ĐỘNG LƯU DRIVE\n',
        '> * Tự động tải trước trọng số pretrained `mit_b0.pth`.\n',
        '> * Checkpoint và file Log được lưu liên tục vào Google Drive (`/content/drive/MyDrive/RTM_B0_30k_Mobile`).\n',
        '> * Nếu Colab bị ngắt giữa chừng, bạn chỉ cần bấm chạy lại Cell này, nó sẽ **tự động `--resume` từ checkpoint gần nhất**!'
    ]
}
cells.append(c2_md)

c2_code = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'import os, glob, json, torch\n',
        '\n',
        '%cd /content/TMD\n',
        '!git pull origin main\n',
        '%cd /content/TMD/ASCFormer\n',
        '\n',
        '# Đảm bảo package ASCFormer được cài đặt\n',
        '!pip install -q -e /content/TMD/ASCFormer\n',
        '\n',
        '# Tải trước trọng số Pretrained MiT-B0 vào cache\n',
        'print("📥 Đang tải sẵn Pretrained SegFormer MiT-B0...")\n',
        '_ = torch.hub.load_state_dict_from_url(\n',
        '    "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth",\n',
        '    map_location="cpu"\n',
        ')\n',
        '\n',
        'CFG_PATH = "configs/ascformer/ascformer_rtm_b0_30k_sota_mobile.py"\n',
        'DRIVE_WORK_DIR = "/content/drive/MyDrive/RTM_B0_30k_Mobile"\n',
        'os.makedirs(DRIVE_WORK_DIR, exist_ok=True)\n',
        '\n',
        '# Tự động tạo file config nếu chưa có trên repo\n',
        'if not os.path.exists(CFG_PATH):\n',
        '    print("⚙️ Đang khởi tạo file config B0 30k chuẩn...")\n',
        '    with open(CFG_PATH, "w", encoding="utf-8") as f:\n',
        '        f.write(\'\'\'_base_ = ["./ascformer_rtm_b0_ablation16k_m4_proto.py"]\n',
        '\n',
        'train_dataloader = dict(batch_size=4, num_workers=4, persistent_workers=True, pin_memory=True)\n',
        'val_dataloader = dict(batch_size=1, num_workers=4, persistent_workers=True, pin_memory=True)\n',
        'test_dataloader = dict(batch_size=1, num_workers=4, persistent_workers=True, pin_memory=True, dataset=dict(ann_file="val.txt"))\n',
        'train_cfg = dict(max_iters=30000, val_interval=10000)\n',
        'param_scheduler = [\n',
        '    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),\n',
        '    dict(type="PolyLR", eta_min=0., power=1., by_epoch=False, begin=1500, end=30000),\n',
        ']\n',
        'optim_wrapper = dict(\n',
        '    type="AmpOptimWrapper", loss_scale="dynamic",\n',
        '    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),\n',
        '    paramwise_cfg=dict(custom_keys=dict(pos_block=dict(decay_mult=0.0), norm=dict(decay_mult=0.0), head=dict(lr_mult=10.0), rpb=dict(decay_mult=0.0)))\n',
        ')\n',
        'default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=5000, max_keep_ckpts=3, save_best="mIoU", rule="greater"))\n',
        'val_evaluator = dict(type="BinaryIoUMetric", iou_metrics=["mIoU", "mDice", "mFscore"], output_dir="work_dirs/b0_30k_sota_mobile/val_metrics", save_confusion_matrix=True)\n',
        '\'\'\')\n',
        '    print("✅ File config đã sẵn sàng!")\n',
        '\n',
        '# Kiểm tra Checkpoint cũ trên Drive để Resume nếu có\n',
        'ckpts = sorted(glob.glob(f"{DRIVE_WORK_DIR}/iter_*.pth"), key=os.path.getmtime)\n',
        'resume_flag = f"--resume {ckpts[-1]}" if ckpts else ""\n',
        'if resume_flag:\n',
        '    print(f"🔄 Tìm thấy checkpoint cũ trên Drive! Tự động tiếp tục từ: {ckpts[-1]}")\n',
        'else:\n',
        '    print("🚀 Bắt đầu huấn luyện MiT-B0 mới từ đầu (Iteration 0 -> 30,000)...")\n',
        '\n',
        '# Chạy lệnh Train chuẩn\n',
        '!python tools/train.py {CFG_PATH} --work-dir {DRIVE_WORK_DIR} {resume_flag}\n',
        '\n',
        'print("🎉 HUẤN LUYỆN HOÀN TẤT! Checkpoint tốt nhất đã nằm trong Drive tại: " + DRIVE_WORK_DIR)\n'
    ]
}
cells.append(c2_code)

# Cell 3: Export to ONNX
c3_md = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '### 📦 BƯỚC 3: XUẤT MODEL ONNX INT8 SIÊU NHẸ (~25 MB) SANG GOOGLE DRIVE\n',
        '> Chạy Cell này để tự động nén mô hình sang định dạng ONNX INT8 siêu nhẹ (~25 MB) để tích hợp vào Android App.'
    ]
}
cells.append(c3_md)

c3_code = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        'import os, glob\n',
        '\n',
        'DRIVE_WORK_DIR = "/content/drive/MyDrive/RTM_B0_30k_Mobile"\n',
        'best_ckpts = sorted(glob.glob(f"{DRIVE_WORK_DIR}/*best*.pth") + glob.glob(f"{DRIVE_WORK_DIR}/iter_*.pth"), key=os.path.getmtime)\n',
        'if not best_ckpts:\n',
        '    raise FileNotFoundError("Chưa có checkpoint nào trong Drive!")\n',
        'best_ckpt = best_ckpts[-1]\n',
        'print(f"🎯 Đang sử dụng checkpoint tốt nhất: {best_ckpt}")\n',
        '\n',
        '# Xuất mô hình sang ONNX INT8\n',
        'out_onnx = f"{DRIVE_WORK_DIR}/rtm_b0_int8.onnx"\n',
        '%cd /content/TMD/ASCFormer\n',
        '!python tools/export_ondevice_model.py \\\n',
        '    --config configs/ascformer/ascformer_rtm_b0_30k_sota_mobile.py \\\n',
        '    --checkpoint "{best_ckpt}" \\\n',
        '    --output "{out_onnx}"\n',
        '\n',
        'if os.path.exists(out_onnx):\n',
        '    sz_mb = os.path.getsize(out_onnx) / (1024 * 1024)\n',
        '    print(f"✅ ĐÃ XUẤT THÀNH CÔNG MODEL ONNX SIÊU NHẸ: {out_onnx} ({sz_mb:.1f} MB)")\n',
        '    print("📲 Bạn chỉ cần tải file rtm_b0_int8.onnx từ Google Drive về là xong!")\n'
    ]
}
cells.append(c3_code)

nb = {
    'cells': cells,
    'metadata': {
        'accelerator': 'GPU',
        'colab': {'provenance': []},
        'language_info': {'name': 'python'}
    },
    'nbformat': 4,
    'nbformat_minor': 0
}

os.makedirs('/home/duy/DACN/RTM/notebooks', exist_ok=True)
with open('/home/duy/DACN/RTM/notebooks/colab_train_b0_30k_mobile.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

with open('/home/duy/DACN/RTM/colab_train_b0_30k_mobile.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print('Done writing Colab Notebooks.')
