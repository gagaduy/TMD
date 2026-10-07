import os
import cv2
import numpy as np
import time
from multiprocessing import Pool, cpu_count

def run_simulation(seed, num_trials, names, mask_dir, crop_size=512):
    np.random.seed(seed)
    
    rc_pos = 0
    rc_pix = []
    
    fc_pos = 0
    fc_pix = []
    
    for _ in range(num_trials):
        # Lấy mẫu ngẫu nhiên trên TOÀN BỘ 5,803 ảnh
        name = np.random.choice(names)
        mask_path = os.path.join(mask_dir, f"{name}.png")
        if not os.path.exists(mask_path):
            continue
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        h, w = mask.shape
        if h < crop_size or w < crop_size:
            mask = cv2.resize(mask, (max(w, crop_size), max(h, crop_size)), interpolation=cv2.INTER_NEAREST)
            h, w = mask.shape
            
        # 1. Random Crop (Baseline)
        ry = np.random.randint(0, h - crop_size + 1)
        rx = np.random.randint(0, w - crop_size + 1)
        rc_crop = mask[ry:ry+crop_size, rx:rx+crop_size]
        rc_cnt = int(np.sum(rc_crop > 0))
        if rc_cnt > 0:
            rc_pos += 1
            rc_pix.append(rc_cnt)
            
        # 2. Focused Crop (pos_prob = 0.70)
        pos_coords = np.argwhere(mask > 0)
        has_pos = len(pos_coords) > 0
        
        if has_pos and (np.random.rand() < 0.70):
            idx = np.random.randint(len(pos_coords))
            cy, cx = pos_coords[idx]
            y1 = max(0, cy - crop_size + 1)
            y2 = min(h - crop_size, cy)
            fy = np.random.randint(y1, y2 + 1) if y2 >= y1 else y1
            x1 = max(0, cx - crop_size + 1)
            x2 = min(w - crop_size, cx)
            fx = np.random.randint(x1, x2 + 1) if x2 >= x1 else x1
        else:
            fy = np.random.randint(0, h - crop_size + 1)
            fx = np.random.randint(0, w - crop_size + 1)
            
        fc_crop = mask[fy:fy+crop_size, fx:fx+crop_size]
        fc_cnt = int(np.sum(fc_crop > 0))
        if fc_cnt > 0:
            fc_pos += 1
            fc_pix.append(fc_cnt)
            
    return rc_pos, rc_pix, fc_pos, fc_pix

def main():
    train_file = 'ASCFormer/data/ttd/RealTextMan/train.txt'
    mask_dir = 'ASCFormer/data/ttd/RealTextMan/SegmentationClass'
    
    with open(train_file) as f:
        names = [line.strip() for line in f if line.strip()]
        
    total_trials = 16000 # Đúng bằng 8,000 iters x batch size 2 của quá trình huấn luyện thật
    num_workers = min(8, cpu_count())
    trials_per_worker = total_trials // num_workers
    
    print(f"Bắt đầu đo đạc trên TOÀN BỘ {len(names)} ảnh của tập Train...")
    print(f"Tổng số mẩu cắt mô phỏng: {total_trials} (tương ứng 8,000 iters x batch 2)")
    print(f"Sử dụng {num_workers} tiến trình song song...")
    
    t0 = time.time()
    args = [(i * 1000 + 42, trials_per_worker, names, mask_dir) for i in range(num_workers)]
    
    with Pool(num_workers) as pool:
        results = pool.starmap(run_simulation, args)
        
    total_rc_pos = sum(r[0] for r in results)
    all_rc_pix = [p for r in results for p in r[1]]
    total_fc_pos = sum(r[2] for r in results)
    all_fc_pix = [p for r in results for p in r[3]]
    
    actual_trials = trials_per_worker * num_workers
    rc_rate = total_rc_pos / actual_trials * 100
    fc_rate = total_fc_pos / actual_trials * 100
    
    avg_rc_pix = np.mean(all_rc_pix) if all_rc_pix else 0
    avg_fc_pix = np.mean(all_fc_pix) if all_fc_pix else 0
    
    print("\n========================================================")
    print(f"KẾT QUẢ THỰC TẾ TRÊN TOÀN BỘ TẬP TRAIN ({len(names)} ẢNH, {actual_trials} MẪU CẮT):")
    print("========================================================")
    print(f"1. Random Crop (Baseline của Paper):")
    print(f"   - Tỷ lệ mẩu ảnh CÓ chữ giả (Positive Crops):   {rc_rate:.2f}% ({total_rc_pos}/{actual_trials})")
    print(f"   - Tỷ lệ mẩu ảnh TOÀN NỀN TRẮNG (Negative/Empty): {100-rc_rate:.2f}% ({actual_trials - total_rc_pos}/{actual_trials})")
    print(f"   - Số pixel chữ giả trung bình trong mỗi mẩu:    {avg_rc_pix:.0f} px")
    print("--------------------------------------------------------")
    print(f"2. Focused Crop 70% + Random 30% (Đề xuất của Luận văn):")
    print(f"   - Tỷ lệ mẩu ảnh CÓ chữ giả (Positive Crops):   {fc_rate:.2f}% ({total_fc_pos}/{actual_trials})")
    print(f"   - Tỷ lệ mẩu ảnh TOÀN NỀN TRẮNG (Negative/Empty): {100-fc_rate:.2f}% ({actual_trials - total_fc_pos}/{actual_trials})")
    print(f"   - Số pixel chữ giả trung bình trong mỗi mẩu:    {avg_fc_pix:.0f} px")
    print("========================================================")
    print(f"Thời gian hoàn thành: {time.time() - t0:.1f} giây")

if __name__ == '__main__':
    main()
