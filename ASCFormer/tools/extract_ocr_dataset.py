import os
import sys
import argparse
import time
from multiprocessing import Process, Queue, cpu_count
import cv2
import numpy as np

# Disable model check online hang
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

def worker_fn(worker_id, image_names, img_dir, out_dir, queue):
    import paddlex
    det_model = paddlex.create_model(model_name='PP-OCRv4_mobile_det')
    
    count = 0
    start_time = time.time()
    
    for name in image_names:
        img_name = name.strip()
        if not img_name:
            continue
        
        out_path = os.path.join(out_dir, f'{img_name}.png')
        if os.path.exists(out_path):
            count += 1
            if count % 20 == 0:
                queue.put((worker_id, count))
            continue
            
        img_path = os.path.join(img_dir, f'{img_name}.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, f'{img_name}.png')
            if not os.path.exists(img_path):
                count += 1
                continue
                
        img = cv2.imread(img_path)
        if img is None:
            count += 1
            continue
        h, w, _ = img.shape
        
        # Predict text polygons
        try:
            preds = list(det_model.predict(img_path, batch_size=1))
            res = preds[0]
            polys = res.get('dt_polys', [])
        except Exception as e:
            polys = []
            
        # 1. Text Region Mask: filled polygons
        mask_region = np.zeros((h, w), dtype=np.uint8)
        for poly in polys:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask_region, [pts], 255)
            
        # 2. Text Boundary Mask: outline of polygons
        mask_boundary = np.zeros((h, w), dtype=np.uint8)
        for poly in polys:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(mask_boundary, [pts], isClosed=True, color=255, thickness=2)
            
        # 3. Distance Transform: geometric field
        if mask_region.max() > 0:
            dist_trans = cv2.distanceTransform(mask_region, cv2.DIST_L2, 5)
            d_max = dist_trans.max()
            if d_max > 0:
                dist_trans = (dist_trans / d_max * 255.0).astype(np.uint8)
            else:
                dist_trans = dist_trans.astype(np.uint8)
        else:
            dist_trans = np.zeros((h, w), dtype=np.uint8)
            
        # Stack to 3 channels: [Region, Boundary, Distance]
        ocr_spatial = np.stack([mask_region, mask_boundary, dist_trans], axis=-1)
        
        cv2.imwrite(out_path, ocr_spatial, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        count += 1
        
        if count % 20 == 0:
            queue.put((worker_id, count))
            
    queue.put((worker_id, count))
    queue.put(None) # Done signal

def main():
    parser = argparse.ArgumentParser(description='Extract OCR spatial maps for RTM dataset')
    parser.add_argument('--split', type=str, default='val.txt', help='Split file name in data root')
    parser.add_argument('--data-root', type=str, default='ASCFormer/data/ttd/RealTextMan')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel worker processes')
    args = parser.parse_args()
    
    split_path = os.path.join(args.data_root, args.split)
    img_dir = os.path.join(args.data_root, 'JPEGImages')
    out_dir = os.path.join(args.data_root, 'ocr_spatial')
    os.makedirs(out_dir, exist_ok=True)
    
    with open(split_path, 'r') as f:
        all_names = [line.strip() for line in f if line.strip()]
        
    total_imgs = len(all_names)
    print(f'=== Extracting OCR Spatial Maps ===')
    print(f'Split: {split_path} ({total_imgs} images)')
    print(f'Output dir: {out_dir}')
    print(f'Workers: {args.workers}')
    
    chunks = np.array_split(all_names, args.workers)
    
    queue = Queue()
    processes = []
    
    t0 = time.time()
    for w_id in range(args.workers):
        chunk = chunks[w_id].tolist()
        p = Process(target=worker_fn, args=(w_id, chunk, img_dir, out_dir, queue))
        p.start()
        processes.append(p)
        
    finished_workers = 0
    worker_counts = {w_id: 0 for w_id in range(args.workers)}
    
    while finished_workers < args.workers:
        item = queue.get()
        if item is None:
            finished_workers += 1
        else:
            w_id, c = item
            worker_counts[w_id] = c
            total_done = sum(worker_counts.values())
            elapsed = time.time() - t0
            speed = total_done / max(elapsed, 0.001)
            eta = (total_imgs - total_done) / max(speed, 0.001)
            print(f'Progress: {total_done}/{total_imgs} ({total_done/total_imgs*100:.1f}%) | '
                  f'Speed: {speed:.1f} img/s | ETA: {eta/60:.1f} mins', end='\r')
                  
    for p in processes:
        p.join()
        
    total_elapsed = time.time() - t0
    print(f'\nDone! Processed {total_imgs} images in {total_elapsed:.1f}s ({total_elapsed/60:.2f} mins).')

if __name__ == '__main__':
    main()
