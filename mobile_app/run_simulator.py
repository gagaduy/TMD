#!/usr/bin/env python3
"""Interactive Mobile App Simulator & APK Download Server for RTM Text Forensic.

Serves the exact mobile UI (`android_app/app/src/main/assets/index.html`) backed by the
exported On-Device ONNX model (`models/rtm_sota_b2_int8.onnx`), and also serves
`RTM_TamperDetector.apk` at `/RTM_TamperDetector.apk` for easy phone installation.
"""
import argparse
import base64
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import onnxruntime as ort

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(APP_DIR, 'android_app', 'app', 'src', 'main', 'assets')
SAMPLES_DIR = os.path.join(ASSETS_DIR, 'samples')
# Default to newly trained ultra-lightweight MiT-B0 INT8 model (60.8 MB, 0.9s CPU)
MODEL_B0_PATH = os.path.join(APP_DIR, 'models', 'rtm_b0_int8.onnx')
MODEL_B2_PATH = os.path.join(APP_DIR, 'models', 'rtm_sota_b2_int8.onnx')
MODEL_PATH = MODEL_B0_PATH if os.path.exists(MODEL_B0_PATH) else MODEL_B2_PATH
APK_PATH = os.path.join(APP_DIR, 'RTM_TamperDetector.apk')
TAMPER_THRESHOLD = 0.20


class ForensicEngine:
    def __init__(self, model_path: str):
        print(f'[Engine] Loading ONNX model: {model_path}')
        self.sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    def build_smart_windows(self, work_bgr: np.ndarray, h: int, w: int):
        stride = 480 if (w > 1500 or h > 1500) else 384
        ys = sorted(set([0] + list(range(stride, max(1, h - 512), stride)) + [max(0, ((h - 512) // 8) * 8)]))
        xs = sorted(set([0] + list(range(stride, max(1, w - 512), stride)) + [max(0, ((w - 512) // 8) * 8)]))

        gray = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.abs(gx) + np.abs(gy)

        candidates = []
        for y0 in ys:
            for x0 in xs:
                p_grad = grad[y0:y0+512, x0:x0+512]
                p_gray = gray[y0:y0+512, x0:x0+512]
                energy = float(p_grad.mean())
                ink_ratio = float((p_gray < 215).mean())
                if energy >= 0.8 or ink_ratio >= 0.002:
                    cy = (y0 + 256) / h
                    cx = (x0 + 256) / w
                    sy = 0 if cy < 0.35 else (1 if cy < 0.65 else 2)
                    sx = 0 if cx < 0.35 else (1 if cx < 0.65 else 2)
                    sector_id = sy * 3 + sx

                    tile_score = min(energy, 80.0) + ink_ratio * 60.0
                    candidates.append({
                        "coords": (y0, x0),
                        "sector": sector_id,
                        "score": tile_score
                    })

        # Sector Priority Wave Scheduling:
        # Prioritizes body text, amounts, dates, signatures before static logos/borders
        SECTOR_ORDER = [4, 2, 5, 8, 3, 1, 6, 7, 0]
        sector_buckets = {s: [] for s in range(9)}
        for c in candidates:
            sector_buckets[c["sector"]].append(c)
        for s in range(9):
            sector_buckets[s].sort(key=lambda c: -c["score"])

        scheduled = []
        for wave in range(len(candidates)):
            added = False
            for s in SECTOR_ORDER:
                if wave < len(sector_buckets[s]):
                    scheduled.append(sector_buckets[s][wave]["coords"])
                    added = True
            if not added:
                break

        max_w = min(18, len(scheduled))
        return scheduled[:max_w]

    def extract_boxes(self, prob: np.ndarray):
        h, w = prob.shape[:2]
        mask = (prob >= TAMPER_THRESHOLD).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        boxes = []
        for lbl in range(1, num_labels):
            x, y, bw, bh, _ = stats[lbl]
            x1 = max(0, int(x - 4))
            y1 = max(0, int(y - 4))
            x2 = min(w - 1, int(x + bw + 4))
            y2 = min(h - 1, int(y + bh + 4))
            sub_p = prob[y1:y2 + 1, x1:x2 + 1]
            sub_m = sub_p >= TAMPER_THRESHOLD
            area = int(sub_m.sum())
            if area >= 48:
                max_p = float(sub_p.max())
                mean_p = float(sub_p[sub_m].mean())
                norm_p = (max_p - TAMPER_THRESHOLD) / (0.40 - TAMPER_THRESHOLD)
                calib_conf = 65.0 + min(34.0, max(0.0, norm_p) * 30.0)
                boxes.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'area': area, 'conf': round(calib_conf, 1),
                    'raw_prob': round(max_p * 100.0, 1),
                    'mean_conf': round(mean_p * 100.0, 1),
                    '_sort': max_p,
                })
        boxes.sort(key=lambda b: -b['_sort'])
        boxes = boxes[:8]
        for idx, b in enumerate(boxes):
            b['id'] = idx + 1
            b.pop('_sort', None)
        return boxes

    def render_bbox(self, orig_bgr: np.ndarray, prob: np.ndarray, boxes: list) -> np.ndarray:
        out = orig_bgr.copy()
        mask = (prob >= TAMPER_THRESHOLD).astype(np.uint8)
        if mask.max() > 0:
            # Soft crimson tint inside tampered mask
            tint = np.zeros_like(out)
            tint[:, :] = (65, 35, 255)  # BGR
            out = np.where(mask[..., None] > 0, (out * 0.62 + tint * 0.38).astype(np.uint8), out)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, (80, 50, 255), 2, cv2.LINE_AA)

        for b in boxes:
            x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
            cv2.rectangle(out, (x1, y1), (x2, y2), (92, 59, 255), 2, cv2.LINE_AA)
            label = f"#{b['id']}  {b['conf']:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            by2 = max(th + 10, y1)
            by1 = by2 - th - 10
            cv2.rectangle(out, (x1, by1), (x1 + tw + 14, by2), (72, 29, 225), -1)
            cv2.putText(out, label, (x1 + 7, by2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def render_heatmap(self, orig_bgr: np.ndarray, prob: np.ndarray) -> np.ndarray:
        p_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
        hm = cv2.applyColorMap(p_u8, cv2.COLORMAP_JET)
        alpha = np.where(prob < 0.10, 0.22, np.minimum(0.82, 0.55 + prob * 0.28))[..., None]
        blended = (orig_bgr.astype(np.float32) * (1.0 - alpha) + hm.astype(np.float32) * alpha).astype(np.uint8)
        return blended

    @staticmethod
    def to_b64(img_bgr: np.ndarray) -> str:
        _, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')

    def analyze(self, raw_bgr: np.ndarray, source_name: str = 'image.jpg') -> dict:
        t0 = time.time()
        orig_bgr = raw_bgr.copy()
        oh, ow = orig_bgr.shape[:2]

        pad_h = max(0, 512 - oh)
        pad_w = max(0, 512 - ow)
        work_bgr = cv2.copyMakeBorder(orig_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0) if (pad_h > 0 or pad_w > 0) else orig_bgr
        wh, ww = work_bgr.shape[:2]

        _, enc = cv2.imencode('.jpg', work_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        recomp = cv2.imdecode(enc, 1)
        ela_bgr = cv2.absdiff(work_bgr, recomp)

        windows = self.build_smart_windows(work_bgr, wh, ww)
        prob_full = np.zeros((wh, ww), dtype=np.float32)

        strong_found = False
        tiles_after_strong = 0

        for y0, x0 in windows:
            ci = work_bgr[y0:y0 + 512, x0:x0 + 512].transpose(2, 0, 1)[None].astype(np.float32)
            ce = ela_bgr[y0:y0 + 512, x0:x0 + 512].transpose(2, 0, 1)[None].astype(np.float32)
            p = self.sess.run(['prob_tamper'], {'img_bgr': ci, 'ela_bgr': ce})[0][0]
            prob_full[y0:y0 + 512, x0:x0 + 512] = np.maximum(prob_full[y0:y0 + 512, x0:x0 + 512], p)

            tile_tamper_px = int((p >= TAMPER_THRESHOLD).sum())
            tile_max_p = float(p.max())
            if tile_max_p >= TAMPER_THRESHOLD and tile_tamper_px >= 48:
                # Early exit: Tamper detected with high confidence, finish immediately
                break
        prob_orig = prob_full[:oh, :ow]
        boxes = self.extract_boxes(prob_orig)
        max_prob = float(prob_orig.max())
        tamper_pixels = int((prob_orig >= TAMPER_THRESHOLD).sum())
        is_tampered = (len(boxes) > 0) and (tamper_pixels >= 48) and (max_prob >= TAMPER_THRESHOLD)
        if is_tampered:
            # Calibrated confidence: map [TAMPER_THRESHOLD, 0.40] -> [65%, 95%], and >=0.40 -> 95%-99%
            norm = (max_prob - TAMPER_THRESHOLD) / (0.40 - TAMPER_THRESHOLD)
            confidence = 65.0 + min(34.0, max(0.0, norm) * 30.0)
        else:
            # Clean authentic document: 90% - 99% confidence
            confidence = 99.0 - (max_prob / TAMPER_THRESHOLD) * 15.0

        tab1_bgr = self.render_bbox(orig_bgr, prob_orig, boxes)
        tab2_bgr = self.render_heatmap(orig_bgr, prob_orig)
        elapsed_ms = int((time.time() - t0) * 1000)

        return {
            'source_name': source_name,
            'width': ow,
            'height': oh,
            'is_tampered': is_tampered,
            'verdict': 'PHÁT HIỆN GIẢ MẠO' if is_tampered else 'ẢNH NGUYÊN BẢN',
            'confidence': round(confidence, 1),
            'max_tamper_prob': round(max_prob * 100.0, 1),
            'tamper_pixels': tamper_pixels,
            'inference_ms': elapsed_ms,
            'boxes': boxes,
            'bbox_b64': self.to_b64(tab1_bgr),
            'heatmap_b64': self.to_b64(tab2_bgr),
            'orig_b64': self.to_b64(orig_bgr),
            '_tab1_bgr': tab1_bgr,
            '_tab2_bgr': tab2_bgr,
        }


ENGINE = None


class SimulatorHandler(BaseHTTPRequestHandler):
    def _send_json(self, obj: dict, status: int = 200):
        clean = {k: v for k, v in obj.items() if not k.startswith('_')}
        body = json.dumps(clean).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self):
        parsed = urlparse(self.path)
        if parsed.path == '/RTM_TamperDetector.apk' and os.path.exists(APK_PATH):
            size = os.path.getsize(APK_PATH)
            self.send_response(200)
            self.send_header('Content-Type', 'application/vnd.android.package-archive')
            self.send_header('Content-Disposition', 'attachment; filename="RTM_TamperDetector.apk"')
            self.send_header('Content-Length', str(size))
            self.end_headers()
        else:
            self.send_response(200)
            self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ('/', '/index.html'):
            html_path = os.path.join(ASSETS_DIR, 'index.html')
            with open(html_path, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif parsed.path == '/api/sample':
            qs = parse_qs(parsed.query)
            name = os.path.basename(qs.get('name', ['sample_tamper_1.jpg'])[0])
            sp = os.path.join(SAMPLES_DIR, name)
            img = cv2.imread(sp)
            res = ENGINE.analyze(img, source_name=name)
            self._send_json(res)
        elif parsed.path == '/RTM_TamperDetector.apk' and os.path.exists(APK_PATH):
            size = os.path.getsize(APK_PATH)
            self.send_response(200)
            self.send_header('Content-Type', 'application/vnd.android.package-archive')
            self.send_header('Content-Disposition', 'attachment; filename="RTM_TamperDetector.apk"')
            self.send_header('Content-Length', str(size))
            self.end_headers()
            with open(APK_PATH, 'rb') as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/api/analyze':
            length = int(self.headers.get('Content-Length', '0'))
            payload = json.loads(self.rfile.read(length).decode('utf-8'))
            b64_str = payload['image_b64']
            if ',' in b64_str:
                b64_str = b64_str.split(',', 1)[1]
            raw_bytes = base64.b64decode(b64_str)
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            res = ENGINE.analyze(img, source_name=payload.get('filename', 'upload.jpg'))
            self._send_json(res)
        else:
            self.send_error(404)


def generate_demos(engine: ForensicEngine):
    demo_dir = os.path.join(APP_DIR, 'demo_outputs')
    os.makedirs(demo_dir, exist_ok=True)
    for fn in ['sample_tamper_1.jpg', 'sample_tamper_2.jpg', 'sample_tamper_3.jpg', 'sample_authentic.jpg']:
        p = os.path.join(SAMPLES_DIR, fn)
        img = cv2.imread(p)
        res = engine.analyze(img, source_name=fn)
        stem = os.path.splitext(fn)[0]
        cv2.imwrite(os.path.join(demo_dir, f'{stem}_tab1_bbox.jpg'), res['_tab1_bgr'])
        cv2.imwrite(os.path.join(demo_dir, f'{stem}_tab2_heatmap.jpg'), res['_tab2_bgr'])
        print(f"  -> {fn}: verdict={res['verdict']}, conf={res['confidence']}%, boxes={len(res['boxes'])}, time={res['inference_ms']}ms")


def main():
    global ENGINE
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--generate-demo', action='store_true')
    args = parser.parse_args()

    ENGINE = ForensicEngine(MODEL_PATH)
    if args.generate_demo:
        generate_demos(ENGINE)
        return

    server = HTTPServer(('0.0.0.0', args.port), SimulatorHandler)
    print(f'[Simulator] Running at http://localhost:{args.port}')
    server.serve_forever()


if __name__ == '__main__':
    main()
