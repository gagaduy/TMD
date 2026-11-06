package com.rtm.tamperdetector;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.webkit.JavascriptInterface;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class MainActivity extends Activity {

    private static final int REQ_GALLERY = 1001;
    private static final int REQ_CAMERA = 1002;
    private static final int REQ_PERM_CAMERA = 2001;
    private static final float TAMPER_THRESHOLD = 0.20f;

    private WebView webView;
    private ExecutorService executor = Executors.newSingleThreadExecutor();
    private OrtEnvironment ortEnv;
    private OrtSession ortSession;
    private File cameraTempFile;
    private ValueCallback<Uri[]> filePathCallback;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        webView = new WebView(this);
        setContentView(webView);

        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        settings.setMediaPlaybackRequiresUserGesture(false);

        webView.setWebViewClient(new WebViewClient());
        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onShowFileChooser(WebView webView, ValueCallback<Uri[]> callback, FileChooserParams params) {
                if (filePathCallback != null) {
                    filePathCallback.onReceiveValue(null);
                }
                filePathCallback = callback;
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                intent.setType("image/*");
                startActivityForResult(Intent.createChooser(intent, "Chọn ảnh kiểm tra"), REQ_GALLERY);
                return true;
            }
        });

        webView.addJavascriptInterface(new AndroidBridge(), "AndroidBridge");
        webView.loadUrl("file:///android_asset/index.html");

        // Pre-initialize ONNX Runtime model in background
        executor.submit(() -> {
            try {
                ensureModelLoaded();
            } catch (Exception e) {
                sendError("Lỗi khởi tạo model: " + e.getMessage());
            }
        });
    }

    private synchronized void ensureModelLoaded() throws Exception {
        if (ortSession != null) return;
        sendStatus("Đang khởi tạo mô hình siêu nhẹ MiT-B0 SOTA (ONNX INT8) trên thiết bị...");
        File modelFile = new File(getFilesDir(), "rtm_b0_int8.onnx");
        if (!modelFile.exists() || modelFile.length() < 10_000_000) {
            try (InputStream is = getAssets().open("rtm_b0_int8.onnx");
                 FileOutputStream fos = new FileOutputStream(modelFile)) {
                byte[] buf = new byte[65536];
                int n;
                while ((n = is.read(buf)) != -1) {
                    fos.write(buf, 0, n);
                }
            }
        }
        ortEnv = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        int availableCores = Runtime.getRuntime().availableProcessors();
        // Optimal thread count for mobile big.LITTLE CPUs (avoids slow efficiency cores)
        int numThreads = Math.min(4, Math.max(2, availableCores / 2));
        opts.setIntraOpNumThreads(numThreads);
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

        // Hardware Acceleration: NNAPI (NPU/GPU/DSP) & XNNPACK (ARM NEON microkernels)
        try {
            opts.addNnapi();
            Log.i("TamperDetector", "NNAPI hardware acceleration enabled successfully");
        } catch (Throwable t) {
            Log.w("TamperDetector", "NNAPI not available on this device: " + t.getMessage());
        }

        try {
            Map<String, String> xnnOpts = new HashMap<>();
            xnnOpts.put("intra_op_num_threads", String.valueOf(numThreads));
            opts.addXnnpack(xnnOpts);
            Log.i("TamperDetector", "XNNPACK ARM NEON execution provider enabled successfully");
        } catch (Throwable t) {
            Log.w("TamperDetector", "XNNPACK not available: " + t.getMessage());
        }

        ortSession = ortEnv.createSession(modelFile.getAbsolutePath(), opts);
        sendStatus("Sẵn sàng (Offline On-Device Engine • Turbo HW Accel)");
    }

    private void sendStatus(final String msg) {
        runOnUiThread(() -> {
            String safe = JSONObject.quote(msg);
            webView.evaluateJavascript("if(window.onEngineStatus) window.onEngineStatus(" + safe + ");", null);
        });
    }

    private void sendError(final String msg) {
        runOnUiThread(() -> {
            String safe = JSONObject.quote(msg);
            webView.evaluateJavascript("if(window.onForensicError) window.onForensicError(" + safe + ");", null);
        });
    }

    public class AndroidBridge {
        @JavascriptInterface
        public boolean isNativeApp() {
            return true;
        }

        @JavascriptInterface
        public void pickGallery() {
            runOnUiThread(() -> {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                intent.setType("image/*");
                startActivityForResult(Intent.createChooser(intent, "Chọn ảnh hoặc ảnh chụp màn hình"), REQ_GALLERY);
            });
        }

        @JavascriptInterface
        public void captureCamera() {
            runOnUiThread(() -> {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                        requestPermissions(new String[]{Manifest.permission.CAMERA}, REQ_PERM_CAMERA);
                        return;
                    }
                }
                launchCameraIntent();
            });
        }

        @JavascriptInterface
        public void runSample(final String sampleFile) {
            executor.submit(() -> {
                try {
                    sendStatus("Đang nạp ảnh mẫu: " + sampleFile + "...");
                    try (InputStream is = getAssets().open("samples/" + sampleFile)) {
                        Bitmap bmp = BitmapFactory.decodeStream(is);
                        analyzeBitmap(bmp, sampleFile);
                    }
                } catch (Exception e) {
                    sendError("Lỗi xử lý ảnh mẫu: " + e.getMessage());
                }
            });
        }
    }

    private void launchCameraIntent() {
        try {
            cameraTempFile = new File(getCacheDir(), "capture_" + System.currentTimeMillis() + ".jpg");
            Uri photoUri = Uri.parse("content://" + getPackageName() + ".fileprovider/" + cameraTempFile.getName());
            Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
            takePictureIntent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION | Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivityForResult(takePictureIntent, REQ_CAMERA);
        } catch (Exception e) {
            sendError("Không thể mở Camera: " + e.getMessage());
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == REQ_PERM_CAMERA) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                launchCameraIntent();
            } else {
                sendError("Bạn cần cấp quyền Camera để chụp ảnh trực tiếp.");
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == REQ_GALLERY) {
            if (filePathCallback != null) {
                Uri[] results = null;
                if (resultCode == Activity.RESULT_OK && data != null && data.getData() != null) {
                    results = new Uri[]{data.getData()};
                }
                filePathCallback.onReceiveValue(results);
                filePathCallback = null;
            }
            if (resultCode == Activity.RESULT_OK && data != null && data.getData() != null) {
                final Uri uri = data.getData();
                executor.submit(() -> {
                    try {
                        sendStatus("Đang đọc ảnh từ thiết bị...");
                        try (InputStream is = getContentResolver().openInputStream(uri)) {
                            Bitmap bmp = BitmapFactory.decodeStream(is);
                            analyzeBitmap(bmp, "uploaded_image.jpg");
                        }
                    } catch (Exception e) {
                        sendError("Lỗi đọc ảnh: " + e.getMessage());
                    }
                });
            }
        } else if (requestCode == REQ_CAMERA && resultCode == Activity.RESULT_OK) {
            executor.submit(() -> {
                try {
                    Bitmap bmp = null;
                    if (cameraTempFile != null && cameraTempFile.exists() && cameraTempFile.length() > 0) {
                        bmp = BitmapFactory.decodeFile(cameraTempFile.getAbsolutePath());
                    } else if (data != null && data.getExtras() != null) {
                        bmp = (Bitmap) data.getExtras().get("data");
                    }
                    if (bmp != null) {
                        analyzeBitmap(bmp, "camera_capture.jpg");
                    } else {
                        sendError("Không nhận được dữ liệu ảnh từ Camera.");
                    }
                } catch (Exception e) {
                    sendError("Lỗi xử lý ảnh chụp: " + e.getMessage());
                }
            });
        }
    }

    private void analyzeBitmap(Bitmap rawBmp, String sourceName) throws Exception {
        ensureModelLoaded();
        long tStart = System.currentTimeMillis();

        // Keep 100% original resolution so 8x8 JPEG block grid is never distorted
        Bitmap origBmp = rawBmp;
        int origW = origBmp.getWidth();
        int origH = origBmp.getHeight();

        // Pad if smaller than 512x512
        int workW = Math.max(512, origW);
        int workH = Math.max(512, origH);
        Bitmap workBmp;
        if (workW != origW || workH != origH) {
            workBmp = Bitmap.createBitmap(workW, workH, Bitmap.Config.ARGB_8888);
            Canvas c = new Canvas(workBmp);
            c.drawColor(Color.BLACK);
            c.drawBitmap(origBmp, 0, 0, null);
        } else {
            workBmp = origBmp.copy(Bitmap.Config.ARGB_8888, false);
        }

        sendStatus("Đang trích xuất đặc trưng ELA (JPEG Q=80) & lưới 8×8 DCT...");
        ByteArrayOutputStream elaBaos = new ByteArrayOutputStream();
        workBmp.compress(Bitmap.CompressFormat.JPEG, 80, elaBaos);
        byte[] jpegBytes = elaBaos.toByteArray();
        Bitmap recompressedBmp = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.length);

        int[] pixelsOrig = new int[workW * workH];
        int[] pixelsRecomp = new int[workW * workH];
        workBmp.getPixels(pixelsOrig, 0, workW, 0, 0, workW, workH);
        recompressedBmp.getPixels(pixelsRecomp, 0, workW, 0, 0, workW, workH);

        // Smart adaptive sliding windows (energy-ranked, margin-filtered, zero-OCR)
        List<int[]> windows = buildSmartSlidingWindows(workW, workH, pixelsOrig);

        float[] probFull = new float[workW * workH];
        long[] tensorShape = new long[]{1, 3, 512, 512};
        FloatBuffer imgBuf = FloatBuffer.allocate(3 * 512 * 512);
        FloatBuffer elaBuf = FloatBuffer.allocate(3 * 512 * 512);

        boolean strongForgeryFound = false;
        int tilesSinceStrong = 0;

        for (int tIdx = 0; tIdx < windows.size(); tIdx++) {
            int y0 = windows.get(tIdx)[0];
            int x0 = windows.get(tIdx)[1];
            sendStatus("Đang quét thông minh (ưu tiên vùng chữ): Ô " + (tIdx + 1) + "/" + windows.size() + "...");

            imgBuf.clear();
            elaBuf.clear();
            int plane = 512 * 512;
            float[] imgArr = imgBuf.array();
            float[] elaArr = elaBuf.array();

            for (int dy = 0; dy < 512; dy++) {
                int rowOff = (y0 + dy) * workW + x0;
                int outRow = dy * 512;
                for (int dx = 0; dx < 512; dx++) {
                    int c1 = pixelsOrig[rowOff + dx];
                    int c2 = pixelsRecomp[rowOff + dx];
                    int r1 = (c1 >> 16) & 0xFF, g1 = (c1 >> 8) & 0xFF, b1 = c1 & 0xFF;
                    int r2 = (c2 >> 16) & 0xFF, g2 = (c2 >> 8) & 0xFF, b2 = c2 & 0xFF;
                    int idx = outRow + dx;
                    // BGR order expected by OnDeviceRTMModel
                    imgArr[idx] = b1;
                    imgArr[plane + idx] = g1;
                    imgArr[2 * plane + idx] = r1;

                    elaArr[idx] = Math.abs(b1 - b2);
                    elaArr[plane + idx] = Math.abs(g1 - g2);
                    elaArr[2 * plane + idx] = Math.abs(r1 - r2);
                }
            }

            float tileMaxP = 0f;
            int tileTamperPx = 0;

            try (OnnxTensor imgTensor = OnnxTensor.createTensor(ortEnv, imgBuf, tensorShape);
                 OnnxTensor elaTensor = OnnxTensor.createTensor(ortEnv, elaBuf, tensorShape)) {
                Map<String, OnnxTensor> feeds = new HashMap<>();
                feeds.put("img_bgr", imgTensor);
                feeds.put("ela_bgr", elaTensor);
                try (OrtSession.Result res = ortSession.run(feeds)) {
                    float[][][] outProb = (float[][][]) res.get(0).getValue();
                    float[][] map2d = outProb[0];
                    for (int dy = 0; dy < 512; dy++) {
                        int rowOff = (y0 + dy) * workW + x0;
                        float[] srcRow = map2d[dy];
                        for (int dx = 0; dx < 512; dx++) {
                            float p = srcRow[dx];
                            if (p > probFull[rowOff + dx]) {
                                probFull[rowOff + dx] = p;
                            }
                            if (p >= TAMPER_THRESHOLD) {
                                tileTamperPx++;
                                if (p > tileMaxP) tileMaxP = p;
                            }
                        }
                    }
                }
            }

            if (tileMaxP >= TAMPER_THRESHOLD && tileTamperPx >= 48) {
                strongForgeryFound = true;
                sendStatus("Đã phát hiện vùng giả mạo (" + Math.round(tileMaxP * 100) + "%), dừng quét sớm!");
                break;
            }
        }

        // Extract connected regions & bounding boxes on original dimensions (origW x origH)
        List<TamperBox> boxes = extractTamperBoxes(probFull, workW, origW, origH);
        float maxProb = 0f;
        int tamperPixels = 0;
        for (int y = 0; y < origH; y++) {
            int row = y * workW;
            for (int x = 0; x < origW; x++) {
                float p = probFull[row + x];
                if (p > maxProb) maxProb = p;
                if (p >= TAMPER_THRESHOLD) tamperPixels++;
            }
        }

        boolean isTampered = (!boxes.isEmpty()) && (tamperPixels >= 48) && (maxProb >= TAMPER_THRESHOLD);
        float modelConfidence;
        if (isTampered) {
            float norm = (maxProb - TAMPER_THRESHOLD) / (0.40f - TAMPER_THRESHOLD);
            modelConfidence = 65.0f + Math.min(34.0f, Math.max(0.0f, norm) * 30.0f);
        } else {
            modelConfidence = 99.0f - (maxProb / TAMPER_THRESHOLD) * 15.0f;
        }

        // Render Tab 1 (Bounding Boxes & Contours) and Tab 2 (Forensic Heatmap)
        Bitmap tab1Bmp = renderBBoxOverlay(origBmp, probFull, workW, boxes);
        Bitmap tab2Bmp = renderHeatmapOverlay(origBmp, probFull, workW, 0.58f);

        long elapsedMs = System.currentTimeMillis() - tStart;

        JSONObject result = new JSONObject();
        result.put("source_name", sourceName);
        result.put("width", origW);
        result.put("height", origH);
        result.put("is_tampered", isTampered);
        result.put("verdict", isTampered ? "PHÁT HIỆN GIẢ MẠO" : "ẢNH NGUYÊN BẢN");
        result.put("confidence", Math.round(modelConfidence * 10.0) / 10.0);
        result.put("max_tamper_prob", Math.round(maxProb * 1000.0) / 10.0);
        result.put("tamper_pixels", tamperPixels);
        result.put("inference_ms", elapsedMs);
        result.put("bbox_b64", bitmapToBase64(tab1Bmp));
        result.put("heatmap_b64", bitmapToBase64(tab2Bmp));
        result.put("orig_b64", bitmapToBase64(origBmp));

        JSONArray boxArr = new JSONArray();
        for (int i = 0; i < boxes.size(); i++) {
            TamperBox b = boxes.get(i);
            JSONObject bo = new JSONObject();
            bo.put("id", i + 1);
            bo.put("x1", b.x1);
            bo.put("y1", b.y1);
            bo.put("x2", b.x2);
            bo.put("y2", b.y2);
            float boxCalib = 65.0f + Math.min(34.0f, Math.max(0.0f, (b.maxProb - TAMPER_THRESHOLD) / (0.40f - TAMPER_THRESHOLD)) * 30.0f);
            bo.put("conf", Math.round(boxCalib * 10.0) / 10.0);
            bo.put("raw_prob", Math.round(b.maxProb * 1000.0) / 10.0);
            bo.put("mean_conf", Math.round(b.meanProb * 1000.0) / 10.0);
            bo.put("area", b.area);
            boxArr.put(bo);
        }
        result.put("boxes", boxArr);

        final String jsonStr = result.toString();
        runOnUiThread(() -> {
            webView.evaluateJavascript("window.onForensicResult(" + jsonStr + ");", null);
        });
        sendStatus("Hoàn tất phân tích (" + elapsedMs + " ms)");
    }

    private static class ScoredWindow {
        int y0, x0;
        int sector;
        float score;
        ScoredWindow(int y0, int x0, int sector, float score) {
            this.y0 = y0;
            this.x0 = x0;
            this.sector = sector;
            this.score = score;
        }
    }

    private List<int[]> buildSmartSlidingWindows(int w, int h, int[] pixelsOrig) {
        int stride = (w > 1500 || h > 1500) ? 480 : 384;
        List<Integer> ys = new ArrayList<>();
        ys.add(0);
        for (int y = stride; y < h - 512; y += stride) {
            if (!ys.contains(y)) ys.add(y);
        }
        int lastY = Math.max(0, ((h - 512) / 8) * 8);
        if (!ys.contains(lastY)) ys.add(lastY);

        List<Integer> xs = new ArrayList<>();
        xs.add(0);
        for (int x = stride; x < w - 512; x += stride) {
            if (!xs.contains(x)) xs.add(x);
        }
        int lastX = Math.max(0, ((w - 512) / 8) * 8);
        if (!xs.contains(lastX)) xs.add(lastX);

        List<ScoredWindow> candidates = new ArrayList<>();
        for (int y0 : ys) {
            for (int x0 : xs) {
                // Compute content energy by sampling every 4th pixel inside this 512x512 tile
                float diffSum = 0f;
                int nonPaperCount = 0;
                int sampleCount = 0;
                for (int dy = 0; dy < 508; dy += 4) {
                    int rowOff = (y0 + dy) * w + x0;
                    int nextRowOff = (y0 + dy + 4) * w + x0;
                    for (int dx = 0; dx < 508; dx += 4) {
                        int c = pixelsOrig[rowOff + dx];
                        int cRight = pixelsOrig[rowOff + dx + 4];
                        int cDown = pixelsOrig[nextRowOff + dx];

                        int r = (c >> 16) & 0xFF, g = (c >> 8) & 0xFF, b = c & 0xFF;
                        int lum = (r + 2 * g + b) >> 2;

                        int rR = (cRight >> 16) & 0xFF, gR = (cRight >> 8) & 0xFF, bR = cRight & 0xFF;
                        int lumR = (rR + 2 * gR + bR) >> 2;

                        int rD = (cDown >> 16) & 0xFF, gD = (cDown >> 8) & 0xFF, bD = cDown & 0xFF;
                        int lumD = (rD + 2 * gD + bD) >> 2;

                        diffSum += Math.abs(lum - lumR) + Math.abs(lum - lumD);
                        if (lum < 215) nonPaperCount++;
                        sampleCount++;
                    }
                }
                float avgGrad = diffSum / Math.max(1, sampleCount);
                float inkRatio = (float) nonPaperCount / Math.max(1, sampleCount);

                // Filter out blank margins (pure paper without text/stamps)
                if (avgGrad >= 0.8f || inkRatio >= 0.002f) {
                    float cy = (float) (y0 + 256) / (float) h;
                    float cx = (float) (x0 + 256) / (float) w;
                    int sy = cy < 0.35f ? 0 : (cy < 0.65f ? 1 : 2);
                    int sx = cx < 0.35f ? 0 : (cx < 0.65f ? 1 : 2);
                    int sectorId = sy * 3 + sx;

                    float tileScore = Math.min(avgGrad, 80.0f) + inkRatio * 60.0f;
                    candidates.add(new ScoredWindow(y0, x0, sectorId, tileScore));
                }
            }
        }

        // Smart Sector-Priority Wave Scheduling:
        // Prioritizes regions where fraud occurs (body text, amounts, dates, signatures)
        // while preventing header/logo monopolies from delaying body scans.
        // 4: Mid-Center (Body main)
        // 2: Top-Right (Dates, Invoice numbers, Receiver details)
        // 5: Mid-Right (Amounts, totals, body table right)
        // 8: Bot-Right (Signatures, seals)
        // 3: Mid-Left (Body labels)
        // 1: Top-Center (Title)
        // 6: Bot-Left (Signer title, notes)
        // 7: Bot-Center (Footer)
        // 0: Top-Left (Org Logo)
        int[] sectorOrder = {4, 2, 5, 8, 3, 1, 6, 7, 0};

        List<List<ScoredWindow>> sectorBuckets = new ArrayList<>(9);
        for (int s = 0; s < 9; s++) {
            sectorBuckets.add(new ArrayList<>());
        }
        for (ScoredWindow sw : candidates) {
            sectorBuckets.get(sw.sector).add(sw);
        }
        for (int s = 0; s < 9; s++) {
            Collections.sort(sectorBuckets.get(s), (a, b) -> Float.compare(b.score, a.score));
        }

        List<int[]> scheduled = new ArrayList<>();
        for (int wave = 0; wave < candidates.size(); wave++) {
            boolean added = false;
            for (int s : sectorOrder) {
                List<ScoredWindow> b = sectorBuckets.get(s);
                if (wave < b.size()) {
                    scheduled.add(new int[]{b.get(wave).y0, b.get(wave).x0});
                    added = true;
                    if (scheduled.size() >= 18) break;
                }
            }
            if (!added || scheduled.size() >= 18) break;
        }
        return scheduled;
    }

    private static class TamperBox {
        int x1, y1, x2, y2, area;
        float maxProb, meanProb;
    }

    private List<TamperBox> extractTamperBoxes(float[] prob, int strideW, int w, int h) {
        // Downsample 4x4 grid for fast connected-component clustering
        int gw = (w + 3) / 4;
        int gh = (h + 3) / 4;
        boolean[] mask = new boolean[gw * gh];
        for (int gy = 0; gy < gh; gy++) {
            for (int gx = 0; gx < gw; gx++) {
                float m = 0f;
                for (int dy = 0; dy < 4 && (gy * 4 + dy) < h; dy++) {
                    int row = (gy * 4 + dy) * strideW + gx * 4;
                    for (int dx = 0; dx < 4 && (gx * 4 + dx) < w; dx++) {
                        if (prob[row + dx] > m) m = prob[row + dx];
                    }
                }
                if (m >= TAMPER_THRESHOLD) {
                    mask[gy * gw + gx] = true;
                }
            }
        }

        boolean[] visited = new boolean[gw * gh];
        List<TamperBox> rawBoxes = new ArrayList<>();
        int[] q = new int[gw * gh];

        for (int i = 0; i < gw * gh; i++) {
            if (!mask[i] || visited[i]) continue;
            int head = 0, tail = 0;
            q[tail++] = i;
            visited[i] = true;
            int minX = gw, minY = gh, maxX = 0, maxY = 0;

            while (head < tail) {
                int curr = q[head++];
                int cy = curr / gw;
                int cx = curr % gw;
                if (cx < minX) minX = cx;
                if (cx > maxX) maxX = cx;
                if (cy < minY) minY = cy;
                if (cy > maxY) maxY = cy;

                for (int ny = Math.max(0, cy - 2); ny <= Math.min(gh - 1, cy + 2); ny++) {
                    for (int nx = Math.max(0, cx - 2); nx <= Math.min(gw - 1, cx + 2); nx++) {
                        int ni = ny * gw + nx;
                        if (mask[ni] && !visited[ni]) {
                            visited[ni] = true;
                            q[tail++] = ni;
                        }
                    }
                }
            }

            int x1 = Math.max(0, minX * 4 - 4);
            int y1 = Math.max(0, minY * 4 - 4);
            int x2 = Math.min(w - 1, (maxX + 1) * 4 + 4);
            int y2 = Math.min(h - 1, (maxY + 1) * 4 + 4);

            int area = 0;
            float maxP = 0f, sumP = 0f;
            for (int y = y1; y <= y2; y++) {
                int row = y * strideW;
                for (int x = x1; x <= x2; x++) {
                    float p = prob[row + x];
                    if (p >= TAMPER_THRESHOLD) {
                        area++;
                        sumP += p;
                        if (p > maxP) maxP = p;
                    }
                }
            }
            if (area >= 48) {
                TamperBox tb = new TamperBox();
                tb.x1 = x1;
                tb.y1 = y1;
                tb.x2 = x2;
                tb.y2 = y2;
                tb.area = area;
                tb.maxProb = maxP;
                tb.meanProb = sumP / area;
                rawBoxes.add(tb);
            }
        }

        Collections.sort(rawBoxes, (a, b) -> Float.compare(b.maxProb, a.maxProb));
        return rawBoxes.size() > 8 ? rawBoxes.subList(0, 8) : rawBoxes;
    }

    private Bitmap renderBBoxOverlay(Bitmap orig, float[] prob, int strideW, List<TamperBox> boxes) {
        int w = orig.getWidth();
        int h = orig.getHeight();
        Bitmap out = orig.copy(Bitmap.Config.ARGB_8888, true);
        int[] px = new int[w * h];
        out.getPixels(px, 0, w, 0, 0, w, h);

        // Highlight tampered mask pixels with soft crimson tint & bright contour
        for (int y = 0; y < h; y++) {
            int pRow = y * strideW;
            int iRow = y * w;
            for (int x = 0; x < w; x++) {
                float p = prob[pRow + x];
                if (p >= TAMPER_THRESHOLD) {
                    int c = px[iRow + x];
                    int r = (c >> 16) & 0xFF;
                    int g = (c >> 8) & 0xFF;
                    int b = c & 0xFF;
                    boolean isEdge = (x == 0 || y == 0 || x == w - 1 || y == h - 1)
                            || prob[pRow + x - 1] < TAMPER_THRESHOLD
                            || prob[pRow + x + 1] < TAMPER_THRESHOLD
                            || prob[pRow - strideW + x] < TAMPER_THRESHOLD
                            || prob[pRow + strideW + x] < TAMPER_THRESHOLD;
                    if (isEdge) {
                        px[iRow + x] = Color.rgb(255, 50, 80);
                    } else {
                        int nr = (int) (r * 0.62f + 255 * 0.38f);
                        int ng = (int) (g * 0.62f + 35 * 0.38f);
                        int nb = (int) (b * 0.62f + 65 * 0.38f);
                        px[iRow + x] = Color.rgb(nr, ng, nb);
                    }
                }
            }
        }
        out.setPixels(px, 0, w, 0, 0, w, h);

        Canvas canvas = new Canvas(out);
        Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(Math.max(3f, Math.min(w, h) * 0.005f));
        boxPaint.setColor(Color.rgb(255, 59, 92));

        Paint badgePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        badgePaint.setStyle(Paint.Style.FILL);
        badgePaint.setColor(Color.argb(230, 225, 29, 72));

        Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        textPaint.setColor(Color.WHITE);
        textPaint.setTypeface(Typeface.create(Typeface.DEFAULT, Typeface.BOLD));
        textPaint.setTextSize(Math.max(20f, Math.min(w, h) * 0.028f));

        for (int i = 0; i < boxes.size(); i++) {
            TamperBox b = boxes.get(i);
            RectF rect = new RectF(b.x1, b.y1, b.x2, b.y2);
            canvas.drawRoundRect(rect, 8f, 8f, boxPaint);

            float boxCalib = 65.0f + Math.min(34.0f, Math.max(0.0f, (b.maxProb - TAMPER_THRESHOLD) / (0.40f - TAMPER_THRESHOLD)) * 30.0f);
            String label = String.format(Locale.US, "#%d  %.0f%%", i + 1, boxCalib);
            float tw = textPaint.measureText(label);
            float th = textPaint.getTextSize();
            float bx1 = b.x1;
            float by2 = Math.max(th + 14f, b.y1);
            float by1 = by2 - th - 12f;
            RectF badgeRect = new RectF(bx1, by1, bx1 + tw + 20f, by2);
            canvas.drawRoundRect(badgeRect, 8f, 8f, badgePaint);
            canvas.drawText(label, bx1 + 10f, by2 - 8f, textPaint);
        }
        return out;
    }

    private Bitmap renderHeatmapOverlay(Bitmap orig, float[] prob, int strideW, float alpha) {
        int w = orig.getWidth();
        int h = orig.getHeight();
        Bitmap out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        int[] px = new int[w * h];
        orig.getPixels(px, 0, w, 0, 0, w, h);

        for (int y = 0; y < h; y++) {
            int pRow = y * strideW;
            int iRow = y * w;
            for (int x = 0; x < w; x++) {
                float v = Math.max(0f, Math.min(1f, prob[pRow + x]));
                int hc = jetColor(v);
                int hr = (hc >> 16) & 0xFF;
                int hg = (hc >> 8) & 0xFF;
                int hb = hc & 0xFF;

                int c = px[iRow + x];
                int r = (c >> 16) & 0xFF;
                int g = (c >> 8) & 0xFF;
                int b = c & 0xFF;

                // Emphasize high probability regions while keeping background document readable
                float wAlpha = v < 0.10f ? (alpha * 0.35f) : Math.min(0.85f, alpha + v * 0.25f);
                int nr = (int) (r * (1f - wAlpha) + hr * wAlpha);
                int ng = (int) (g * (1f - wAlpha) + hg * wAlpha);
                int nb = (int) (b * (1f - wAlpha) + hb * wAlpha);
                px[iRow + x] = Color.rgb(nr, ng, nb);
            }
        }
        out.setPixels(px, 0, w, 0, 0, w, h);
        return out;
    }

    private int jetColor(float v) {
        // Smooth forensic colormap: Deep Navy (0.0) -> Cyan (0.25) -> Green/Yellow (0.55) -> Red (1.0)
        int r, g, b;
        if (v < 0.25f) {
            float t = v / 0.25f;
            r = 15;
            g = (int) (25 + t * 200);
            b = (int) (130 + t * 125);
        } else if (v < 0.50f) {
            float t = (v - 0.25f) / 0.25f;
            r = (int) (15 + t * 230);
            g = 235;
            b = (int) (255 * (1f - t));
        } else if (v < 0.75f) {
            float t = (v - 0.50f) / 0.25f;
            r = 250;
            g = (int) (235 * (1f - t * 0.65f));
            b = 20;
        } else {
            float t = (v - 0.75f) / 0.25f;
            r = 255;
            g = (int) (82 * (1f - t));
            b = (int) (20 + t * 40);
        }
        return Color.rgb(r, g, b);
    }

    private String bitmapToBase64(Bitmap bmp) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        bmp.compress(Bitmap.CompressFormat.JPEG, 90, baos);
        return "data:image/jpeg;base64," + Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP);
    }
}
