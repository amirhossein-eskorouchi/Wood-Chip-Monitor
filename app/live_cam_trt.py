# live_cam_trt.py
"""
Unified drop-in: DETR (size + histogram + calibration + alarm) + Moisture (TRT) in ONE loop.

Design goals:
- Keep your previous DETR pipeline behavior (stats/hist/overlay/ref-object scale/alarm)
- Add moisture inference using the TOP-K detected chip crops (fallback: full frame if no boxes)
- Avoid TensorRT / CUDA segfaults:
  * single global TRT logger
  * buffers allocated per FULL binding shape after set_binding_shape
  * destroy TRT objects BEFORE ctx.pop()
  * one CUDA context per inference thread
- Thread-safe shared states for FastAPI:
  LATEST_FRAME, LATEST_STATS, HIST_DATA, LATEST_MOISTURE

Updates in this version (as requested by user):
- Histogram ONLY for diameter (D)
- Bounding boxes only TWO colors:
  * GREEN for normal chips
  * RED for oversized chips
- Apply performance/stability numbers:
  * Camera FPS = 20
  * Processed FPS ≈ 10 via infer every 2 frames
  * Window via ROLLING_MAX = 2000
  * Histogram update cadence = 1 Hz
  * Bins = 20
  * Fixed range 0→D_max (in mm once calibrated), optional EMA smoothing (alpha=0.2)
"""

import os
import time
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import threading
import traceback



# =================== PATHS / DEVICE CONFIG ===================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

MODEL_DIR = os.environ.get(
    "WOODCHIP_MODEL_DIR",
    os.path.join(REPO_ROOT, "models"),
)

ENGINE_PATH = os.environ.get(
    "WOODCHIP_DETR_ENGINE",
    os.path.join(MODEL_DIR, "detr_resnet101_fp16.engine"),
)

MOIST_ENGINE_PATH = os.environ.get(
    "WOODCHIP_MOISTURE_ENGINE",
    os.path.join(MODEL_DIR, "moistnetlite_fp16.engine"),
)

MOIST_CLASSES_PATH = os.environ.get(
    "WOODCHIP_MOISTURE_CLASSES",
    os.path.join(MODEL_DIR, "moistnetlite_classes.txt"),
)

OUT_DIR = os.environ.get(
    "WOODCHIP_OUTPUT_DIR",
    os.path.join(REPO_ROOT, "outputs"),
)
os.makedirs(OUT_DIR, exist_ok=True)

# Camera configuration
CAM_DEV = os.environ.get("WOODCHIP_CAMERA_DEVICE", "/dev/video0")
CAM_W = int(os.environ.get("WOODCHIP_CAMERA_WIDTH", "1480"))
CAM_H = int(os.environ.get("WOODCHIP_CAMERA_HEIGHT", "900"))
CAM_FPS = int(os.environ.get("WOODCHIP_CAMERA_FPS", "20"))

# UPDATED: processed FPS control (camera 20 FPS, infer every 2 frames => ~10 FPS)
INFER_EVERY_N_FRAMES = 2

# DETR input handling (your old settings)
ENGINE_INPUT_MODE = "fixed"
FIXED_H, FIXED_W = 800, 1333
SHORTEST_EDGE, LONGEST_EDGE = 800, 1333

CONF_THR, NMS_IOU = 0.5, 0.5

# --- Reference object config ---
REF_DIAM_MM = 110.0   # real diameter of your white disk in millimeters (user-configurable)
PIXELS_PER_MM = None  # estimated online from the white object
UNITS = "pixels"

# Rolling buffer of px/mm estimates
SCALE_BUF = []
SCALE_ROLLING_MAX = 500

# Alarm
ALARM_THRESHOLD_MM = 60.0
ALARM_ENABLED = True
ALARM_ACTIVE = False
ALARM_MAX_D_MM = None

# Histogram mode (kept for API compatibility; will publish only diameter histogram)
HISTOGRAM_MODE = "diameter"  # "length" | "width" | "diameter"

# UPDATED: rolling sample window
ROLLING_MAX = 2000

# UPDATED: histogram update cadence
HIST_UPDATE_INTERVAL = 1.0  # 1 Hz

# UPDATED: histogram bins and fixed range (in mm once calibrated)
HIST_NUM_BINS = 20
HIST_D_MAX_MM = 120.0  # choose based on your spec (e.g., 80–120 mm); used when UNITS == "mm"

# UPDATED: optional EMA smoothing for histogram stability
HIST_EMA_ENABLED = True
HIST_EMA_ALPHA = 0.2

# Moisture runtime config (unchanged except its cadence is tied to processed frames below)
MOISTURE_ENABLED = True
MOISTURE_TOPK = 8
MOISTURE_EVERY_N_FRAMES = 3
MOISTURE_INPUT_SIZE = 224
MOISTURE_MIN_BOX_PX = 12
MOISTURE_FALLBACK_FULLFRAME = True  # if no boxes, still run moisture on full frame

# Normalization (ImageNet)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# =================== SHARED STATE for backend_app.py ===================
STATE_LOCK = threading.Lock()

LATEST_FRAME = None   # BGR np.uint8 image (annotated)
LATEST_STATS = None   # dict
HIST_DATA = None      # dict
LATEST_MOISTURE = None  # dict
LATEST_HEALTH = None


# =================== TRT LOGGER (single global) ===================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# =================== WEB CONFIG HELPERS ===================
def get_runtime_config():
    return {
        "conf_thr": float(CONF_THR),
        "nms_iou": float(NMS_IOU),
        "alarm_threshold_mm": float(ALARM_THRESHOLD_MM),
        "alarm_enabled": bool(ALARM_ENABLED),
        "ref_diam_mm": float(REF_DIAM_MM),
        "histogram_mode": HISTOGRAM_MODE,

        "moisture_enabled": bool(MOISTURE_ENABLED),
        "moisture_topk": int(MOISTURE_TOPK),
        "moisture_every_n_frames": int(MOISTURE_EVERY_N_FRAMES),

        # expose agreed knobs (safe extras; clients can ignore)
        "camera_fps": int(CAM_FPS),
        "infer_every_n_frames": int(INFER_EVERY_N_FRAMES),
        "rolling_max": int(ROLLING_MAX),
        "hist_update_interval": float(HIST_UPDATE_INTERVAL),
        "hist_num_bins": int(HIST_NUM_BINS),
        "hist_d_max_mm": float(HIST_D_MAX_MM),
        "hist_ema_enabled": bool(HIST_EMA_ENABLED),
        "hist_ema_alpha": float(HIST_EMA_ALPHA),
    }


def get_health():
    with STATE_LOCK:
        return LATEST_HEALTH.copy() if isinstance(LATEST_HEALTH, dict) else None

def update_runtime_config(
    conf_thr=None,
    nms_iou=None,
    alarm_threshold_mm=None,
    alarm_enabled=None,
    ref_diam_mm=None,
    histogram_mode=None,
    moisture_enabled=None,
    moisture_topk=None,
    moisture_every_n_frames=None,
    **_ignored
):
    global CONF_THR, NMS_IOU, ALARM_THRESHOLD_MM, ALARM_ENABLED, REF_DIAM_MM, HISTOGRAM_MODE
    global MOISTURE_ENABLED, MOISTURE_TOPK, MOISTURE_EVERY_N_FRAMES

    if conf_thr is not None:
        try:
            CONF_THR = max(0.0, min(1.0, float(conf_thr)))
        except (ValueError, TypeError):
            pass

    if nms_iou is not None:
        try:
            NMS_IOU = max(0.0, min(1.0, float(nms_iou)))
        except (ValueError, TypeError):
            pass

    if alarm_threshold_mm is not None:
        try:
            ALARM_THRESHOLD_MM = max(0.0, float(alarm_threshold_mm))
        except (ValueError, TypeError):
            pass

    if alarm_enabled is not None:
        ALARM_ENABLED = bool(alarm_enabled)

    if ref_diam_mm is not None:
        try:
            v = float(ref_diam_mm)
            if v > 0:
                REF_DIAM_MM = v
        except (ValueError, TypeError):
            pass

    # kept for compatibility; UI can still set it, but we publish only diameter histogram anyway
    if histogram_mode is not None:
        hm = str(histogram_mode).lower()
        if hm in ("length", "width", "diameter"):
            HISTOGRAM_MODE = hm

    if moisture_enabled is not None:
        MOISTURE_ENABLED = bool(moisture_enabled)
    if moisture_topk is not None:
        try:
            MOISTURE_TOPK = max(1, int(moisture_topk))
        except Exception:
            pass
    if moisture_every_n_frames is not None:
        try:
            MOISTURE_EVERY_N_FRAMES = max(1, int(moisture_every_n_frames))
        except Exception:
            pass


# =================== MATH / HELPERS ===================
def _now():
    return float(time.time())


def softmax_lastdim(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=-1, keepdims=True) + 1e-12)


def safe_softmax_2d(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)


def nms_np(boxes_xyxy, scores, iou_thr=0.5):
    if boxes_xyxy.shape[0] == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter + 1e-12
        iou = inter / union
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def resize_keep_aspect(h0, w0, shortest=SHORTEST_EDGE, longest=LONGEST_EDGE):
    s = shortest / float(min(h0, w0))
    if max(h0, w0) * s > longest:
        s = longest / float(max(h0, w0))
    return s, int(round(h0 * s)), int(round(w0 * s))


def preprocess_fixed(img_bgr):
    h0, w0 = img_bgr.shape[:2]
    s, Hr, Wr = resize_keep_aspect(h0, w0)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_resized = cv2.resize(img_rgb, (Wr, Hr), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((FIXED_H, FIXED_W, 3), np.float32)
    canvas[:Hr, :Wr] = img_resized
    canvas = (canvas - MEAN) / STD
    x = np.transpose(canvas, (2, 0, 1))[None, ...].astype(np.float32)
    return x, (h0, w0, Hr, Wr)


def postprocess_from_fixed(pred_boxes, logits, meta, conf_thr=0.5):
    h0, w0, Hr, Wr = meta

    logits = logits.reshape(1, -1, logits.shape[-1])
    probs = softmax_lastdim(logits)[0]

    scores = probs[:, :-1].max(axis=1)
    keep = np.where(scores >= conf_thr)[0]
    if keep.size == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

    boxes = pred_boxes[0][keep]
    scores = scores[keep]

    cx = boxes[:, 0] * FIXED_W
    cy = boxes[:, 1] * FIXED_H
    ww = boxes[:, 2] * FIXED_W
    hh = boxes[:, 3] * FIXED_H

    x1 = cx - ww / 2.0
    y1 = cy - hh / 2.0
    x2 = cx + ww / 2.0
    y2 = cy + hh / 2.0

    x1 = np.clip(x1, 0, Wr - 1)
    x2 = np.clip(x2, 0, Wr - 1)
    y1 = np.clip(y1, 0, Hr - 1)
    y2 = np.clip(y2, 0, Hr - 1)

    sx = w0 / float(Wr)
    sy = h0 / float(Hr)

    x1 *= sx
    x2 *= sx
    y1 *= sy
    y2 *= sy

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    boxes_xyxy[:, 0::2] = np.clip(boxes_xyxy[:, 0::2], 0, w0 - 1)
    boxes_xyxy[:, 1::2] = np.clip(boxes_xyxy[:, 1::2], 0, h0 - 1)

    return boxes_xyxy.astype(np.float32), scores.astype(np.float32)


def compute_LWD(boxes_xyxy):
    """Return L,W,D in pixels or mm depending on PIXELS_PER_MM."""
    if boxes_xyxy.shape[0] == 0:
        return (np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))
    arr = np.asarray(boxes_xyxy, np.float32)
    L = np.abs(arr[:, 2] - arr[:, 0])
    W = np.abs(arr[:, 3] - arr[:, 1])
    D = np.sqrt(L * L + W * W)
    if PIXELS_PER_MM:
        s = float(PIXELS_PER_MM)
        L, W, D = L / s, W / s, D / s
    return L, W, D


# =================== REFERENCE OBJECT SCALE ===================
def detect_white_reference_object(frame_bgr):
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 80, 40], np.uint8)
    upper = np.array([130, 225, 255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 2:
        contours, _ = cnts
    else:
        _, contours, _ = cnts

    best = None
    best_score = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        if area > 0.5 * h * w:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / float(bh + 1e-6)
        if aspect < 0.6 or aspect > 1.6:
            continue

        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        circ = 4.0 * np.pi * area / (perim * perim)

        score = circ * area
        if score > best_score:
            (cx_f, cy_f), radius = cv2.minEnclosingCircle(cnt)
            best = (int(cx_f), int(cy_f), float(2.0 * radius))
            best_score = score

    return best


def update_scale_from_reference(frame_bgr):
    global PIXELS_PER_MM, UNITS, SCALE_BUF

    ref = detect_white_reference_object(frame_bgr)
    if ref is None or REF_DIAM_MM <= 0:
        return None

    cx, cy, diameter_px = ref
    ppm_new = diameter_px / float(REF_DIAM_MM)

    SCALE_BUF.append(ppm_new)
    if len(SCALE_BUF) > SCALE_ROLLING_MAX:
        del SCALE_BUF[: len(SCALE_BUF) - SCALE_ROLLING_MAX]

    if PIXELS_PER_MM is None:
        PIXELS_PER_MM = ppm_new
    else:
        PIXELS_PER_MM = 0.9 * PIXELS_PER_MM + 0.1 * ppm_new

    UNITS = "mm"
    return cx, cy, diameter_px


# =================== UI/STAT HELPERS ===================
def classify_batch(D_buf):
    if len(D_buf) < 10:
        return "Collecting..."
    std = np.std(D_buf)
    if std < 30:
        return "Uniform"
    elif std < 60:
        return "Moderate"
    else:
        return "Diverse"


def compute_diameter_stats(D_buf, units, scale_mean=None, scale_std=None):
    if len(D_buf) == 0:
        return None
    D_arr = np.array(D_buf, dtype=np.float32)
    stats = {
        "ready": True,
        "mean_d": float(np.mean(D_arr)),
        "median_d": float(np.median(D_arr)),
        "std_d": float(np.std(D_arr)),
        "min_d": float(np.min(D_arr)),
        "max_d": float(np.max(D_arr)),
        "batch_label": classify_batch(D_buf),
        "units": units,
        "px_per_mm_mean": float(scale_mean) if scale_mean is not None else None,
        "px_per_mm_std": float(scale_std) if scale_std is not None else None,
    }
    return stats


def draw_stats_panel(frame, D_buf, units, scale_mean=None, scale_std=None):
    h, w = frame.shape[:2]
    panel_w, panel_h = 280, 190
    x0, y0 = w - panel_w - 10, 30
    panel = np.zeros((panel_h, panel_w, 3), np.uint8)
    panel[:] = (20, 20, 20)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 25
    cv2.putText(panel, "Diameter Stats", (10, y), font, 0.8, (0, 255, 255), 2)

    if len(D_buf) > 0:
        D_arr = np.array(D_buf, dtype=np.float32)
        mean = float(np.mean(D_arr))
        med = float(np.median(D_arr))
        std = float(np.std(D_arr))
        dmin = float(np.min(D_arr))
        dmax = float(np.max(D_arr))
        batch = classify_batch(D_buf)

        y += 25
        cv2.putText(panel, f"Mean : {mean:.1f} {units}", (10, y), font, 0.7, (0, 255, 0), 1)
        y += 22
        cv2.putText(panel, f"Median: {med:.1f} {units}", (10, y), font, 0.7, (0, 255, 0), 1)
        y += 22
        cv2.putText(panel, f"Std  : {std:.1f} {units}", (10, y), font, 0.7, (0, 255, 0), 1)
        y += 22
        cv2.putText(panel, f"Min  : {dmin:.1f} {units}", (10, y), font, 0.7, (0, 255, 0), 1)
        y += 22
        cv2.putText(panel, f"Max  : {dmax:.1f} {units}", (10, y), font, 0.7, (0, 255, 0), 1)
        y += 22
        cv2.putText(panel, f"Batch: {batch}", (10, y), font, 0.7, (255, 200, 0), 1)

        if scale_mean is not None and scale_mean > 0:
            y += 22
            if scale_std is not None:
                cv2.putText(panel, f"px/mm: {scale_mean:.1f} +/- {scale_std:.1f}", (10, y), font, 0.7, (200, 200, 255), 1)
            else:
                cv2.putText(panel, f"px/mm: {scale_mean:.1f}", (10, y), font, 0.7, (200, 200, 255), 1)
    else:
        y += 50
        cv2.putText(panel, "No data yet...", (10, y), font, 0.7, (0, 0, 255), 1)

    y1, y2 = y0, y0 + panel_h
    x1, x2 = x0, x0 + panel_w
    frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.4, panel, 0.6, 0)
    return frame


def draw_color_legend(frame):
    # UPDATED: two-color legend (Green=Normal, Red=Oversized)
    h, w = frame.shape[:2]
    legend = np.zeros((30, 300, 3), np.uint8)
    legend[:] = (40, 40, 40)

    # green block
    legend[:, 0:140] = (0, 200, 0)
    cv2.putText(legend, "Normal", (10, 20), 0, 0.55, (0, 0, 0), 1)

    # separator
    legend[:, 140:160] = (40, 40, 40)
    cv2.putText(legend, "|", (148, 20), 0, 0.55, (255, 255, 255), 1)

    # red block
    legend[:, 160:300] = (0, 0, 255)
    cv2.putText(legend, "Oversize", (170, 20), 0, 0.55, (255, 255, 255), 1)

    frame[h - 40 : h - 10, 10:310] = legend
    return frame


# =================== TRT WRAPPERS (stable) ===================
class GenericTRTWrapper:
    """
    Stable TRT wrapper:
    - single global TRT_LOGGER
    - re-alloc bindings when FULL input shape changes
    - uses binding names to resolve outputs if needed
    """
    def __init__(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        self.logger = TRT_LOGGER
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.num_bindings = self.engine.num_bindings
        self.binding_names = [self.engine.get_binding_name(i) for i in range(self.num_bindings)]

        self.input_idx = [i for i in range(self.num_bindings) if self.engine.binding_is_input(i)][0]
        self.output_indices = [i for i in range(self.num_bindings) if not self.engine.binding_is_input(i)]

        self.bindings = [None] * self.num_bindings
        self.host_mem = {}
        self.device_mem = {}
        self.last_input_shape = None

    def _alloc(self, idx: int, shape: tuple):
        dtype = trt.nptype(self.engine.get_binding_dtype(idx))
        n_elts = int(np.prod(shape))
        host = cuda.pagelocked_empty(n_elts, dtype)
        dev = cuda.mem_alloc(host.nbytes)
        self.host_mem[idx] = host
        self.device_mem[idx] = dev
        self.bindings[idx] = int(dev)

    def _ensure_buffers(self, input_shape: tuple):
        input_shape = tuple(input_shape)
        if self.last_input_shape == input_shape:
            return

        self.context.set_binding_shape(self.input_idx, input_shape)

        # allocate ALL bindings based on shapes AFTER set_binding_shape
        for i in range(self.num_bindings):
            shape = tuple(self.context.get_binding_shape(i))
            self._alloc(i, shape)

        self.last_input_shape = input_shape

    def infer_all(self, x: np.ndarray):
        x = np.ascontiguousarray(x, dtype=np.float32)
        self._ensure_buffers(x.shape)

        np.copyto(self.host_mem[self.input_idx], x.ravel())
        cuda.memcpy_htod_async(self.device_mem[self.input_idx], self.host_mem[self.input_idx], self.stream)

        self.context.execute_async_v2(self.bindings, self.stream.handle)

        for oi in self.output_indices:
            cuda.memcpy_dtoh_async(self.host_mem[oi], self.device_mem[oi], self.stream)

        self.stream.synchronize()

        outs = {}
        for oi in self.output_indices:
            shape = tuple(self.context.get_binding_shape(oi))
            name = self.binding_names[oi] or f"out_{oi}"
            outs[name] = np.array(self.host_mem[oi], copy=True).reshape(shape)
        return outs


class DETRTRT(GenericTRTWrapper):
    """Return (logits, pred_boxes) exactly like your old code expected."""
    def infer(self, x: np.ndarray):
        outs = self.infer_all(x)

        logits = None
        pred_boxes = None

        for k, v in outs.items():
            lk = (k or "").lower()
            if ("logit" in lk) and (logits is None):
                logits = v
            if (("pred_boxes" in lk) or ("box" in lk)) and (pred_boxes is None):
                pred_boxes = v

        # Fallback if names aren't helpful
        if logits is None or pred_boxes is None:
            vals = list(outs.values())
            if len(vals) == 2:
                a, b = vals
                if a.shape[-1] == 4:
                    pred_boxes, logits = a, b
                elif b.shape[-1] == 4:
                    pred_boxes, logits = b, a
                else:
                    logits, pred_boxes = a, b
            else:
                raise RuntimeError("Unable to resolve DETR TRT outputs (logits/pred_boxes).")

        return logits, pred_boxes


class MoistureTRT(GenericTRTWrapper):
    """Return probabilities (N,C)."""
    def infer_probs_bhwc(self, crops_bhwc: np.ndarray):
        if crops_bhwc is None or len(crops_bhwc) == 0:
            return None

        # engine may expect BCHW or BHWC
        input_shape = tuple(self.engine.get_binding_shape(self.input_idx))
        if len(input_shape) == 4 and input_shape[-1] == 3:
            x = crops_bhwc
        else:
            x = np.transpose(crops_bhwc, (0, 3, 1, 2))

        outs = self.infer_all(x)
        out = list(outs.values())[0]
        if out.ndim > 2:
            out = out.reshape(out.shape[0], -1)

        # softmax if not already probs
        row_sum = np.sum(out, axis=1)
        if np.any(row_sum < 0.8) or np.any(row_sum > 1.2) or np.any(out < 0.0) or np.any(out > 1.0):
            out = safe_softmax_2d(out, axis=1)

        return out


# =================== MOISTURE PRE/POST ===================
def _read_moist_classes(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]

MOISTURE_CLASSES = _read_moist_classes(MOIST_CLASSES_PATH)


def _prep_moist_crop_bgr(crop_bgr: np.ndarray, size: int):
    crop = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    crop = (crop - MEAN) / STD
    return crop.astype(np.float32)  # HWC


def _boxes_to_topk_crops(frame_bgr: np.ndarray, boxes_xyxy: np.ndarray, scores: np.ndarray, topk: int):
    """
    Return:
      crops_bhwc (N,H,W,3) float32 normalized
      used_boxes list of dicts (coords + score)
    """
    h, w = frame_bgr.shape[:2]
    if boxes_xyxy is None or boxes_xyxy.shape[0] == 0:
        return None, []

    order = np.argsort(scores)[::-1]
    crops = []
    used = []

    for idx in order[: max(1, topk)]:
        x1, y1, x2, y2 = boxes_xyxy[idx].astype(int).tolist()
        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, 0, h - 1))

        bw = x2 - x1
        bh = y2 - y1
        if bw < MOISTURE_MIN_BOX_PX or bh < MOISTURE_MIN_BOX_PX:
            continue

        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue

        crops.append(_prep_moist_crop_bgr(crop_bgr, MOISTURE_INPUT_SIZE))
        used.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": float(scores[idx])})

        if len(crops) >= topk:
            break

    if len(crops) == 0:
        return None, []
    return np.stack(crops, axis=0).astype(np.float32), used


def _moisture_aggregate(probs: np.ndarray):
    mean_probs = np.mean(probs, axis=0)
    pred_idx = int(np.argmax(mean_probs))
    if MOISTURE_CLASSES and pred_idx < len(MOISTURE_CLASSES):
        pred_label = MOISTURE_CLASSES[pred_idx]
    else:
        pred_label = str(pred_idx)
    return mean_probs, pred_idx, pred_label


# =================== MAIN LOOP ===================
def run_inference_loop(headless: bool = True):
    global UNITS, SCALE_BUF, LATEST_FRAME, LATEST_STATS, HIST_DATA, LATEST_MOISTURE, LATEST_HEALTH
    global ALARM_ACTIVE, ALARM_MAX_D_MM

    print("[live_cam_trt] Initializing CUDA in inference thread...", flush=True)
    cuda.init()
    dev = cuda.Device(0)
    ctx = dev.make_context()

    cap = None
    detr = None
    moist = None

    # --- Health/telemetry state (for /api/devices/status) ---
    start_ts = _now()
    last_frame_ts = None
    last_detr_ts = None
    last_moist_ts = None
    last_hist_pub_ts = None
    last_error = None

    try:
        print("[live_cam_trt] Opening camera:", CAM_DEV, flush=True)
        cap = cv2.VideoCapture(CAM_DEV, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

        for _ in range(5):
            cap.read()

        if not cap.isOpened():
            print("[live_cam_trt] ERROR: camera failed to open.", flush=True)
            err = f"camera failed to open: {CAM_DEV}"
            last_error = err

            with STATE_LOCK:
                LATEST_STATS = {"ready": False, "error": err, "timestamp": _now()}
                HIST_DATA = {"ready": False, "error": "camera failed to open", "timestamp": _now()}
                LATEST_MOISTURE = {"ready": False, "error": "camera failed to open", "timestamp": _now()}
                LATEST_HEALTH = {
                    "ready": False,
                    "timestamp": _now(),
                    "uptime_sec": float(_now() - start_ts),
                    "camera_ok": False,
                    "cam_dev": str(CAM_DEV),
                    "last_frame_ts": None,
                    "last_detr_ts": None,
                    "last_moist_ts": None,
                    "last_hist_pub_ts": None,
                    "fps_smoothed": 0.0,
                    "frame_idx": 0,
                    "processed_idx": 0,
                    "infer_every_n_frames": int(INFER_EVERY_N_FRAMES),
                    "units": str(UNITS),
                    "alarm_active": bool(ALARM_ACTIVE),
                    "alarm_max_d_mm": None,
                    "last_error": err,
                }
            return

        print("[live_cam_trt] Loading DETR TensorRT engine:", ENGINE_PATH, flush=True)
        detr = DETRTRT(ENGINE_PATH)
        print("[live_cam_trt] DETR engine loaded OK", flush=True)

        if MOISTURE_ENABLED:
            print("[live_cam_trt] Loading Moisture TensorRT engine:", MOIST_ENGINE_PATH, flush=True)
            moist = MoistureTRT(MOIST_ENGINE_PATH)
            print("[live_cam_trt] Moisture engine loaded OK", flush=True)

        L_buf, W_buf, D_buf = [], [], []
        fps, t_prev = 0.0, time.time()

        frame_idx = 0
        processed_idx = 0

        # Keep last inference results to overlay on skipped frames
        last_boxes_xyxy = np.zeros((0, 4), np.float32)
        last_scores = np.zeros((0,), np.float32)
        last_D = np.zeros((0,), np.float32)

        # Histogram throttling + EMA state (diameter only)
        last_hist_time = 0.0
        prev_hist_counts = None  # np.float32 array
        prev_hist_bins = None    # centers

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # keep loop alive, but publish health that we are not receiving frames
                last_error = "cap.read() failed"
                print("[live_cam_trt] WARNING: cap.read() failed, continuing...", flush=True)
                time.sleep(0.05)

                with STATE_LOCK:
                    LATEST_HEALTH = {
                        "ready": True,
                        "timestamp": _now(),
                        "uptime_sec": float(_now() - start_ts),
                        "camera_ok": True,  # camera object exists; frame read failed transiently
                        "cam_dev": str(CAM_DEV),
                        "last_frame_ts": last_frame_ts,
                        "last_detr_ts": last_detr_ts,
                        "last_moist_ts": last_moist_ts,
                        "last_hist_pub_ts": last_hist_pub_ts,
                        "fps_smoothed": float(fps),
                        "frame_idx": int(frame_idx),
                        "processed_idx": int(processed_idx),
                        "infer_every_n_frames": int(INFER_EVERY_N_FRAMES),
                        "units": str(UNITS),
                        "alarm_active": bool(ALARM_ACTIVE),
                        "alarm_max_d_mm": float(ALARM_MAX_D_MM) if ALARM_MAX_D_MM is not None else None,
                        "last_error": str(last_error),
                    }
                continue

            last_frame_ts = _now()
            frame_idx += 1

            # --- scale estimation from reference object ---
            ref_info = update_scale_from_reference(frame)

            if len(SCALE_BUF) > 0:
                scale_mean = float(np.mean(SCALE_BUF))
                scale_std = float(np.std(SCALE_BUF))
            else:
                scale_mean = None
                scale_std = None

            # Processed FPS control
            do_infer = (frame_idx % max(1, INFER_EVERY_N_FRAMES) == 0)

            if do_infer:
                processed_idx += 1

                # --- DETR inference ---
                x, meta = preprocess_fixed(frame)
                try:
                    logits, pred_boxes = detr.infer(x)
                    boxes_xyxy, scores = postprocess_from_fixed(pred_boxes, logits, meta, conf_thr=CONF_THR)
                    last_detr_ts = _now()
                    # clear last_error only on a successful infer (keeps latest meaningful error otherwise)
                    if last_error and str(last_error).startswith("DETR infer failed"):
                        last_error = None
                except Exception as e:
                    err = f"DETR infer failed: {e}"
                    last_error = err
                    print("[live_cam_trt] " + err, flush=True)
                    print(traceback.format_exc(), flush=True)
                    with STATE_LOCK:
                        LATEST_STATS = {"ready": False, "error": err, "timestamp": _now()}
                    boxes_xyxy = np.zeros((0, 4), np.float32)
                    scores = np.zeros((0,), np.float32)

                if boxes_xyxy.shape[0] > 0:
                    keep = nms_np(boxes_xyxy, scores, iou_thr=NMS_IOU)
                    boxes_xyxy = boxes_xyxy[keep]
                    scores = scores[keep]

                # --- compute sizes ---
                L, W, D = compute_LWD(boxes_xyxy)

                if D.size:
                    L_buf.extend(L.tolist())
                    W_buf.extend(W.tolist())
                    D_buf.extend(D.tolist())

                    if len(L_buf) > ROLLING_MAX:
                        del L_buf[: len(L_buf) - ROLLING_MAX]
                    if len(W_buf) > ROLLING_MAX:
                        del W_buf[: len(W_buf) - ROLLING_MAX]
                    if len(D_buf) > ROLLING_MAX:
                        del D_buf[: len(D_buf) - ROLLING_MAX]

                # --- Oversize alarm logic (mm after calibration) ---
                alarm_active = False
                max_d_mm = None
                if ALARM_ENABLED and PIXELS_PER_MM and D.size > 0:
                    max_d_mm = float(np.max(D))
                    if max_d_mm > ALARM_THRESHOLD_MM:
                        alarm_active = True
                ALARM_ACTIVE = alarm_active
                ALARM_MAX_D_MM = max_d_mm

                # Save last results for overlay on skipped frames
                last_boxes_xyxy = boxes_xyxy.copy()
                last_scores = scores.copy()
                last_D = D.copy()
            else:
                boxes_xyxy = last_boxes_xyxy
                scores = last_scores
                D = last_D

            vis = frame.copy()

            # --- draw reference object ---
            if ref_info is not None:
                cx, cy, diam_px = ref_info
                r = int(diam_px / 2.0)
                cv2.circle(vis, (cx, cy), r, (255, 255, 255), 2)
                cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

            # --- draw boxes with TWO colors only ---
            for (x1, y1, x2, y2), sc, d_val in zip(boxes_xyxy.astype(int), scores, D):
                is_oversized = (ALARM_ENABLED and (PIXELS_PER_MM is not None) and (d_val > ALARM_THRESHOLD_MM))

                color_bgr = (0, 0, 255) if is_oversized else (0, 255, 0)  # RED if oversized else GREEN
                cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr, 2)

                # oversized badge (kept)
                if is_oversized:
                    box_w = max(1, x2 - x1)
                    box_h = max(1, y2 - y1)
                    badge_r = int(max(6, min(12, min(box_w, box_h) * 0.18)))
                    cx_badge = x2 - badge_r - 2
                    cy_badge = y1 + badge_r + 2
                    cv2.circle(vis, (cx_badge, cy_badge), badge_r, (0, 0, 255), -1)
                    cv2.putText(
                        vis,
                        "!",
                        (cx_badge - badge_r // 3, cy_badge + badge_r // 2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        lineType=cv2.LINE_AA,
                    )

                if not headless:
                    cv2.putText(vis, f"{sc:.2f}", (x1, max(0, y1 - 6)), 0, 0.5, color_bgr, 1)

            # --- FPS ---
            now = time.time()
            fps_inst = 1.0 / max(1e-3, now - t_prev)
            fps = 0.9 * fps + 0.1 * fps_inst
            t_prev = now
            if not headless:
                cv2.putText(vis, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # =================== Moisture (Top-K crops) ===================
            # Updated: cadence tied to PROCESSED frames (do_infer) to keep it consistent after frame skipping
            if MOISTURE_ENABLED and moist and do_infer and (processed_idx % max(1, MOISTURE_EVERY_N_FRAMES) == 0):
                try:
                    crops_bhwc, used_boxes = _boxes_to_topk_crops(frame, boxes_xyxy, scores, MOISTURE_TOPK)

                    if (crops_bhwc is None or len(used_boxes) == 0) and MOISTURE_FALLBACK_FULLFRAME:
                        crops_bhwc = np.stack([_prep_moist_crop_bgr(frame, MOISTURE_INPUT_SIZE)], axis=0).astype(np.float32)
                        used_boxes = []

                    if crops_bhwc is not None:
                        probs = moist.infer_probs_bhwc(crops_bhwc)
                        if probs is not None:
                            mean_probs, pred_idx, pred_label = _moisture_aggregate(probs)
                            last_moist_ts = _now()
                            if last_error and str(last_error).startswith("Moisture infer failed"):
                                last_error = None

                            per_box = []
                            if used_boxes:
                                for i, ub in enumerate(used_boxes):
                                    pi = int(np.argmax(probs[i]))
                                    pl = MOISTURE_CLASSES[pi] if MOISTURE_CLASSES and pi < len(MOISTURE_CLASSES) else str(pi)
                                    per_box.append({
                                        "box": ub,
                                        "pred": pl,
                                        "pred_index": pi,
                                        "probs": (
                                            {MOISTURE_CLASSES[j]: float(probs[i][j]) for j in range(probs.shape[1])}
                                            if MOISTURE_CLASSES and len(MOISTURE_CLASSES) == probs.shape[1]
                                            else [float(x) for x in probs[i]]
                                        )
                                    })

                            with STATE_LOCK:
                                LATEST_MOISTURE = {
                                    "ready": True,
                                    "timestamp": _now(),
                                    "classes": MOISTURE_CLASSES if MOISTURE_CLASSES else None,
                                    "mean_pred": pred_label,
                                    "mean_pred_index": pred_idx,
                                    "mean_probs": (
                                        {MOISTURE_CLASSES[i]: float(mean_probs[i]) for i in range(len(mean_probs))}
                                        if MOISTURE_CLASSES and len(MOISTURE_CLASSES) == len(mean_probs)
                                        else [float(x) for x in mean_probs]
                                    ),
                                    "topk": int(MOISTURE_TOPK),
                                    "boxes_used": int(len(used_boxes)),
                                    "per_box": per_box,
                                }
                    else:
                        with STATE_LOCK:
                            LATEST_MOISTURE = {
                                "ready": False,
                                "timestamp": _now(),
                                "error": "No crops available for moisture (no boxes and fallback disabled).",
                                "topk": int(MOISTURE_TOPK),
                                "boxes_used": 0,
                            }
                except Exception as e:
                    err = f"Moisture infer failed: {e}"
                    last_error = err
                    print("[live_cam_trt] " + err, flush=True)
                    print(traceback.format_exc(), flush=True)
                    with STATE_LOCK:
                        LATEST_MOISTURE = {"ready": False, "timestamp": _now(), "error": err}

            # =================== Update SHARED STATE ===================
            stats_dict = compute_diameter_stats(D_buf, UNITS, scale_mean=scale_mean, scale_std=scale_std)
            if stats_dict is not None:
                stats_dict["alarm_active"] = bool(ALARM_ACTIVE)
                stats_dict["alarm_threshold_mm"] = float(ALARM_THRESHOLD_MM)
                stats_dict["alarm_max_d_mm"] = float(ALARM_MAX_D_MM) if ALARM_MAX_D_MM is not None else None
                stats_dict["ref_diam_mm"] = float(REF_DIAM_MM)

            # histogram package + threshold line info (UPDATED: diameter only, throttled to 1 Hz)
            hist_data = None
            should_update_hist = (now - last_hist_time) >= float(HIST_UPDATE_INTERVAL)

            if should_update_hist and len(D_buf) > 10:
                last_hist_time = now

                def make_diameter_hist(values):
                    vals = np.array(values, dtype=np.float32)
                    if len(vals) < 5:
                        return None

                    # Fixed range when calibrated (mm); percentile range when still in pixels
                    if UNITS == "mm":
                        upper = float(HIST_D_MAX_MM)
                        upper = max(1.0, upper)
                        edges = np.linspace(0.0, upper, HIST_NUM_BINS + 1, dtype=np.float32)
                    else:
                        upper = float(np.percentile(vals, 99.0))
                        if upper <= 0:
                            upper = float(np.max(vals)) if len(vals) > 0 else 1.0
                        upper = max(1.0, upper)
                        edges = np.linspace(0.0, upper, HIST_NUM_BINS + 1, dtype=np.float32)

                    hist, _ = np.histogram(vals, bins=edges)
                    centers = 0.5 * (edges[:-1] + edges[1:])

                    nonlocal prev_hist_counts, prev_hist_bins  # type: ignore
                    if HIST_EMA_ENABLED:
                        if prev_hist_counts is not None and prev_hist_bins is not None and len(prev_hist_counts) == len(hist) and np.allclose(prev_hist_bins, centers):
                            sm = (1.0 - HIST_EMA_ALPHA) * prev_hist_counts + HIST_EMA_ALPHA * hist.astype(np.float32)
                            hist_out = np.rint(sm).astype(int)
                        else:
                            hist_out = hist.astype(int)

                        prev_hist_counts = hist_out.astype(np.float32)
                        prev_hist_bins = centers.copy()
                    else:
                        hist_out = hist.astype(int)

                    return {"bins": centers.tolist(), "counts": hist_out.tolist()}

                h_dia = make_diameter_hist(D_buf)

                threshold_x = None
                pct_over_threshold = None
                if ALARM_ENABLED and UNITS == "mm" and len(D_buf) > 0:
                    thr = float(ALARM_THRESHOLD_MM)
                    threshold_x = thr
                    darr = np.array(D_buf, dtype=np.float32)
                    pct_over_threshold = float(100.0 * np.mean(darr > thr))

                hist_data = {
                    "ready": True,
                    "timestamp": _now(),
                    "units": UNITS,
                    "mode": "diameter",
                    "diameter": h_dia,

                    "threshold_x": threshold_x,
                    "pct_over_threshold": pct_over_threshold,
                    "alarm_threshold_mm": float(ALARM_THRESHOLD_MM),
                    "alarm_enabled": bool(ALARM_ENABLED),

                    "hist_update_interval": float(HIST_UPDATE_INTERVAL),
                    "hist_num_bins": int(HIST_NUM_BINS),
                    "hist_d_max_mm": float(HIST_D_MAX_MM) if UNITS == "mm" else None,
                    "hist_ema_enabled": bool(HIST_EMA_ENABLED),
                    "hist_ema_alpha": float(HIST_EMA_ALPHA) if HIST_EMA_ENABLED else None,
                }
                last_hist_pub_ts = _now()

            # --- publish shared state (frame/stats/hist/health) ---
            health_dict = {
                "ready": True,
                "timestamp": _now(),
                "uptime_sec": float(_now() - start_ts),
                "camera_ok": True,
                "cam_dev": str(CAM_DEV),

                "last_frame_ts": last_frame_ts,
                "last_detr_ts": last_detr_ts,
                "last_moist_ts": last_moist_ts,
                "last_hist_pub_ts": last_hist_pub_ts,

                "fps_smoothed": float(fps),
                "frame_idx": int(frame_idx),
                "processed_idx": int(processed_idx),
                "infer_every_n_frames": int(INFER_EVERY_N_FRAMES),

                "units": str(UNITS),
                "alarm_active": bool(ALARM_ACTIVE),
                "alarm_threshold_mm": float(ALARM_THRESHOLD_MM),
                "alarm_max_d_mm": float(ALARM_MAX_D_MM) if ALARM_MAX_D_MM is not None else None,

                "last_error": str(last_error) if last_error else None,
            }

            with STATE_LOCK:
                LATEST_FRAME = vis.copy()
                LATEST_STATS = stats_dict if stats_dict is not None else {"ready": False, "timestamp": _now(), "error": "No detections yet."}
                if hist_data is not None:
                    HIST_DATA = hist_data
                # else keep last HIST_DATA to avoid blanking between 1Hz updates

                LATEST_HEALTH = health_dict

            # Debug GUI
            if not headless:
                disp = draw_stats_panel(vis.copy(), D_buf, UNITS, scale_mean=scale_mean, scale_std=scale_std)
                disp = draw_color_legend(disp)
                cv2.imshow("woodchip-live", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.001)

        print("[live_cam_trt] Exiting inference loop.", flush=True)

    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            del moist
        except Exception:
            pass
        try:
            del detr
        except Exception:
            pass

        try:
            ctx.pop()
        except Exception:
            pass


def main():
    run_inference_loop(headless=False)


if __name__ == "__main__":
    main()

