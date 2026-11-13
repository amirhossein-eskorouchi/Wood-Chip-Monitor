# woodChipGUI.py
"""
Wood-Chip Monitor - Live Motion Tracking and Inference System
Fully compatible with both Windows and Linux.

Key Features:
- Cross-platform hardware capture (Windows DirectShow/MSMF and Linux V4L2)
- Automatically loads GUI window displaying live camera feed
- Real-time ONLINE / OFFLINE camera status indicator
- Live auto-reconnection and seamless hot-plugging (handles USB plug/unplug)
- Multi-tier inference engine:
    1. TensorRT (Linux / Jetson GPU optimized)
    2. PyTorch DETR (Windows / Linux GPU/CPU)
    3. OpenCV Contour & Motion Tracker (fast zero-dependency fallback)
- Real-time diameter statistics, histogram analysis, and oversize alarm
- Fast startup with lazy model loading
- Thread-safe shared state for FastAPI / Web Dashboard integration
"""

import os
import sys
import time
from datetime import datetime, timezone
import platform
import math
import csv
import queue
from collections import deque
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import threading
import traceback
import warnings

# Suppress warning noise
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# =================== PLATFORM DETECTION ===================
IS_WINDOWS = (platform.system() == "Windows")
IS_LINUX = (platform.system() == "Linux")
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Lazy-loaded inference backends
_HAS_TRT = False
trt = None
cuda = None

_HAS_TORCH = False
torch = None
nn = None
transformers = None
DetrForObjectDetection = None
DetrImageProcessor = None


def _check_and_import_trt():
    """Attempt to import TensorRT and PyCUDA if available."""
    global _HAS_TRT, trt, cuda
    if _HAS_TRT:
        return True
    try:
        import tensorrt as _trt
        import pycuda.driver as _cuda
        trt = _trt
        cuda = _cuda
        _HAS_TRT = True
        return True
    except (ImportError, Exception):
        _HAS_TRT = False
        return False


def _check_and_import_torch():
    """Attempt to import PyTorch and Transformers if available."""
    global _HAS_TORCH, torch, nn, transformers, DetrForObjectDetection, DetrImageProcessor
    if _HAS_TORCH:
        return True
    try:
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        import torch as _torch
        import torch.nn as _nn
        import transformers as _transformers
        from transformers import DetrForObjectDetection as _DetrForObjectDetection
        from transformers import DetrImageProcessor as _DetrImageProcessor

        torch = _torch
        nn = _nn
        transformers = _transformers
        DetrForObjectDetection = _DetrForObjectDetection
        DetrImageProcessor = _DetrImageProcessor
        _HAS_TORCH = True
        return True
    except (ImportError, Exception):
        _HAS_TORCH = False
        return False


# =================== CONFIG & PATH RESOLUTION ===================
def _find_candidate_path(env_var_name: str, filename: str, search_dirs: list) -> str:
    """Find file in environment variable or common candidate directories."""
    env_val = os.environ.get(env_var_name)
    if env_val and os.path.exists(env_val):
        return env_val
    for s in search_dirs:
        p = os.path.join(_BASE_DIR, s, filename) if s else os.path.join(_BASE_DIR, filename)
        if os.path.exists(p):
            return p
    return os.path.join(_BASE_DIR, search_dirs[0] if search_dirs else "", filename)


ENGINE_PATH = _find_candidate_path(
    "WOODCHIP_ENGINE_PATH",
    "detr_resnet101_fp16.engine",
    ["models/artifacts", "models", "best models", ""]
)
CKPT_PATH = _find_candidate_path(
    "WOODCHIP_CKPT_PATH",
    "best-detr.ckpt",
    ["models/artifacts", "models", "best models", ""]
)
MOIST_ENGINE_PATH = _find_candidate_path(
    "WOODCHIP_MOIST_ENGINE_PATH",
    "moistnetlite_fp16.engine",
    ["models/artifacts", "models", "best models", ""]
)
MOIST_CLASSES_PATH = _find_candidate_path(
    "WOODCHIP_MOIST_CLASSES_PATH",
    "moistnetlite_classes.txt",
    ["models/artifacts", "models", "best models", ""]
)

OUT_DIR = os.environ.get("WOODCHIP_OUT_DIR", os.path.join(_BASE_DIR, "live_outputs"))
os.makedirs(OUT_DIR, exist_ok=True)


# Camera device configuration (handles both Windows integer indices and Linux /dev/video nodes)
def _resolve_camera_dev(raw_dev):
    if raw_dev is None:
        return 0
    if isinstance(raw_dev, int):
        return raw_dev
    raw_str = str(raw_dev).strip()
    if raw_str.isdigit():
        return int(raw_str)
    if IS_WINDOWS:
        # On Windows, /dev/videoN cannot be opened directly by OpenCV. Extract index.
        import re
        match = re.search(r"video(\d+)", raw_str, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0
    return raw_str


CAM_DEV = _resolve_camera_dev(os.environ.get("WOODCHIP_CAM_DEV", 0))
CAM_W = int(os.environ.get("WOODCHIP_CAMERA_WIDTH", "1280"))
CAM_H = int(os.environ.get("WOODCHIP_CAMERA_HEIGHT", "720"))
CAM_FPS = int(os.environ.get("WOODCHIP_CAMERA_FPS", "20"))

# Processed FPS control (camera 20 FPS, infer every 2 frames => ~10 FPS)
INFER_EVERY_N_FRAMES = 2

# DETR input handling
ENGINE_INPUT_MODE = "fixed"
FIXED_H, FIXED_W = 800, 1333
SHORTEST_EDGE, LONGEST_EDGE = 800, 1333

CONF_THR, NMS_IOU = 0.5, 0.5

# Reference object config
REF_DIAM_MM = 110.0   # Real diameter of white calibration disk in mm
PIXELS_PER_MM = None  # Estimated online from the white object
DEFAULT_PIXELS_PER_MM = float(os.environ.get("WOODCHIP_PIXELS_PER_MM", "2.0"))  # Default scale when reference disk not in view
UNITS = "mm"

# Rolling buffer of px/mm estimates
SCALE_BUF = []
SCALE_ROLLING_MAX = 500

# Alarm
ALARM_THRESHOLD_MM = 60.0
ALARM_ENABLED = True
ALARM_ACTIVE = False
ALARM_MAX_D_MM = None

# Max Bounding Box Filter (Optional toggle: ignore false positives > 1000mm)
FILTER_MAX_1000MM = True
MAX_SIZE_LIMIT_MM = 800.0

# High-Speed Conveyor Motion Tracker & Lock-In Config
CONVEYOR_TRACKING_ENABLED = True
TRACKER_MAX_MATCH_DIST = 180.0  # Pixels (accommodates high-speed conveyor belt motion)
TRACKER_MAX_LOST_FRAMES = 8     # Retain track through brief occlusion/misses
TRACKER_MIN_HITS_TO_LOCK = 2    # Detections required to confirm and lock in

# CSV Data Logging ('yyyymmdd_woodChips.csv')
CSV_LOGGING_ENABLED = True
CSV_LOG_THROTTLE_SEC = 0.25     # Throttle write interval per chip to prevent oversized files

# Histogram mode
HISTOGRAM_MODE = "diameter"
ROLLING_MAX = 2000
HIST_UPDATE_INTERVAL = 1.0  # 1 Hz
HIST_NUM_BINS = 20
HIST_D_MAX_MM = 120.0
HIST_EMA_ENABLED = True
HIST_EMA_ALPHA = 0.2

# Moisture runtime config
MOISTURE_ENABLED = True
MOISTURE_TOPK = 8
MOISTURE_EVERY_N_FRAMES = 3
MOISTURE_INPUT_SIZE = 224
MOISTURE_MIN_BOX_PX = 12
MOISTURE_FALLBACK_FULLFRAME = True

# Camera reconnection interval (seconds)
CAMERA_RECONNECT_INTERVAL = 2.0

# Normalization (ImageNet)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# =================== SHARED STATE for backend_app.py ===================
STATE_LOCK = threading.Lock()

LATEST_FRAME = None     # BGR np.uint8 image (annotated / status screen)
LATEST_STATS = None     # dict
HIST_DATA = None        # dict
LATEST_MOISTURE = None  # dict
LATEST_HEALTH = None    # dict
GLOBAL_T_CAP = None     # ThreadedCamera instance


def force_camera_reconnect():
    """Manual trigger to restart camera hardware capture."""
    global GLOBAL_T_CAP
    if GLOBAL_T_CAP:
        print("[motionTracking] Manual reconnect triggered.", flush=True)
        GLOBAL_T_CAP.trigger_reconnect()
        return True
    return False


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
        "camera_fps": int(CAM_FPS),
        "infer_every_n_frames": int(INFER_EVERY_N_FRAMES),
        "rolling_max": int(ROLLING_MAX),
        "hist_update_interval": float(HIST_UPDATE_INTERVAL),
        "hist_num_bins": int(HIST_NUM_BINS),
        "hist_d_max_mm": float(HIST_D_MAX_MM),
        "hist_ema_enabled": bool(HIST_EMA_ENABLED),
        "hist_ema_alpha": float(HIST_EMA_ALPHA),
        "filter_max_1000mm": bool(FILTER_MAX_1000MM),
        "max_size_limit_mm": float(MAX_SIZE_LIMIT_MM),
        "conveyor_tracking_enabled": bool(CONVEYOR_TRACKING_ENABLED),
        "csv_logging_enabled": bool(CSV_LOGGING_ENABLED),
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
    filter_max_1000mm=None,
    csv_logging_enabled=None,
    **_ignored
):
    global CONF_THR, NMS_IOU, ALARM_THRESHOLD_MM, ALARM_ENABLED, REF_DIAM_MM, HISTOGRAM_MODE
    global MOISTURE_ENABLED, MOISTURE_TOPK, MOISTURE_EVERY_N_FRAMES, FILTER_MAX_1000MM, CSV_LOGGING_ENABLED

    if filter_max_1000mm is not None:
        FILTER_MAX_1000MM = bool(filter_max_1000mm)

    if csv_logging_enabled is not None:
        CSV_LOGGING_ENABLED = bool(csv_logging_enabled)

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


# =================== MATH / PREPROCESSING ===================
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
    """Return L, W, D in mm (using dynamic reference calibration or default px/mm scale)."""
    if boxes_xyxy.shape[0] == 0:
        return (np.zeros(0, np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32))
    arr = np.asarray(boxes_xyxy, np.float32)
    L = np.abs(arr[:, 2] - arr[:, 0])
    W = np.abs(arr[:, 3] - arr[:, 1])
    D = np.sqrt(L * L + W * W)
    scale = PIXELS_PER_MM if PIXELS_PER_MM is not None else DEFAULT_PIXELS_PER_MM
    if scale and scale > 0:
        s = float(scale)
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
    contours = cnts[0] if len(cnts) == 2 else cnts[1]

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


# =================== STATS & HUD HELPERS ===================
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


# =================== MODERN HUD & GRAPHICS SYSTEM ===================
def draw_rounded_rect(img, pt1, pt2, color, thickness=1, radius=6):
    """Draw a smooth rounded rectangle with anti-aliasing."""
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    w = x2 - x1
    h = y2 - y1
    r = max(1, min(radius, w // 2, h // 2))

    if thickness == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)


def draw_glass_card(img, pt1, pt2, bg_color=(18, 22, 28), border_color=(55, 70, 90), alpha=0.82, radius=8, highlight=True):
    """Render a modern translucent frosted-glass card with anti-aliased corners and top sheen."""
    x1, y1 = max(0, int(pt1[0])), max(0, int(pt1[1]))
    x2, y2 = min(img.shape[1] - 1, int(pt2[0])), min(img.shape[0] - 1, int(pt2[1]))
    if x2 <= x1 or y2 <= y1:
        return
    sub = img[y1:y2, x1:x2]
    overlay = sub.copy()
    draw_rounded_rect(overlay, (0, 0), (x2 - x1, y2 - y1), bg_color, -1, radius=radius)
    cv2.addWeighted(overlay, alpha, sub, 1.0 - alpha, 0, dst=sub)
    draw_rounded_rect(img, (x1, y1), (x2, y2), border_color, 1, radius=radius)
    if highlight and (x2 - x1) > 2 * radius:
        hl_color = (min(255, border_color[0] + 45), min(255, border_color[1] + 45), min(255, border_color[2] + 45))
        cv2.line(img, (x1 + radius, y1 + 1), (x2 - radius, y1 + 1), hl_color, 1, cv2.LINE_AA)


def draw_stats_panel(frame, D_buf, units, scale_mean=None, scale_std=None, num_objects=0, normal_count=0, oversize_count=0,
                     conveyor_speed=0.0, locked_count=0, filter_max_1000=True):
    """Render modern high-precision telemetry panel on top-right of frame."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 270, 316
    if w < panel_w + 20 or h < panel_h + 20:
        return frame

    x0, y0 = w - panel_w - 12, 12
    x1, y1 = x0 + panel_w, y0 + panel_h
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Glass container card
    draw_glass_card(frame, (x0, y0), (x1, y1), bg_color=(16, 19, 25), border_color=(55, 70, 90), alpha=0.85, radius=8)

    # Accent header bar
    cv2.rectangle(frame, (x0 + 10, y0 + 10), (x0 + 13, y0 + 26), (255, 200, 40), -1)
    cv2.putText(frame, "DIAMETER STATS", (x0 + 20, y0 + 23), font, 0.48, (245, 248, 252), 2, cv2.LINE_AA)

    # Unit badge pill [mm]
    draw_rounded_rect(frame, (x1 - 48, y0 + 10), (x1 - 12, y0 + 26), (35, 45, 58), -1, radius=4)
    draw_rounded_rect(frame, (x1 - 48, y0 + 10), (x1 - 12, y0 + 26), (255, 200, 40), 1, radius=4)
    cv2.putText(frame, units, (x1 - 40, y0 + 22), font, 0.36, (255, 200, 40), 1, cv2.LINE_AA)

    # Divider line
    cv2.line(frame, (x0 + 10, y0 + 34), (x1 - 10, y0 + 34), (45, 55, 70), 1, cv2.LINE_AA)

    if len(D_buf) > 0:
        D_arr = np.array(D_buf, dtype=np.float32)
        mean = float(np.mean(D_arr))
        med = float(np.median(D_arr))
        std = float(np.std(D_arr))
        dmin = float(np.min(D_arr))
        dmax = float(np.max(D_arr))
        batch = classify_batch(D_buf)

        cal_str = f"{scale_mean:.1f} px/mm" if (scale_mean and scale_mean > 0) else f"{DEFAULT_PIXELS_PER_MM:.1f} (def)"
        thr_col = (45, 60, 255) if ALARM_ACTIVE else (0, 230, 255)

        rows = [
            ("Mean Diameter", f"{mean:.1f} {units}", (50, 230, 130)),
            ("Median Diameter", f"{med:.1f} {units}", (50, 230, 130)),
            ("Std Deviation", f"{std:.1f} {units}", (235, 240, 248)),
            ("Min / Max", f"{dmin:.1f} / {dmax:.1f} {units}", (235, 240, 248)),
            ("Batch Quality", batch, (30, 195, 255)),
            ("Alarm Limit", f"{ALARM_THRESHOLD_MM:.0f} {units}", thr_col),
            ("Locked Chips", f"{locked_count} locked", (50, 230, 130) if locked_count > 0 else (160, 170, 180)),
            ("Conveyor Motion", f"{abs(conveyor_speed):.0f} px/s", (0, 230, 255)),
            ("Max Filter", "< 1000 mm (ON)" if filter_max_1000 else "OFF", (50, 230, 130) if filter_max_1000 else (255, 190, 50)),
            ("CSV Logging", "Active (Daily)", (50, 230, 130)),
        ]

        y_cur = y0 + 56
        for label, val, val_col in rows:
            cv2.putText(frame, label, (x0 + 12, y_cur), font, 0.36, (150, 165, 180), 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(val, font, 0.38, 1)
            cv2.putText(frame, val, (x1 - 14 - tw, y_cur), font, 0.38, val_col, 1, cv2.LINE_AA)
            y_cur += 25
    else:
        cv2.putText(frame, "Waiting for video stream...", (x0 + 16, y0 + 75), font, 0.44, (140, 155, 170), 1, cv2.LINE_AA)

    return frame


def draw_bottom_ribbon(vis, normal_count=0, oversize_count=0, alarm_active=False, locked_count=0, filter_max_1000=True):
    """Render modern docked glass status ribbons at the bottom of the feed."""
    global GUI_MAX_FILTER_BTN_RECT
    h, w = vis.shape[:2]
    if h < 80 or w < 400:
        return vis

    font = cv2.FONT_HERSHEY_SIMPLEX
    total_cnt = normal_count + oversize_count

    # Left Ribbon: Classification metrics & Locked count
    ribbon_w = 428
    draw_glass_card(vis, (12, h - 38), (ribbon_w, h - 10), bg_color=(16, 20, 26), border_color=(55, 68, 85), radius=6)

    # Emerald Normal Dot
    cv2.circle(vis, (26, h - 24), 5, (50, 230, 130), -1, cv2.LINE_AA)
    cv2.putText(vis, f"Normal: {normal_count}", (36, h - 20), font, 0.38, (230, 240, 248), 1, cv2.LINE_AA)

    # Coral Oversize Dot
    cv2.circle(vis, (136, h - 24), 5, (45, 60, 255), -1, cv2.LINE_AA)
    cv2.putText(vis, f"Oversize: {oversize_count}", (146, h - 20), font, 0.38, (230, 240, 248), 1, cv2.LINE_AA)

    # Cyan Total Dot
    cv2.circle(vis, (248, h - 24), 5, (255, 200, 40), -1, cv2.LINE_AA)
    cv2.putText(vis, f"Tot: {total_cnt}", (258, h - 20), font, 0.38, (230, 240, 248), 1, cv2.LINE_AA)

    # Lock-In Dot & Count
    lock_col = (50, 230, 130) if locked_count > 0 else (120, 135, 145)
    cv2.circle(vis, (336, h - 24), 5, lock_col, -1, cv2.LINE_AA)
    cv2.putText(vis, f"Lock: {locked_count}", (346, h - 20), font, 0.38, lock_col, 1, cv2.LINE_AA)

    # Center Filter Toggle Button: [M] Max <1m (Clickable / 'M' key)
    btn_w = 170
    btn_x1 = (w - btn_w) // 2
    btn_x2 = btn_x1 + btn_w
    btn_y1 = h - 38
    btn_y2 = h - 10
    GUI_MAX_FILTER_BTN_RECT = (btn_x1, btn_y1, btn_x2, btn_y2)

    bg_filter = (20, 36, 26) if filter_max_1000 else (36, 30, 20)
    border_filter = (50, 230, 130) if filter_max_1000 else (255, 190, 50)
    draw_glass_card(vis, (btn_x1, btn_y1), (btn_x2, btn_y2), bg_color=bg_filter, border_color=border_filter, radius=6)
    fltr_status_text = "Max <1m: ON" if filter_max_1000 else "Max <1m: OFF"
    cv2.putText(vis, "[M]", (btn_x1 + 8, h - 20), font, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, fltr_status_text, (btn_x1 + 38, h - 20), font, 0.36, (240, 245, 252), 1, cv2.LINE_AA)

    # Right Ribbon: Shortcuts & CSV recording indicator
    right_w = 410
    draw_glass_card(vis, (w - right_w - 12, h - 38), (w - 12, h - 10), bg_color=(16, 20, 26), border_color=(55, 68, 85), radius=6)

    # Pulsing CSV recording LED
    t_pulse = time.time()
    csv_led = (50, 230, 130) if (int(t_pulse * 2) % 2 == 0) else (30, 160, 90)
    cv2.circle(vis, (w - right_w + 2, h - 24), 4, csv_led, -1, cv2.LINE_AA)
    cv2.putText(vis, "CSV", (w - right_w + 12, h - 20), font, 0.36, (200, 215, 230), 1, cv2.LINE_AA)

    shortcuts = [
        ("[M]", "Fltr"),
        ("[+/-]", "Thr"),
        ("[R]", "Reconn"),
        ("[Q]", "Exit"),
    ]
    cur_x = w - right_w + 54
    for key_cap, act in shortcuts:
        cv2.putText(vis, key_cap, (cur_x, h - 20), font, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
        (kw, _), _ = cv2.getTextSize(key_cap, font, 0.38, 1)
        cur_x += kw + 4
        cv2.putText(vis, act, (cur_x, h - 20), font, 0.36, (160, 175, 195), 1, cv2.LINE_AA)
        (aw, _), _ = cv2.getTextSize(act, font, 0.36, 1)
        cur_x += aw + 10

    return vis


def draw_color_legend(frame, normal_count=0, oversize_count=0, locked_count=0, filter_max_1000=True):
    """Backward compatible wrapper calling draw_bottom_ribbon."""
    return draw_bottom_ribbon(frame, normal_count=normal_count, oversize_count=oversize_count, alarm_active=ALARM_ACTIVE,
                              locked_count=locked_count, filter_max_1000=filter_max_1000)


def draw_online_status_badge(vis, cam_dev, fps, w, h, model_name="DETR"):
    """Render sleek modern status card with pulsating live indicator."""
    badge_w, badge_h = 308, 48
    x0, y0 = 12, 12
    x1, y1 = x0 + badge_w, y0 + badge_h
    if vis.shape[1] < x1 + 10 or vis.shape[0] < y1 + 10:
        return vis

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Glass container card
    draw_glass_card(vis, (x0, y0), (x1, y1), bg_color=(16, 20, 26), border_color=(45, 120, 70), radius=8)

    # Pulsing glowing LED dot
    t_pulse = time.time()
    glow_r = int(7 + 1.5 * math.sin(t_pulse * 4.0))
    cv2.circle(vis, (x0 + 18, y0 + 24), glow_r, (40, 180, 80), -1, cv2.LINE_AA)
    cv2.circle(vis, (x0 + 18, y0 + 24), 5, (50, 240, 130), -1, cv2.LINE_AA)
    cv2.circle(vis, (x0 + 18, y0 + 24), glow_r + 2, (70, 255, 160), 1, cv2.LINE_AA)

    # Title & live tag
    cv2.putText(vis, "WOOD-CHIP AI", (x0 + 36, y0 + 20), font, 0.48, (245, 248, 252), 2, cv2.LINE_AA)

    # Mini LIVE badge
    draw_rounded_rect(vis, (x0 + 162, y0 + 8), (x0 + 218, y0 + 24), (20, 42, 28), -1, radius=3)
    draw_rounded_rect(vis, (x0 + 162, y0 + 8), (x0 + 218, y0 + 24), (50, 220, 110), 1, radius=3)
    cv2.putText(vis, "ONLINE", (x0 + 169, y0 + 20), font, 0.35, (50, 230, 120), 1, cv2.LINE_AA)

    # Subtitle details
    details = f"CAM {cam_dev}  |  {w}x{h}  |  {fps:.1f} FPS  |  {model_name}"
    cv2.putText(vis, details, (x0 + 36, y0 + 38), font, 0.34, (160, 175, 195), 1, cv2.LINE_AA)
    return vis


def draw_objects_badge(vis, num_objects=0, normal_count=0, oversize_count=0, locked_count=0, x0=524, y0=12):
    """Render modern 'Objects Detected' telemetry capsule on the top HUD."""
    badge_w, badge_h = 210, 48
    x1, y1 = x0 + badge_w, y0 + badge_h
    if vis.shape[1] < x1 + 10 or vis.shape[0] < y1 + 10:
        return vis

    font = cv2.FONT_HERSHEY_SIMPLEX
    border_col = (45, 60, 255) if oversize_count > 0 else (60, 75, 95)
    draw_glass_card(vis, (x0, y0), (x1, y1), bg_color=(16, 20, 26), border_color=border_col, radius=8)

    # 3D isometric cube vector icon
    ix, iy = x0 + 12, y0 + 12
    cv2.rectangle(vis, (ix, iy), (ix + 12, iy + 10), (50, 230, 130), 1, cv2.LINE_AA)
    cv2.line(vis, (ix + 3, iy - 3), (ix + 15, iy - 3), (50, 230, 130), 1, cv2.LINE_AA)
    cv2.line(vis, (ix + 12, iy), (ix + 15, iy - 3), (50, 230, 130), 1, cv2.LINE_AA)
    cv2.line(vis, (ix, iy), (ix + 3, iy - 3), (50, 230, 130), 1, cv2.LINE_AA)

    header_text = f"OBJECTS ({locked_count} LKD)" if locked_count > 0 else "OBJECTS DETECTED"
    cv2.putText(vis, header_text, (x0 + 32, y0 + 17), font, 0.35, (50, 230, 130) if locked_count > 0 else (170, 185, 200), 1, cv2.LINE_AA)

    # Count display
    cv2.putText(vis, f"{num_objects}", (x0 + 10, y0 + 39), font, 0.54, (250, 252, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "chips", (x0 + 44, y0 + 38), font, 0.33, (140, 155, 175), 1, cv2.LINE_AA)

    # Normal count pill [N: X]
    draw_rounded_rect(vis, (x0 + 96, y0 + 24), (x0 + 150, y0 + 44), (20, 36, 26), -1, radius=4)
    draw_rounded_rect(vis, (x0 + 96, y0 + 24), (x0 + 150, y0 + 44), (50, 200, 100), 1, radius=4)
    cv2.putText(vis, f"N: {normal_count}", (x0 + 102, y0 + 38), font, 0.36, (50, 230, 130), 1, cv2.LINE_AA)

    # Oversize count pill [O: Y]
    o_bg = (38, 20, 26) if oversize_count > 0 else (26, 28, 34)
    o_border = (45, 60, 255) if oversize_count > 0 else (75, 85, 100)
    o_text_col = (45, 60, 255) if oversize_count > 0 else (160, 170, 180)
    draw_rounded_rect(vis, (x0 + 154, y0 + 24), (x0 + 204, y0 + 44), o_bg, -1, radius=4)
    draw_rounded_rect(vis, (x0 + 154, y0 + 24), (x0 + 204, y0 + 44), o_border, 1, radius=4)
    cv2.putText(vis, f"O: {oversize_count}", (x0 + 160, y0 + 38), font, 0.36, o_text_col, 1, cv2.LINE_AA)

    return vis


# =================== INTERACTIVE GUI CONTROLS ===================
GUI_MOUSE_POS = (0, 0)
GUI_MOUSE_CLICKED = False
GUI_RECONNECT_BTN_RECT = None
GUI_THRESH_MINUS_RECT = None
GUI_THRESH_PLUS_RECT = None
GUI_MAX_FILTER_BTN_RECT = None
GUI_RECONNECTING_UNTIL = 0.0


def _on_mouse(event, x, y, flags, param):
    global GUI_MOUSE_POS, GUI_MOUSE_CLICKED
    GUI_MOUSE_POS = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        GUI_MOUSE_CLICKED = True


def on_alarm_trackbar(val):
    """Callback when Alarm Threshold trackbar slider moves."""
    global ALARM_THRESHOLD_MM
    ALARM_THRESHOLD_MM = max(5.0, float(val))


def on_conf_trackbar(val):
    """Callback when Confidence Threshold trackbar slider moves."""
    global CONF_THR
    CONF_THR = max(0.05, min(1.0, float(val) / 100.0))


def _sync_trackbars(window_name, can_show_gui):
    """Keep OpenCV trackbar sliders synchronized with internal threshold values."""
    if not can_show_gui:
        return
    try:
        cur_pos = cv2.getTrackbarPos("Alarm (mm)", window_name)
        target_pos = int(round(ALARM_THRESHOLD_MM))
        if cur_pos != target_pos:
            cv2.setTrackbarPos("Alarm (mm)", window_name, target_pos)

        cur_conf = cv2.getTrackbarPos("Confidence %", window_name)
        target_conf = int(round(CONF_THR * 100))
        if cur_conf != target_conf:
            cv2.setTrackbarPos("Confidence %", window_name, target_conf)
    except Exception:
        pass


def draw_threshold_widget(vis, threshold_val=60.0, conf_val=0.5, x0=746, y0=12, is_hover_minus=False, is_hover_plus=False):
    """Render modern interactive alarm threshold stepper with glass card and keycaps."""
    badge_w, badge_h = 240, 48
    x1, y1 = x0 + badge_w, y0 + badge_h
    if vis.shape[1] < x1 + 10 or vis.shape[0] < y1 + 10:
        return vis

    font = cv2.FONT_HERSHEY_SIMPLEX
    draw_glass_card(vis, (x0, y0), (x1, y1), bg_color=(16, 20, 26), border_color=(60, 75, 95), radius=8)

    # Header row
    cv2.putText(vis, "ALARM THRESHOLD", (x0 + 10, y0 + 15), font, 0.36, (170, 185, 200), 1, cv2.LINE_AA)
    cv2.putText(vis, "[+/-5]", (x1 - 44, y0 + 15), font, 0.32, (120, 140, 160), 1, cv2.LINE_AA)

    # Minus Button [-]
    m_x1, m_y1, m_x2, m_y2 = x0 + 10, y0 + 21, x0 + 44, y0 + 43
    m_bg = (48, 58, 72) if is_hover_minus else (30, 36, 46)
    m_border = (255, 190, 50) if is_hover_minus else (80, 100, 125)
    draw_rounded_rect(vis, (m_x1, m_y1), (m_x2, m_y2), m_bg, -1, radius=4)
    draw_rounded_rect(vis, (m_x1, m_y1), (m_x2, m_y2), m_border, 1, radius=4)
    cv2.line(vis, (m_x1 + 8, m_y1 + 11), (m_x2 - 8, m_y1 + 11), (255, 255, 255), 2, cv2.LINE_AA)

    # Threshold Value Display
    val_str = f"{threshold_val:.0f} mm"
    cv2.putText(vis, val_str, (x0 + 52, y0 + 37), font, 0.52, (0, 230, 255), 2, cv2.LINE_AA)

    # Plus Button [+]
    p_x1, p_y1, p_x2, p_y2 = x0 + 144, y0 + 21, x0 + 178, y0 + 43
    p_bg = (48, 58, 72) if is_hover_plus else (30, 36, 46)
    p_border = (255, 190, 50) if is_hover_plus else (80, 100, 125)
    draw_rounded_rect(vis, (p_x1, p_y1), (p_x2, p_y2), p_bg, -1, radius=4)
    draw_rounded_rect(vis, (p_x1, p_y1), (p_x2, p_y2), p_border, 1, radius=4)
    cv2.line(vis, (p_x1 + 8, p_y1 + 11), (p_x2 - 8, p_y1 + 11), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(vis, (p_x1 + 17, p_y1 + 4), (p_x1 + 17, p_y2 - 4), (255, 255, 255), 2, cv2.LINE_AA)

    # Keycap hint
    cv2.putText(vis, "1mm: [ / ]", (x0 + 185, y0 + 35), font, 0.32, (130, 145, 165), 1, cv2.LINE_AA)
    return vis


def draw_reconnect_button(frame, rect, is_hover=False, is_reconnecting=False):
    """Render modern interactive reconnect button with glass styling and keycap badge."""
    x1, y1, x2, y2 = rect
    btn_w = x2 - x1
    btn_h = y2 - y1
    if x2 > frame.shape[1] or y2 > frame.shape[0] or x1 < 0 or y1 < 0:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX

    if is_reconnecting:
        bg_col = (24, 38, 54)
        border_col = (255, 200, 40)
        text_col = (255, 220, 80)
        text = "Reconnecting..."
    elif is_hover:
        bg_col = (34, 44, 58)
        border_col = (255, 190, 50)
        text_col = (255, 235, 150)
        text = "Reconnect"
    else:
        bg_col = (18, 22, 28)
        border_col = (65, 80, 100)
        text_col = (230, 240, 250)
        text = "Reconnect"

    draw_glass_card(frame, (x1, y1), (x2, y2), bg_color=bg_col, border_color=border_col, radius=8)

    # Sync circular arrow
    cx = x1 + 18
    cy = y1 + btn_h // 2
    r = 7
    if is_reconnecting:
        t_spin = time.time() * 360.0
        cv2.ellipse(frame, (cx, cy), (r, r), t_spin, 0, 270, text_col, 2, cv2.LINE_AA)
    else:
        cv2.ellipse(frame, (cx, cy), (r, r), 0, -45, 225, text_col, 2, cv2.LINE_AA)
        rad = math.radians(-45)
        ax = int(cx + r * math.cos(rad))
        ay = int(cy + r * math.sin(rad))
        cv2.line(frame, (ax, ay), (ax + 3, ay - 3), text_col, 2, cv2.LINE_AA)
        cv2.line(frame, (ax, ay), (ax - 3, ay - 2), text_col, 2, cv2.LINE_AA)

    # Button label
    cv2.putText(frame, text, (x1 + 33, y1 + btn_h // 2 + 5), font, 0.44, text_col, 1, cv2.LINE_AA)

    # Keycap badge [ R ]
    if not is_reconnecting and btn_w >= 160:
        draw_rounded_rect(frame, (x2 - 38, y1 + 14), (x2 - 12, y1 + 34), (30, 36, 46), -1, radius=4)
        draw_rounded_rect(frame, (x2 - 38, y1 + 14), (x2 - 12, y1 + 34), border_col, 1, radius=4)
        cv2.putText(frame, "R", (x2 - 29, y1 + 28), font, 0.38, (0, 230, 255) if not is_hover else (255, 230, 100), 1, cv2.LINE_AA)

    return frame


def draw_alarm_banner(vis, max_d_mm=None, threshold_mm=60.0):
    """Render high-visibility warning banner when oversize chip alarm triggers."""
    w = vis.shape[1]
    banner_w = min(660, w - 24)
    banner_h = 32
    x0 = (w - banner_w) // 2
    y0 = 68
    x1, y1 = x0 + banner_w, y0 + banner_h

    draw_glass_card(vis, (x0, y0), (x1, y1), bg_color=(36, 16, 24), border_color=(45, 60, 255), alpha=0.92, radius=6)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Warning icon
    cv2.putText(vis, "!", (x0 + 14, y0 + 22), font, 0.50, (45, 60, 255), 2, cv2.LINE_AA)
    cv2.circle(vis, (x0 + 17, y0 + 16), 11, (45, 60, 255), 1, cv2.LINE_AA)

    max_str = f"{max_d_mm:.1f} mm" if max_d_mm is not None else "> limit"
    msg = f"OVERSIZE CHIP DETECTED (Max: {max_str} > Limit: {threshold_mm:.0f} mm) - SORTING ALERT"
    cv2.putText(vis, msg, (x0 + 36, y0 + 21), font, 0.40, (255, 235, 130), 1, cv2.LINE_AA)
    return vis


def create_offline_frame(width=1280, height=720, cam_dev=0, reconnect_interval=2.0):
    """
    Generate futuristic high-aesthetic standby slate when camera is disconnected.
    Features subtle tech grid, glowing radar aperture, and animated scanning sweeps.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (18, 20, 25)  # Obsidian slate base

    # Subtle background tech grid
    grid_color = (25, 28, 36)
    for gx in range(0, width, 60):
        cv2.line(frame, (gx, 0), (gx, height), grid_color, 1)
    for gy in range(0, height, 60):
        cv2.line(frame, (0, gy), (width, gy), grid_color, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Top glass header bar
    draw_glass_card(frame, (0, 0), (width, 50), bg_color=(15, 18, 24), border_color=(45, 55, 70), alpha=0.90, radius=0)
    cv2.putText(frame, "WOOD-CHIP MONITOR", (20, 32), font, 0.62, (245, 248, 252), 2, cv2.LINE_AA)

    # Sub-tag [ SYSTEM STANDBY ]
    draw_rounded_rect(frame, (230, 16), (360, 36), (32, 24, 20), -1, radius=4)
    draw_rounded_rect(frame, (230, 16), (360, 36), (30, 195, 255), 1, radius=4)
    cv2.putText(frame, "STANDBY", (242, 31), font, 0.38, (30, 195, 255), 1, cv2.LINE_AA)

    # Target info on right
    os_info = f"OS: {platform.system()}  |  TARGET: {cam_dev}"
    cv2.putText(frame, os_info, (width - 290, 32), font, 0.44, (160, 170, 195), 1, cv2.LINE_AA)

    cx = width // 2
    cy = height // 2 - 40

    # Animated radar scan rings
    now = time.time()
    for r_ring in [85, 60, 35]:
        cv2.circle(frame, (cx, cy), r_ring, (38, 44, 56), 1, cv2.LINE_AA)

    # Rotating radar sweep ray
    sweep_angle = (now * 1.8) % (2 * math.pi)
    ray_x = int(cx + 85 * math.cos(sweep_angle))
    ray_y = int(cy + 85 * math.sin(sweep_angle))
    cv2.line(frame, (cx, cy), (ray_x, ray_y), (60, 95, 135), 1, cv2.LINE_AA)

    # Camera body graphic
    body_w, body_h = 100, 64
    bx1, by1 = cx - body_w // 2, cy - body_h // 2
    bx2, by2 = cx + body_w // 2, cy + body_h // 2
    draw_glass_card(frame, (bx1, by1), (bx2, by2), bg_color=(25, 30, 38), border_color=(75, 90, 110), radius=8)
    # Lens
    cv2.circle(frame, (cx, cy), 22, (20, 24, 30), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 22, (100, 115, 135), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 12, (12, 14, 18), -1, cv2.LINE_AA)
    # Red diagonal disconnect slash
    cv2.line(frame, (bx1 - 6, by2 + 6), (bx2 + 6, by1 - 6), (45, 60, 255), 3, cv2.LINE_AA)

    # Status badge: CAMERA SIGNAL OFFLINE
    badge_w, badge_h = 320, 42
    bx = (width - badge_w) // 2
    by = cy + 64
    draw_glass_card(frame, (bx, by), (bx + badge_w, by + badge_h), bg_color=(36, 18, 24), border_color=(45, 60, 255), radius=6)
    cv2.circle(frame, (bx + 22, by + 21), 6, (45, 60, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (bx + 22, by + 21), 9, (70, 90, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "CAMERA SIGNAL OFFLINE", (bx + 38, by + 27), font, 0.58, (45, 60, 255), 2, cv2.LINE_AA)

    # Dynamic status line with dots
    dot_count = int(now * 2) % 4
    dots = "." * dot_count + " " * (3 - dot_count)
    msg1 = f"Camera device '{cam_dev}' is disconnected or not detected."
    msg2 = f"Searching for video stream{dots} (auto-polling every {reconnect_interval:.1f}s)"
    msg3 = "Please plug in your camera. Live video feed will start automatically."

    cv2.putText(frame, msg1, (cx - int(len(msg1) * 4.6), by + 70), font, 0.50, (210, 220, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, msg2, (cx - int(len(msg2) * 4.2), by + 98), font, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, msg3, (cx - int(len(msg3) * 4.4), by + 128), font, 0.46, (140, 220, 160), 1, cv2.LINE_AA)

    return frame


# =================== CONVEYOR MOTION TRACKER & CSV LOGGER ===================
def _resolve_csv_dir() -> str:
    """Return working directory or configured output directory for wood chip CSV logs."""
    env_dir = os.environ.get("WOODCHIP_CSV_DIR")
    if env_dir and os.path.exists(env_dir):
        return env_dir
    return os.getcwd()


class TrackedChip:
    """
    Represents an individual physical wood chip moving on a high-speed conveyor belt.
    Maintains persistent track ID, smoothed coordinates, motion velocity vector,
    size measurements, and lock status.
    """
    def __init__(self, track_id: int, box: np.ndarray, score: float, d_val: float,
                 L_val: float, W_val: float, velocity=(0.0, 0.0), moisture="N/A"):
        self.track_id = int(track_id)
        self.box = np.array(box, dtype=np.float32)  # [x1, y1, x2, y2]
        self.score = float(score)
        self.d_val = float(d_val)
        self.L_val = float(L_val)
        self.W_val = float(W_val)
        self.vx = float(velocity[0])  # pixels / second along X
        self.vy = float(velocity[1])  # pixels / second along Y
        self.moisture = str(moisture)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.locked = False
        self.last_update_ts = time.time()
        self.last_csv_log_ts = 0.0

        cx = (self.box[0] + self.box[2]) / 2.0
        cy = (self.box[1] + self.box[3]) / 2.0
        self.history = deque(maxlen=15)
        self.history.append((cx, cy, self.last_update_ts))

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    @property
    def speed_px_s(self) -> float:
        return math.hypot(self.vx, self.vy)

    def predict(self, dt: float):
        """Propagate bounding box forward using estimated conveyor velocity."""
        if dt > 0:
            dx = self.vx * dt
            dy = self.vy * dt
            self.box[0] += dx
            self.box[2] += dx
            self.box[1] += dy
            self.box[3] += dy
            cx, cy = self.centroid
            self.history.append((cx, cy, time.time()))
        self.time_since_update += 1
        self.age += 1

    def update(self, det_box: np.ndarray, score: float, d_val: float, L_val: float, W_val: float, dt: float, moisture="N/A"):
        """Update track state when detection matches."""
        now = time.time()
        prev_cx, prev_cy = self.centroid
        new_cx = (det_box[0] + det_box[2]) / 2.0
        new_cy = (det_box[1] + det_box[3]) / 2.0

        if dt > 0.001:
            inst_vx = (new_cx - prev_cx) / dt
            inst_vy = (new_cy - prev_cy) / dt
            alpha_v = 0.4
            self.vx = (1.0 - alpha_v) * self.vx + alpha_v * inst_vx
            self.vy = (1.0 - alpha_v) * self.vy + alpha_v * inst_vy

        # Smooth position and size
        alpha_pos = 0.6
        self.box = (1.0 - alpha_pos) * self.box + alpha_pos * det_box
        self.score = float(score)

        alpha_d = 0.35
        self.d_val = (1.0 - alpha_d) * self.d_val + alpha_d * float(d_val)
        self.L_val = (1.0 - alpha_d) * self.L_val + alpha_d * float(L_val)
        self.W_val = (1.0 - alpha_d) * self.W_val + alpha_d * float(W_val)

        if moisture and moisture != "N/A":
            self.moisture = str(moisture)

        self.hits += 1
        self.time_since_update = 0
        self.last_update_ts = now
        if self.hits >= TRACKER_MIN_HITS_TO_LOCK:
            self.locked = True

        cx, cy = self.centroid
        self.history.append((cx, cy, now))


class ConveyorLockTracker:
    """
    Multi-object motion tracker engineered for high-speed conveyor belt operations.
    Maintains persistent tracks, estimates bulk conveyor velocity, handles high-speed motion compensation,
    and locks onto wood chips across successive frames.
    """
    def __init__(self, max_match_dist=180.0, max_lost_frames=8):
        self.max_match_dist = max_match_dist
        self.max_lost_frames = max_lost_frames
        self.tracks: List[TrackedChip] = []
        self.next_id = 1
        self.bulk_conveyor_vx = 0.0  # Estimated belt speed in px/s
        self.bulk_conveyor_vy = 0.0
        self.last_t = time.time()

    @property
    def bulk_speed_px_s(self) -> float:
        return math.hypot(self.bulk_conveyor_vx, self.bulk_conveyor_vy)

    def step(self, det_boxes: np.ndarray, det_scores: np.ndarray,
             det_L: np.ndarray, det_W: np.ndarray, det_D: np.ndarray,
             moistures: list = None, frame_w: int = 1280, frame_h: int = 720) -> List[TrackedChip]:
        now = time.time()
        dt = max(0.001, min(0.5, now - self.last_t))
        self.last_t = now

        # 1. Predict position for all active tracks using velocity model
        for trk in self.tracks:
            trk.predict(dt)

        # 2. Association between predicted tracks and newly detected boxes
        N_det = len(det_boxes) if det_boxes is not None else 0
        N_trk = len(self.tracks)

        matched_tracks = set()
        matched_dets = set()

        if N_trk > 0 and N_det > 0:
            # Build cost matrix based on Euclidean distance & IOU
            cost_matrix = np.zeros((N_trk, N_det), dtype=np.float32)
            for i, trk in enumerate(self.tracks):
                tcx, tcy = trk.centroid
                tb = trk.box
                for j in range(N_det):
                    db = det_boxes[j]
                    dcx = (db[0] + db[2]) / 2.0
                    dcy = (db[1] + db[3]) / 2.0
                    dist = math.hypot(tcx - dcx, tcy - dcy)

                    # Calculate IOU
                    ix1 = max(tb[0], db[0])
                    iy1 = max(tb[1], db[1])
                    ix2 = min(tb[2], db[2])
                    iy2 = min(tb[3], db[3])
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    area_t = max(1.0, (tb[2] - tb[0]) * (tb[3] - tb[1]))
                    area_d = max(1.0, (db[2] - db[0]) * (db[3] - db[1]))
                    iou = inter / (area_t + area_d - inter + 1e-6)

                    # Primary metric: distance minus IOU bonus
                    cost_matrix[i, j] = dist - (80.0 * iou)

            # Greedy bipartite matching
            for _ in range(min(N_trk, N_det)):
                min_val = np.min(cost_matrix)
                if min_val > self.max_match_dist:
                    break
                trk_idx, det_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                if cost_matrix[trk_idx, det_idx] >= 1e5:
                    break

                matched_tracks.add(trk_idx)
                matched_dets.add(det_idx)

                # Update matched track
                m_label = moistures[det_idx] if moistures and det_idx < len(moistures) else "N/A"
                self.tracks[trk_idx].update(
                    det_boxes[det_idx], det_scores[det_idx],
                    det_D[det_idx], det_L[det_idx], det_W[det_idx],
                    dt, moisture=m_label
                )

                # Invalidate row and column
                cost_matrix[trk_idx, :] = 1e6
                cost_matrix[:, det_idx] = 1e6

        # 3. Create new tracks for unmatched detections
        for j in range(N_det):
            if j not in matched_dets:
                m_label = moistures[j] if moistures and j < len(moistures) else "N/A"
                new_trk = TrackedChip(
                    track_id=self.next_id,
                    box=det_boxes[j],
                    score=det_scores[j],
                    d_val=det_D[j],
                    L_val=det_L[j],
                    W_val=det_W[j],
                    velocity=(self.bulk_conveyor_vx, self.bulk_conveyor_vy),
                    moisture=m_label
                )
                self.next_id += 1
                self.tracks.append(new_trk)

        # 4. Clean up tracks (out of bounds or lost too long)
        surviving_tracks = []
        margin = 60
        for trk in self.tracks:
            if trk.time_since_update > self.max_lost_frames:
                continue
            b = trk.box
            if b[2] < -margin or b[0] > frame_w + margin or b[3] < -margin or b[1] > frame_h + margin:
                continue
            surviving_tracks.append(trk)
        self.tracks = surviving_tracks

        # 5. Update bulk conveyor velocity estimate from locked tracks
        locked_vels_x = [t.vx for t in self.tracks if t.locked and abs(t.vx) > 1.0]
        locked_vels_y = [t.vy for t in self.tracks if t.locked and abs(t.vy) > 1.0]
        if locked_vels_x:
            self.bulk_conveyor_vx = float(np.median(locked_vels_x))
        if locked_vels_y:
            self.bulk_conveyor_vy = float(np.median(locked_vels_y))

        return self.tracks


class WoodChipCSVLogger:
    """
    High-performance, asynchronous CSV logger for wood chip monitoring and sorting.
    Rotates daily into 'YYYYMMDD_woodChips.csv'. Writes in a non-blocking background thread
    so video capture and inference never stall on file I/O.
    """
    def __init__(self, out_dir=None, throttle_sec=0.25):
        self.out_dir = out_dir if out_dir else os.getcwd()
        self.throttle_sec = throttle_sec
        self.queue = queue.Queue(maxsize=10000)
        self.current_date_str = ""
        self.current_file_path = ""
        self.running = True
        self.fieldnames = [
            "time",
            "detected_object_index",
            "track_id",
            "estimated_size_mm",
            "threshold_size_mm",
            "threshold_status",
            "moisture_level",
            "length_mm",
            "width_mm",
            "aspect_ratio",
            "confidence",
            "conveyor_speed_px_s",
            "conveyor_speed_mm_s",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "batch_quality"
        ]

        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _ensure_file(self):
        """Check for daily rotation (yyyymmdd_woodChips.csv)."""
        date_str = datetime.now().strftime("%Y%m%d")
        if date_str != self.current_date_str:
            self.current_date_str = date_str
            filename = f"{date_str}_woodChips.csv"
            self.current_file_path = os.path.join(self.out_dir, filename)
            file_exists = os.path.exists(self.current_file_path)
            if not file_exists:
                try:
                    with open(self.current_file_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                        writer.writeheader()
                    print(f"[motionTracking] Daily CSV log initialized: {self.current_file_path}", flush=True)
                except Exception as e:
                    print(f"[motionTracking] Error creating CSV file {self.current_file_path}: {e}", flush=True)

    def log_detection(self, row_dict: dict):
        """Enqueue row for non-blocking background write."""
        if not self.running:
            return
        try:
            self.queue.put_nowait(row_dict)
        except queue.Full:
            pass  # Avoid blocking pipeline if queue full

    def _worker(self):
        while self.running:
            try:
                batch = []
                item = self.queue.get(timeout=0.5)
                batch.append(item)
                while len(batch) < 100:
                    try:
                        batch.append(self.queue.get_nowait())
                    except queue.Empty:
                        break

                if batch:
                    self._ensure_file()
                    if self.current_file_path:
                        try:
                            file_exists = os.path.exists(self.current_file_path)
                            with open(self.current_file_path, "a", newline="", encoding="utf-8") as f:
                                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                                if not file_exists or os.path.getsize(self.current_file_path) == 0:
                                    writer.writeheader()
                                for r in batch:
                                    writer.writerow(r)
                                f.flush()
                        except Exception as e:
                            print(f"[motionTracking] Error writing to CSV {self.current_file_path}: {e}", flush=True)
            except queue.Empty:
                continue
            except Exception:
                time.sleep(0.1)

    def close(self):
        self.running = False


# =================== TRACKING & INFERENCE ENGINES ===================
class ContourMotionTracker:
    """
    High-performance OpenCV-based wood chip motion and contour tracker.
    Runs on any platform (Windows & Linux) without GPU, CUDA, or weights files.
    """
    def __init__(self):
        self.name = "OpenCV-Contour"
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def detect_boxes(self, frame_bgr, conf_thr=0.4):
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Robust segmentation using Otsu thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts[0] if len(cnts) == 2 else cnts[1]

        min_area = (h * w) * 0.0004
        max_area = (h * w) * 0.35

        boxes = []
        scores = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area < max_area:
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = bw / float(bh + 1e-6)
                if 0.15 < aspect < 6.5:
                    boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
                    score = min(0.99, max(0.45, float(area / (bw * bh + 1e-5))))
                    scores.append(score)

        if not boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        return np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32)


class GenericTRTWrapper:
    """Stable TensorRT wrapper."""
    def __init__(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        _check_and_import_trt()
        self.logger = trt.Logger(trt.Logger.WARNING)
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
                raise RuntimeError("Unable to resolve DETR TRT outputs.")

        return logits, pred_boxes


class DETRTorch:
    """DETR PyTorch inference using huggingface Transformers."""
    def __init__(self, checkpoint_path: str = None):
        _check_and_import_torch()
        print("[motionTracking/Torch] Initializing DETR PyTorch model...", flush=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-101",
            num_labels=1,
            ignore_mismatched_sizes=True
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[motionTracking/Torch] Loading weights from {checkpoint_path}...", flush=True)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    clean_state_dict[k[6:]] = v
                else:
                    clean_state_dict[k] = v
            self.model.load_state_dict(clean_state_dict)

        self.model.to(self.device)
        self.model.eval()
        print(f"[motionTracking/Torch] Model ready on {self.device}", flush=True)

    def infer(self, x: np.ndarray):
        with torch.no_grad():
            t_x = torch.from_numpy(x).to(self.device)
            outputs = self.model(pixel_values=t_x)
            logits = outputs.logits.cpu().numpy()
            pred_boxes = outputs.pred_boxes.cpu().numpy()
        return logits, pred_boxes


def _create_moistnet_torch_class():
    if not _check_and_import_torch():
        return None

    class MoistNetTorch(nn.Module):
        def __init__(self, num_filters=32, dense_layer_size=256):
            super(MoistNetTorch, self).__init__()
            self.conv0 = nn.Conv2d(3, num_filters, kernel_size=3, padding=0)
            self.relu0 = nn.ReLU()
            self.pool0 = nn.MaxPool2d(kernel_size=2)

            self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=0)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2)

            self.conv2 = nn.Conv2d(num_filters, 512, kernel_size=3, padding=0)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2)

            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.dense = nn.Linear(512, dense_layer_size)
            self.relu_dense = nn.ReLU()
            self.head = nn.Linear(dense_layer_size, 3)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.pool0(self.relu0(self.conv0(x)))
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.gap(x)
            x = x.view(x.size(0), -1)
            x = self.relu_dense(self.dense(x))
            x = self.head(x)
            x = self.softmax(x)
            return x

    return MoistNetTorch


class MoistureTorch:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Moisture model not found: {model_path}")
        _check_and_import_torch()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MoistClass = _create_moistnet_torch_class()
        self.model = MoistClass()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def infer_probs_bhwc(self, crops_bhwc: np.ndarray):
        with torch.no_grad():
            x = np.transpose(crops_bhwc, (0, 3, 1, 2))
            t_x = torch.from_numpy(x).to(self.device)
            probs = self.model(t_x)
            return probs.cpu().numpy()


class MoistureTRT(GenericTRTWrapper):
    def infer_probs_bhwc(self, crops_bhwc: np.ndarray):
        if crops_bhwc is None or len(crops_bhwc) == 0:
            return None
        input_shape = tuple(self.engine.get_binding_shape(self.input_idx))
        if len(input_shape) == 4 and input_shape[-1] == 3:
            x = crops_bhwc
        else:
            x = np.transpose(crops_bhwc, (0, 3, 1, 2))

        outs = self.infer_all(x)
        out = list(outs.values())[0]
        if out.ndim > 2:
            out = out.reshape(out.shape[0], -1)

        row_sum = np.sum(out, axis=1)
        if np.any(row_sum < 0.8) or np.any(row_sum > 1.2) or np.any(out < 0.0) or np.any(out > 1.0):
            out = safe_softmax_2d(out, axis=1)
        return out


class MoistureSim:
    """Simulation fallback for moisture."""
    def infer_probs_bhwc(self, crops_bhwc: np.ndarray):
        if crops_bhwc is None or len(crops_bhwc) == 0:
            return None
        n = crops_bhwc.shape[0]
        return np.random.dirichlet([5, 2, 1], size=n).astype(np.float32)


# =================== MOISTURE CROPS ===================
def _read_moist_classes(path: str):
    if not path or not os.path.exists(path):
        return ["dry", "moist", "wet"]
    with open(path, "r") as f:
        return [l.strip() for l in f if l.strip()]

MOISTURE_CLASSES = _read_moist_classes(MOIST_CLASSES_PATH)


def _prep_moist_crop_bgr(crop_bgr: np.ndarray, size: int, use_imagenet=False):
    crop = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if use_imagenet:
        crop = (crop - MEAN) / STD
    return crop.astype(np.float32)


def _boxes_to_topk_crops(frame_bgr: np.ndarray, boxes_xyxy: np.ndarray, scores: np.ndarray, topk: int):
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

        crops.append(_prep_moist_crop_bgr(crop_bgr, MOISTURE_INPUT_SIZE, use_imagenet=False))
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


# =================== CROSS-PLATFORM THREADED CAMERA ===================
class ThreadedCamera:
    """
    Cross-platform threaded camera reader.
    Handles device opening, frame grabbing, disconnection, and hot-plugging
    on both Windows (DirectShow/MSMF) and Linux (V4L2).
    """
    def __init__(self, dev=0, w=1280, h=720, fps=20):
        self.dev = dev
        self.active_dev = dev
        self.w = w
        self.h = h
        self.fps = fps
        self.cap = None
        self.frame = None
        self.ok = False
        self.last_grab_ts = 0.0
        self.last_reconnect_ts = time.time()
        self.stopped = False
        self.force_reconnect_now = True  # Triggers immediate open in background thread
        self.lock = threading.Lock()

        global GLOBAL_T_CAP
        GLOBAL_T_CAP = self

        self.t = threading.Thread(target=self._update, daemon=True)
        self.t.start()

    def trigger_reconnect(self):
        """Signal the background thread to immediately re-open the camera."""
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None
            self.ok = False
            self.force_reconnect_now = True

    def _open_cap(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        # Build candidate devices: requested device first
        dev_candidates = [self.dev]
        if self.dev == 0 or self.dev == "/dev/video0":
            if isinstance(self.dev, int):
                for extra in [1, 2]:
                    if extra not in dev_candidates:
                        dev_candidates.append(extra)
            elif IS_LINUX:
                for extra in ["/dev/video1", 0, 1]:
                    if extra not in dev_candidates:
                        dev_candidates.append(extra)

        # Backends prioritized by OS
        backends = []
        if IS_WINDOWS:
            backends = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "MediaFoundation"),
                (cv2.CAP_ANY, "Auto")
            ]
        elif IS_LINUX:
            backends = [
                (cv2.CAP_V4L2, "V4L2"),
                (cv2.CAP_ANY, "Auto")
            ]
        else:
            backends = [(cv2.CAP_ANY, "Auto")]

        opened_cap = None
        opened_dev = None

        for d in dev_candidates:
            for b_id, b_name in backends:
                try:
                    c = cv2.VideoCapture(d, b_id)
                    if c.isOpened():
                        # Verify we can grab an actual frame
                        ret, test_frame = c.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            opened_cap = c
                            opened_dev = d
                            break
                        c.release()
                except Exception:
                    pass
            if opened_cap is not None:
                break

        if opened_cap is not None:
            try:
                opened_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
                opened_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
                opened_cap.set(cv2.CAP_PROP_FPS, self.fps)
            except Exception:
                pass

            ret, frame = opened_cap.read()
            if ret and frame is not None and frame.size > 0:
                with self.lock:
                    self.cap = opened_cap
                    self.active_dev = opened_dev
                    self.frame = frame.copy()
                    self.ok = True
                    self.last_grab_ts = time.time()
                    self.last_reconnect_ts = time.time()
                print(f"[ThreadedCamera] Connected to camera on device '{self.active_dev}' ({frame.shape[1]}x{frame.shape[0]}).", flush=True)
                return True
            else:
                try:
                    opened_cap.release()
                except Exception:
                    pass

        with self.lock:
            self.ok = False
        return False

    def _update(self):
        consecutive_failures = 0
        last_reconnect_time = 0.0

        while not self.stopped:
            need_reconnect = (self.cap is None or not self.ok or self.force_reconnect_now)
            if need_reconnect:
                now = time.time()
                if self.force_reconnect_now or (now - last_reconnect_time > CAMERA_RECONNECT_INTERVAL):
                    self.force_reconnect_now = False
                    opened = self._open_cap()
                    last_reconnect_time = time.time()
                    if opened:
                        consecutive_failures = 0
                time.sleep(0.1)
                continue

            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    with self.lock:
                        self.frame = frame.copy()
                        self.ok = True
                        self.last_grab_ts = time.time()
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            except Exception:
                consecutive_failures += 1

            if consecutive_failures >= 3:
                print(f"[ThreadedCamera] Camera disconnected from device '{self.active_dev}'. Switching to OFFLINE mode...", flush=True)
                with self.lock:
                    self.ok = False
                if self.cap:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                self.cap = None
                consecutive_failures = 0
                last_reconnect_time = time.time()

            time.sleep(max(0.001, 1.0 / (self.fps * 2)))

    def read(self):
        with self.lock:
            if not self.ok or self.frame is None:
                return False, None
            # If last successful grab is older than 2.5s, consider it stale/disconnected
            if time.time() - self.last_grab_ts > 2.5:
                return False, self.frame
            return True, self.frame

    def stop(self):
        self.stopped = True
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


# =================== MAIN TRACKING & INFERENCE LOOP ===================
def run_inference_loop(headless: bool = False):
    global UNITS, SCALE_BUF, LATEST_FRAME, LATEST_STATS, HIST_DATA, LATEST_MOISTURE, LATEST_HEALTH
    global ALARM_ACTIVE, ALARM_MAX_D_MM, GLOBAL_T_CAP, ALARM_THRESHOLD_MM, CONF_THR
    global GUI_MOUSE_POS, GUI_MOUSE_CLICKED, GUI_RECONNECT_BTN_RECT, GUI_RECONNECTING_UNTIL
    global GUI_THRESH_MINUS_RECT, GUI_THRESH_PLUS_RECT, GUI_MAX_FILTER_BTN_RECT
    global FILTER_MAX_1000MM, MAX_SIZE_LIMIT_MM, CONVEYOR_TRACKING_ENABLED, CSV_LOGGING_ENABLED

    print(f"[motionTracking] Starting Wood-Chip Monitor on {platform.system()}...", flush=True)

    # Initialize CUDA context for TRT if engine exists and TRT available
    ctx = None
    detr = None
    model_name = "Contour"

    if os.path.exists(ENGINE_PATH) and _check_and_import_trt():
        try:
            print("[motionTracking] Initializing CUDA context...", flush=True)
            cuda.init()
            dev = cuda.Device(0)
            ctx = dev.make_context()
            print(f"[motionTracking] Loading DETR TensorRT engine: {ENGINE_PATH}", flush=True)
            detr = DETRTRT(ENGINE_PATH)
            model_name = "DETR (TensorRT)"
        except Exception as e:
            print(f"[motionTracking] Failed to load TensorRT engine: {e}", flush=True)
            detr = None

    if detr is None and os.path.exists(CKPT_PATH) and _check_and_import_torch():
        try:
            print(f"[motionTracking] Loading DETR PyTorch model: {CKPT_PATH}", flush=True)
            detr = DETRTorch(CKPT_PATH)
            model_name = "DETR (PyTorch)"
        except Exception as e:
            print(f"[motionTracking] Failed to load PyTorch model: {e}", flush=True)
            detr = None

    if detr is None:
        print("[motionTracking] Operating in OpenCV Contour & Motion Tracking mode.", flush=True)
        detr = ContourMotionTracker()
        model_name = "Contour / Motion"

    # Load Moisture Model
    moist = None
    if MOISTURE_ENABLED:
        if os.path.exists(MOIST_ENGINE_PATH) and _check_and_import_trt():
            try:
                print(f"[motionTracking] Loading Moisture TensorRT engine: {MOIST_ENGINE_PATH}", flush=True)
                moist = MoistureTRT(MOIST_ENGINE_PATH)
            except Exception:
                moist = None
        if moist is None:
            moist_pt_path = _find_candidate_path("WOODCHIP_MOIST_PT_PATH", "moistnetlite.pt", ["models/artifacts", "models", "best models", ""])
            if os.path.exists(moist_pt_path) and _check_and_import_torch():
                try:
                    print(f"[motionTracking] Loading Moisture PyTorch model: {moist_pt_path}", flush=True)
                    moist = MoistureTorch(moist_pt_path)
                except Exception:
                    moist = None
        if moist is None:
            moist = MoistureSim()

    # Start Camera
    t_cap = ThreadedCamera(CAM_DEV, CAM_W, CAM_H, CAM_FPS)
    GLOBAL_T_CAP = t_cap

    start_ts = _now()
    last_frame_ts = None
    last_detr_ts = None
    last_moist_ts = None
    last_hist_pub_ts = None
    last_error = None

    L_buf, W_buf, D_buf = [], [], []
    fps, t_prev = 0.0, time.time()
    frame_idx = 0
    processed_idx = 0
    last_boxes_xyxy = np.zeros((0, 4), np.float32)
    last_scores = np.zeros((0,), np.float32)
    last_D = np.zeros((0,), np.float32)
    last_hist_time = 0.0
    prev_hist_counts = None
    prev_hist_bins = None

    # Initialize High-Speed Conveyor Motion Tracker & Daily CSV Logger
    tracker = ConveyorLockTracker(max_match_dist=TRACKER_MAX_MATCH_DIST, max_lost_frames=TRACKER_MAX_LOST_FRAMES)
    csv_logger = WoodChipCSVLogger(out_dir=_resolve_csv_dir(), throttle_sec=CSV_LOG_THROTTLE_SEC) if CSV_LOGGING_ENABLED else None
    bulk_conveyor_speed = 0.0
    locked_count = 0

    # GUI Display Setup
    window_name = "Wood-Chip Monitor - Live Tracking"
    has_display = True
    if not IS_WINDOWS:
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            has_display = False
            print("[motionTracking] Headless environment detected (no $DISPLAY). Running without GUI.", flush=True)

    can_show_gui = (not headless) and has_display
    if can_show_gui:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
            cv2.setMouseCallback(window_name, _on_mouse)
            cv2.createTrackbar("Alarm (mm)", window_name, int(round(ALARM_THRESHOLD_MM)), 200, on_alarm_trackbar)
            cv2.createTrackbar("Confidence %", window_name, int(round(CONF_THR * 100)), 100, on_conf_trackbar)
        except Exception as e:
            print(f"[motionTracking] Could not create GUI window: {e}. Running headless.", flush=True)
            can_show_gui = False

    try:
        while True:
            now_ts = _now()
            dt = now_ts - t_prev
            t_prev = now_ts
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            ok, frame = t_cap.read()

            # =================== CAMERA OFFLINE ===================
            if not ok or frame is None:
                last_error = f"Camera device '{t_cap.active_dev}' disconnected or not detected."
                offline_frame = create_offline_frame(
                    width=CAM_W, height=CAM_H,
                    cam_dev=t_cap.active_dev,
                    reconnect_interval=CAMERA_RECONNECT_INTERVAL
                )

                # Render Reconnect Button and Threshold Widget on Offline Slate
                btn_w, btn_h = 240, 46
                bx = (CAM_W - btn_w) // 2
                by_btn = min(CAM_H - 100, CAM_H // 2 + 155)
                offline_btn_rect = (bx, by_btn, bx + btn_w, by_btn + btn_h)
                GUI_RECONNECT_BTN_RECT = offline_btn_rect

                # Threshold adjustment widget on offline screen
                thr_w, thr_h = 240, 48
                thr_x0 = max(12, CAM_W - thr_w - 20)
                thr_y0 = 60
                offline_thresh_minus_rect = (thr_x0 + 10, thr_y0 + 21, thr_x0 + 44, thr_y0 + 43)
                offline_thresh_plus_rect = (thr_x0 + 144, thr_y0 + 21, thr_x0 + 178, thr_y0 + 43)
                GUI_THRESH_MINUS_RECT = offline_thresh_minus_rect
                GUI_THRESH_PLUS_RECT = offline_thresh_plus_rect

                mx, my = GUI_MOUSE_POS
                is_reconnect_hover = (offline_btn_rect[0] <= mx <= offline_btn_rect[2] and offline_btn_rect[1] <= my <= offline_btn_rect[3])
                is_reconnecting = (_now() < GUI_RECONNECTING_UNTIL)
                offline_frame = draw_reconnect_button(offline_frame, offline_btn_rect, is_hover=is_reconnect_hover, is_reconnecting=is_reconnecting)

                is_minus_hover = (offline_thresh_minus_rect[0] <= mx <= offline_thresh_minus_rect[2] and offline_thresh_minus_rect[1] <= my <= offline_thresh_minus_rect[3])
                is_plus_hover = (offline_thresh_plus_rect[0] <= mx <= offline_thresh_plus_rect[2] and offline_thresh_plus_rect[1] <= my <= offline_thresh_plus_rect[3])
                offline_frame = draw_threshold_widget(
                    offline_frame, threshold_val=ALARM_THRESHOLD_MM, conf_val=CONF_THR,
                    x0=thr_x0, y0=thr_y0,
                    is_hover_minus=is_minus_hover, is_hover_plus=is_plus_hover
                )
                offline_frame = draw_bottom_ribbon(offline_frame, normal_count=0, oversize_count=0, alarm_active=False,
                                                   locked_count=0, filter_max_1000=FILTER_MAX_1000MM)

                with STATE_LOCK:
                    LATEST_FRAME = offline_frame.copy()
                    LATEST_HEALTH = {
                        "ready": True,
                        "timestamp": now_ts,
                        "uptime_sec": float(now_ts - start_ts),
                        "camera_ok": False,
                        "cam_dev": str(t_cap.active_dev),
                        "last_frame_ts": last_frame_ts,
                        "last_detr_ts": last_detr_ts,
                        "last_moist_ts": last_moist_ts,
                        "last_hist_pub_ts": last_hist_pub_ts,
                        "last_reconnect_ts": t_cap.last_reconnect_ts,
                        "last_reconnect_iso": datetime.fromtimestamp(t_cap.last_reconnect_ts, tz=timezone.utc).isoformat() if t_cap.last_reconnect_ts else None,
                        "fps_smoothed": float(fps),
                        "frame_idx": int(frame_idx),
                        "processed_idx": int(processed_idx),
                        "infer_every_n_frames": int(INFER_EVERY_N_FRAMES),
                        "units": str(UNITS),
                        "alarm_active": bool(ALARM_ACTIVE),
                        "alarm_threshold_mm": float(ALARM_THRESHOLD_MM),
                        "alarm_max_d_mm": float(ALARM_MAX_D_MM) if ALARM_MAX_D_MM is not None else None,
                        "filter_max_1000mm": bool(FILTER_MAX_1000MM),
                        "last_error": str(last_error),
                    }

                if can_show_gui:
                    cv2.imshow(window_name, offline_frame)
                    key = cv2.waitKey(30) & 0xFF

                    is_max_filter_hover = (GUI_MAX_FILTER_BTN_RECT and
                                           GUI_MAX_FILTER_BTN_RECT[0] <= mx <= GUI_MAX_FILTER_BTN_RECT[2] and
                                           GUI_MAX_FILTER_BTN_RECT[1] <= my <= GUI_MAX_FILTER_BTN_RECT[3])

                    # Check for mouse click on Reconnect, Threshold, or Max Filter buttons
                    if GUI_MOUSE_CLICKED:
                        GUI_MOUSE_CLICKED = False
                        if is_reconnect_hover:
                            print("[motionTracking] Reconnect button clicked! Attempting camera reconnection...", flush=True)
                            GUI_RECONNECTING_UNTIL = _now() + 1.2
                            t_cap.trigger_reconnect()
                        elif is_minus_hover:
                            ALARM_THRESHOLD_MM = max(5.0, ALARM_THRESHOLD_MM - 5.0)
                            print(f"[motionTracking] Alarm threshold decreased (-5mm): {ALARM_THRESHOLD_MM:.1f} mm", flush=True)
                            _sync_trackbars(window_name, can_show_gui)
                        elif is_plus_hover:
                            ALARM_THRESHOLD_MM = min(200.0, ALARM_THRESHOLD_MM + 5.0)
                            print(f"[motionTracking] Alarm threshold increased (+5mm): {ALARM_THRESHOLD_MM:.1f} mm", flush=True)
                            _sync_trackbars(window_name, can_show_gui)
                        elif is_max_filter_hover:
                            FILTER_MAX_1000MM = not FILTER_MAX_1000MM
                            print(f"[motionTracking] Filter bounding boxes > 1000mm: {'ENABLED' if FILTER_MAX_1000MM else 'DISABLED'}", flush=True)

                    # Keyboard shortcut: R to reconnect
                    if key == ord('r') or key == ord('R'):
                        print("[motionTracking] 'R' pressed! Attempting camera reconnection...", flush=True)
                        GUI_RECONNECTING_UNTIL = _now() + 1.2
                        t_cap.trigger_reconnect()

                    # Keyboard shortcut: M to toggle max size filter (>1000mm)
                    if key == ord('m') or key == ord('M'):
                        FILTER_MAX_1000MM = not FILTER_MAX_1000MM
                        print(f"[motionTracking] Filter bounding boxes > 1000mm: {'ENABLED' if FILTER_MAX_1000MM else 'DISABLED'}", flush=True)

                    # Keyboard shortcuts for threshold adjustment:
                    if key == ord('+') or key == ord('='):
                        ALARM_THRESHOLD_MM = min(200.0, ALARM_THRESHOLD_MM + 5.0)
                        print(f"[motionTracking] Alarm threshold increased (+5mm): {ALARM_THRESHOLD_MM:.1f} mm", flush=True)
                        _sync_trackbars(window_name, can_show_gui)
                    elif key == ord('-') or key == ord('_'):
                        ALARM_THRESHOLD_MM = max(5.0, ALARM_THRESHOLD_MM - 5.0)
                        print(f"[motionTracking] Alarm threshold decreased (-5mm): {ALARM_THRESHOLD_MM:.1f} mm", flush=True)
                        _sync_trackbars(window_name, can_show_gui)
                    elif key == ord(']'):
                        ALARM_THRESHOLD_MM = min(200.0, ALARM_THRESHOLD_MM + 1.0)
                        print(f"[motionTracking] Alarm threshold fine-tuned (+1mm): {ALARM_THRESHOLD_MM:.1f} mm", flush=True)
                        _sync_trackbars(window_name, can_show_gui)
                    elif key == ord('['):
                        ALARM_THRESHOLD_MM = max(5.0, ALARM_THRESHOLD_MM - 1.0)
                        print(f"[motionTracking] Alarm threshold fine-tuned (-1mm): {ALARM_THRESHOLD_MM:.1f} mm", flush=True)
                        _sync_trackbars(window_name, can_show_gui)
                    elif key == ord('c') or key == ord('C'):
                        conf_steps = [0.25, 0.35, 0.50, 0.65, 0.75]
                        cur_idx = 0
                        min_diff = 999.0
                        for idx, c_val in enumerate(conf_steps):
                            if abs(CONF_THR - c_val) < min_diff:
                                min_diff = abs(CONF_THR - c_val)
                                cur_idx = idx
                        CONF_THR = conf_steps[(cur_idx + 1) % len(conf_steps)]
                        print(f"[motionTracking] Confidence threshold updated: {CONF_THR:.2f}", flush=True)
                        _sync_trackbars(window_name, can_show_gui)

                    if key == ord('q') or key == 27:
                        print("[motionTracking] Exit requested via keyboard.", flush=True)
                        break
                    try:
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            print("[motionTracking] Window closed by user.", flush=True)
                            break
                    except Exception:
                        pass
                else:
                    time.sleep(0.1)

                continue

            # =================== CAMERA ONLINE ===================
            last_frame_ts = _now()
            frame_idx += 1
            h_f, w_f = frame.shape[:2]

            # Scale estimation from reference object
            ref_info = update_scale_from_reference(frame)
            if len(SCALE_BUF) > 0:
                scale_mean = float(np.mean(SCALE_BUF))
                scale_std = float(np.std(SCALE_BUF))
            else:
                scale_mean = None
                scale_std = None

            do_infer = (frame_idx % max(1, INFER_EVERY_N_FRAMES) == 0)

            det_boxes_xyxy = np.zeros((0, 4), np.float32)
            det_scores = np.zeros((0,), np.float32)
            det_L = np.zeros(0, np.float32)
            det_W = np.zeros(0, np.float32)
            det_D = np.zeros(0, np.float32)
            box_moistures = []

            if do_infer:
                processed_idx += 1
                try:
                    if isinstance(detr, ContourMotionTracker):
                        det_boxes_xyxy, det_scores = detr.detect_boxes(frame, conf_thr=CONF_THR)
                    else:
                        x, meta = preprocess_fixed(frame)
                        logits, pred_boxes = detr.infer(x)
                        det_boxes_xyxy, det_scores = postprocess_from_fixed(pred_boxes, logits, meta, conf_thr=CONF_THR)

                    last_detr_ts = _now()
                    last_error = None
                except Exception as e:
                    err = f"Detection failed: {e}"
                    last_error = err
                    det_boxes_xyxy = np.zeros((0, 4), np.float32)
                    det_scores = np.zeros((0,), np.float32)

                if det_boxes_xyxy.shape[0] > 0:
                    keep = nms_np(det_boxes_xyxy, det_scores, iou_thr=NMS_IOU)
                    det_boxes_xyxy = det_boxes_xyxy[keep]
                    det_scores = det_scores[keep]

                # Compute chip sizes (L, W, D)
                det_L, det_W, det_D = compute_LWD(det_boxes_xyxy)

                # Optional Toggle: filter out any bounding boxes > 1000mm
                if FILTER_MAX_1000MM and det_D.size > 0:
                    valid_mask = (det_D <= MAX_SIZE_LIMIT_MM)
                    det_boxes_xyxy = det_boxes_xyxy[valid_mask]
                    det_scores = det_scores[valid_mask]
                    det_L = det_L[valid_mask]
                    det_W = det_W[valid_mask]
                    det_D = det_D[valid_mask]

                # Moisture Inference
                if MOISTURE_ENABLED and moist and (processed_idx % max(1, MOISTURE_EVERY_N_FRAMES) == 0):
                    try:
                        crops_bhwc, used_boxes = _boxes_to_topk_crops(frame, det_boxes_xyxy, det_scores, MOISTURE_TOPK)
                        if (crops_bhwc is None or len(used_boxes) == 0) and MOISTURE_FALLBACK_FULLFRAME:
                            crops_bhwc = np.stack([_prep_moist_crop_bgr(frame, MOISTURE_INPUT_SIZE, use_imagenet=False)], axis=0).astype(np.float32)
                            used_boxes = []

                        if crops_bhwc is not None:
                            probs = moist.infer_probs_bhwc(crops_bhwc)
                            if probs is not None:
                                mean_probs, pred_idx, pred_label = _moisture_aggregate(probs)
                                last_moist_ts = _now()
                                per_box = []
                                if used_boxes:
                                    for i, ub in enumerate(used_boxes):
                                        pi = int(np.argmax(probs[i]))
                                        pl = MOISTURE_CLASSES[pi] if MOISTURE_CLASSES and pi < len(MOISTURE_CLASSES) else str(pi)
                                        per_box.append({
                                            "box": ub,
                                            "pred": pl,
                                            "pred_index": pi,
                                            "probs": {MOISTURE_CLASSES[j]: float(probs[i][j]) for j in range(probs.shape[1])}
                                        })

                                with STATE_LOCK:
                                    LATEST_MOISTURE = {
                                        "ready": True,
                                        "timestamp": _now(),
                                        "classes": MOISTURE_CLASSES if MOISTURE_CLASSES else None,
                                        "mean_pred": pred_label,
                                        "mean_pred_index": pred_idx,
                                        "mean_probs": {MOISTURE_CLASSES[i]: float(mean_probs[i]) for i in range(len(mean_probs))},
                                        "topk": int(MOISTURE_TOPK),
                                        "boxes_used": int(len(used_boxes)),
                                        "per_box": per_box,
                                        "simulation": isinstance(moist, MoistureSim),
                                    }

                                for dbox in det_boxes_xyxy:
                                    assigned_m = pred_label
                                    if used_boxes:
                                        dcx, dcy = (dbox[0] + dbox[2]) / 2.0, (dbox[1] + dbox[3]) / 2.0
                                        best_dist = 999.0
                                        for pb in per_box:
                                            ub = pb["box"]
                                            ucx, ucy = (ub["x1"] + ub["x2"]) / 2.0, (ub["y1"] + ub["y2"]) / 2.0
                                            dist = math.hypot(dcx - ucx, dcy - ucy)
                                            if dist < best_dist:
                                                best_dist = dist
                                                assigned_m = pb["pred"]
                                    box_moistures.append(assigned_m)
                    except Exception as e:
                        last_error = f"Moisture error: {e}"

            # Step the High-Speed Conveyor Motion Tracker & Lock-In System
            active_tracks = tracker.step(
                det_boxes_xyxy if do_infer else np.zeros((0, 4), np.float32),
                det_scores if do_infer else np.zeros((0,), np.float32),
                det_L if do_infer else np.zeros(0, np.float32),
                det_W if do_infer else np.zeros(0, np.float32),
                det_D if do_infer else np.zeros(0, np.float32),
                moistures=box_moistures if do_infer else None,
                frame_w=w_f, frame_h=h_f
            )

            # Filter visible tracks (enforce optional max size toggle: < 1000mm)
            visible_tracks = []
            for t in active_tracks:
                if FILTER_MAX_1000MM and t.d_val > MAX_SIZE_LIMIT_MM:
                    continue
                visible_tracks.append(t)

            # Update rolling diameter buffers
            if visible_tracks:
                L_buf.extend([t.L_val for t in visible_tracks])
                W_buf.extend([t.W_val for t in visible_tracks])
                D_buf.extend([t.d_val for t in visible_tracks])
                if len(L_buf) > ROLLING_MAX:
                    del L_buf[: len(L_buf) - ROLLING_MAX]
                if len(W_buf) > ROLLING_MAX:
                    del W_buf[: len(W_buf) - ROLLING_MAX]
                if len(D_buf) > ROLLING_MAX:
                    del D_buf[: len(D_buf) - ROLLING_MAX]

            # Oversize Alarm Check
            alarm_active = False
            max_d_mm = None
            if ALARM_ENABLED and visible_tracks:
                max_d_mm = float(max(t.d_val for t in visible_tracks))
                if max_d_mm > ALARM_THRESHOLD_MM:
                    alarm_active = True
            ALARM_ACTIVE = alarm_active
            ALARM_MAX_D_MM = max_d_mm

            vis = frame.copy()

            # Draw reference object if detected
            if ref_info is not None:
                cx, cy, diam_px = ref_info
                r = int(diam_px / 2.0)
                cv2.circle(vis, (cx, cy), r, (255, 255, 255), 2)
                cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

            # Count and classify tracked chips
            num_objects = len(visible_tracks)
            normal_count = sum(1 for t in visible_tracks if not (ALARM_ENABLED and t.d_val > ALARM_THRESHOLD_MM))
            oversize_count = sum(1 for t in visible_tracks if (ALARM_ENABLED and t.d_val > ALARM_THRESHOLD_MM))
            locked_count = sum(1 for t in visible_tracks if t.locked)
            bulk_conveyor_speed = tracker.bulk_speed_px_s

            # Render Bounding Boxes, Reticles, Motion Velocity Vectors & Size Pill Tags
            unit_str = "mm" if UNITS == "mm" else "px"
            for trk in visible_tracks:
                x1, y1, x2, y2 = trk.box.astype(int)
                x1 = max(0, min(vis.shape[1] - 1, x1))
                x2 = max(0, min(vis.shape[1] - 1, x2))
                y1 = max(0, min(vis.shape[0] - 1, y1))
                y2 = max(0, min(vis.shape[0] - 1, y2))
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                d_val = trk.d_val

                is_oversized = (ALARM_ENABLED and (d_val > ALARM_THRESHOLD_MM))
                accent_bgr = (45, 60, 255) if is_oversized else ((50, 230, 130) if trk.locked else (0, 220, 255))
                subtle_bgr = (40, 45, 130) if is_oversized else ((35, 80, 50) if trk.locked else (30, 60, 75))

                # Subtle bounding rect outline
                cv2.rectangle(vis, (x1, y1), (x2, y2), subtle_bgr, 1, cv2.LINE_AA)

                # Targeting Corner Brackets (HUD Reticles)
                c_len = min(14, max(5, min(bw, bh) // 4))
                # Top-Left corner
                cv2.line(vis, (x1, y1), (x1 + c_len, y1), accent_bgr, 2, cv2.LINE_AA)
                cv2.line(vis, (x1, y1), (x1, y1 + c_len), accent_bgr, 2, cv2.LINE_AA)
                # Top-Right corner
                cv2.line(vis, (x2, y1), (x2 - c_len, y1), accent_bgr, 2, cv2.LINE_AA)
                cv2.line(vis, (x2, y1), (x2, y1 + c_len), accent_bgr, 2, cv2.LINE_AA)
                # Bottom-Left corner
                cv2.line(vis, (x1, y2), (x1 + c_len, y2), accent_bgr, 2, cv2.LINE_AA)
                cv2.line(vis, (x1, y2), (x1, y2 - c_len), accent_bgr, 2, cv2.LINE_AA)
                # Bottom-Right corner
                cv2.line(vis, (x2, y2), (x2 - c_len, y2), accent_bgr, 2, cv2.LINE_AA)
                cv2.line(vis, (x2, y2), (x2, y2 - c_len), accent_bgr, 2, cv2.LINE_AA)

                # Lock-In Crosshair Reticle on Centroid
                tcx = int((x1 + x2) / 2)
                tcy = int((y1 + y2) / 2)
                if trk.locked:
                    cv2.circle(vis, (tcx, tcy), 3, accent_bgr, -1, cv2.LINE_AA)
                    cv2.line(vis, (tcx - 6, tcy), (tcx + 6, tcy), accent_bgr, 1, cv2.LINE_AA)
                    cv2.line(vis, (tcx, tcy - 6), (tcx, tcy + 6), accent_bgr, 1, cv2.LINE_AA)

                # Conveyor Motion Velocity Vector Arrow
                speed = trk.speed_px_s
                if speed > 15.0:
                    arrow_len = min(36, max(12, int(speed * 0.08)))
                    end_x = int(tcx + (trk.vx / speed) * arrow_len)
                    end_y = int(tcy + (trk.vy / speed) * arrow_len)
                    cv2.arrowedLine(vis, (tcx, tcy), (end_x, end_y), (0, 230, 255), 2, cv2.LINE_AA, tipLength=0.35)

                # Oversize exclamation warning badge on top-right
                if is_oversized:
                    badge_r = int(max(7, min(12, min(bw, bh) * 0.18)))
                    cx_badge = x2 - badge_r - 2
                    cy_badge = y1 + badge_r + 2
                    cv2.circle(vis, (cx_badge, cy_badge), badge_r + 2, (30, 30, 180), -1, cv2.LINE_AA)
                    cv2.circle(vis, (cx_badge, cy_badge), badge_r, (45, 60, 255), -1, cv2.LINE_AA)
                    cv2.putText(vis, "!", (cx_badge - 3, cy_badge + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, lineType=cv2.LINE_AA)

                # Modern Glass Size Pill Tag (shows Lock status & estimated size)
                lock_prefix = f"LOCK #{trk.track_id} " if trk.locked else f"ID #{trk.track_id} "
                tag_str = f"{lock_prefix}| {d_val:.0f} {unit_str}"
                font_pill = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), baseline = cv2.getTextSize(tag_str, font_pill, 0.38, 1)

                pill_h = 20
                pill_w = tw + 18
                if y1 - pill_h - 4 >= 0:
                    pill_y1 = y1 - pill_h - 4
                else:
                    pill_y1 = y2 + 4
                pill_y2 = pill_y1 + pill_h

                pill_x1 = max(0, min(vis.shape[1] - pill_w - 2, x1))
                pill_x2 = pill_x1 + pill_w

                # Draw frosted glass card for tag
                draw_glass_card(vis, (pill_x1, pill_y1), (pill_x2, pill_y2),
                                bg_color=(18, 22, 28), border_color=accent_bgr, alpha=0.88, radius=4)

                # Vertical accent indicator on left of pill
                cv2.rectangle(vis, (pill_x1 + 3, pill_y1 + 3), (pill_x1 + 6, pill_y2 - 3), accent_bgr, -1)

                # Size text inside pill
                text_y = pill_y1 + 14
                cv2.putText(vis, tag_str, (pill_x1 + 10, text_y), font_pill, 0.38, (245, 248, 252), 1, cv2.LINE_AA)

            # Asynchronous CSV Logging to 'yyyymmdd_woodChips.csv'
            if CSV_LOGGING_ENABLED and csv_logger:
                cur_batch_str = classify_batch(D_buf) if len(D_buf) > 0 else "Pending"
                for obj_idx, trk in enumerate(visible_tracks):
                    if now_ts - trk.last_csv_log_ts >= CSV_LOG_THROTTLE_SEC:
                        trk.last_csv_log_ts = now_ts
                        is_exceeded = (ALARM_ENABLED and trk.d_val > ALARM_THRESHOLD_MM)
                        row_data = {
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                            "detected_object_index": int(obj_idx),
                            "track_id": int(trk.track_id),
                            "estimated_size_mm": round(float(trk.d_val), 1),
                            "threshold_size_mm": round(float(ALARM_THRESHOLD_MM), 1),
                            "threshold_status": "EXCEEDED" if is_exceeded else "NOT_EXCEEDED",
                            "moisture_level": str(trk.moisture) if trk.moisture else "N/A",
                            "length_mm": round(float(trk.L_val), 1),
                            "width_mm": round(float(trk.W_val), 1),
                            "aspect_ratio": round(float(trk.L_val / max(1.0, trk.W_val)), 2),
                            "confidence": round(float(trk.score), 2),
                            "conveyor_speed_px_s": round(float(trk.speed_px_s), 1),
                            "conveyor_speed_mm_s": round(float(trk.speed_px_s / scale_mean), 1) if (scale_mean and scale_mean > 0) else "N/A",
                            "bbox_x1": int(round(trk.box[0])),
                            "bbox_y1": int(round(trk.box[1])),
                            "bbox_x2": int(round(trk.box[2])),
                            "bbox_y2": int(round(trk.box[3])),
                            "batch_quality": str(cur_batch_str)
                        }
                        csv_logger.log_detection(row_data)

            # Stats Dictionary
            stats_dict = compute_diameter_stats(D_buf, UNITS, scale_mean=scale_mean, scale_std=scale_std)
            if stats_dict is not None:
                stats_dict["num_objects"] = int(num_objects)
                stats_dict["normal_count"] = int(normal_count)
                stats_dict["oversize_count"] = int(oversize_count)
                stats_dict["locked_count"] = int(locked_count)
                stats_dict["conveyor_speed_px_s"] = float(bulk_conveyor_speed)
                stats_dict["alarm_active"] = bool(ALARM_ACTIVE)
                stats_dict["alarm_threshold_mm"] = float(ALARM_THRESHOLD_MM)
                stats_dict["alarm_max_d_mm"] = float(ALARM_MAX_D_MM) if ALARM_MAX_D_MM is not None else None
                stats_dict["ref_diam_mm"] = float(REF_DIAM_MM)
                stats_dict["filter_max_1000mm"] = bool(FILTER_MAX_1000MM)

            # Histogram
            hist_data = None
            should_update_hist = (now_ts - last_hist_time) >= float(HIST_UPDATE_INTERVAL)
            if should_update_hist and len(D_buf) > 10:
                last_hist_time = now_ts
                vals = np.array(D_buf, dtype=np.float32)
                upper = float(HIST_D_MAX_MM) if UNITS == "mm" else max(1.0, float(np.percentile(vals, 99.0)))
                edges = np.linspace(0.0, upper, HIST_NUM_BINS + 1, dtype=np.float32)
                hist, _ = np.histogram(vals, bins=edges)
                centers = 0.5 * (edges[:-1] + edges[1:])

                if HIST_EMA_ENABLED and prev_hist_counts is not None and len(prev_hist_counts) == len(hist):
                    sm = (1.0 - HIST_EMA_ALPHA) * prev_hist_counts + HIST_EMA_ALPHA * hist.astype(np.float32)
                    hist_out = np.rint(sm).astype(int)
                else:
                    hist_out = hist.astype(int)

                prev_hist_counts = hist_out.astype(np.float32)
                prev_hist_bins = centers.copy()

                thr = float(ALARM_THRESHOLD_MM)
                pct_over = float(100.0 * np.mean(vals > thr)) if (ALARM_ENABLED and UNITS == "mm") else None

                hist_data = {
                    "ready": True,
                    "timestamp": _now(),
                    "units": UNITS,
                    "mode": "diameter",
                    "diameter": {"bins": centers.tolist(), "counts": hist_out.tolist()},
                    "threshold_x": thr if (ALARM_ENABLED and UNITS == "mm") else None,
                    "pct_over_threshold": pct_over,
                    "alarm_threshold_mm": float(ALARM_THRESHOLD_MM),
                    "alarm_enabled": bool(ALARM_ENABLED),
                }
                last_hist_pub_ts = _now()

            # Health dictionary
            health_dict = {
                "ready": True,
                "timestamp": now_ts,
                "uptime_sec": float(now_ts - start_ts),
                "camera_ok": True,
                "cam_dev": str(t_cap.active_dev),
                "num_objects": int(num_objects),
                "normal_count": int(normal_count),
                "oversize_count": int(oversize_count),
                "locked_count": int(locked_count),
                "conveyor_speed_px_s": float(bulk_conveyor_speed),
                "filter_max_1000mm": bool(FILTER_MAX_1000MM),
                "last_frame_ts": last_frame_ts,
                "last_detr_ts": last_detr_ts,
                "last_moist_ts": last_moist_ts,
                "last_hist_pub_ts": last_hist_pub_ts,
                "last_reconnect_ts": t_cap.last_reconnect_ts,
                "last_reconnect_iso": datetime.fromtimestamp(t_cap.last_reconnect_ts, tz=timezone.utc).isoformat() if t_cap.last_reconnect_ts else None,
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

            # Draw HUD Overlays: Status Badge, Reconnect Button, Objects Badge, Threshold Stepper, Stats Panel, Alarm Banner, and Bottom Ribbon
            disp = draw_online_status_badge(vis, t_cap.active_dev, fps, w_f, h_f, model_name=model_name)

            # Modern Top HUD Layout:
            # [12..320] Status Badge (w=308, h=48)
            # [332..512] Reconnect Button (w=180, h=48)
            # [524..734] Objects Badge (w=210, h=48)
            # [746..986] Threshold Stepper (w=240, h=48)
            # [w-282..w-12] Stats Panel (w=270, h=316)
            if w_f >= 1000:
                online_btn_rect = (332, 12, 512, 60)
                obj_x0, obj_y0 = 524, 12
                thr_x0, thr_y0 = 746, 12
            elif w_f >= 760:
                online_btn_rect = (332, 12, 512, 60)
                obj_x0, obj_y0 = 524, 12
                thr_x0, thr_y0 = 12, 68
            else:
                online_btn_rect = (12, 68, 192, 116)
                obj_x0, obj_y0 = 12, 124
                thr_x0, thr_y0 = 12, 180

            GUI_RECONNECT_BTN_RECT = online_btn_rect

            # Accurate hitboxes for Minus and Plus in Threshold Stepper
            online_thresh_minus_rect = (thr_x0 + 10, thr_y0 + 21, thr_x0 + 44, thr_y0 + 43)
            online_thresh_plus_rect = (thr_x0 + 144, thr_y0 + 21, thr_x0 + 178, thr_y0 + 43)
            GUI_THRESH_MINUS_RECT = online_thresh_minus_rect
            GUI_THRESH_PLUS_RECT = online_thresh_plus_rect

            # Objects Detected Badge (includes locked count)
            disp = draw_objects_badge(disp, num_objects=num_objects, normal_count=normal_count,
                                      oversize_count=oversize_count, locked_count=locked_count,
                                      x0=obj_x0, y0=obj_y0)

            # Reconnect Button
            mx, my = GUI_MOUSE_POS
            is_reconnect_hover = (online_btn_rect[0] <= mx <= online_btn_rect[2] and online_btn_rect[1] <= my <= online_btn_rect[3])
            is_reconnecting = (_now() < GUI_RECONNECTING_UNTIL)
            disp = draw_reconnect_button(disp, online_btn_rect, is_hover=is_reconnect_hover, is_reconnecting=is_reconnecting)

            # Threshold Widget
            is_minus_hover = (online_thresh_minus_rect[0] <= mx <= online_thresh_minus_rect[2] and online_thresh_minus_rect[1] <= my <= online_thresh_minus_rect[3])
            is_plus_hover = (online_thresh_plus_rect[0] <= mx <= online_thresh_plus_rect[2] and online_thresh_plus_rect[1] <= my <= online_thresh_plus_rect[3])
            disp = draw_threshold_widget(
                disp, threshold_val=ALARM_THRESHOLD_MM, conf_val=CONF_THR,
                x0=thr_x0, y0=thr_y0,
                is_hover_minus=is_minus_hover, is_hover_plus=is_plus_hover
            )

            # Telemetry Stats Card (includes conveyor motion, locked chips, max filter, csv status)
            disp = draw_stats_panel(disp, D_buf, UNITS, scale_mean=scale_mean, scale_std=scale_std,
                                    num_objects=num_objects, normal_count=normal_count, oversize_count=oversize_count,
                                    conveyor_speed=bulk_conveyor_speed, locked_count=locked_count,
                                    filter_max_1000=FILTER_MAX_1000MM)

            # High-visibility alarm banner when oversize chip triggers
            if ALARM_ACTIVE:
                disp = draw_alarm_banner(disp, ALARM_MAX_D_MM, ALARM_THRESHOLD_MM)

            # Docked bottom status ribbon with color legend, locked count, max filter toggle, and keycaps
            disp = draw_bottom_ribbon(disp, normal_count=normal_count, oversize_count=oversize_count,
                                      alarm_active=ALARM_ACTIVE, locked_count=locked_count,
                                      filter_max_1000=FILTER_MAX_1000MM)

            with STATE_LOCK:
                LATEST_FRAME = disp.copy()
                LATEST_STATS = stats_dict if stats_dict is not None else {"ready": False, "timestamp": _now(), "error": "No detections yet."}
                if hist_data is not None:
                    HIST_DATA = hist_data
                LATEST_HEALTH = health_dict

            # GUI Display
            if can_show_gui:
                cv2.imshow(window_name, disp)
                key = cv2.waitKey(1) & 0xFF

                is_max_filter_hover = (GUI_MAX_FILTER_BTN_RECT and
                                       GUI_MAX_FILTER_BTN_RECT[0] <= mx <= GUI_MAX_FILTER_BTN_RECT[2] and
                                       GUI_MAX_FILTER_BTN_RECT[1] <= my <= GUI_MAX_FILTER_BTN_RECT[3])

                # Check for mouse click on Reconnect, Threshold, or Max Filter buttons
                if GUI_MOUSE_CLICKED:
                    GUI_MOUSE_CLICKED = False
                    if is_reconnect_hover:
                        print("[motionTracking] Reconnect button clicked! Attempting camera reconnection...", flush=True)
                        GUI_RECONNECTING_UNTIL = _now() + 1.2
                        t_cap.trigger_reconnect()
                    elif is_minus_hover:
                        ALARM_THRESHOLD_MM = max(5.0, ALARM_THRESHOLD_MM - 5.0)
                        print(f"[motionTracking] Alarm threshold decreased (-5mm): {ALARM_THRESHOLD_MM:.1f} mm", flush=True)
                        _sync_trackbars(window_name, can_show_gui)
                    elif is_plus_hover:
                        ALARM_THRESHOLD_MM = min(200.0, ALARM_THRESHOLD_MM + 5.0)
                        print(f"[motionTracking] Alarm threshold increased (+5mm): {ALARM_THRESHOLD_MM:.1f} mm", flush=True)
                        _sync_trackbars(window_name, can_show_gui)
                    elif is_max_filter_hover:
                        FILTER_MAX_1000MM = not FILTER_MAX_1000MM
                        print(f"[motionTracking] Filter bounding boxes > 1000mm: {'ENABLED' if FILTER_MAX_1000MM else 'DISABLED'}", flush=True)

                # Keyboard shortcut: R to reconnect
                if key == ord('r') or key == ord('R'):
                    print("[motionTracking] 'R' pressed! Attempting camera reconnection...", flush=True)
                    GUI_RECONNECTING_UNTIL = _now() + 1.2
                    t_cap.trigger_reconnect()

                # Keyboard shortcut: M to toggle max size filter (>1000mm)
                if key == ord('m') or key == ord('M'):
                    FILTER_MAX_1000MM = not FILTER_MAX_1000MM
                    print(f"[motionTracking] Filter bounding boxes > 1000mm: {'ENABLED' if FILTER_MAX_1000MM else 'DISABLED'}", flush=True)

                # Keyboard shortcuts for threshold adjustment:
                if key == ord('+') or key == ord('='):
                    ALARM_THRESHOLD_MM = min(200.0, ALARM_THRESHOLD_MM + 5.0)
                    print(f"[motionTracking] Alarm threshold increased (+5mm): {ALARM_THRESHOLD_MM:.1f} mm", flush=True)
                    _sync_trackbars(window_name, can_show_gui)
                elif key == ord('-') or key == ord('_'):
                    ALARM_THRESHOLD_MM = max(5.0, ALARM_THRESHOLD_MM - 5.0)
                    print(f"[motionTracking] Alarm threshold decreased (-5mm): {ALARM_THRESHOLD_MM:.1f} mm", flush=True)
                    _sync_trackbars(window_name, can_show_gui)
                elif key == ord(']'):
                    ALARM_THRESHOLD_MM = min(200.0, ALARM_THRESHOLD_MM + 1.0)
                    print(f"[motionTracking] Alarm threshold fine-tuned (+1mm): {ALARM_THRESHOLD_MM:.1f} mm", flush=True)
                    _sync_trackbars(window_name, can_show_gui)
                elif key == ord('['):
                    ALARM_THRESHOLD_MM = max(5.0, ALARM_THRESHOLD_MM - 1.0)
                    print(f"[motionTracking] Alarm threshold fine-tuned (-1mm): {ALARM_THRESHOLD_MM:.1f} mm", flush=True)
                    _sync_trackbars(window_name, can_show_gui)
                elif key == ord('c') or key == ord('C'):
                    conf_steps = [0.25, 0.35, 0.50, 0.65, 0.75]
                    cur_idx = 0
                    min_diff = 999.0
                    for idx, c_val in enumerate(conf_steps):
                        if abs(CONF_THR - c_val) < min_diff:
                            min_diff = abs(CONF_THR - c_val)
                            cur_idx = idx
                    CONF_THR = conf_steps[(cur_idx + 1) % len(conf_steps)]
                    print(f"[motionTracking] Confidence threshold updated: {CONF_THR:.2f}", flush=True)
                    _sync_trackbars(window_name, can_show_gui)

                if key == ord('q') or key == 27:
                    print("[motionTracking] Exit requested via keyboard.", flush=True)
                    break
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("[motionTracking] Window closed by user.", flush=True)
                        break
                except Exception:
                    pass
            else:
                time.sleep(0.001)

    finally:
        print("[motionTracking] Cleaning up resources...", flush=True)
        if csv_logger:
            csv_logger.close()
        try:
            t_cap.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if ctx is not None:
            try:
                ctx.pop()
            except Exception:
                pass
        print("[motionTracking] Exited cleanly.", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Wood-Chip Monitor - Live Motion Tracking")
    parser.add_argument("--camera", "--dev", dest="camera_dev", default=None,
                        help="Camera device index (e.g. 0) or path (e.g. /dev/video0)")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Run in headless mode without opening GUI window")
    parser.add_argument("--alarm", type=float, default=None,
                        help="Oversize chip alarm threshold in mm (default: 60.0)")
    args = parser.parse_args()

    global CAM_DEV, ALARM_THRESHOLD_MM
    if args.camera_dev is not None:
        CAM_DEV = _resolve_camera_dev(args.camera_dev)
    if args.alarm is not None:
        ALARM_THRESHOLD_MM = float(args.alarm)

    run_inference_loop(headless=args.headless)


if __name__ == "__main__":
    main()
