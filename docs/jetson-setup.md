# Jetson Deployment Environment

Wood-Chip Monitor was originally deployed and validated on an NVIDIA Jetson Nano platform. This document summarizes the reference hardware and software environment used for the edge-AI prototype and explains the expected deployment configuration.

## Reference Hardware Environment

The validated prototype environment used:

- NVIDIA Jetson Nano
- NVIDIA Tegra X1 integrated GPU
- ARMv8 64-bit processor
- approximately 4 GB system memory
- approximately 60 GB system storage
- Ubuntu 18.04.5 LTS
- Python 3.6.9
- NVIDIA L4T R32.6.1

The original deployment environment reflects the Jetson Nano software ecosystem available during development of the Wood-Chip Monitor prototype.

> **Note:** The reference deployment uses a legacy Jetson Nano software stack. Users should not assume that TensorRT engines generated with newer CUDA, TensorRT, or JetPack versions will be binary-compatible with the original environment.

---

## Edge-AI Deployment Architecture

The deployed system performs the main quality-monitoring operations directly on the Jetson device.

The edge pipeline consists of:

1. image acquisition from the monitoring camera,
2. TensorRT-based wood-chip detection,
3. physical chip-size estimation,
4. rolling size-distribution analysis,
5. oversize-chip identification and alerts,
6. TensorRT-based moisture assessment,
7. local event and quality-data management, and
8. visualization through the Wood-Chip Monitor web dashboard.

All primary inference and monitoring operations are designed to run locally without requiring cloud-based processing.

---

## TensorRT Deployment

Wood-Chip Monitor uses TensorRT-optimized inference engines for two primary machine-learning components:

1. wood-chip object detection, and
2. vision-based moisture assessment.

TensorRT engine files are intentionally **not stored in this repository** because serialized TensorRT engines depend on the target:

- hardware platform,
- CUDA version,
- TensorRT version,
- JetPack/L4T environment, and
- model export configuration.

The default runtime expects the following deployment artifacts:

```text
models/
├── detr_resnet101_fp16.engine
├── moistnetlite_fp16.engine
└── moistnetlite_classes.txt
```

Alternative model locations can be provided through environment variables.

For additional information, see:

```text
models/README.md
configs/device.env.example
```

---

## Model Configuration

By default, the application looks for model artifacts inside the repository's `models/` directory.

The relevant environment variables are:

```text
WOODCHIP_MODEL_DIR
WOODCHIP_DETR_ENGINE
WOODCHIP_MOISTURE_ENGINE
WOODCHIP_MOISTURE_CLASSES
```

For example:

```bash
export WOODCHIP_MODEL_DIR=/home/user/woodchip-monitor/models
```

Individual model paths can also be overridden:

```bash
export WOODCHIP_DETR_ENGINE=/path/to/detr_resnet101_fp16.engine
export WOODCHIP_MOISTURE_ENGINE=/path/to/moistnetlite_fp16.engine
export WOODCHIP_MOISTURE_CLASSES=/path/to/moistnetlite_classes.txt
```

Machine-specific paths should not be committed to the repository.

---

## Camera Configuration

The reference prototype used a Linux video device such as:

```text
/dev/video0
```

The default runtime camera configuration is:

```text
width:  1480
height: 900
fps:    20
```

These settings can be overridden using the following environment variables:

```text
WOODCHIP_CAMERA_DEVICE
WOODCHIP_CAMERA_WIDTH
WOODCHIP_CAMERA_HEIGHT
WOODCHIP_CAMERA_FPS
```

Example:

```bash
export WOODCHIP_CAMERA_DEVICE=/dev/video0
export WOODCHIP_CAMERA_WIDTH=1480
export WOODCHIP_CAMERA_HEIGHT=900
export WOODCHIP_CAMERA_FPS=20
```

Camera settings should be adjusted when a different imaging device or acquisition configuration is used.

---

## Runtime Storage

Generated application data should remain outside version control.

By default, Wood-Chip Monitor uses:

```text
data/
outputs/
```

for runtime data.

The `data/` directory is intended for application state such as the local SQLite database.

The `outputs/` directory is intended for generated inference outputs, monitoring results, and other runtime artifacts.

Both directories are excluded from Git through `.gitignore`.

The locations can be changed using:

```text
WOODCHIP_DB_PATH
WOODCHIP_OUTPUT_DIR
```

For example:

```bash
export WOODCHIP_DB_PATH=/home/user/woodchip-data/woodchip_app.sqlite3
export WOODCHIP_OUTPUT_DIR=/home/user/woodchip-data/outputs
```

---

## Device Configuration

A deployment configuration template is provided at:

```text
configs/device.env.example
```

The template includes settings for:

- device identity,
- administrator initialization,
- camera configuration,
- model locations,
- application database location, and
- runtime output storage.

The example file should be copied and customized locally for the target system.

Sensitive values such as passwords or private machine-specific information should never be committed to the repository.

---

## Application Components

The deployed application currently consists of three primary components:

```text
app/
├── backend_app.py
├── live_cam_trt.py
└── web/
    └── index.html
```

### `live_cam_trt.py`

Implements the primary edge-AI monitoring pipeline, including:

- camera acquisition,
- TensorRT detector inference,
- chip measurement,
- physical calibration,
- size-distribution statistics,
- oversize monitoring,
- moisture inference, and
- runtime monitoring information.

### `backend_app.py`

Provides the local FastAPI backend used by the monitoring system.

The backend supports functionality including:

- application API endpoints,
- authentication,
- role-based access,
- event management,
- audit records,
- quality-rule configuration,
- device information, and
- communication with the live inference pipeline.

### `web/index.html`

Provides the browser-based operator interface for monitoring the Wood-Chip Monitor system.

---

## Standalone TensorRT Inference

A standalone detector inference utility is provided at:

```text
tools/infer_trt.py
```

The utility can be used to run the deployed DETR TensorRT engine on example images independently from the complete monitoring application.

Example:

```bash
python tools/infer_trt.py \
    --engine models/detr_resnet101_fp16.engine \
    --images examples/images \
    --output outputs/predictions
```

Optional parameters include:

```text
--confidence
--nms-iou
```

This tool is primarily intended for deployment verification and inference testing.

---

## Deployment Verification

Before launching the complete monitoring application, verify that:

- the Jetson camera is detected,
- the required TensorRT engines are available,
- the TensorRT runtime matches the engine build environment,
- CUDA is available,
- PyCUDA is functioning,
- OpenCV can access the camera,
- the model class file is available,
- the runtime output directory is writable, and
- sufficient storage and system memory are available.

A basic Linux camera check can be performed with:

```bash
ls -l /dev/video*
```

The Python version can be checked with:

```bash
python3 --version
```

The L4T version can be checked with:

```bash
cat /etc/nv_tegra_release
```

System memory can be inspected with:

```bash
free -h
```

Disk availability can be inspected with:

```bash
df -h
```

---

## Reference Deployment Notes

The public repository preserves the architecture and deployment workflow of the validated Wood-Chip Monitor prototype while removing:

- developer-specific absolute paths,
- private runtime data,
- local databases,
- TensorRT engine binaries,
- intermediate numerical arrays, and
- machine-specific development artifacts.

The goal is to provide a clear and reproducible description of the deployed research system without treating hardware-specific binary artifacts as portable source code.

---

## Compatibility

The reference system was developed using an older Jetson Nano software stack.

Because TensorRT serialized engines are environment-specific, users working with newer Jetson devices or newer JetPack versions should regenerate model engines for their target platform.

The following components may require platform-specific adaptation:

- TensorRT API calls,
- CUDA compatibility,
- PyCUDA installation,
- Python version,
- OpenCV camera support,
- engine binding behavior, and
- model serialization.

Future repository updates may document additional deployment environments when they are independently validated.

---

## Next Documentation

Additional deployment documentation will be added for:

- model export,
- TensorRT engine generation,
- complete application startup,
- system validation,
- calibration,
- operator usage, and
- troubleshooting.