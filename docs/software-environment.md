# Software Environment

Wood-Chip Monitor was developed and validated on an NVIDIA Jetson Nano using the NVIDIA Jetson software ecosystem available during the original prototype deployment.

Because the reference deployment uses a legacy Jetson platform, reproducing the environment requires more care than installing a conventional modern Python package.

This document distinguishes between:

1. the validated device environment,
2. NVIDIA platform dependencies,
3. Python application dependencies,
4. model-development dependencies,
5. and future reproducibility requirements.

---

## Reference Deployment Platform

The documented Wood-Chip Monitor prototype used:

```text
Hardware:        NVIDIA Jetson Nano
Architecture:    ARMv8 / aarch64
GPU:             NVIDIA Tegra X1
System memory:   approximately 4 GB
Operating system: Ubuntu 18.04.5 LTS
Python:          3.6.9
NVIDIA L4T:      R32.6.1
```

This environment reflects the software stack used during the original deployed prototype.

It should therefore be treated as the **reference deployment environment**, not as a recommendation to use an outdated software stack for new projects.

---

## Why the Environment Requires Special Handling

Jetson applications rely on several tightly coupled NVIDIA components.

These may include:

- JetPack,
- Linux for Tegra (L4T),
- CUDA,
- cuDNN,
- TensorRT,
- GPU drivers,
- PyCUDA,
- OpenCV,
- and Python bindings supplied for the target system.

Unlike ordinary Python packages, these components cannot always be upgraded independently.

For example, a TensorRT engine generated using one TensorRT/CUDA/hardware combination may not deserialize correctly in another environment.

For this reason, Wood-Chip Monitor distinguishes between:

```text
Application source code
        │
        ▼
Python dependencies
        │
        ▼
NVIDIA platform dependencies
        │
        ▼
Target Jetson environment
        │
        ▼
Device-specific TensorRT engines
```

---

## NVIDIA Platform Components

The deployed application relies on the NVIDIA edge-inference ecosystem.

The primary platform-level components are:

### NVIDIA TensorRT

Used for optimized inference of:

- the wood-chip detector,
- and MoistNetLite.

The deployed model artifacts use the TensorRT engine format:

```text
*.engine
```

These binaries are not committed to the repository.

---

### CUDA

CUDA provides the underlying GPU-compute platform used by TensorRT and PyCUDA.

The exact CUDA version should remain compatible with the JetPack/L4T and TensorRT environment installed on the target Jetson device.

---

### cuDNN

cuDNN may be present as part of the NVIDIA deep-learning software stack used by the device.

Its version should follow the compatibility requirements of the installed JetPack/L4T environment.

---

### PyCUDA

The deployed Python inference code uses PyCUDA for GPU memory allocation, transfers, and TensorRT execution.

Relevant source files include:

```text
app/live_cam_trt.py
tools/infer_trt.py
```

---

## Core Python Dependencies

Wood-Chip Monitor uses several categories of Python dependencies.

---

## Backend and Application Services

### FastAPI

FastAPI provides the backend API used by the monitoring application.

It supports functions including:

- runtime-data access,
- application configuration,
- authentication,
- event access,
- audit information,
- quality-rule management,
- and device-health information.

Primary source:

```text
app/backend_app.py
```

---

### Uvicorn

Uvicorn is used to serve the FastAPI application.

The backend can therefore expose the monitoring API and browser dashboard through a local web server.

---

### Python Standard Library

Several standard-library modules are used throughout the application, including functionality for:

- file-system operations,
- environment variables,
- time handling,
- JSON processing,
- threading,
- SQLite persistence,
- and general application utilities.

SQLite support is provided through Python's standard library.

---

## Computer Vision and Numerical Processing

### NumPy

NumPy is used extensively for:

- tensor preparation,
- numerical operations,
- bounding-box processing,
- geometric calculations,
- statistical analysis,
- image-array manipulation,
- TensorRT input/output processing.

---

### OpenCV

OpenCV is used for:

- camera capture,
- image resizing,
- image conversion,
- drawing detections,
- visualization,
- image encoding,
- image saving,
- crop handling.

Primary usage appears in:

```text
app/live_cam_trt.py
tools/infer_trt.py
```

---

## Edge Inference Dependencies

### TensorRT Python Bindings

TensorRT Python bindings are required to:

- deserialize `.engine` files,
- create execution contexts,
- inspect model bindings,
- execute inference,
- retrieve model outputs.

TensorRT should generally be installed through the NVIDIA-supported Jetson environment rather than by blindly installing an arbitrary PyPI package.

---

### PyCUDA

PyCUDA is used to manage:

- GPU buffers,
- host-device transfers,
- CUDA streams,
- asynchronous inference operations.

Its compatibility should be verified against the CUDA environment installed on the device.

---

## MoistNetLite Source Dependencies

The source architecture for MoistNetLite is provided in:

```text
models/moistnetlite.py
```

The architecture definition uses TensorFlow/Keras components.

Relevant functionality includes:

- convolution layers,
- pooling,
- dropout,
- global average pooling,
- dense layers,
- image translation augmentation,
- Adam optimization,
- classification metrics.

TensorFlow/Keras is required when working with the source model architecture.

It is **not necessarily required for normal live TensorRT execution** once the optimized moisture engine has already been generated.

---

## Runtime vs. Model-Development Dependencies

The repository distinguishes between runtime dependencies and model-development dependencies.

### Live runtime

The primary Jetson runtime requires components such as:

```text
Python
NumPy
OpenCV
TensorRT
PyCUDA
FastAPI
Uvicorn
```

### MoistNetLite architecture or model preparation

Additional dependencies may include:

```text
TensorFlow
Keras
```

### Detector training and research

The full detector training and benchmarking environment is maintained in the separate UOT-DETR repository rather than duplicated here.

This keeps Wood-Chip Monitor focused on edge deployment and application integration.

---

## Browser Interface

The operator dashboard is served from:

```text
app/web/index.html
```

The interface is designed to run in a standard browser communicating with the local FastAPI backend.

No separate desktop application framework is required.

---

## Local Database

Operational persistence uses SQLite.

The default conceptual database location is:

```text
data/woodchip_app.sqlite3
```

The location can be overridden using:

```text
WOODCHIP_DB_PATH
```

Database files are excluded from Git because they contain local runtime state rather than source code.

---

## Runtime Output Directory

Generated inference artifacts can be stored under:

```text
outputs/
```

or another configured directory.

The location can be changed using:

```text
WOODCHIP_OUTPUT_DIR
```

The output directory is ignored by Git.

---

## Model Directory

The default local deployment model location is:

```text
models/
```

Expected deployment artifacts include:

```text
models/
├── detr_resnet101_fp16.engine
├── moistnetlite_fp16.engine
└── moistnetlite_classes.txt
```

Alternative paths can be configured using:

```text
WOODCHIP_MODEL_DIR
WOODCHIP_DETR_ENGINE
WOODCHIP_MOISTURE_ENGINE
WOODCHIP_MOISTURE_CLASSES
```

See:

```text
configs/device.env.example
```

for an example configuration.

---

## Camera Configuration

The reference runtime uses configuration variables for camera access:

```text
WOODCHIP_CAMERA_DEVICE
WOODCHIP_CAMERA_WIDTH
WOODCHIP_CAMERA_HEIGHT
WOODCHIP_CAMERA_FPS
```

Default values correspond to the original deployment configuration:

```text
Device: /dev/video0
Width:  1480
Height: 900
FPS:    20
```

These settings should be adjusted for the target camera and deployment environment when necessary.

---

## Why There Is No Generic `requirements.txt` Yet

The repository does not currently claim that the validated Jetson environment can be reproduced using:

```bash
pip install -r requirements.txt
```

There are several reasons.

### Python 3.6 compatibility

The original deployment used:

```text
Python 3.6.9
```

Many current releases of commonly used Python libraries no longer support Python 3.6.

Installing the latest versions would therefore not reproduce the original system.

---

### TensorRT is platform-dependent

TensorRT is tightly connected to:

- JetPack,
- L4T,
- CUDA,
- cuDNN,
- GPU architecture,
- and NVIDIA's Jetson package ecosystem.

A normal desktop `pip install` process does not fully represent these dependencies.

---

### PyCUDA compatibility

PyCUDA must be compatible with the installed CUDA environment.

Using an arbitrary package version could produce an environment different from the validated deployment.

---

### OpenCV differences

Jetson environments may use NVIDIA-provided or platform-specific OpenCV builds.

These may differ from current desktop PyPI distributions.

---

### TensorFlow on Jetson

TensorFlow compatibility on Jetson platforms may depend on NVIDIA-provided builds and the corresponding CUDA/cuDNN environment.

The repository should therefore avoid claiming that a generic modern TensorFlow package reproduces the original model-development environment.

---

## Dependency Documentation Strategy

The repository will document dependencies in separate categories rather than pretending that every component can be installed with one universal command.

The intended organization is:

```text
Device-provided NVIDIA stack
        │
        ├── CUDA
        ├── cuDNN
        ├── TensorRT
        └── JetPack / L4T
        │
        ▼
Runtime Python dependencies
        │
        ├── NumPy
        ├── OpenCV
        ├── PyCUDA
        ├── FastAPI
        └── Uvicorn
        │
        ▼
Optional model-development dependencies
        │
        ├── TensorFlow / Keras
        └── detector export tools
```

---

## Exact Version Policy

Exact dependency versions should only be published when they can be supported by evidence from:

1. the original Jetson environment,
2. package records from the deployed device,
3. development-environment records,
4. or successful reproduction on an equivalent system.

Versions should **not** be guessed solely to create a visually complete `requirements.txt`.

This is especially important for:

- TensorRT,
- CUDA,
- PyCUDA,
- TensorFlow,
- OpenCV,
- FastAPI,
- Uvicorn.

---

## Reference Environment vs. Modern Development Environment

There are two different reproducibility goals.

### Reference deployment reproduction

The first goal is to preserve the environment of the original validated prototype:

```text
Jetson Nano
Ubuntu 18.04.5
Python 3.6.9
L4T R32.6.1
Tegra X1
```

This provides historical reproducibility for the deployed system.

### Modern development support

A future release may also define a separate environment for:

- source-code inspection,
- development,
- testing,
- modern NVIDIA devices,
- or non-Jetson systems.

That environment should be documented separately rather than silently replacing the validated reference configuration.

---

## TensorRT Engine Compatibility

Serialized TensorRT engines are intentionally excluded from the repository because compatibility may depend on:

- GPU architecture,
- TensorRT version,
- CUDA version,
- JetPack version,
- L4T version,
- precision mode,
- optimization profile,
- input dimensions,
- engine-building configuration.

The model source or intermediate representation is therefore more portable than the generated `.engine` file.

See:

```text
models/README.md
```

for additional information.

---

## FP16 Deployment

The reference deployment uses FP16 engine naming:

```text
detr_resnet101_fp16.engine
moistnetlite_fp16.engine
```

FP16 inference can reduce memory usage and improve throughput on supported NVIDIA hardware.

Actual precision support and performance depend on the target device and TensorRT configuration.

---

## Repository Dependency Boundaries

The repository intentionally separates dependencies according to purpose.

### Application source

```text
app/
```

Contains the integrated monitoring application.

### Models

```text
models/
```

Contains model architecture documentation and local deployment expectations.

### Tools

```text
tools/
```

Contains standalone deployment and inference utilities.

### Research detector implementation

Maintained separately in:

```text
UOT-DETR
```

This prevents the deployment repository from duplicating the much larger detector research environment.

---

## Suggested Environment Verification

Before attempting deployment on a Jetson device, verify at minimum:

```bash
python --version
```

and confirm the operating system/L4T release.

TensorRT availability should also be checked from Python:

```bash
python -c "import tensorrt as trt; print(trt.__version__)"
```

PyCUDA can be checked with:

```bash
python -c "import pycuda.driver as cuda; print('PyCUDA available')"
```

OpenCV can be checked using:

```bash
python -c "import cv2; print(cv2.__version__)"
```

NumPy can be checked using:

```bash
python -c "import numpy as np; print(np.__version__)"
```

FastAPI availability can be checked using:

```bash
python -c "import fastapi; print(fastapi.__version__)"
```

These checks verify package availability but do not by themselves establish compatibility with the original prototype.

---

## Recommended Version Capture

When preparing a deployment environment for reproducibility, capture:

```bash
python --version
```

```bash
pip freeze
```

and relevant NVIDIA platform information.

On Jetson systems, L4T information should also be recorded.

This provides a traceable record of the exact environment used for a given deployment.

---

## Reproducibility Status

The current repository documents the known reference deployment environment and the software-component boundaries.

A complete reproducible environment definition will be finalized only after package versions and NVIDIA dependencies have been verified against the original deployment or an equivalent reproduced system.

Until that validation is complete, the repository intentionally avoids presenting unverified package versions as authoritative.

---

## Related Documentation

For overall system architecture, see:

```text
docs/architecture.md
```

For Jetson device setup, see:

```text
docs/jetson-setup.md
```

For model artifacts and TensorRT deployment, see:

```text
models/README.md
```

For configurable device paths and runtime settings, see:

```text
configs/device.env.example
```

For standalone detector inference, see:

```text
tools/infer_trt.py
```