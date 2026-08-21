# Wood-Chip Monitor

**An Edge-AI System for Real-Time Wood-Chip Quality Evaluation**

Wood-Chip Monitor is a portable edge-AI platform for real-time wood-chip quality assessment on NVIDIA Jetson hardware. The system integrates computer vision, physical size estimation, size-distribution monitoring, oversize detection, vision-based moisture assessment, and an operator-facing web dashboard in a local edge deployment.

> **Repository status:** Public research-software release in preparation. The repository currently preserves the core deployed application, model architecture, deployment utilities, examples, and system documentation while reproducibility materials are being finalized.

<p align="center">
  <img src="assets/dashboard.png" alt="Wood-Chip Monitor dashboard" width="100%">
</p>

---

## Overview

Conventional wood-chip quality assessment often relies on periodic manual sampling and offline measurements. Wood-Chip Monitor was developed to move visual quality assessment closer to the production environment by combining computer vision, edge computing, physical measurement, and operator-facing decision support in a portable monitoring platform.

The system is designed to provide:

- real-time wood-chip detection,
- physical chip-size estimation,
- rolling size-distribution analysis,
- oversize-chip identification,
- configurable quality thresholds,
- image-based moisture assessment,
- TensorRT-accelerated edge inference,
- local event and audit records,
- device-status monitoring,
- and a browser-based operator dashboard.

The primary inspection workflow is designed to operate locally on the edge device without requiring cloud inference.

---

## Key Capabilities

### Real-Time Wood-Chip Detection

A TensorRT-optimized DETR-based detector identifies individual wood chips in the camera stream.

Detector outputs support:

- dense chip localization,
- downstream geometric measurement,
- live visualization,
- oversize assessment,
- and chip-region extraction for moisture inference.

### Physical Size Estimation

Detected bounding boxes are converted from image-space measurements to physical chip dimensions using scene calibration.

These measurements are aggregated to characterize the observed chip population rather than treating each detection independently.

### Size-Distribution Monitoring

The system continuously maintains chip-size statistics and distribution information that can be visualized through the monitoring interface.

### Oversize Detection

Physical chip measurements are evaluated against configurable quality thresholds.

Oversize material can be highlighted visually and summarized through quality-control metrics.

### Moisture Assessment

Wood-Chip Monitor integrates **MoistNetLite**, a lightweight vision model for image-based moisture classification.

Selected high-confidence chip regions are processed using a configurable Top-K strategy to limit moisture-inference cost in dense scenes.

## Quick Start

Wood-Chip Monitor is intended to run on the configured NVIDIA Jetson deployment platform with the required TensorRT engines available locally.

```bash
# Clone and enter the repository
git clone https://github.com/amirhossein-eskorouchi/Wood-Chip-Monitor.git
cd Wood-Chip-Monitor

# Create a local device configuration
cp configs/device.env.example configs/device.env

# Edit device-specific settings and credentials
nano configs/device.env

# Load configuration
set -a
source configs/device.env
set +a

# Start the monitoring application
python -m app.backend_app
```

Then open:

```text
http://localhost:8000
```

in a browser on the Jetson, or use `http://<JETSON-IP>:8000` from another device on the same network.

> **Note:** TensorRT engines and trained model artifacts are not distributed in this repository. See [`models/README.md`](models/README.md) and [`docs/jetson-setup.md`](docs/jetson-setup.md) for model and deployment requirements.

### Edge Deployment

The reference system executes the primary AI pipeline locally using NVIDIA Jetson hardware and TensorRT-optimized models.

### Operator Dashboard

A local browser interface provides access to:

- live inspection,
- annotated camera imagery,
- size-distribution information,
- oversize status,
- moisture assessment,
- historical events,
- quality-rule configuration,
- audit information,
- and device status.

---

## End-to-End Workflow

```text
RGB Camera
    │
    ▼
Frame Acquisition
    │
    ▼
Image Preprocessing
    │
    ▼
DETR TensorRT Inference
    │
    ├──────────────────────────────┐
    │                              │
    ▼                              ▼
Chip Localization             Chip ROI Selection
    │                              │
    ▼                              ▼
Physical Size                 MoistNetLite
Estimation                    TensorRT Inference
    │                              │
    ▼                              ▼
Rolling Size                  Moisture
Statistics                    Assessment
    │                              │
    └──────────────┬───────────────┘
                   │
                   ▼
             Quality Metrics
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
   Distribution  Oversize    System
    Analysis      Rules      Health
        │          │          │
        └──────────┼──────────┘
                   │
                   ▼
             FastAPI Backend
                   │
        ┌──────────┼────────────┐
        │          │            │
        ▼          ▼            ▼
      Events     Audit        Device
                 Logs         Status
        │          │            │
        └──────────┼────────────┘
                   │
                   ▼
            Browser Dashboard
```

A detailed description of the software and data-flow architecture is available in:

**[`docs/architecture.md`](docs/architecture.md)**

---

## Hardware Prototype

<p align="center">
  <img src="assets/prototype.png" alt="Wood-Chip Monitor hardware prototype" width="70%">
</p>

The reference prototype integrates the edge-computing platform, camera, local display, and custom enclosure required for portable wood-chip monitoring.

The original deployed platform used an NVIDIA Jetson Nano.

Reference environment information is documented in:

**[`docs/jetson-setup.md`](docs/jetson-setup.md)**

Additional information about the integrated edge-computing platform and mechanical prototype is available in:

**[`hardware/README.md`](hardware/README.md)**

---

## AI Components

Wood-Chip Monitor integrates two primary computer-vision components.

### Wood-Chip Detector

The detector provides the object localizations used for:

- physical chip measurement,
- rolling size statistics,
- oversize analysis,
- visualization,
- and moisture-model crop selection.

The default deployment artifact is expected to be:

```text
models/detr_resnet101_fp16.engine
```

The complete research implementation and experimental framework associated with the detector are maintained separately in the **UOT-DETR** repository:

https://github.com/amirhossein-eskorouchi/UOT-DETR

### MoistNetLite

MoistNetLite provides image-based moisture assessment for selected detected chip regions.

The source architecture is included in:

```text
models/moistnetlite.py
```

The reference deployment expects:

```text
models/moistnetlite_fp16.engine
models/moistnetlite_classes.txt
```

Detailed model information is available in:

**[`models/README.md`](models/README.md)**

---

## Moisture-Inference Strategy

Dense wood-chip scenes may contain many simultaneous detections.

Rather than running the moisture model on every detected object, the live pipeline supports a configurable Top-K selection strategy:

```text
DETR detections
       │
       ▼
Confidence filtering
       │
       ▼
Candidate chip regions
       │
       ▼
Top-K selection
       │
       ▼
Crop extraction
       │
       ▼
MoistNetLite inference
       │
       ▼
Moisture assessment
```

This design bounds the number of moisture-model evaluations performed during an inference cycle.

---

## Dashboard

The browser-based dashboard communicates with the local FastAPI backend and provides an operator-facing view of the monitoring process.

The interface supports functions including:

- live annotated inspection,
- chip-size visualization,
- size-distribution summaries,
- oversize monitoring,
- moisture information,
- system status,
- quality-rule configuration,
- event history,
- audit records,
- and device information.

The frontend implementation is maintained in:

```text
app/web/index.html
```

The backend service is maintained in:

```text
app/backend_app.py
```

## User Documentation

Operator-facing functionality, role-based access, monitoring controls, events, quality rules, audit logs, and device-health features are documented in:

**[`docs/user-guide.md`](docs/user-guide.md)**

---

## Example Detection Results

Example input images and corresponding detector outputs are included in [`examples/`](examples/).

| Input | Prediction |
|---|---|
| ![](examples/images/14_29.jpg) | ![](examples/predictions/pred_14_29.jpg) |
| ![](examples/images/1_23.jpg) | ![](examples/predictions/pred_1_23.jpg) |
| ![](examples/images/25_51.jpg) | ![](examples/predictions/pred_25_51.jpg) |
| ![](examples/images/2_2.jpg) | ![](examples/predictions/pred_2_2.jpg) |

These examples illustrate the dense wood-chip localization component used by the downstream monitoring pipeline.

---

## Quality Analytics

The deployed system converts object detections into population-level quality information.

### Example Size Distribution

<p align="center">
  <img src="assets/size_distribution.png" alt="Wood-chip size distribution" width="75%">
</p>

### Example Size Statistics

<p align="center">
  <img src="assets/size_boxplot.png" alt="Wood-chip size statistics" width="75%">
</p>

These visualizations illustrate the type of statistical information generated from detected and physically measured chips.

---

## Software Architecture

The application is organized around three primary runtime components:

```text
app/
├── live_cam_trt.py
├── backend_app.py
└── web/
    └── index.html
```

### `app/live_cam_trt.py`

Responsible for:

- camera acquisition,
- image preprocessing,
- TensorRT detector execution,
- prediction filtering,
- physical size estimation,
- rolling statistics,
- oversize evaluation,
- MoistNetLite TensorRT execution,
- visualization,
- and shared runtime state.

### `app/backend_app.py`

Responsible for:

- FastAPI services,
- authentication,
- role-based access,
- event persistence,
- audit logging,
- quality-rule management,
- runtime configuration,
- device-health information,
- and dashboard delivery.

### `app/web/index.html`

Responsible for the browser-based operator interface.

Detailed architecture documentation is available in:

**[`docs/architecture.md`](docs/architecture.md)**

---

## Edge Deployment

The reference prototype was deployed on an NVIDIA Jetson Nano using TensorRT-accelerated inference.

The documented deployment environment included:

```text
Hardware:          NVIDIA Jetson Nano
GPU:               NVIDIA Tegra X1
Operating system:  Ubuntu 18.04.5 LTS
Python:            3.6.9
NVIDIA L4T:        R32.6.1
System memory:     approximately 4 GB
```

This environment represents the original validated prototype and should not be interpreted as a recommendation to use an outdated software stack for new deployments.

Deployment documentation is available in:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/jetson-setup.md`](docs/jetson-setup.md)
- [`docs/software-environment.md`](docs/software-environment.md)
- [`models/README.md`](models/README.md)
- [`configs/device.env.example`](configs/device.env.example)

---

## Configuration

Machine-specific deployment paths have been moved out of the source code where practical and can be supplied through environment variables.

An example configuration is provided in:

```text
configs/device.env.example
```

Supported configuration includes:

### Camera

```text
WOODCHIP_CAMERA_DEVICE
WOODCHIP_CAMERA_WIDTH
WOODCHIP_CAMERA_HEIGHT
WOODCHIP_CAMERA_FPS
```

### Model Artifacts

```text
WOODCHIP_MODEL_DIR
WOODCHIP_DETR_ENGINE
WOODCHIP_MOISTURE_ENGINE
WOODCHIP_MOISTURE_CLASSES
```

### Runtime Storage

```text
WOODCHIP_DB_PATH
WOODCHIP_OUTPUT_DIR
```

### Device Identity

```text
WOODCHIP_DEVICE_ID
```

Deployment-specific credentials should never be committed to the repository.

---

## Model Artifacts

TensorRT engines and trained model binaries are intentionally excluded from version control.

The default local model structure is:

```text
models/
├── detr_resnet101_fp16.engine
├── moistnetlite_fp16.engine
├── moistnetlite_classes.txt
├── moistnetlite.py
└── README.md
```

The repository `.gitignore` excludes model artifacts such as:

```text
*.engine
*.onnx
*.pt
*.pth
*.ckpt
*.trt
*.weights
```

TensorRT engines are treated as device-specific deployment artifacts rather than canonical portable research files.

See:

**[`models/README.md`](models/README.md)**

for additional details.

---

## Standalone TensorRT Inference

A standalone inference utility is provided in:

```text
tools/infer_trt.py
```

It can be used to run the deployed detector on a directory of images independently of the full live monitoring application.

Example usage:

```bash
python tools/infer_trt.py \
    --engine models/detr_resnet101_fp16.engine \
    --images examples/images \
    --output outputs/predictions
```

Additional information is available in:

**[`tools/README.md`](tools/README.md)**

---

## Software Environment

The original Jetson deployment uses a legacy and tightly coupled NVIDIA software stack.

For this reason, the repository does not currently claim that a generic:

```bash
pip install -r requirements.txt
```

command reproduces the validated device environment.

TensorRT, CUDA, PyCUDA, OpenCV, JetPack/L4T, and related components must be handled according to the target NVIDIA platform.

The current software-environment policy and dependency boundaries are documented in:

**[`docs/software-environment.md`](docs/software-environment.md)**

Exact dependency versions will only be published when they have been verified against the original deployment environment or successfully reproduced on an equivalent system.

---

## Validation

The deployment pipeline was developed through staged validation from reference workstation processing to manual preprocessing/post-processing reconstruction, Jetson TensorRT inference, and integrated size analytics.

A compact record of these development stages is preserved in:

**[`validation/README.md`](validation/README.md)**

Representative historical outputs are included for transparency and provenance without committing the large intermediate tensors and duplicated development artifacts from the original archive.
---

## Repository Structure

Wood-Chip-Monitor/
├── app/
│   ├── backend_app.py
│   ├── live_cam_trt.py
│   └── web/
│       └── index.html
├── assets/
├── configs/
│   └── device.env.example
├── docs/
│   ├── architecture.md
│   ├── jetson-setup.md
│   ├── software-environment.md
│   └── user-guide.md
├── examples/
│   ├── images/
│   └── predictions/
├── models/
│   ├── moistnetlite.py
│   └── README.md
├── tools/
│   ├── infer_trt.py
│   └── README.md
├── validation/
│   ├── jetson_tensorrt.csv
│   ├── manual_processing.csv
│   ├── reference_processing.csv
│   ├── size_statistics.csv
│   └── README.md
├── .gitignore
└── README.md



---

## Documentation

Current technical documentation includes:

| Document | Purpose |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | End-to-end system and software architecture |
| [`docs/jetson-setup.md`](docs/jetson-setup.md) | Reference Jetson deployment environment |
| [`docs/software-environment.md`](docs/software-environment.md) | Dependency boundaries and reproducibility guidance |
| [`models/README.md`](models/README.md) | AI-model components and deployment artifacts |
| [`tools/README.md`](tools/README.md) | Standalone deployment utilities |
| [`configs/device.env.example`](configs/device.env.example) | Example device-level configuration |

---

## Related Research

Wood-Chip Monitor is part of a broader research effort on vision-based wood-chip quality assessment.

| Resource | Role in the Research Program |
|---|---|
| **UOT-DETR** | Distribution-aware dense object detection and wood-chip size-distribution estimation |
| **MoistNet / MoistNetLite** | Vision-based wood-chip moisture assessment |
| **WoodChip-Detection** | Public dataset for dense wood-chip detection and instance analysis |
| **Edge-AI System Paper** | Integrated deployment of geometry and moisture assessment on embedded hardware |

Detailed publication and resource information is available in:

**[`docs/related-research.md`](docs/related-research.md)**

### UOT-DETR

The UOT-DETR project focuses on the underlying object-detection and distribution-aware visual measurement methodology.

It contains the research implementation, benchmarks, experimental comparisons, ablation studies, and reproducibility resources associated with the detection framework.

**Repository:**  
https://github.com/amirhossein-eskorouchi/UOT-DETR

### MoistNet / MoistNetLite

MoistNetLite provides the lightweight moisture-assessment component integrated into the edge system.

The public Wood-Chip Monitor repository includes the lightweight model architecture and deployment integration.

Additional publication and research-resource links will be added as the repository release is finalized.

### Wood-Chip Detection Dataset

The underlying research effort also includes annotated wood-chip imagery used for model development and evaluation.

The corresponding dataset and publication links will be added to the final research-resource section.

---

## Research Code vs. Deployment Code

The project ecosystem intentionally separates methodological research from deployed application software.

### UOT-DETR

Focuses on:

- model development,
- distribution-aware learning,
- detector training,
- experimental benchmarking,
- ablation studies,
- sensitivity analysis,
- scientific reproducibility.

### Wood-Chip Monitor

Focuses on:

- camera integration,
- TensorRT inference,
- physical chip measurement,
- size-distribution monitoring,
- oversize detection,
- moisture-model integration,
- backend services,
- operator interaction,
- system governance,
- and Jetson deployment.

This separation keeps both repositories focused while preserving the connection between methodological research and real-world technology implementation.

---

## Data and Runtime Privacy

Local application databases, runtime output, credentials, and machine-specific configuration should not be committed to Git.

The repository excludes items such as:

```text
data/
outputs/
*.db
*.sqlite
*.sqlite3
.env
```

Users deploying the application should store credentials and local operational information outside version control.

---

## Project Status

The repository currently includes:

- [x] core Jetson inference pipeline,
- [x] FastAPI backend,
- [x] browser dashboard,
- [x] configurable model and runtime paths,
- [x] MoistNetLite architecture,
- [x] standalone TensorRT detector inference,
- [x] example images and predictions,
- [x] system architecture documentation,
- [x] Jetson environment documentation,
- [x] software-environment guidance,
- [ ] concise validation documentation,
- [ ] operator/user documentation,
- [ ] finalized model-export workflow,
- [ ] verified dependency-version manifest,
- [ ] publication and dataset links,
- [ ] citation metadata,
- [ ] final licensing review.

The repository will remain under preparation until the public release materials are validated.

---


## Acknowledgments

Wood-Chip Monitor was developed through contributions spanning computer vision, edge-AI deployment, quality analytics, and physical system integration.

Contributor roles and project acknowledgments are documented in:

**[`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md)**


## Citation

Citation metadata will be added before the public release.

The final repository will include citation information for the Wood-Chip Monitor system and links to the associated scholarly resources.

---

## License

A source-code license will be added before the public release.

Research publications, datasets, trained model artifacts, and externally developed components remain subject to their respective licenses and usage terms.

---

## Acknowledgments

Wood-Chip Monitor was developed as part of research on computer vision, edge AI, and decision support for wood-chip quality assessment.

Contributor, collaborator, institutional, and funding acknowledgments will be finalized with the public release.