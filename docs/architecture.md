# System Architecture

Wood-Chip Monitor is a self-contained edge-AI inspection platform for continuous wood-chip quality evaluation.

The deployed system combines image acquisition, TensorRT-accelerated computer-vision inference, physical measurement, quality analytics, backend services, local data persistence, and a browser-based operator dashboard on NVIDIA Jetson hardware.

The architecture is designed to keep the primary inspection workflow local to the edge device rather than depending on cloud inference.

---

## Architecture Overview

The system is organized into four primary layers:

1. **Image acquisition and hardware**
2. **Edge-AI inference and quality analytics**
3. **Backend services and local persistence**
4. **Operator-facing web dashboard**

The end-to-end workflow is:

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
    ├───────────────────────────────┐
    │                               │
    ▼                               ▼
Chip Localization              Chip ROI Selection
    │                               │
    ▼                               ▼
Physical Size                  MoistNetLite
Estimation                     TensorRT Inference
    │                               │
    ▼                               ▼
Rolling Size                   Moisture
Statistics                     Assessment
    │                               │
    └──────────────┬────────────────┘
                   │
                   ▼
             Quality Metrics
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
   Histogram   Oversize     System
   Analysis     Rules       Health
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

---

## 1. Hardware and Image Acquisition

The first layer captures the visual information required for quality assessment.

The reference prototype integrates:

- an RGB camera,
- NVIDIA Jetson edge-computing hardware,
- a local display,
- and a custom portable enclosure.

The camera continuously observes the wood-chip scene and supplies frames to the inference pipeline.

The imaging geometry is important because object detections are used not only for localization but also for physical chip-size estimation.

---

## Camera Configuration

The reference application uses a Linux video device such as:

```text
/dev/video0
```

The default runtime configuration is:

```text
Width:  1480 pixels
Height: 900 pixels
FPS:    20
```

These values are configurable through environment variables:

```text
WOODCHIP_CAMERA_DEVICE
WOODCHIP_CAMERA_WIDTH
WOODCHIP_CAMERA_HEIGHT
WOODCHIP_CAMERA_FPS
```

The corresponding example configuration is provided in:

```text
configs/device.env.example
```

---

## 2. Edge-AI Inference Layer

The edge-inference layer is responsible for processing incoming camera frames and extracting the visual information required for downstream quality assessment.

The primary implementation is maintained in:

```text
app/live_cam_trt.py
```

The live pipeline performs:

1. frame acquisition,
2. image preprocessing,
3. TensorRT detector inference,
4. confidence filtering,
5. bounding-box processing,
6. geometric measurement,
7. rolling size-distribution analysis,
8. oversize-condition evaluation,
9. chip-region selection,
10. MoistNetLite inference,
11. result visualization,
12. runtime-state updates.

---

## Wood-Chip Detection

The detection component identifies individual wood chips within the camera frame.

The deployed system uses a TensorRT-optimized DETR-based detector.

The default deployment artifact is:

```text
models/detr_resnet101_fp16.engine
```

Detector outputs are used for several downstream functions:

```text
Detector
   │
   ├──► chip localization
   │
   ├──► physical size estimation
   │
   ├──► rolling size statistics
   │
   ├──► oversize detection
   │
   ├──► visualization
   │
   └──► moisture-model crop selection
```

The methodological research and full experimental framework associated with the detector are maintained separately in the UOT-DETR project.

---

## Physical Size Estimation

Object detections are converted from image-space bounding boxes into physical chip measurements using the system calibration process.

Conceptually, the measurement workflow is:

```text
Detected bounding box
        │
        ▼
Pixel-space dimensions
        │
        ▼
Scene calibration
        │
        ▼
Physical dimensions
        │
        ▼
Chip-size statistics
```

This enables the system to move beyond object counting and provide measurements directly relevant to wood-chip quality evaluation.

---

## Rolling Size Statistics

The system continuously aggregates measurements from detected chips rather than treating every frame independently.

These measurements support statistics such as:

- observed chip dimensions,
- size-distribution summaries,
- histogram generation,
- oversize percentages,
- rolling population statistics.

The resulting information is made available to the operator dashboard.

---

## Oversize Detection

Detected chip measurements are compared with configurable quality thresholds.

The system can therefore classify or highlight material that exceeds the selected size limits.

The general process is:

```text
Physical chip measurement
          │
          ▼
Configured quality threshold
          │
          ▼
Normal / Oversize decision
          │
          ▼
Visualization + quality metrics
```

Oversize detections can be visually distinguished in the live monitoring interface and incorporated into aggregated quality statistics.

---

## Moisture Assessment

Wood-Chip Monitor integrates a second neural model for image-based moisture assessment.

The model architecture is maintained in:

```text
models/moistnetlite.py
```

The deployed TensorRT artifact is expected at:

```text
models/moistnetlite_fp16.engine
```

The associated class labels are expected at:

```text
models/moistnetlite_classes.txt
```

---

## Top-K Moisture Sampling

Dense wood-chip scenes may contain many simultaneous detections.

Running the moisture model on every detected chip could increase inference cost unnecessarily.

The deployed pipeline therefore supports a configurable **Top-K chip-selection strategy**.

The general workflow is:

```text
Detector outputs
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

This design limits the number of moisture-model evaluations performed during an inference cycle while preserving selected high-confidence chip regions for assessment.

---

## 3. Quality Analytics Layer

The analytics layer converts model predictions into information useful for process monitoring.

The system combines:

- chip detections,
- physical measurements,
- size statistics,
- distribution information,
- oversize conditions,
- moisture predictions,
- and runtime health information.

The resulting state is used by both the visualization pipeline and the backend service.

---

## Live Runtime State

The inference component maintains the latest inspection results in memory.

This allows the backend to expose current system information without launching a separate inference process.

Runtime information can include:

- the latest annotated image,
- detection counts,
- chip measurements,
- histogram data,
- oversize statistics,
- moisture predictions,
- inference timing,
- and system-health information.

The architecture therefore separates:

```text
Inference generation
        │
        ▼
Shared runtime state
        │
        ▼
Backend access
        │
        ▼
Dashboard presentation
```

---

## 4. Backend Service Layer

The application backend is implemented using FastAPI.

The primary source file is:

```text
app/backend_app.py
```

The backend acts as the control and information layer between the live inference process and the browser-based user interface.

Its responsibilities include:

- serving current inspection results,
- exposing runtime configuration,
- serving or streaming annotated imagery,
- user authentication,
- role-based access control,
- event persistence,
- audit logging,
- quality-rule management,
- device-health reporting,
- and static dashboard delivery.

---

## Authentication and Role-Based Access

The application includes authentication and role-based access-control functionality.

This allows different system functions to be restricted according to user permissions.

The backend therefore supports more than inference alone; it also provides the governance functionality required for an operator-facing monitoring application.

---

## Event Management

Operational events can be persisted for later review.

Examples may include:

- quality-rule events,
- oversize conditions,
- monitoring events,
- and system-related events.

This provides a historical record rather than limiting the system to a live-only view.

---

## Audit Logging

The backend includes audit functionality for recording relevant system and user actions.

Audit records can support traceability of configuration and operational changes.

This functionality is exposed through the web dashboard.

---

## Quality-Rule Management

Quality-control parameters can be managed through the application rather than being permanently hard-coded into the inference source.

The backend supports quality-rule configuration and versioning so that monitoring behavior can be adjusted while preserving traceability.

---

## Device Health

The system exposes device and runtime status information through the backend.

This allows the dashboard to report whether major application components are operating normally.

Health information can include inference state and other device-level status information available to the application.

---

## Local Persistence

Persistent application information is stored locally using SQLite.

The default database location is conceptually:

```text
data/woodchip_app.sqlite3
```

The actual location can be overridden through:

```text
WOODCHIP_DB_PATH
```

The `data/` directory and local database files are excluded from Git version control.

This prevents runtime state and potentially sensitive operational records from being committed to the repository.

---

## 5. Browser Dashboard

The operator-facing interface is implemented as a browser-based application.

The primary frontend source is:

```text
app/web/index.html
```

The dashboard communicates with the local FastAPI backend.

No specialized desktop software is required for normal interaction with the monitoring interface.

---

## Dashboard Functions

The application interface provides access to system functions such as:

- live monitoring,
- annotated camera imagery,
- chip-size information,
- size-distribution visualization,
- oversize status,
- moisture assessment,
- system health,
- historical events,
- quality-rule configuration,
- audit records,
- device information.

The dashboard screenshot used in the main repository documentation is stored at:

```text
assets/dashboard.png
```

---

## Local-First Operation

The primary monitoring workflow is designed for edge execution.

The major functions remain local to the Jetson device:

```text
Camera
   │
   ▼
Edge inference
   │
   ▼
Quality analytics
   │
   ▼
FastAPI service
   │
   ▼
Local browser dashboard
```

This architecture reduces reliance on external network connectivity and avoids requiring cloud inference for the core quality-monitoring process.

---

## Model and Application Separation

The repository intentionally separates the underlying research models from the deployed monitoring application.

### UOT-DETR

The UOT-DETR repository focuses on:

- detection methodology,
- model training,
- experimental benchmarking,
- distribution-aware learning,
- ablation studies,
- sensitivity analysis,
- scientific reproducibility.

### Wood-Chip Monitor

This repository focuses on:

- TensorRT edge inference,
- camera integration,
- physical measurement,
- size analytics,
- moisture-model integration,
- quality rules,
- backend services,
- operator interaction,
- and deployment.

This separation prevents duplication while preserving a clear connection between methodological research and deployed technology.

---

## Repository Mapping

The principal runtime components are organized as:

```text
Wood-Chip-Monitor/
│
├── app/
│   ├── backend_app.py
│   ├── live_cam_trt.py
│   ├── __init__.py
│   │
│   └── web/
│       └── index.html
│
├── models/
│   ├── moistnetlite.py
│   └── README.md
│
├── tools/
│   ├── infer_trt.py
│   └── README.md
│
├── configs/
│   └── device.env.example
│
├── docs/
│   ├── architecture.md
│   ├── jetson-setup.md
│   └── software-environment.md
│
│   ├── images/
│   └── predictions/
│
└── assets/
```

---

## Component Responsibilities

### `app/live_cam_trt.py`

Responsible for:

- camera acquisition,
- detector TensorRT execution,
- moisture TensorRT execution,
- prediction processing,
- physical sizing,
- quality analytics,
- oversize evaluation,
- live visualization,
- shared runtime state.

### `app/backend_app.py`

Responsible for:

- FastAPI services,
- authentication,
- role-based access,
- event handling,
- audit logging,
- quality-rule management,
- runtime configuration,
- device status,
- dashboard delivery.

### `app/web/index.html`

Responsible for:

- browser-based operator interaction,
- live inspection display,
- quality information,
- events,
- quality rules,
- audit records,
- device information.

### `models/moistnetlite.py`

Provides the source architecture definition for the lightweight moisture-classification model.

### `tools/infer_trt.py`

Provides standalone TensorRT detector inference for image-based testing outside the full live application.

### `configs/device.env.example`

Documents configurable deployment paths and device-level runtime settings.

---

## Deployment Artifacts

The following files are expected during deployment but are intentionally not committed:

```text
models/
├── detr_resnet101_fp16.engine
├── moistnetlite_fp16.engine
└── moistnetlite_classes.txt
```

TensorRT engines are treated as device-specific deployment artifacts.

See:

```text
models/README.md
```

for details.

---

## Runtime Artifacts

The application may generate local files such as:

```text
data/
outputs/
```

These directories are excluded from version control.

They may contain:

- SQLite application state,
- generated inference outputs,
- temporary runtime results,
- or other deployment-specific data.

---

## Design Principles

The public repository follows several architectural principles.

### Local inference

The core inspection process should remain available without cloud inference.

### Separation of concerns

Research training code, edge deployment, backend services, and frontend interaction are maintained as distinct responsibilities.

### Configurable deployment

Machine-specific paths should be supplied through configuration or environment variables rather than embedded directly in source code.

### Traceability

Events, audit records, and quality-rule management support an operator-facing system rather than a model-only demonstration.

### Reproducibility

Research methodology and deployment artifacts are documented separately so that device-specific binaries are not presented as universally portable software.

---

## Further Documentation

For target-device configuration, see:

```text
docs/jetson-setup.md
```

For software dependencies and environment considerations, see:

```text
docs/software-environment.md
```

For model artifacts and deployment expectations, see:

```text
models/README.md
```

For example deployment configuration, see:

```text
configs/device.env.example
```