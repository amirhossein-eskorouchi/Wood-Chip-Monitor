# Wood-Chip Monitor

**An Edge-AI System for Real-Time Wood Chip Quality Evaluation**

Wood-Chip Monitor is a portable edge-AI system for real-time wood-chip quality assessment on NVIDIA Jetson hardware. The system integrates visual chip detection, physical size estimation, size-distribution monitoring, oversize alerts, and vision-based moisture assessment in a fully local deployment.

> **Status:** This repository is being prepared as the public research-software release of the Wood-Chip Monitor system.

<p align="center">
  <img src="assets/dashboard.png" alt="Wood-Chip Monitor dashboard" width="100%">
</p>

## Overview

Conventional wood-chip quality assessment often relies on periodic manual sampling and offline measurements. Wood-Chip Monitor was developed to move quality assessment closer to the production environment by combining computer vision, edge computing, and operator-facing decision support in a portable system.

The platform provides:

- real-time wood-chip detection,
- physical chip-size estimation,
- live size-distribution monitoring,
- oversize-chip identification and alerts,
- vision-based moisture assessment,
- local edge inference without cloud dependence,
- configurable quality-control parameters,
- and a browser-based monitoring dashboard.

## System Capabilities

### Visual Size Assessment

Detected wood chips are localized in the camera stream and converted from image-space measurements to physical dimensions using scene calibration. The resulting measurements are used to continuously characterize the observed chip-size distribution.

### Oversize Monitoring

The system evaluates detected chips against configurable quality thresholds and visually identifies oversize material during operation.

### Moisture Assessment

A lightweight vision model is integrated into the edge pipeline to provide image-based moisture assessment alongside chip-size measurements.

### Local Decision Support

Inference and monitoring are performed locally on NVIDIA Jetson hardware. The accompanying dashboard provides live visualization, system status, quality information, events, and configurable monitoring rules.

## Hardware Prototype

<p align="center">
  <img src="assets/prototype.png" alt="Wood-Chip Monitor hardware prototype" width="70%">
</p>

The portable prototype integrates the edge-computing platform, imaging hardware, display, and custom enclosure required for on-site wood-chip quality monitoring.

## Example Detection Results

Example inputs and corresponding detector outputs are included in [`examples/`](examples/).

| Input | Prediction |
|---|---|
| ![](examples/images/14_29.jpg) | ![](examples/predictions/pred_14_29.jpg) |
| ![](examples/images/1_23.jpg) | ![](examples/predictions/pred_1_23.jpg) |
| ![](examples/images/25_51.jpg) | ![](examples/predictions/pred_25_51.jpg) |
| ![](examples/images/2_2.jpg) | ![](examples/predictions/pred_2_2.jpg) |

## Quality Analytics

The deployed inference pipeline produces continuous statistics describing the observed chip population.

<p align="center">
  <img src="assets/size_distribution.png" alt="Wood-chip size distribution" width="70%">
</p>

<p align="center">
  <img src="assets/size_boxplot.png" alt="Wood-chip size statistics" width="70%">
</p>

## System Architecture

The system integrates four main components:

1. **Image acquisition** from the monitoring camera.
2. **Edge-AI inference** for chip detection and moisture assessment.
3. **Quality analytics** for physical sizing, distribution monitoring, and oversize detection.
4. **Operator dashboard** for visualization, configuration, events, and system status.

A detailed architecture diagram and software design description will be added in the documentation.

## Software

The public software release will include the deployed edge-inference pipeline, monitoring backend, browser dashboard, model-conversion utilities, and deployment documentation.

## Edge Deployment

The original prototype was deployed on NVIDIA Jetson hardware using TensorRT-accelerated inference. Reproducible device setup and deployment instructions will be provided in [`docs/`](docs/).

## Validation

Development included staged validation from reference workstation inference through manual preprocessing/postprocessing verification and Jetson TensorRT deployment. A concise reproducibility and validation record will be included in this repository without the large intermediate development artifacts.

## Related Research

Wood-Chip Monitor builds on a broader research effort in vision-based wood-chip quality assessment.

- **UOT-DETR** — distribution-aware object detection and wood-chip size-distribution estimation
- **MoistNet / MoistNetLite** — vision-based wood-chip moisture assessment
- **Wood-chip detection dataset** — annotated imagery supporting model development and evaluation

Publication, dataset, code, and project links will be added before public release.

## Repository Structure

```text
Wood-Chip-Monitor/
├── app/
│   ├── backend_app.py       # FastAPI monitoring backend
│   ├── live_cam_trt.py      # Jetson/TensorRT inference pipeline
│   └── web/
│       └── index.html       # Operator dashboard
├── assets/                  # README figures and system visuals
├── configs/
│   └── device.env.example   # Example deployment configuration
├── docs/                    # Architecture, deployment, and user documentation
├── examples/
│   ├── images/              # Example wood-chip images
│   └── predictions/         # Example detection outputs
├── models/
│   └── README.md            # Model artifact requirements
├── .gitignore
└── README.md