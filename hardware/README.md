# Hardware Prototype

Wood-Chip Monitor was implemented as a portable edge-AI inspection platform integrating the sensing, computation, display, and enclosure components required for on-site wood-chip quality monitoring.

<p align="center">
  <img src="../assets/prototype.png" alt="Wood-Chip Monitor prototype" width="70%">
</p>

## System Components

The prototype integrates:

- NVIDIA Jetson Nano edge-computing hardware,
- RGB image acquisition,
- local display and operator interface,
- custom camera mounting,
- portable enclosure components,
- local TensorRT inference,
- and browser-based monitoring software.

## Design Objective

The hardware platform was designed to support deployment close to the wood-chip production environment rather than requiring images to be transferred to a separate workstation or cloud service.

The integrated form factor supports:

1. stable image acquisition,
2. local AI processing,
3. operator visualization,
4. device-level quality monitoring.

## Edge Computing Platform

The reference prototype used an NVIDIA Jetson Nano with:

- NVIDIA Tegra X1,
- approximately 4 GB memory,
- Ubuntu 18.04.5 LTS,
- NVIDIA L4T R32.6.1,
- Python 3.6.9.

Detailed software information is available in:

- [`../docs/jetson-setup.md`](../docs/jetson-setup.md)
- [`../docs/software-environment.md`](../docs/software-environment.md)

## Mechanical Design

Custom mechanical components integrate the camera, Jetson Nano, touchscreen,
mounting hardware, and enclosure.

The public repository preserves native SolidWorks files under
[`cad/`](cad/) as part of the versioned engineering record.

## Hardware-release scope

The CAD package is publicly available with version 0.1.0 for research,
engineering reference, reproducibility, and continued development.

Its inclusion does not constitute certification by an open-hardware standards
organization and does not override terms applicable to third-party hardware
or software components.

See [`../NOTICE`](../NOTICE) and [`cad/README.md`](cad/README.md).

## Attribution

Prashant Bhattarai contributed to the mechanical design and physical prototype
development associated with the AI2F Summer 2025 program.

See [`../ACKNOWLEDGMENTS.md`](../ACKNOWLEDGMENTS.md).
