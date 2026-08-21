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

Custom mechanical components were developed to integrate the camera, edge-computing hardware, display, and enclosure into the portable prototype.

The original development archive contains SolidWorks part and assembly files. These source CAD files are not currently distributed through this repository.

This repository focuses on the research software and system architecture while preserving visual documentation of the physical prototype.

## Open-Hardware Scope

The inclusion of prototype photographs or mechanical-design figures should not be interpreted as an open-source release of the complete mechanical CAD package.

If the mechanical design is released separately in the future, corresponding licensing and attribution information will be provided.