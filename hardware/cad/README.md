# Mechanical CAD Source

This directory contains the native SolidWorks source files used to design and prototype the mechanical enclosure and component integration for the Wood-Chip Monitor portable edge-AI system.

## Overview

The Wood-Chip Monitor hardware was developed as a compact, portable inspection platform integrating the edge-computing device, camera, touchscreen, mounting hardware, and power components required for on-site wood-chip quality monitoring.

The CAD package in this directory preserves the original mechanical design files used during prototype development.

## Included CAD Files

The package contains the following SolidWorks source files:

```text
hardware/cad/
├── 7_inch_LCD_CASE.SLDPRT
├── CAMERA_HOLDER.SLDPRT
├── Camera_Lock_System.SLDPRT
├── Handle.SLDPRT
├── Hook.SLDPRT
├── Intergrated_camera.SLDPRT
├── Jetson_nano_case.SLDPRT
├── NEW ASSEMBLY.SLDASM
├── port_lid.SLDPRT
└── README.md
```

## Primary Assembly

The main assembly file is:

```text
NEW ASSEMBLY.SLDASM
```

The corresponding `.SLDPRT` files should remain in the same directory, or in paths that SolidWorks can resolve, for the assembly to load correctly.

Because SolidWorks assemblies maintain references to individual component files, renaming or relocating the part files may result in unresolved assembly references.

## Mechanical Components

The CAD design includes components for:

- NVIDIA Jetson Nano enclosure,
- 7-inch touchscreen enclosure,
- RGB camera mounting,
- camera locking and positioning,
- portable handle,
- mounting hook,
- enclosure access and port management,
- and complete prototype assembly.

Together, these components form the mechanical structure of the portable Wood-Chip Monitor prototype.

## Reference Hardware

The prototype was designed around the following major hardware components:

- NVIDIA Jetson Nano edge-computing platform,
- 8-megapixel USB RGB camera,
- 7-inch touchscreen display,
- NP-F970 rechargeable battery,
- battery adapter,
- voltage-regulation / buck-converter components,
- and custom 3D-printed enclosure components.

The hardware platform supports local sensing, edge-AI inference, visualization, and operator interaction without requiring a separate workstation during normal operation.

Additional system and deployment information is available in:

- [`../README.md`](../README.md)
- [`../../docs/architecture.md`](../../docs/architecture.md)
- [`../../docs/jetson-setup.md`](../../docs/jetson-setup.md)
- [`../../docs/software-environment.md`](../../docs/software-environment.md)

## Design Objectives

The mechanical system was designed to support several practical requirements of the Wood-Chip Monitor prototype:

1. stable and repeatable camera positioning,
2. physical protection of the embedded computing hardware,
3. integration of the touchscreen interface,
4. portable system operation,
5. convenient access to power and device ports,
6. support for camera positioning and locking,
7. compact integration of sensing and computation,
8. and suitability for laboratory, field, and industrial evaluation.

Stable camera geometry is particularly important because the monitoring system uses visual detections to estimate wood-chip dimensions and can perform pixel-to-millimeter calibration using a reference object in the scene.

## Design and Fabrication

The mechanical prototype was developed using CAD and rapid-prototyping workflows.

Project documentation indicates the use of:

- SolidWorks,
- Fusion 360,
- additive manufacturing / 3D printing,
- PLA-based prototype fabrication,
- Prusa XL 3D printing,
- and Bambu Lab X1 Carbon 3D printing.

The use of modular CAD components allowed individual parts of the prototype to be designed, fabricated, tested, and revised independently before integration into the complete assembly.

## Relationship to the Edge-AI System

The mechanical design provides the physical platform supporting the software and AI components maintained elsewhere in this repository.

The complete system can be summarized as:

```text
Mechanical Prototype
        |
        v
RGB Camera + Jetson Hardware
        |
        v
Frame Acquisition
        |
        v
TensorRT Edge-AI Inference
        |
        +----------------------+
        |                      |
        v                      v
DETR-Based Detection      MoistNetLite
        |                 Moisture Inference
        v                      |
Chip Geometry                  |
and Size Analysis              |
        |                      |
        +----------+-----------+
                   |
                   v
             Quality Metrics
                   |
                   v
             FastAPI Backend
                   |
                   v
          Browser-Based Dashboard
```

The CAD files therefore represent the hardware layer of the broader Wood-Chip Monitor architecture.

## File Formats

The CAD source is stored in native SolidWorks formats:

```text
.SLDPRT    SolidWorks part file
.SLDASM    SolidWorks assembly file
```

A compatible version of SolidWorks is recommended for viewing, editing, and reconstructing the complete assembly.

These are binary files and therefore cannot be reviewed through line-by-line Git diffs in the same way as source-code files.

## Working with the Assembly

When opening the complete assembly:

1. keep all `.SLDPRT` files together with `NEW ASSEMBLY.SLDASM`,
2. avoid renaming component files unless SolidWorks references are updated,
3. allow SolidWorks to resolve referenced parts when prompted,
4. verify component mates and alignment after moving the CAD directory,
5. save a new revision rather than overwriting an important validated design when making major changes.

For substantial mechanical revisions, consider preserving the previous design as a tagged repository version or clearly named revision.

## Version-Control Notes

Native SolidWorks files are binary artifacts. Git can store and version them, but it cannot provide meaningful line-level differences between revisions.

When modifying the CAD files:

- avoid unnecessary resaves that create large binary changes,
- commit related mechanical changes together,
- use descriptive commit messages,
- avoid renaming files referenced by the main assembly unless necessary,
- verify that the assembly still resolves all part references before committing,
- and document major mechanical revisions in the commit history.

Example commit messages include:

```text
Update camera-holder geometry
Revise Jetson enclosure ventilation
Adjust touchscreen housing dimensions
Update complete prototype assembly
```

## Repository Scope

These CAD files are stored as part of the private Wood-Chip Monitor research and development repository.

They are included to preserve the complete engineering record of the prototype alongside the software, deployment, and validation components of the project.

The presence of the CAD source in this private repository does not constitute a public open-hardware release.

## Intellectual Property Status

The Wood-Chip Monitor project includes research software, AI models, mechanical design, deployment methods, and system-integration work that may be subject to collaborator, institutional, publication, or intellectual-property considerations.

Accordingly:

- these CAD files should remain within the private repository at this stage,
- they should not be publicly redistributed without appropriate review,
- they should not be interpreted as being released under an open-hardware license,
- and public release should be considered separately from the source-code release strategy.

If the mechanical design is released publicly in the future, its licensing, attribution, and intellectual-property status should be defined explicitly.

## Attribution

The mechanical design and physical prototype development were carried out as part of the Wood-Chip Monitor hardware-development effort associated with the AI2F Summer 2025 program.

Prashant Bhattarai contributed to the mechanical design and physical prototype development of the system.

The broader Wood-Chip Monitor project integrates contributions spanning:

- computer vision,
- wood-chip detection and geometric measurement,
- moisture assessment,
- edge-AI deployment,
- backend and web-application development,
- quality analytics,
- and physical prototype design.

Repository-level contributor information and acknowledgments are maintained in:

[`../../ACKNOWLEDGMENTS.md`](../../ACKNOWLEDGMENTS.md)

## Related Repository Components

The CAD source is one component of the complete Wood-Chip Monitor repository.

Related resources include:

```text
app/
    Core edge-inference pipeline, backend service, and web dashboard

models/
    Model definitions and deployment-artifact documentation

tools/
    TensorRT inference and deployment utilities

configs/
    Device configuration templates

docs/
    Architecture, deployment, environment, and user documentation

validation/
    Historical deployment-validation records

examples/
    Representative input images and prediction outputs

assets/
    Dashboard, prototype, and analytical visualizations
```

For a system-level description of how these components interact, see:

[`../../docs/architecture.md`](../../docs/architecture.md)

## Private Repository Notice

At the current stage, this directory is maintained for internal research, engineering, reproducibility, and intellectual-property documentation.

Public disclosure, redistribution, licensing, or external release of these CAD files should occur only after appropriate project, collaborator, institutional, and intellectual-property review.