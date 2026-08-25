# Mechanical CAD Source

This directory contains the native SolidWorks source files used to design and prototype the mechanical enclosure and component integration for the Wood-Chip Monitor portable edge-AI system.

## Overview

The Wood-Chip Monitor hardware was developed as a compact, portable inspection platform integrating the edge-computing device, camera, touchscreen, mounting hardware, and power components required for on-site wood-chip quality monitoring.

The CAD package in this directory preserves the original mechanical design files used during prototype development.

## Included CAD Files

The package contains the following SolidWorks source files:

```text
hardware/cad/
â”œâ”€â”€ 7_inch_LCD_CASE.SLDPRT
â”œâ”€â”€ CAMERA_HOLDER.SLDPRT
â”œâ”€â”€ Camera_Lock_System.SLDPRT
â”œâ”€â”€ Handle.SLDPRT
â”œâ”€â”€ Hook.SLDPRT
â”œâ”€â”€ Intergrated_camera.SLDPRT
â”œâ”€â”€ Jetson_nano_case.SLDPRT
â”œâ”€â”€ NEW ASSEMBLY.SLDASM
â”œâ”€â”€ port_lid.SLDPRT
â””â”€â”€ README.md
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

The SolidWorks files are included in the public Wood-Chip Monitor repository
to preserve the complete versioned engineering record.

The package should be treated as a coordinated assembly. Component files
should remain together so SolidWorks can resolve assembly references.

## Release and licensing boundary

The CAD files are publicly downloadable as engineering records. Their presence
does not constitute certification by an open-hardware standards organization.

The MIT License applies to independently authored software and documentation.
See [`../../NOTICE`](../../NOTICE) for the mechanical-design and third-party
boundaries.

Users are responsible for fabrication suitability, dimensions, tolerances,
materials, assembly references, third-party hardware terms, safety validation,
and applicable attribution.

## Attribution

Prashant Bhattarai contributed to the mechanical design and physical prototype
development associated with the AI2F Summer 2025 program.

See [`../../ACKNOWLEDGMENTS.md`](../../ACKNOWLEDGMENTS.md).

## Related repository components

- `app/`: edge inference, backend, and dashboard;
- `models/`: model definitions and deployment artifacts;
- `tools/`: TensorRT utilities;
- `configs/`: configuration templates;
- `docs/`: architecture and deployment documentation;
- `validation/`: deployment-validation records;
- `examples/`: representative inputs and predictions; and
- `assets/`: prototype and analytical figures.

See [`../../docs/architecture.md`](../../docs/architecture.md).
