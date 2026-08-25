# Changelog

All notable public changes to Wood-Chip Monitor are documented here.

## [0.1.0] - 2026-08-25

### Added

- Jetson-based real-time wood-chip detection pipeline.
- FastAPI backend and browser-based operator dashboard.
- Physical chip-size estimation and size-distribution monitoring.
- Oversize detection and quality-statistics reporting.
- MoistNetLite architecture and moisture-inference integration.
- TensorRT deployment and standalone inference utilities.
- Git LFS-managed model deployment artifacts.
- SolidWorks mechanical part and assembly files.
- Representative example images and prediction outputs.
- Historical deployment-validation records.
- Architecture, deployment, environment, hardware, and user documentation.
- MIT License, release notice, citation metadata, and citation guidance.

### Changed

- Replaced a workstation-specific development path with an optional
  environment-controlled extension path.
- Replaced a local workstation account identifier with the contributor's
  public name.
- Updated private-development language for the public release.
- Clarified model, CAD, dataset, publication, and third-party boundaries.

### Security

- Confirmed that no concrete credentials, private environment file, database,
  runtime output, archive, or private workstation path is tracked.
- Retained only an explicit placeholder password in the example configuration.

### Known limitations

- The reference deployment targets a legacy NVIDIA Jetson Nano environment.
- TensorRT engines are device- and software-stack-specific.
- Model outputs require deployment-specific calibration and validation.
- The system is a research prototype rather than a certified industrial
  measurement instrument.
