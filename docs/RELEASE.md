# Public Release

## Version

The initial public release is version 0.1.0, dated August 25, 2026.

## Release contents

The release includes:

- maintained Jetson application and dashboard code;
- model-definition and deployment utilities;
- Git LFS-managed ONNX and HDF5 model artifacts;
- SolidWorks parts and assembly files;
- configuration templates;
- example images and predictions;
- validation records;
- system, deployment, environment, hardware, and user documentation;
- citation metadata;
- MIT licensing for independently authored software and documentation; and
- a notice describing model, CAD, dataset, publication, and third-party terms.

## Validation boundary

The release audit checks:

- required release files;
- forbidden credentials, caches, archives, databases, and runtime output;
- ordinary tracked-file size;
- exact Git LFS model pointers, hashes, and expected payload sizes;
- presence of all nine CAD files;
- license and notice content;
- citation metadata;
- README sections and local links;
- private workstation paths;
- concrete secret patterns;
- Python syntax without importing deployment dependencies; and
- Git whitespace integrity.

## Git LFS

The model artifacts are managed with Git LFS. Users requiring full model
payloads should install Git LFS before cloning.

Continuous integration intentionally checks out LFS pointers without
downloading approximately 492 MB of model payloads. The audit validates each
pointer's recorded SHA-256 and byte size.

## Environment boundary

The reference deployment uses a legacy NVIDIA Jetson Nano software stack.
Continuous integration validates portable repository structure and syntax; it
does not claim to reproduce TensorRT execution on GitHub-hosted runners.

See:

- [`software-environment.md`](software-environment.md)
- [`jetson-setup.md`](jetson-setup.md)
- [`../models/README.md`](../models/README.md)
- [`../NOTICE`](../NOTICE)

## Release procedure

Before publishing a release:

1. run `python scripts/audit_release.py` with an available Python runtime;
2. verify that GitHub Actions passes;
3. verify Git LFS object integrity;
4. confirm that the working tree is clean;
5. create tag `v0.1.0` from `main`;
6. publish the release without attaching private runtime material; and
7. verify repository description, topics, citation, license, and visibility.
