# Contributing

Thank you for your interest in Wood-Chip Monitor.

## Before contributing

Open an issue before making a substantial change. Describe the proposed
capability, affected component, Jetson compatibility, and any dataset,
licensing, privacy, model, or hardware implications.

## Development expectations

Contributions should:

- avoid workstation-specific absolute paths;
- avoid committed credentials and local configuration;
- keep databases and runtime output outside Git;
- preserve documented configuration interfaces;
- document behavior changes;
- avoid device-specific TensorRT engines;
- use Git LFS for approved model artifacts;
- preserve CAD assembly references; and
- document the provenance of third-party content.

## Validation

Before submitting a pull request:

1. run the release audit;
2. run applicable syntax checks and tests;
3. verify that example credentials remain placeholders;
4. confirm that runtime files remain ignored;
5. run `git diff --check`; and
6. summarize validation results.

## Binary contributions

Model artifacts, datasets, CAD files, and other substantial binaries should
include origin, ownership, public-release status, applicable terms, hash,
size, and reproducibility requirements.

Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
