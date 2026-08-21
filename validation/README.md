# Deployment Validation

This directory preserves a compact record of the staged validation used while translating the Wood-Chip Monitor detection pipeline from workstation development to NVIDIA Jetson TensorRT deployment.

Large intermediate tensors, duplicate visualization files, and development-only artifacts from the original validation archive are intentionally excluded.

## Validation Progression

The deployment workflow progressed through several stages:

```text
Reference DETR Processing
        ↓
Manual Pre/Post-Processing
        ↓
Jetson TensorRT Inference
        ↓
Size-Distribution Analysis
        ↓
Statistical Validation
        ↓
Integrated Inference Pipeline
```

## 1. Reference Processing

[`reference_processing.csv`](reference_processing.csv) contains outputs generated using the original reference processing pipeline on four representative wood-chip images.

This stage established baseline image dimensions and prediction counts before deployment-specific preprocessing was introduced.

The recorded fields include:

- image filename,
- number of retained predictions,
- processed image height,
- processed image width.

## 2. Manual Pre/Post-Processing Validation

[`manual_processing.csv`](manual_processing.csv) records results from the manually reconstructed preprocessing and post-processing workflow.

This stage was used during development to verify that image resizing, model-input preparation, and prediction reconstruction could be reproduced independently of the original reference processor.

The comparison helped validate the transition from the workstation development pipeline toward the deployment-specific implementation.

## 3. Jetson TensorRT Inference

[`jetson_tensorrt.csv`](jetson_tensorrt.csv) contains archived outputs from an early TensorRT deployment test on the NVIDIA Jetson Nano.

Recorded fields include:

- original image dimensions,
- TensorRT input dimensions,
- valid resized image dimensions,
- number of retained predictions,
- inference latency measured by the historical test script.

These values are retained as development provenance and should not be interpreted as a controlled benchmark of the final live monitoring application.

Prediction counts across validation stages may differ because preprocessing, post-processing, confidence filtering, non-maximum suppression, and deployment logic evolved during system integration.

Similarly, the archived latency values reflect the historical standalone validation script and are not presented as the final end-to-end real-time throughput of Wood-Chip Monitor.

## 4. Size-Analysis Validation

[`size_statistics.csv`](size_statistics.csv) contains descriptive statistics produced during the integrated size-analysis validation stage.

The archived analysis reports three geometric measurements:

- `L` — estimated chip length,
- `W` — estimated chip width,
- `D` — estimated diagonal or characteristic diameter.

The stored validation values are expressed in pixel units before physical calibration is applied.

The statistics include:

- sample count,
- mean,
- standard deviation,
- coefficient of variation,
- minimum,
- selected percentiles,
- median,
- maximum.

These development outputs were used to verify the downstream statistical-analysis pipeline before integration into the live monitoring system.

## 5. Distribution and Statistical Analytics

The validation workflow also verified the analytical components used to summarize the observed chip population.

These components include:

- descriptive statistics,
- chip-size distributions,
- percentile summaries,
- size buckets,
- histograms,
- box plots,
- rolling statistics,
- and oversize-related summaries.

Selected visual outputs are included in the main repository under [`../assets/`](../assets/).

For example:

- [`../assets/size_distribution.png`](../assets/size_distribution.png)
- [`../assets/size_boxplot.png`](../assets/size_boxplot.png)

## 6. Representative Visual Examples

Representative input images and detector outputs are available in:

```text
examples/
├── images/
└── predictions/
```

The source images are stored in:

[`../examples/images/`](../examples/images/)

The corresponding archived prediction visualizations are stored in:

[`../examples/predictions/`](../examples/predictions/)

These examples provide a compact visual record of detector behavior without including the much larger collection of intermediate development artifacts from the original project archive.

## Validation Files

The compact validation record included in this repository is:

```text
validation/
├── reference_processing.csv
├── manual_processing.csv
├── jetson_tensorrt.csv
├── size_statistics.csv
└── README.md
```

### `reference_processing.csv`

Reference workstation processing results used during early deployment validation.

### `manual_processing.csv`

Results from the manually reconstructed preprocessing and post-processing workflow.

### `jetson_tensorrt.csv`

Historical TensorRT inference outputs generated during Jetson deployment testing.

### `size_statistics.csv`

Archived descriptive statistics from the integrated size-analysis stage.

## What Is Intentionally Excluded

The original development archive contains substantially more intermediate material, including:

- NumPy tensor dumps,
- repeated prediction visualizations,
- duplicated output folders,
- intermediate preprocessing results,
- development-only diagnostics,
- temporary model-conversion artifacts.

These files are intentionally excluded from the public repository because they are large, redundant, or unnecessary for understanding the validation workflow.

The goal of this directory is to preserve meaningful provenance without turning the repository into a raw development archive.

## Interpretation of the Validation Record

The files in this directory document the historical engineering progression used to move the system from workstation development toward an embedded TensorRT deployment.

They should not be interpreted as a new controlled benchmark of:

- final detection accuracy,
- final model equivalence,
- production throughput,
- end-to-end system latency,
- or final calibrated measurement accuracy.

Those claims would require reproduction under a clearly defined hardware, software, model, and evaluation configuration.

Instead, this validation record demonstrates the staged engineering process used during system translation:

```text
Research Model
      ↓
Reference Processing
      ↓
Manual Pipeline Reconstruction
      ↓
TensorRT Deployment
      ↓
Geometric Measurement
      ↓
Statistical Analysis
      ↓
Integrated Edge-AI Monitoring
```

## Reproducibility Scope

The repository preserves enough information to document how the deployment pipeline evolved and how major stages were checked during development.

A controlled deployment benchmark may be added separately if the complete reference hardware and software environment is available for repeat testing.

For the original Jetson deployment environment, see:

[`../docs/jetson-setup.md`](../docs/jetson-setup.md)

For the broader software architecture, see:

[`../docs/architecture.md`](../docs/architecture.md)

For the standalone TensorRT inference utility, see:

[`../tools/infer_trt.py`](../tools/infer_trt.py)