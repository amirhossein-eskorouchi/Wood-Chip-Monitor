# Deployment Tools

This directory contains utilities used to prepare, test, and validate the Wood-Chip Monitor edge-AI deployment.

## TensorRT Inference

### `infer_trt.py`

`infer_trt.py` provides standalone image inference using the TensorRT-optimized DETR detector deployed as part of Wood-Chip Monitor.

The utility:

- loads a serialized TensorRT DETR engine,
- preprocesses input images using the deployment image pipeline,
- executes TensorRT inference,
- applies confidence filtering,
- applies non-maximum suppression,
- maps detections back to the original image coordinates,
- and saves annotated prediction images.

### Expected Model

By default, the Wood-Chip Monitor deployment uses a detector engine such as:

```text
models/detr_resnet101_fp16.engine
```

TensorRT engines are intentionally not committed to this repository because they depend on the target hardware and software environment.

See:

- [`../models/README.md`](../models/README.md)
- [`../docs/jetson-setup.md`](../docs/jetson-setup.md)

for deployment details.

## Usage

From the repository root:

```bash
python tools/infer_trt.py \
    --engine models/detr_resnet101_fp16.engine \
    --images examples/images \
    --output outputs/predictions
```

On Windows PowerShell, the same command can be written as:

```powershell
python tools\infer_trt.py `
    --engine models\detr_resnet101_fp16.engine `
    --images examples\images `
    --output outputs\predictions
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--engine` | Path to the TensorRT DETR engine | Required |
| `--images` | Directory containing input images | `examples/images` |
| `--output` | Directory for annotated predictions | `outputs/predictions` |
| `--confidence` | Detection confidence threshold | `0.5` |
| `--nms-iou` | Non-maximum suppression IoU threshold | `0.5` |

Example with custom thresholds:

```bash
python tools/infer_trt.py \
    --engine models/detr_resnet101_fp16.engine \
    --images examples/images \
    --output outputs/predictions \
    --confidence 0.60 \
    --nms-iou 0.50
```

## Output

Annotated predictions are written to the specified output directory.

For example:

```text
outputs/
└── predictions/
    ├── pred_14_29.jpg
    ├── pred_1_23.jpg
    ├── pred_25_51.jpg
    └── pred_2_2.jpg
```

Runtime-generated outputs are excluded from version control.

Representative prediction examples are instead maintained under:

```text
examples/predictions/
```

## Model Export

The historical project archive includes an early ONNX-export utility used during system development. That script depends on a project-specific DETR model wrapper that is not contained in the deployment archive.

Rather than publishing an incomplete historical export workflow, the public repository will provide a reproducible model-export procedure based on the canonical detector implementation.

This workflow will document:

1. loading the trained detector checkpoint,
2. exporting the model to ONNX,
3. validating ONNX inference,
4. generating the TensorRT engine for the target Jetson environment,
5. and checking inference consistency before deployment.

## Deployment Environment

The original Wood-Chip Monitor prototype was deployed on NVIDIA Jetson Nano hardware using a legacy Jetson software stack.

Because TensorRT engines are environment-dependent, users should generate deployment engines for their own target platform rather than assuming binary compatibility across CUDA, TensorRT, JetPack/L4T, or Jetson versions.

See [`../docs/jetson-setup.md`](../docs/jetson-setup.md) for the reference deployment environment.