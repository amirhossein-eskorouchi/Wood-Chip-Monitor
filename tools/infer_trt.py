"""
Standalone TensorRT inference utility for Wood-Chip Monitor.

This script runs the deployed DETR TensorRT engine on a directory
of images and writes annotated predictions to an output directory.

The TensorRT engine itself is not stored in this repository because
TensorRT engines depend on the target hardware and software stack.
"""

import argparse
import os
import time

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt


TARGET_HEIGHT = 800
TARGET_WIDTH = 1333

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


def resize_and_letterbox_bgr(image):
    """Resize an image while preserving aspect ratio and pad to model size."""

    original_height, original_width = image.shape[:2]

    scale = min(
        TARGET_HEIGHT / original_height,
        TARGET_WIDTH / original_width,
    )

    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.zeros(
        (TARGET_HEIGHT, TARGET_WIDTH, 3),
        dtype=np.uint8,
    )

    canvas[:resized_height, :resized_width] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb /= 255.0

    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

    tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
    tensor = tensor.astype(np.float32)

    return (
        tensor,
        (original_height, original_width),
        scale,
        resized_height,
        resized_width,
    )


def nms(boxes, scores, iou_threshold=0.5):
    """Apply non-maximum suppression."""

    if boxes.shape[0] == 0:
        return []

    x1, y1, x2, y2 = boxes.T

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        index = order[0]
        keep.append(index)

        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])

        intersection = (
            np.maximum(0.0, xx2 - xx1)
            * np.maximum(0.0, yy2 - yy1)
        )

        iou = intersection / (
            areas[index]
            + areas[order[1:]]
            - intersection
            + 1e-6
        )

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


class TensorRTDetector:
    """Minimal TensorRT wrapper for the deployed DETR engine."""

    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as engine_file:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(
                engine_file.read()
            )

        if self.engine is None:
            raise RuntimeError(
                f"Unable to deserialize TensorRT engine: {engine_path}"
            )

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_index = None
        self.output_indices = []

        self.host_buffers = []
        self.device_buffers = []
        self.binding_names = {}

        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            self.binding_names[index] = name

            dtype = trt.nptype(
                self.engine.get_binding_dtype(index)
            )

            shape = tuple(
                self.context.get_binding_shape(index)
            )

            size = int(np.prod(shape))

            host_buffer = cuda.pagelocked_empty(size, dtype)
            device_buffer = cuda.mem_alloc(host_buffer.nbytes)

            self.host_buffers.append(host_buffer)
            self.device_buffers.append(device_buffer)

            if self.engine.binding_is_input(index):
                self.input_index = index
            else:
                self.output_indices.append(index)

        if self.input_index is None:
            raise RuntimeError(
                "TensorRT engine does not contain an input binding."
            )

        if len(self.output_indices) != 2:
            raise RuntimeError(
                "Expected DETR engine to contain exactly two outputs."
            )

    def infer(self, input_tensor):
        """Execute TensorRT inference."""

        self.context.set_binding_shape(
            self.input_index,
            input_tensor.shape,
        )

        np.copyto(
            self.host_buffers[self.input_index],
            input_tensor.ravel(),
        )

        cuda.memcpy_htod_async(
            self.device_buffers[self.input_index],
            self.host_buffers[self.input_index],
            self.stream,
        )

        self.context.execute_async_v2(
            [int(buffer) for buffer in self.device_buffers],
            self.stream.handle,
        )

        outputs = {}

        for output_index in self.output_indices:
            cuda.memcpy_dtoh_async(
                self.host_buffers[output_index],
                self.device_buffers[output_index],
                self.stream,
            )

        self.stream.synchronize()

        for output_index in self.output_indices:
            shape = tuple(
                self.context.get_binding_shape(output_index)
            )

            array = np.array(
                self.host_buffers[output_index],
                copy=True,
            ).reshape(shape)

            outputs[
                self.binding_names[output_index]
            ] = array

        logits = outputs.get("logits")
        pred_boxes = outputs.get("pred_boxes")

        if logits is None or pred_boxes is None:
            raise RuntimeError(
                "Expected TensorRT outputs named "
                "'logits' and 'pred_boxes'."
            )

        return logits, pred_boxes


def process_image(
    detector,
    image_path,
    output_path,
    confidence_threshold,
    nms_iou,
):
    """Run inference and save an annotated prediction."""

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    (
        input_tensor,
        (original_height, original_width),
        scale,
        resized_height,
        resized_width,
    ) = resize_and_letterbox_bgr(image)

    start = time.time()

    logits, pred_boxes = detector.infer(input_tensor)

    inference_ms = (time.time() - start) * 1000.0

    probabilities = logits.reshape(1, -1, 2)

    probabilities = np.exp(
        probabilities
        - probabilities.max(axis=2, keepdims=True)
    )

    probabilities /= probabilities.sum(
        axis=2,
        keepdims=True,
    )

    scores = probabilities[0, :, 1]
    boxes = pred_boxes[0]

    keep = np.where(
        scores >= confidence_threshold
    )[0]

    max_index = min(
        boxes.shape[0],
        scores.shape[0],
    )

    keep = keep[keep < max_index]

    boxes = boxes[keep]
    scores = scores[keep]

    if boxes.size > 0:
        center_x = boxes[:, 0] * TARGET_WIDTH
        center_y = boxes[:, 1] * TARGET_HEIGHT
        width = boxes[:, 2] * TARGET_WIDTH
        height = boxes[:, 3] * TARGET_HEIGHT

        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2

        x1 = np.clip(x1, 0, resized_width - 1)
        x2 = np.clip(x2, 0, resized_width - 1)
        y1 = np.clip(y1, 0, resized_height - 1)
        y2 = np.clip(y2, 0, resized_height - 1)

        x1 /= scale
        x2 /= scale
        y1 /= scale
        y2 /= scale

        boxes_xyxy = np.stack(
            [x1, y1, x2, y2],
            axis=1,
        )

        boxes_xyxy[:, 0::2] = np.clip(
            boxes_xyxy[:, 0::2],
            0,
            original_width - 1,
        )

        boxes_xyxy[:, 1::2] = np.clip(
            boxes_xyxy[:, 1::2],
            0,
            original_height - 1,
        )

        keep_nms = nms(
            boxes_xyxy,
            scores,
            iou_threshold=nms_iou,
        )

        boxes_xyxy = boxes_xyxy[keep_nms].astype(int)
        scores = scores[keep_nms]

    else:
        boxes_xyxy = np.zeros(
            (0, 4),
            dtype=int,
        )

        scores = np.zeros(
            (0,),
            dtype=np.float32,
        )

    visualization = image.copy()

    for (x1, y1, x2, y2), score in zip(
        boxes_xyxy,
        scores,
    ):
        cv2.rectangle(
            visualization,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            visualization,
            f"{score:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        output_path,
        visualization,
    )

    return inference_ms, len(scores)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Wood-Chip Monitor DETR TensorRT "
            "engine on a directory of images."
        )
    )

    parser.add_argument(
        "--engine",
        required=True,
        help="Path to the TensorRT DETR engine.",
    )

    parser.add_argument(
        "--images",
        default="examples/images",
        help="Directory containing input images.",
    )

    parser.add_argument(
        "--output",
        default="outputs/predictions",
        help="Directory for annotated predictions.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold.",
    )

    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(
        args.output,
        exist_ok=True,
    )

    detector = TensorRTDetector(
        args.engine
    )

    image_names = sorted(
        name
        for name in os.listdir(args.images)
        if name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    )

    for image_name in image_names:
        input_path = os.path.join(
            args.images,
            image_name,
        )

        output_path = os.path.join(
            args.output,
            f"pred_{image_name}",
        )

        inference_ms, detection_count = process_image(
            detector=detector,
            image_path=input_path,
            output_path=output_path,
            confidence_threshold=args.confidence,
            nms_iou=args.nms_iou,
        )

        print(
            f"{image_name}: "
            f"{inference_ms:.1f} ms, "
            f"{detection_count} detections"
        )


if __name__ == "__main__":
    main()