"""
ONNX-only inference utilities for RD-DETR.

Replaces the previous RF-DETR Python-model + supervision dependency with a
lightweight ONNXRuntime pipeline to reduce resource usage. Provides:
- RFDETR_ONNX: minimal ONNX wrapper
- run_rfdetr_inference: single-image inference and visualization
- run_rfdetr_inference_tiled: simple pass-through to normal (tiling not used)
"""

import io
import os
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MAX_NUMBER_BOXES = 300


def open_image(path: str) -> Image.Image:
    """Load image from local path or URL."""
    if path.startswith("http://") or path.startswith("https://"):
        return Image.open(io.BytesIO(requests.get(path).content))
    if os.path.exists(path):
        return Image.open(path)
    raise FileNotFoundError(f"The file {path} does not exist.")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def box_cxcywh_to_xyxyn(x):
    cx, cy, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2
    return np.stack([xmin, ymin, xmax, ymax], axis=-1)


class RFDETR_ONNX:
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    def __init__(self, onnx_model_path: str):
        try:
            self.ort_session = ort.InferenceSession(onnx_model_path)
            input_info = self.ort_session.get_inputs()[0]
            self.input_height, self.input_width = input_info.shape[2:]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ONNX model from '{onnx_model_path}'."
            ) from e

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.resize((self.input_width, self.input_height))
        image = np.array(image).astype(np.float32) / 255.0
        image = ((image - self.MEANS) / self.STDS).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image

    def _post_process(
        self,
        outputs,
        origin_height: int,
        origin_width: int,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_number_boxes: int = DEFAULT_MAX_NUMBER_BOXES,
    ):
        masks = outputs[2] if len(outputs) == 3 else None
        prob = sigmoid(outputs[1])

        scores = np.max(prob, axis=2).squeeze()
        labels = np.argmax(prob, axis=2).squeeze()
        sorted_idx = np.argsort(scores)[::-1]
        scores = scores[sorted_idx][:max_number_boxes]
        labels = labels[sorted_idx][:max_number_boxes]
        boxes = outputs[0].squeeze()[sorted_idx][:max_number_boxes]
        if masks is not None:
            masks = masks.squeeze()[sorted_idx][:max_number_boxes]

        boxes = box_cxcywh_to_xyxyn(boxes)
        boxes[..., [0, 2]] *= origin_width
        boxes[..., [1, 3]] *= origin_height

        if masks is not None:
            new_w, new_h = origin_width, origin_height
            masks = np.stack([
                np.array(Image.fromarray(img).resize((new_w, new_h)))
                for img in masks
            ], axis=0)
            masks = (masks > 0).astype(np.uint8) * 255

        confidence_mask = scores > confidence_threshold
        scores = scores[confidence_mask]
        labels = labels[confidence_mask]
        boxes = boxes[confidence_mask]
        if masks is not None:
            masks = masks[confidence_mask]

        return scores, labels, boxes, masks

    def predict(
        self,
        image_path: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_number_boxes: int = DEFAULT_MAX_NUMBER_BOXES,
    ):
        image = open_image(image_path).convert("RGB")
        origin_width, origin_height = image.size
        input_image = self._preprocess(image)
        input_name = self.ort_session.get_inputs()[0].name
        outputs = self.ort_session.run(None, {input_name: input_image})
        return self._post_process(outputs, origin_height, origin_width, confidence_threshold, max_number_boxes)

    def save_detections(self, image_path: str, boxes, labels, masks, save_image_path: Path, class_names=None):
        base = open_image(image_path).convert("RGBA")
        result = base.copy()

        label_colors = {
            label: (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                100,
            )
            for label in np.unique(labels)
        }

        if masks is not None:
            for i in range(masks.shape[0]):
                label = labels[i]
                color = label_colors[label]
                mask_overlay = Image.fromarray(masks[i]).convert("L")
                mask_overlay = ImageOps.autocontrast(mask_overlay)
                overlay_color = Image.new("RGBA", base.size, color)
                overlay_masked = Image.new("RGBA", base.size)
                overlay_masked.paste(overlay_color, (0, 0), mask_overlay)
                result = Image.alpha_composite(result, overlay_masked)

        result_rgb = result.convert("RGB")
        draw = ImageDraw.Draw(result_rgb)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()

        for i, box in enumerate(boxes.astype(int)):
            label = int(labels[i])
            box_color = tuple(label_colors[label][:3])
            text_label = (
                class_names[label]
                if class_names and 0 <= label < len(class_names)
                else str(label)
            )
            draw.rectangle(box.tolist(), outline=box_color, width=4)
            draw.text((box[0] + 5, box[1] + 5), text_label,
                      fill=(255, 255, 255), font=font)

        result_rgb.save(save_image_path)


# ================================================================
#                      NORMAL INFERENCE
# ================================================================
def run_rfdetr_inference(model: RFDETR_ONNX, image_path: str, class_names=None, save_dir="saved_predictions", threshold=0.4):
    """Run ONNX RF-DETR inference on one image and save visualization."""
    scores, labels, boxes, masks = model.predict(
        image_path,
        confidence_threshold=threshold,
    )

    if scores is None or len(scores) == 0:
        print("No detections found.")
        return None, None

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{Path(image_path).stem}_pred.jpg"
    model.save_detections(image_path, boxes, labels, masks,
                          save_path, class_names=class_names)

    detections = {
        "scores": scores.tolist(),
        "labels": labels.tolist(),
        "boxes": boxes.tolist(),
    }
    return detections, str(save_path)


# ================================================================
#                   SAHI-STYLE / TILED (fallback)
# ================================================================
def run_rfdetr_inference_tiled(
    model: RFDETR_ONNX,
    image_path: str,
    class_names=None,
    tile_size=640,
    overlap=0.2,
    conf_thres=0.35,
    save_dir="saved_predictions_tiled",
):
    """Tiled path not specialized; fall back to single-pass inference."""
    return run_rfdetr_inference(
        model,
        image_path,
        class_names=class_names,
        save_dir=save_dir,
        threshold=conf_thres,
    )
