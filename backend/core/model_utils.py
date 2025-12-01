from turtle import st
import requests
import supervision as sv
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from rfdetr import RFDETRBase
from huggingface_hub import hf_hub_download
import warnings
from PIL import Image

# from website_streamlit.app import BACKEND_URL
warnings.filterwarnings("ignore", category=UserWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"


class RFDETR_ONNXWrapper:
    """
    Drop-in ONNXRuntime wrapper that mimics torch.nn.Module.
    Works seamlessly inside RFDETRBase by preserving the same
    pre/post-processing flow.
    """

    def __init__(self, onnx_path, providers=None):
        import onnxruntime as ort
        import numpy as np

        self.onnx_path = onnx_path
        self.providers = providers or [
            "CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(onnx_path), providers=self.providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # print(f"[INFO] ONNX model loaded: {onnx_path}")
        # print(f"[INFO] Detected input name: {self.input_name}")
        # print(f"[INFO] Detected outputs: {self.output_names}")

    def __call__(self, images):
        """
        Acts like a forward() pass for the PyTorch model.
        Accepts torch tensors from RFDETRBase and runs ONNX inference.
        """
        import torch
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()

        ort_inputs = {self.input_name: images}
        outputs = self.session.run(None, ort_inputs)

        # Convert numpy outputs back to torch tensors for postprocess
        torch_outputs = [torch.from_numpy(o).to(device) for o in outputs]

        # Handle both tuple-style and dict-style outputs
        if len(torch_outputs) == 2:
            pred_logits, pred_boxes = torch_outputs
            return pred_logits, pred_boxes
        else:
            return torch_outputs

    def eval(self):
        """Mimic torch.nn.Module.eval()"""
        return self


# ================================================================
#                      NORMAL INFERENCE
# ================================================================
def run_rfdetr_inference(model, image_path: str, class_names=None, save_dir="saved_predictions", threshold=0.4):
    """Run RF-DETR inference on one image and save visualization using supervision."""
    image = Image.open(image_path)

    detections = model.predict(image, threshold=threshold)
    if len(detections) == 0:
        print("No detections found.")
        return None, None
    # print("Class IDs:", detections.class_id)
    # print("Confidences:", detections.confidence)
    # print("Boxes:", detections.xyxy if hasattr(detections, "xyxy") else None)

    if class_names is None:
        class_names = ["damage"]

    labels = []
    for class_id, confidence in zip(detections.class_id, detections.confidence):
        if class_id < len(class_names):
            label = f"{class_names[class_id]} {confidence:.2f}"
        else:
            label = f"Unknown({class_id}) {confidence:.2f}"
        labels.append(label)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated = box_annotator.annotate(image.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections, labels)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{Path(image_path).stem}_pred.jpg"
    annotated.save(save_path)
    # print(f"Saved annotated image to: {save_path}")

    return detections, str(save_path)


# ================================================================
#                      SAHI-STYLE INFERENCE
# ================================================================
def run_rfdetr_inference_tiled(
    model,
    image_path: str,
    class_names=None,
    tile_size=640,
    overlap=0.2,
    conf_thres=0.35,
    save_dir="saved_predictions_tiled"
):
    """Run tiled (SAHI-style) inference for RF-DETR."""
    from shapely.geometry import box as shapely_box
    import numpy as np

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    detections_all = []

    step = int(tile_size * (1 - overlap))
    # Slide over the image
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            x1, y1 = x0 + tile_size, y0 + tile_size
            # skip if out of bounds
            if x1 > w or y1 > h:
                continue
            # crop tile to detect on that single title
            tile = image.crop((x0, y0, x1, y1))
            tile = tile.resize((640, 640), Image.LANCZOS)
            detections = model.predict(tile, threshold=conf_thres)

            if detections is None or len(detections.class_id) == 0:
                continue
            # regular boxes to global image boxes, adjusting coordinate if title start at (x0, y0) detection only inside it so we have to add x0, y0 more to represent global image
            for i in range(len(detections.class_id)):
                scale_x = (x1 - x0) / 640
                scale_y = (y1 - y0) / 640
                x_min = detections.xyxy[i][0] * scale_x + x0
                y_min = detections.xyxy[i][1] * scale_y + y0
                x_max = detections.xyxy[i][2] * scale_x + x0
                y_max = detections.xyxy[i][3] * scale_y + y0
                score = detections.confidence[i]
                cls = detections.class_id[i]
                detections_all.append([
                    x_min, y_min, x_max, y_max, score, cls
                ])

    if len(detections_all) == 0:
        print("No detections found.")
        return None, None

    # NMS merging
    detections_all = np.array(detections_all)
    # each detection_all: [x_min, y_min, x_max, y_max, score, class_id]
    boxes = detections_all[:, :4]  # all rows, first 4 columns
    scores = detections_all[:, 4]  # all rows, 5th column
    cls = detections_all[:, 5]  # all rows, 6th column
    keep = nms(boxes, scores, iou_thres=0.5)
    final_dets = detections_all[keep]

    # Visualization
    dets_xyxy = sv.Detections(
        xyxy=final_dets[:, :4],
        confidence=final_dets[:, 4],
        class_id=final_dets[:, 5].astype(int)
    )

    if class_names is None:
        class_names = ["damage"]

    labels = [f"{class_names[int(c)]} {s:.2f}" for c, s in zip(
        dets_xyxy.class_id, dets_xyxy.confidence)]
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated = box_annotator.annotate(image.copy(), dets_xyxy)
    annotated = label_annotator.annotate(annotated, dets_xyxy, labels)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / f"{Path(image_path).stem}_tiled_pred.jpg"
    annotated.save(save_path)
    # print(f"Saved tiled annotated image to: {save_path}")

    return dets_xyxy, str(save_path)


def nms(boxes, scores, iou_thres=0.5):
    """Simple NMS."""
    x1, y1, x2, y2 = boxes.T  # flip the numpy array so now we have all x1, y1, x2, y2 in separate arrays
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def load_model():
    """
    Try to load ONNX model first.
    If unavailable or invalid, fall back to PyTorch RF-DETR model.
    """

    class_names = ["wind", "hail"]
    onnx_path = Path("exported_models/inference_model.onnx")
    if not onnx_path.exists():
        try:
            # Download ONNX model from Hugging Face if not present
            onnx_path_str = hf_hub_download(
                repo_id="tnkchaseme/rfdetr-roof-assessment",
                filename="inference_model.onnx",
            )
            onnx_path = Path(onnx_path_str)
            print(f"[INFO] Downloaded ONNX model to {onnx_path}")
        except Exception as e:
            print(f"[WARNING] Failed to download ONNX model: {e}")
            # Will not exist
            onnx_path = Path("exported_models/inference_model.onnx")
    checkpoint_path = "merged_annotations/output/checkpoint.pth"
    if not Path(checkpoint_path).exists():
        try:
            # Download PyTorch checkpoint from Hugging Face if not present
            checkpoint_path_str = hf_hub_download(
                repo_id="tnkchaseme/rfdetr-roof-assessment",
                filename="checkpoint.pth",
            )
            checkpoint_path = checkpoint_path_str
            print(f"[INFO] Downloaded checkpoint to {checkpoint_path}")
        except Exception as e:
            print(f"[ERROR] Failed to download checkpoint: {e}")
            raise FileNotFoundError(
                "No valid model checkpoint found for RF-DETR inference."
            )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = None
    model_type = "ONNX"

    # --- Try loading ONNX model ---
    if onnx_path.exists():
        try:
            core_onnx = RFDETR_ONNXWrapper(
                'exported_models/inference_model.onnx')
            # Wrap it inside the existing RFDETRBase structure
            model = RFDETRBase(num_classes=2, device=device)
            model.model.model = core_onnx
            print(f"[INFO] Loaded ONNX model from {onnx_path}")
        except Exception as e:
            print(f"[WARNING] Failed to load ONNX model: {e}")
            model = None

    # --- Fallback to PyTorch model ---
    if model is None:
        model_type = "PyTorch"
        print("[INFO] Falling back to PyTorch checkpoint loading...")

        model = RFDETRBase(
            num_classes=len(class_names),
            pretrain_weights=checkpoint_path,
            device=device
        )
        model.optimize_for_inference()

        core_model = model.model.model
        core_model.to(device)
        core_model.eval()

        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        try:
            compiled_core = torch.compile(
                core_model, mode="reduce-overhead", backend="inductor"
            )
            model.model.model = compiled_core
            print("[INFO] Core RF-DETR network compiled successfully.")
        except Exception as e:
            print(f"[WARNING] TorchDynamo compile skipped: {e}")

    print(f"[INFO] Using {model_type} model for inference.")
    return model, class_names, model_type


def generate_report(files, infer_mode, conf_threshold, tile_size, path):
    model, class_names, model_type = load_model()
    pred_path = None

    for file in Path(files).glob("*.*"):

        if file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        if infer_mode == "Normal":
            print('Chase smells really bad')
            detections, pred_path = run_rfdetr_inference(
                model=model,
                image_path=str(file),
                class_names=class_names,
                save_dir=path, threshold=conf_threshold
            )
        else:
            print('Cason smells really bad')
            detections, pred_path = run_rfdetr_inference_tiled(
                model=model,
                image_path=str(file),
                class_names=class_names,
                tile_size=tile_size,
                overlap=0.4,
                conf_thres=conf_threshold,
                save_dir=path,
            )

    return pred_path
    # if pred_path and Path(pred_path).exists():
    #     st.success("Inference completed successfully.")
    #     st.image(str(pred_path), caption="Detection Result",
    #             width="content")
    # else:
    #     st.warning("No detections found or failed to save output.")

# usage :
# python main.py --mode test --infer_mode normal
# tile side: tiny, small, normal, large
# python main.py --mode test --infer_mode tiled --tile_size small --path datasets/hail_1/test
# python main.py --mode train

# test wind_1
# python main.py --mode test --infer_mode normal --path datasets/wind_1/test
