import supervision as sv
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from rfdetr import RFDETRBase
import warnings
from PIL import Image
warnings.filterwarnings("ignore", category=UserWarning)


# ================================================================
#                      TRAINING FUNCTION
# ================================================================
def run_training(
    num_classes: int = 1,
    path_to_dataset: str = "merged_annotations",
    resume_checkpoint: str | None = None,
    output_dir: str = "merged_annotations/output"
):
    model = RFDETRBase(num_classes=num_classes)

    # Resume path logic
    resume_path = (
        resume_checkpoint
        if resume_checkpoint
        else Path(output_dir) / "checkpoint.pth"
    )
    if Path(resume_path).exists():
        print(f"Resuming from {resume_path}")
    else:
        print("Starting fresh training...")
        resume_path = None

    model.train(
        dataset_dir=path_to_dataset,
        epochs=100,
        batch_size=8,
        grad_accum_steps=4,
        lr=1e-5,
        num_workers=0,
        output_dir=output_dir,
        tensorboard=True,
        resume=resume_path,
        seed=42,
        early_stopping=True,
        early_stopping_patience=10,
        gradient_checkpointing=True,
    )


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
    print("Class IDs:", detections.class_id)
    print("Confidences:", detections.confidence)
    print("Boxes:", detections.xyxy if hasattr(detections, "xyxy") else None)

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
    print(f"Saved annotated image to: {save_path}")

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
    print(f"Saved tiled annotated image to: {save_path}")

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


# ================================================================
#                      MAIN SCRIPT
# ================================================================
if __name__ == "__main__":
    import multiprocessing
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', type=str, choices=['train', 'test'], default='test', help='Mode: train or test')
    parser.add_argument('--infer_mode', type=str, choices=['normal', 'tiled'],
                        default='normal',
                        help='Inference mode: normal or tiled (SAHI-style)')
    parser.add_argument('--tile_size', type=str, choices=["tiny", "small", "normal", "large"], default="normal",
                        help='Tile size for tiled inference')
    parser.add_argument('--path', type=str, help='Path to test images folder')
    args = parser.parse_args()
    mode = args.mode
    infer_mode = args.infer_mode
    tile_size = args.tile_size
    if tile_size == "tiny":
        tile_size = 320
    elif tile_size == "small":
        tile_size = 480
    elif tile_size == "normal":
        tile_size = 640
    elif tile_size == "large":
        tile_size = 800
    else:
        tile_size = 640

    multiprocessing.freeze_support()  # required for Windows

    class_names = ["wind", "hail"]

    if mode == "train":
        checkpoint_path = "merged_annotations/output/checkpoint.pth"
        num_classes = len(class_names)
        run_training(
            num_classes=num_classes,
            path_to_dataset="merged_annotations",
            resume_checkpoint=checkpoint_path,
            output_dir="merged_annotations/output"
        )

    else:
        # === Paths ===
        test_folder_path = args.path if args.path else "datasets/single_test"
        print(f"Testing on images from: {test_folder_path}")
        checkpoint_path = "merged_annotations/output/checkpoint.pth"
        model = RFDETRBase(
            num_classes=len(class_names),
            pretrain_weights=checkpoint_path
        )
        # model = rfdetr
        # print(
        #     rfdetr.model.model.class_embed.bias.shape[0], "classes in model head")
        for img in Path(test_folder_path).glob("*.*"):
            if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            if infer_mode == "normal":
                path = test_folder_path.split('/')[-2:]
                data_path = path[0]
                type_path = path[1]
                run_rfdetr_inference(
                    model=model,
                    image_path=str(img),
                    class_names=class_names,
                    save_dir=f"run/{data_path}/{type_path}_predictions"
                )
            else:
                path = test_folder_path.split('/')[-2:]
                data_path = path[0]
                type_path = path[1]
                run_rfdetr_inference_tiled(
                    model=model,
                    image_path=str(img),
                    class_names=class_names,
                    tile_size=tile_size,
                    overlap=0.4,
                    conf_thres=0.35,
                    save_dir=f"run_tiled/{data_path}/{type_path}"
                )


# usage :
# python main.py --mode test --infer_mode normal
# tile side: tiny, small, normal, large
# python main.py --mode test --infer_mode tiled --tile_size small --path datasets/hail_1/test
# python main.py --mode train

# test wind_1
# python main.py --mode test --infer_mode normal --path datasets/wind_1/test