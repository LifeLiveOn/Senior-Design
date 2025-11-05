import json
import os
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np


def is_black_edge_image(img, threshold=15, edge_ratio=0.5):
    """Quickly check if left or right 10% edges are mostly black."""
    np_img = np.array(img.convert("L"))  # grayscale
    h, w = np_img.shape
    edge_width = max(1, int(w * 0.1))
    left_ratio = np.mean(np_img[:, :edge_width] < threshold)
    right_ratio = np.mean(np_img[:, -edge_width:] < threshold)
    return left_ratio > edge_ratio or right_ratio > edge_ratio


def slice_single_image(
    image_path,
    anns,
    ann_id_start,
    next_img_id,
    tile_size=640,
    overlap=0.0,
    visualize=False,
    black_threshold=15,
    output_dir="tiles/",
    data_type="train",
    area_threshold=15*15,
    edge_threshold=3,
    context_margin=0.2,
    **kwargs
):
    from shapely.geometry import box as shapely_box

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    step = int(tile_size * (1 - overlap))

    new_images, new_annotations = [], []
    ann_id, img_id = ann_id_start, next_img_id

    img_out_dir = os.path.join(output_dir, data_type)
    vis_out_dir = os.path.join(
        output_dir, "visualized", data_type) if visualize else None
    os.makedirs(img_out_dir, exist_ok=True)
    if vis_out_dir:
        os.makedirs(vis_out_dir, exist_ok=True)

    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            x1, y1 = x0 + tile_size, y0 + tile_size
            if x1 > w or y1 > h:
                continue  # skip tiles exceeding image bounds

            tile = image.crop((x0, y0, x1, y1))
            gray = np.array(tile.convert("L"))

            # Skip mostly black tiles quickly
            if np.mean(gray < black_threshold) > 0.25:
                continue

            has_box = False
            tile_filename = f"{Path(image_path).stem}_x{x0}_y{y0}.jpg"
            slice_box = shapely_box(x0, y0, x1, y1)

            tile_annotations = []

            for ann in anns:
                bx, by, bw, bh = ann["bbox"]
                bx2, by2 = bx + bw, by + bh
                ann_box = shapely_box(bx, by, bx2, by2)

                # Check intersection
                inter = ann_box.intersection(slice_box)
                if inter.is_empty:
                    continue

                ixmin, iymin, ixmax, iymax = inter.bounds
                new_x, new_y = ixmin - x0, iymin - y0
                new_w, new_h = ixmax - ixmin, iymax - iymin

                if new_w <= 1 or new_h <= 1:
                    continue

                # skip tiny boxes (area-based)
                if new_w * new_h < area_threshold:
                    continue

                # skip edge bboxes
                if new_x <= edge_threshold or new_y <= edge_threshold or new_x + new_w >= tile_size - edge_threshold or new_y + new_h >= tile_size - edge_threshold:
                    continue

                has_box = True
                tile_annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": ann["category_id"],
                    "bbox": [new_x, new_y, new_w, new_h],
                    "area": new_w * new_h,
                    "iscrowd": ann.get("iscrowd", 0),
                    "segmentation": []
                })
                ann_id += 1

                # ================= SMART ZOOM LOGIC =================
            if has_box and len(tile_annotations) > 0:
                bboxes = np.array([a["bbox"] for a in tile_annotations])
                # Compute cluster area (only for boxes not at tile edges)
                x_min = bboxes[:, 0].min()
                y_min = bboxes[:, 1].min()
                x_max = (bboxes[:, 0] + bboxes[:, 2]).max()
                y_max = (bboxes[:, 1] + bboxes[:, 3]).max()

                # Add 20% context margin
                margin_x = (x_max - x_min) * context_margin
                margin_y = (y_max - y_min) * context_margin
                crop_x0 = max(0, x_min - margin_x)
                crop_y0 = max(0, y_min - margin_y)
                crop_x1 = min(tile_size, x_max + margin_x)
                crop_y1 = min(tile_size, y_max + margin_y)

                crop_w = crop_x1 - crop_x0
                crop_h = crop_y1 - crop_y0

                # Ensure minimum zoom area (avoid excessive zoom)
                min_zoom = 256
                if crop_w < min_zoom or crop_h < min_zoom:
                    cx = (crop_x0 + crop_x1) / 2
                    cy = (crop_y0 + crop_y1) / 2
                    crop_w = max(crop_w, min_zoom)
                    crop_h = max(crop_h, min_zoom)
                    crop_x0 = max(0, cx - crop_w / 2)
                    crop_y0 = max(0, cy - crop_h / 2)
                    crop_x1 = min(tile_size, crop_x0 + crop_w)
                    crop_y1 = min(tile_size, crop_y0 + crop_h)

                crop_box = (int(crop_x0), int(crop_y0),
                            int(crop_x1), int(crop_y1))
                crop_w, crop_h = crop_x1 - crop_x0, crop_y1 - crop_y0

                # Apply zoom crop
                zoom_tile = tile.crop(crop_box).resize(
                    (tile_size, tile_size), Image.LANCZOS)
                scale_x = tile_size / crop_w
                scale_y = tile_size / crop_h

                # Adjust annotations
                zoom_annotations = []
                for a in tile_annotations:
                    x, y, w_, h_ = a["bbox"]
                    # Skip boxes outside the crop area
                    if x + w_ < crop_x0 or y + h_ < crop_y0 or x > crop_x1 or y > crop_y1:
                        continue
                    nx = (x - crop_x0) * scale_x
                    ny = (y - crop_y0) * scale_y
                    nw = w_ * scale_x
                    nh = h_ * scale_y
                    zoom_annotations.append({
                        "id": a["id"],
                        "image_id": img_id,
                        "category_id": a["category_id"],
                        "bbox": [nx, ny, nw, nh],
                        "area": nw * nh,
                        "iscrowd": a.get("iscrowd", 0),
                        "segmentation": []
                    })

                tile = zoom_tile
                tile_annotations = zoom_annotations
            # =====================================================

            if has_box and len(tile_annotations) > 0:
                tile.save(os.path.join(img_out_dir, tile_filename))

                if visualize and vis_out_dir:
                    vis_tile = tile.copy()
                    draw = ImageDraw.Draw(vis_tile)
                    for a in tile_annotations:
                        draw.rectangle(
                            [a["bbox"][0], a["bbox"][1],
                             a["bbox"][0] + a["bbox"][2],
                             a["bbox"][1] + a["bbox"][3]],
                            outline="red", width=2
                        )
                    vis_tile.save(os.path.join(vis_out_dir, tile_filename))

                new_images.append({
                    "id": img_id,
                    "file_name": tile_filename,
                    "width": tile_size,
                    "height": tile_size
                })
                new_annotations.extend(tile_annotations)
                img_id += 1

    return new_images, new_annotations, ann_id, img_id


def slice_folder_images(
    image_dir,
    anno_path,
    output_dir,
    tile_size=640,
    overlap=0.0,
    visualize=False,
    data_type="train",
    black_threshold=15,
    area_threshold=15*15,
    edge_threshold=3, **kwargs
):
    os.makedirs(output_dir, exist_ok=True)

    with open(anno_path, "r") as f:
        coco = json.load(f)

    all_new_images, all_new_annotations = [], []
    next_ann_id, next_img_id = 1, 1
    skipped = 0

    for idx, img_info in enumerate(coco["images"], 1):
        image_path = os.path.join(image_dir, Path(img_info["file_name"]).name)
        if not os.path.exists(image_path):
            print(f"Skipping missing image: {image_path}")
            continue

        img = Image.open(image_path)
        if is_black_edge_image(img):
            skipped += 1
            print(f"Skipping {Path(image_path).name} (black edges)")
            continue

        anns = [a for a in coco["annotations"]
                if a["image_id"] == img_info["id"]]
        new_imgs, new_anns, next_ann_id, next_img_id = slice_single_image(
            image_path, anns, next_ann_id, next_img_id,
            tile_size=tile_size, overlap=overlap,
            visualize=visualize, output_dir=output_dir,
            data_type=data_type, black_threshold=black_threshold,
            area_threshold=area_threshold, edge_threshold=edge_threshold, context_margin=kwargs.get(
                "context_margin", 0.2)
        )

        all_new_images.extend(new_imgs)
        all_new_annotations.extend(new_anns)
        print(
            f"[{idx}/{len(coco['images'])}] {Path(image_path).name} → {len(new_imgs)} tiles", flush=True)

    new_coco = {
        "images": all_new_images,
        "annotations": all_new_annotations,
        "categories": coco["categories"]
    }
    out_json = os.path.join(output_dir, f"{data_type}/_annotations.coco.json")
    with open(out_json, "w") as f:
        json.dump(new_coco, f, indent=2)

    print(f"\n{data_type.upper()} DONE: {len(all_new_images)} tiles, {len(all_new_annotations)} boxes, {skipped} skipped due to black edges.")


if __name__ == "__main__":
    folder = "datasets/hail_1"

    for dt in ["train", "valid", "test"]:
        slice_folder_images(
            image_dir=f"{folder}/{dt}/",
            anno_path=f"{folder}/{dt}/_annotations.coco.json",
            output_dir=f"{folder}_cropped/",
            tile_size=640,
            overlap=0,
            visualize=True,
            data_type=dt,
            black_threshold=15,
            area_threshold=20*20,
            edge_threshold=5,
            context_margin=0.4  # zoom crop margin (0.4 = 40%
        )
