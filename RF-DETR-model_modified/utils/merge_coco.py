import os
import json
from pathlib import Path
from typing import List, Union


def merge_coco_datasets(
    json_list: Union[List[str], str],
    dataset_list: Union[List[str], str],
    output_json: str,
):
    """
    Merge multiple COCO JSON files into a single COCO JSON.

    Args:
      json_list: list of json annotation paths (or a single path). The first
        JSON is treated as the base (its category definitions are used).
      dataset_list: list of dataset path prefixes corresponding to each json
        (or a single path). Each image's `file_name` will be prefixed with the
        corresponding dataset path.
      output_json: path to write the merged COCO json.

    Notes/assumptions:
      - The first JSON provides the canonical `categories` list. Additional
        JSONs are assumed to use compatible categories.
      - The number of jsons and dataset paths must match.
    """

    # Normalize inputs to lists
    if isinstance(json_list, (str, Path)):
        json_list = [str(json_list)]
    if isinstance(dataset_list, (str, Path)):
        dataset_list = [str(dataset_list)]

    if len(json_list) == 0:
        raise ValueError("At least one json file must be provided")
    if len(json_list) != len(dataset_list):
        raise ValueError(
            "The number of json files and dataset paths must match")

    # --- Load the first (base) dataset ---
    with open(json_list[0], "r") as f:
        base = json.load(f)

    merged = {
        "images": [],
        "annotations": [],
        "categories": base.get("categories", []),  # assume same categories
    }

    # --- Track max IDs using base dataset ---
    max_img_id = max((img.get("id", 0)
                     for img in base.get("images", [])), default=0)
    max_ann_id = max((ann.get("id", 0)
                     for ann in base.get("annotations", [])), default=0)

    # --- Copy base dataset images and annotations, prefixing file_name ---
    for img in base.get("images", []):
        new_img = img.copy()  # copy the image dict
        # prefix and use / for windows compatibility
        new_img["file_name"] = f"{dataset_list[0]}/{img['file_name']}".replace(
            "\\", "/")
        merged["images"].append(new_img)
    # use the default annotations from base
    merged["annotations"].extend(base.get("annotations", []))

    # --- Merge remaining datasets one-by-one ---
    for idx in range(1, len(json_list)):
        with open(json_list[idx], "r") as f:
            coco = json.load(f)

        dataset_prefix = dataset_list[idx]

        # Map from original file_name to new image id
        img_name_to_newid = {}

        for img in coco.get("images", []):
            max_img_id += 1
            new_img = img.copy()
            new_img["id"] = max_img_id
            new_img["file_name"] = f"{dataset_prefix}/{new_img['file_name']}".replace(
                "\\", "/")
            # map old name to new id so that annotations can be relinked
            img_name_to_newid[img.get("file_name")] = new_img["id"]
            merged["images"].append(new_img)

        # Relink and append annotations
        for ann in coco.get("annotations", []):
            # find corresponding old image by id in coco's image list
            old_img = next((i for i in coco.get("images", [])
                           if i.get("id") == ann.get("image_id")), None)
            # same as above is for i in coco.get("images", []) if i.get("id") == ann.get("image_id"): old_img = i; break
            if not old_img:
                continue
            old_name = old_img.get("file_name")
            new_img_id = img_name_to_newid.get(
                old_name)  # get the new image id
            if not new_img_id:
                continue

            max_ann_id += 1
            new_ann = ann.copy()
            new_ann["id"] = max_ann_id
            new_ann["image_id"] = new_img_id
            merged["annotations"].append(new_ann)

    # --- Save final merged dataset ---
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged dataset saved to {output_json}")
    print(f"  Total images: {len(merged['images'])}")
    print(f"  Total annotations: {len(merged['annotations'])}")
    print(f"  Categories: {len(merged['categories'])}")


if __name__ == "__main__":
    mode1 = ["train", "valid", "test"]
    prefix = "datasets"
    json_name = "_annotations.coco.json"
    try:
        for m in mode1:
            list_json = [f"{prefix}/hail_1_cropped/{m}/{json_name}",
                         f"{prefix}/hail_2/{m}/{json_name}", f"{prefix}/hail_3/{m}/{json_name}", f"{prefix}/wind_1/{m}/{json_name}"]
            datasets_path = [f"{prefix}/hail_1_cropped/{m}",
                             f"{prefix}/hail_2/{m}", f"{prefix}/hail_3/{m}", f"{prefix}/wind_1/{m}"]
            output = f"merged_annotations/{m}/_annotations.coco.json"
            merge_coco_datasets(list_json, datasets_path, output)
    except Exception as e:
        raise ValueError("No json files or dataset paths provided.") from e
