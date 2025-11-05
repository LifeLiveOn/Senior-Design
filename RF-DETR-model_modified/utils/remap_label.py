import json
from pathlib import Path

# map the category ids in a COCO annotation file according to a a new ID mapping


def remap_coco_category_ids(file_in: str, file_out: str, id_mapping: dict):
    """
    Remap category IDs in a COCO JSON annotation file.

    Args:
      file_in: path to input COCO JSON file.
      file_out: path to output COCO JSON file with remapped category IDs.
      id_mapping: dictionary mapping old category IDs to new category IDs.  
    """
    with open(file_in, "r") as f:
        coco_data = json.load(f)

    # Remap category IDs in annotations
    for ann in coco_data.get("annotations", []):
        old_id = ann.get("category_id")
        if old_id in id_mapping:
            ann["category_id"] = id_mapping[old_id]
        else:
            raise ValueError(f"Old category ID {old_id} not found in mapping.")

    # Write the modified data to the output file
    with open(file_out, "w") as f:
        json.dump(coco_data, f)


if __name__ == "__main__":
    mapping = {
        1: 0
    }
    folder_path = "datasets/wind_1"
    mode = ["train", "valid", "test"]
    for m in mode:
        file_in = f"{folder_path}/{m}/_annotations.coco.json"
        file_out = f"{folder_path}/{m}/_annotations.coco.json"
        remap_coco_category_ids(file_in, file_out, mapping)
