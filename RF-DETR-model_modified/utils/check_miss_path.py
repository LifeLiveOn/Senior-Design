import numpy as np
from collections import Counter, defaultdict
import json
from collections import defaultdict
from pathlib import Path

with open("merged_annotations/train/_annotations.coco.json") as f:
    data = json.load(f)

img_to_anns = defaultdict(list)
for ann in data["annotations"]:
    img_to_anns[ann["image_id"]].append(ann)

num_with = sum(1 for img in data["images"] if img_to_anns[img["id"]])
num_total = len(data["images"])
print(f"{num_with}/{num_total} images contain annotations ({num_with/num_total:.2%})")

img_ids = {img["id"] for img in data["images"]}
ann_ids = {ann["image_id"] for ann in data["annotations"]}
missing = [aid for aid in ann_ids if aid not in img_ids]
print(
    f"Annotations pointing to missing image_ids: {len(missing)} / {len(ann_ids)}")


missing = [img["file_name"]
           for img in data["images"] if not Path(img["file_name"]).exists()]

print(f"Missing {len(missing)} / {len(data['images'])} image files")
print("Example missing paths:", missing[:5])


with open("merged_annotations/train/_annotations.coco.json") as f:
    data = json.load(f)

counts = Counter(ann["category_id"] for ann in data["annotations"])
print("Category distribution:", counts)


hail1 = list(Path("datasets/hail_1_cropped/train").glob("*.jpg"))
hail2 = list(Path("datasets/hail_2/train").glob("*.jpg"))
print("hail_1:", len(hail1), "hail_2:", len(hail2))


areas = [a["bbox"][2]*a["bbox"][3] for a in data["annotations"]]
print("Median bbox area:", np.median(areas))
