# RF-DETR Roof Damage Detection

This project uses the `'RFDETR_base'` model from Roboflow to train a vision model capable of detecting hail damage and wind damage on roofs.

---

## Installation

### Clone the repository

git clone https://github.com/LifeLiveOn/Modification-to-RF-DETR-model

cd Modification-to-RF-DETR-model

### Install `uv`

**On Windows:**
pip install uv

**On Linux:**
curl -LsSf https://astral.sh/uv/install.sh | sh

### Set up a Python 3.10 virtual environment

uv venv --python 3.10

### Install dependencies

uv pip install -r requirements.txt

### Install local RF-DETR modifications

- cd rf-detr-modifications
- uv pip install -e .

---

## Running the Demo

From the project root, launch the Streamlit demo:
streamlit run app.py

---

## Utilities Overview (`utils/`)

### Script: `check_miss_path.py`

**Description:** Ensures that all images referenced in `'annotation.json'` exist in the dataset directories.

### Script: `merge_coco.py`

**Description:** Merges multiple COCO-format JSON files into one unified file, preserving source references.

### Script: `remap_label.py`

**Description:** Remaps category IDs to maintain consistent `'category_id'` values across annotation files.

### Script: `tiles_images.py`

**Description:** Preprocesses images into tiles of specified sizes for training purposes.

---

## Main CLI Script

`main.py` is used for training, testing, and inference through the command line.

### Example Usage

**Inference (normal mode):**
python main.py --mode test --infer_mode normal --tile_size small

**Inference (tiled mode):**
python main.py --mode test --infer_mode tiled --tile_size small --path datasets/hail_1/test

**Training:**
python main.py --mode train

**Test on wind damage dataset:**
python main.py --mode test --infer_mode normal --path datasets/wind_1/test

---

## Exporting to ONNX

**Note:** ONNX export only works with Python 3.10 due to `'onnxsim'` incompatibility with Python 3.12.

### Install ONNX tools:

uv pip install onnx onnxsim onnxscript

### Export the model:

python export_to_onnx.py

---

## Project Summary

- **Purpose:** Detect hail and wind roof damage using the RFDETR model.
- **Framework:** Based on RFDETR (DETR variant) with local modifications.
- **Training Pipeline:** Supports normal and tiled inference, data preprocessing utilities, and ONNX export.
- **Environment:** Requires Python 3.10 for full compatibility.

---

## Repository Structure

Modification-to-RF-DETR-model/
├── app.py
├── main.py
├── requirements.txt
├── rf-detr-modifications/
├── utils/
│ ├── check_miss_path.py
│ ├── merge_coco.py
│ ├── remap_label.py
│ └── tiles_images.py
└── datasets/
├── hail_1/
└── wind_1/

---

**Author:** LifeLiveOn  
**Model Base:** RFDETR (Roboflow)  
**License:** MIT
