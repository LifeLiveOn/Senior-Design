"""
Streamlit Application for RF-DETR Inference
-------------------------------------------
This app allows users to upload an image and perform object detection
using an RF-DETR model. It supports two inference modes:
    1. Normal inference – run directly on the full image.
    2. Tiled inference – divides the image into smaller overlapping tiles.

If the ONNX model exists, it will be loaded for faster inference.
Otherwise, the PyTorch checkpoint will be used.
"""

import streamlit as st
from pathlib import Path
from PIL import Image
import tempfile
import torch
from huggingface_hub import hf_hub_download
# Import inference utilities
from main import run_rfdetr_inference, run_rfdetr_inference_tiled
from rfdetr import RFDETRBase


# -----------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------
st.set_page_config(page_title="RF-DETR Inference", layout="wide")

st.title("RF-DETR Damage Detection")
st.write("Upload an image and choose between **Normal** or **Tiled** inference modes.")


# -----------------------------------------------------------
# Model Loading (cached to prevent reloading on every run)
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    """
    Try to load ONNX model first.
    If unavailable or invalid, fall back to PyTorch RF-DETR model.
    """
    from export_to_onnx import RFDETR_ONNXWrapper

    class_names = ["wind", "hail"]
    onnx_path = Path("exported_models/inference_model.onnx")
    if not onnx_path.exists():
        try:
            # Download ONNX model from Hugging Face if not present
            onnx_path = hf_hub_download(
                repo_id="tnkchaseme/rfdetr-roof-assessment",
                filename="inference_model.onnx",
            )

            onnx_path = Path(onnx_path)
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
            core_onnx = RFDETR_ONNXWrapper(str(onnx_path))
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


model, class_names, model_type = load_model()


# -----------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------
st.sidebar.header("Inference Settings")

st.sidebar.markdown(f"**Model Type:** `{model_type}`")

infer_mode = st.sidebar.radio("Select inference mode:", ["Normal", "Tiled"])

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Minimum confidence required for a detection to be considered valid."
)

tile_size_option = st.sidebar.selectbox(
    "Tile size (for tiled mode only):",
    ["tiny", "small", "normal", "large"],
    index=2
)

tile_size_map = {
    "tiny": 224,
    "small": 448,
    "normal": 560,
    "large": 616,
}
tile_size = tile_size_map[tile_size_option]


# -----------------------------------------------------------
# File Upload
# -----------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)


# -----------------------------------------------------------
# Image Handling and Inference
# -----------------------------------------------------------
if uploaded_file is not None:
    temp_dir = tempfile.mkdtemp()
    img_path = Path(temp_dir) / uploaded_file.name
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(str(img_path), caption="Uploaded Image", width="content")

    if st.button("Run Detection"):
        with st.spinner("Running inference... Please wait..."):
            if infer_mode == "Normal":
                detections, pred_path = run_rfdetr_inference(
                    model=model,
                    image_path=str(img_path),
                    class_names=class_names,
                    save_dir="streamlit_results/normal", threshold=conf_threshold
                )
            else:
                detections, pred_path = run_rfdetr_inference_tiled(
                    model=model,
                    image_path=str(img_path),
                    class_names=class_names,
                    tile_size=tile_size,
                    overlap=0.4,
                    conf_thres=conf_threshold,
                    save_dir="streamlit_results/tiled",
                )

        if pred_path and Path(pred_path).exists():
            st.success("Inference completed successfully.")
            st.image(str(pred_path), caption="Detection Result",
                     width="content")
        else:
            st.warning("No detections found or failed to save output.")
else:
    st.info("Please upload an image to start inference.")
