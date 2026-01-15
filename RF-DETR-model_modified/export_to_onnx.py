"""
RF-DETR Built-in ONNX Export Script
-----------------------------------
Exports RF-DETR to ONNX using the built-in export() method.
Defines RFDETR_ONNXWrapper — a thin shim that behaves like
`model.model.model` so the RFDETRBase pipeline still works.
"""
from rfdetr import RFDETRBase
from pathlib import Path
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())


# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
checkpoint_path = "merged_annotations/output/checkpoint.pth"
output_dir = "exported_models"
Path(output_dir).mkdir(parents=True, exist_ok=True)

class_names = ["wind", "hail"]
device = "cuda"

# -----------------------------------------------------------
# Export Function
# -----------------------------------------------------------


def export_rfdetr_to_onnx():
    print("[INFO] Loading RF-DETR model...")
    model = RFDETRBase(
        num_classes=len(class_names),
        pretrain_weights=checkpoint_path,
        device=device
    )
    model.optimize_for_inference()

    print("[INFO] Exporting RF-DETR model to ONNX using built-in export()...")
    model.export(output_dir=f"{output_dir}_{device}", opset_version=17)

    print(
        f"[INFO] Successfully exported ONNX model to: {output_dir}_{device}/inference_model.onnx")
    print("[INFO] RFDETR_ONNXWrapper ready for integration.")


if __name__ == "__main__":
    export_rfdetr_to_onnx()
