"""
RF-DETR Built-in ONNX Export Script
-----------------------------------
Exports RF-DETR to ONNX using the built-in export() method.
Defines RFDETR_ONNXWrapper — a thin shim that behaves like
`model.model.model` so the RFDETRBase pipeline still works.
"""

import torch
from pathlib import Path
from rfdetr import RFDETRBase


# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
checkpoint_path = "merged_annotations/output/checkpoint.pth"
output_dir = "exported_models"
Path(output_dir).mkdir(parents=True, exist_ok=True)

class_names = ["wind", "hail"]
device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------
# ONNX Wrapper (keeps RFDETRBase predict() working)
# -----------------------------------------------------------
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

        print(f"[INFO] ONNX model loaded: {onnx_path}")
        print(f"[INFO] Detected input name: {self.input_name}")
        print(f"[INFO] Detected outputs: {self.output_names}")

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
        torch_outputs = [torch.from_numpy(o) for o in outputs]

        # Handle both tuple-style and dict-style outputs
        if len(torch_outputs) == 2:
            pred_logits, pred_boxes = torch_outputs
            return pred_logits, pred_boxes
        else:
            return torch_outputs

    def eval(self):
        """Mimic torch.nn.Module.eval()"""
        return self


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
    model.export(output_dir=str(output_dir), opset_version=17)

    print(
        f"[INFO] Successfully exported ONNX model to: {output_dir}/inference_model.onnx")
    print("[INFO] RFDETR_ONNXWrapper ready for integration.")


if __name__ == "__main__":
    export_rfdetr_to_onnx()
