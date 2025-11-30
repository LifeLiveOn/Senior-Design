from pathlib import Path
import tempfile
import requests
from huggingface_hub import hf_hub_download

from rfdetr import RFDETRBase
<<<<<<< HEAD
from model_utils import run_rfdetr_inference, run_rfdetr_inference_tiled
=======
from .model_utils import run_rfdetr_inference, run_rfdetr_inference_tiled
>>>>>>> b0cfbc9867c79ad0ae326012b74cf45fd35c4104


import onnxruntime as ort
import torch
import numpy as np


class RFDETR_ONNXWrapper:
    """
    Clean ONNXRuntime wrapper that mimics torch.nn.Module forward().
    Works as a drop-in replacement inside RFDETRBase.
    """

    def __init__(self, onnx_path: str, use_cuda: bool = False):
        self.onnx_path = onnx_path

        # Providers priority
        if use_cuda:
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            self.providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            onnx_path,
            providers=self.providers
        )

        # ONNX model I/O names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        print(f"[INFO] Loaded ONNX model: {onnx_path}")
        print(f"[INFO] Providers: {self.session.get_providers()}")
        print(f"[INFO] Input name: {self.input_name}")
        print(f"[INFO] Output names: {self.output_names}")

    def __call__(self, images):
        """
        ONNX forward pass. Accepts torch tensors (B,C,H,W).
        Returns torch tensors so RFDETRBase postprocess works unchanged.
        """
        # Convert torch tensor → numpy
        if isinstance(images, torch.Tensor):
            np_input = images.detach().cpu().numpy()
        else:
            np_input = np.asarray(images)

        # Run ONNX inference
        outputs = self.session.run(
            None,
            {self.input_name: np_input}
        )

        # Convert outputs → torch (CPU)
        torch_outputs = [torch.tensor(o) for o in outputs]

        if len(torch_outputs) == 2:
            # Standard RF-DETR output shapes
            pred_logits, pred_boxes = torch_outputs
            return pred_logits, pred_boxes

        return torch_outputs

    def eval(self):
        """Mimic torch.nn.Module.eval() to keep RFDETRBase happy."""
        return self


class RFDETRService:
    _model = None
    _class_names = ["wind", "hail"]
    _model_type = "ONNX"

    @classmethod
    def load_model(cls):
        """Load ONNX RF-DETR model (no fallback)."""
        if cls._model is not None:
            return cls._model, cls._class_names, cls._model_type

        onnx_path = Path("exported_models/inference_model.onnx")

        # Download ONNX if missing
        if not onnx_path.exists():
            try:
                onnx_path = Path(
                    hf_hub_download(
                        repo_id="tnkchaseme/rfdetr-roof-assessment",
                        filename="inference_model.onnx",
                    )
                )
            except Exception:
                raise FileNotFoundError(
                    "ONNX model not found locally or on HuggingFace."
                )

        core_onnx = RFDETR_ONNXWrapper(str(onnx_path))

        # Wrap inside RFDETRBase interface
<<<<<<< HEAD
        model = RFDETRBase(num_classes=2, device="cpu")  # ONNX always CPU
=======
        model = RFDETRBase(num_classes=2, device="cpu",
                           pretrain_weights=None)  # ONNX always CPU
>>>>>>> b0cfbc9867c79ad0ae326012b74cf45fd35c4104
        model.model.model = core_onnx

        cls._model = model
        return cls._model, cls._class_names, cls._model_type

    @staticmethod
    def _download_image(url: str) -> str:
        """Download URL → temp file path."""
        resp = requests.get(url, stream=True)
        resp.raise_for_status()

        suffix = Path(url).suffix or ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(resp.content)
        tmp.close()
        return tmp.name

    @classmethod
    def predict(cls, image_path_or_url: str, mode="normal", threshold=0.4, tile_size=560):
        """Run ONNX inference (normal or tiled)."""
        model, class_names, _ = cls.load_model()

        # Ensure local file (download if needed)
        image_path = (
            cls._download_image(image_path_or_url)
            if image_path_or_url.startswith("http")
            else image_path_or_url
        )

        if mode == "normal":
            detections, pred_path = run_rfdetr_inference(
                model=model,
                image_path=image_path,
                class_names=class_names,
<<<<<<< HEAD
                save_dir="results/normal",
=======
                # save_dir="results/normal",
>>>>>>> b0cfbc9867c79ad0ae326012b74cf45fd35c4104
                threshold=threshold,
            )
        else:
            detections, pred_path = run_rfdetr_inference_tiled(
                model=model,
                image_path=image_path,
                class_names=class_names,
                tile_size=tile_size,
                overlap=0.4,
                conf_thres=threshold,
<<<<<<< HEAD
                save_dir="results/tiled",
=======
                # save_dir="results/tiled",
>>>>>>> b0cfbc9867c79ad0ae326012b74cf45fd35c4104
            )

        return detections, pred_path
