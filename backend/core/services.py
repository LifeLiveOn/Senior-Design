import importlib
from pathlib import Path
import tempfile
import requests
from huggingface_hub import hf_hub_download

from .model_utils import run_rfdetr_inference, run_rfdetr_inference_tiled
import os

import onnxruntime as ort
import torch
import numpy as np

import torch.nn as nn
import sys
from pathlib import Path

# current file: backend/backend/core/xxx.py
BASE_DIR = Path(__file__).resolve().parent

RFDETR_BASE_FILE = (
    BASE_DIR
    / "../../RF-DETR-model_modified/rf-detr-modifications/rfdetr/__init__.py"
).resolve()

spec = importlib.util.spec_from_file_location("rfdetr", RFDETR_BASE_FILE)
rf_detr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf_detr_module)

RFDETRBase = rf_detr_module.RFDETRBase

device = "cuda" if torch.cuda.is_available() else "cpu"


class RFDETR_ONNXWrapper:
    """
    ONNXRuntime inference backend.
    Inference-only. NOT a torch module.
    """

    def __init__(self, onnx_path: str, device: str):
        self.device = device

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        info = self.session.get_inputs()[0]
        self.input_name = info.name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self._input_buf = None

    def __call__(self, images: torch.Tensor):
        # Torch → NumPy
        arr = images.detach().cpu().numpy()

        if self._input_buf is None or self._input_buf.shape != arr.shape:
            self._input_buf = np.empty_like(arr)

        np.copyto(self._input_buf, arr)

        outputs = self.session.run(
            self.output_names,
            {self.input_name: self._input_buf}
        )

        # NumPy → Torch
        t0 = torch.from_numpy(outputs[0]).to(self.device, non_blocking=True)
        t1 = torch.from_numpy(outputs[1]).to(self.device, non_blocking=True)

        return t0, t1



class RFDETRService:
    _model = None
    _class_names = ["wind", "hail"]
    _model_type = "ONNX"

    @classmethod
    def load_model(cls):
        if cls._model is not None:
            return cls._model, cls._class_names, cls._model_type

        onnx_path = Path("exported_models/inference_model.onnx")

        if not onnx_path.exists():
            onnx_path = Path(
                hf_hub_download(
                    repo_id="tnkchaseme/rfdetr-roof-assessment",
                    filename="inference_model.onnx"
                )
            )

        model = RFDETRBase(
            num_classes=2,
            device=device,
            pretrain_weights=None
        )

        onnx_core = RFDETR_ONNXWrapper(str(onnx_path), device)

        def onnx_forward(images):
            return onnx_core(images)

        model.model.model.forward = onnx_forward

        cls._model = model
        return cls._model, cls._class_names, cls._model_type


    @staticmethod
    def _download_image(url: str) -> str:
        """Download URL to temp file path."""
        resp = requests.get(url, timeout=5)
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
                # save_dir="results/normal",
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
                # save_dir="results/tiled",
            )

        return detections, pred_path