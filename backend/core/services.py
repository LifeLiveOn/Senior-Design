from pathlib import Path
import tempfile
import requests
from huggingface_hub import hf_hub_download

from rfdetr import RFDETRBase
from .model_utils import run_rfdetr_inference, run_rfdetr_inference_tiled
import os

import onnxruntime as ort
import torch
import numpy as np


class RFDETR_ONNXWrapper:
    """
    High-performance ONNXRuntime wrapper for CPU.
    Minimizes memory copy, uses optimized session options,
    and avoids unnecessary torch conversions.
    """

    def __init__(self, onnx_path: str):
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(
            1, os.cpu_count() - 1)  # keep 1 core free for OS
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )

        # Cache names
        info = self.session.get_inputs()[0]
        self.input_name = info.name
        self.input_shape = info.shape
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Preallocate reusable numpy buffer
        # shape may include dynamic dims => allocate on first run
        self._input_buf = None

    def __call__(self, images):
        """
        Accepts torch tensor (B,C,H,W)
        Returns torch tensors (to keep RFDETRBase working)
        """
        if isinstance(images, torch.Tensor):
            arr = images.detach().cpu().numpy()
        else:
            arr = np.asarray(images)

        if self._input_buf is None or self._input_buf.shape != arr.shape:
            self._input_buf = np.empty_like(arr)

        np.copyto(self._input_buf, arr)

        outputs = self.session.run(
            self.output_names,
            {self.input_name: self._input_buf}
        )

        # convert to torch only once
        t0 = torch.from_numpy(outputs[0])
        t1 = torch.from_numpy(outputs[1])
        return t0, t1

    def eval(self):
        return self


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

        # Wrap only once
        core = RFDETR_ONNXWrapper(str(onnx_path))

        model = RFDETRBase(num_classes=2, device="cpu", pretrain_weights=None)
        model.model.model = core  # minimal patching

        cls._model = model
        return cls._model, cls._class_names, cls._model_type

    @staticmethod
    def _download_image(url: str) -> str:
        """Download URL → temp file path."""
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
