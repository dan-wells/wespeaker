# Copyright (c) 2025 Dan Wells
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified speaker embedding model interface."""

import numpy as np
import onnxruntime as ort
import torch


def init_session(source, device):
    """Initialize an ONNX Runtime inference session.

    Args:
        source: Path to .onnx model file.
        device: 'cpu' or 'cuda'.

    Returns:
        ort.InferenceSession.
    """
    if device == "cpu":
        providers = ["CPUExecutionProvider"]
    elif device == "cuda":
        providers = ["CUDAExecutionProvider"]
    else:
        raise ValueError("Unknown device: {}".format(device))

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 2
    session = ort.InferenceSession(source,
                                   sess_options=opts,
                                   providers=providers)
    return session


class EmbeddingModel:
    """Unified interface for ONNX and PyTorch speaker embedding models.

    Supports both ONNX Runtime and PyTorch backends, with automatic
    detection based on the model source path.

    Args:
        source: Path to .onnx file or PyTorch model directory.
        device: Inference device, 'cpu' or 'cuda'.
        backend: 'onnx', 'pytorch', or None for auto-detection.
    """

    def __init__(self, source, device="cuda", backend=None):
        if backend is None:
            backend = "onnx" if source.endswith(".onnx") else "pytorch"
        self.backend = backend
        self.device = device

        if backend == "onnx":
            self.session = init_session(source, device)
        elif backend == "pytorch":
            from wespeaker.cli.speaker import load_model_pt
            self.model = load_model_pt(source)
            self.model.to(torch.device(device))
        else:
            raise ValueError("Unknown backend: {}".format(backend))

    def extract(self, fbank):
        """Extract a single embedding from fbank features.

        Args:
            fbank: np.ndarray of shape (T, 80), already CMN'd if desired.

        Returns:
            np.ndarray of shape (emb_dim,).
        """
        if self.backend == "onnx":
            feats = fbank[np.newaxis, :, :].astype(np.float32)
            emb = self.session.run(
                input_feed={"feats": feats},
                output_names=["embs"])[0].squeeze()
            return emb
        else:
            feats = torch.from_numpy(fbank).unsqueeze(0).float().to(
                torch.device(self.device))
            with torch.no_grad():
                outputs = self.model(feats)
                emb = outputs[-1] if isinstance(outputs, tuple) else outputs
            return emb.squeeze().cpu().numpy()
