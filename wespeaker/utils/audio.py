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

"""Audio I/O and feature extraction utilities."""

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi


SAMPLE_RATE = 16000


def load_audio(wav_path):
    """Load audio file as mono 16kHz float32 numpy array.

    Args:
        wav_path: Path to audio file.

    Returns:
        np.ndarray of shape (n_samples,), float32.
    """
    signal, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        signal = torchaudio.functional.resample(signal, sr, SAMPLE_RATE)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    return signal.squeeze(0).numpy()


def compute_fbank(audio_chunk):
    """Compute 80-dim fbank features for an audio chunk.

    Args:
        audio_chunk: np.ndarray of shape (n_samples,), float32, 16kHz.

    Returns:
        np.ndarray of shape (T, 80), float32. No CMN applied.
    """
    wav = torch.from_numpy(audio_chunk).unsqueeze(0).float() * (1 << 15)
    feat = kaldi.fbank(
        wav,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=SAMPLE_RATE,
        window_type="hamming",
        use_energy=False)
    return feat.numpy()
