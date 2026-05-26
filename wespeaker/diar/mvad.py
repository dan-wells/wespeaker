"""MVAD_V2 inference and post-processing for multi-class VAD.

Provides model loading, log-mel feature extraction at 48kHz, frame-level
inference (silence/single-speaker/overlap), and post-filters for use in
diarization pipelines.
"""

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

from math import gcd

import numpy as np
import scipy.signal
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

V2_SAMPLE_RATE = 48000
V2_HOP_SAMPLES = 480
V2_ANALYSIS_WINDOW_SAMPLES = 1200
V2_N_FFT = 2048
V2_N_MELS = 40
V2_FMIN = 80
V2_FMAX = 8000
V2_NUM_CLASSES = 3

# Class labels
CLASS_SILENCE = 0
CLASS_SINGLE = 1
CLASS_OVERLAP = 2


# ---------------------------------------------------------------------------
# Mel filterbank (numpy-only, no librosa)
# ---------------------------------------------------------------------------

def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _create_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """Create a mel filterbank matrix.

    Args:
        sr: Sample rate in Hz.
        n_fft: FFT size.
        n_mels: Number of mel bands.
        fmin: Minimum frequency in Hz.
        fmax: Maximum frequency in Hz.

    Returns:
        np.ndarray of shape (n_mels, n_fft // 2 + 1).
    """
    n_freqs = n_fft // 2 + 1
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        for j in range(left, center):
            if center > left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


_MEL_FB = _create_mel_filterbank(
    V2_SAMPLE_RATE, V2_N_FFT, V2_N_MELS, V2_FMIN, V2_FMAX)


# ---------------------------------------------------------------------------
# Log-mel feature extraction
# ---------------------------------------------------------------------------

def _compute_log_mel(audio, sr):
    """Compute log-mel spectrogram at 48kHz / 10ms hop.

    Resamples internally if input sr != 48kHz.

    Args:
        audio: np.ndarray of shape (n_samples,), float32.
        sr: Input sample rate.

    Returns:
        np.ndarray of shape (T, V2_N_MELS), float32.
    """
    if sr != V2_SAMPLE_RATE:
        up = V2_SAMPLE_RATE // gcd(sr, V2_SAMPLE_RATE)
        down = sr // gcd(sr, V2_SAMPLE_RATE)
        audio = scipy.signal.resample_poly(audio, up, down).astype(np.float32)

    # STFT with hann window
    _, _, Zxx = scipy.signal.stft(
        audio,
        fs=V2_SAMPLE_RATE,
        window="hann",
        nperseg=V2_ANALYSIS_WINDOW_SAMPLES,
        noverlap=V2_ANALYSIS_WINDOW_SAMPLES - V2_HOP_SAMPLES,
        nfft=V2_N_FFT,
        boundary=None,
        padded=False)

    # Power spectrum
    power = np.abs(Zxx) ** 2

    # Apply mel filterbank
    mel_spec = _MEL_FB @ power  # (n_mels, T)

    # Log compression
    log_mel = np.log(np.maximum(mel_spec, 1e-10))

    return log_mel.T.astype(np.float32)  # (T, n_mels)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class DSConvBlock(nn.Module):
    """Depthwise-separable 1D convolution block.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        kernel_size: Kernel size for depthwise conv.
        dropout: Dropout probability.
    """

    def __init__(self, in_ch, out_ch, kernel_size, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.dw_conv = nn.Conv1d(
            in_ch, in_ch, kernel_size, padding=padding, groups=in_ch,
            bias=False)
        self.dw_bn = nn.BatchNorm1d(in_ch)
        self.pw_conv = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.pw_bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.has_residual = (in_ch == out_ch)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Tensor of shape (B, C, T).

        Returns:
            Tensor of shape (B, out_ch, T).
        """
        residual = x
        out = self.dw_conv(x)
        out = self.dw_bn(out)
        out = torch.relu(out)
        out = self.pw_conv(out)
        out = self.pw_bn(out)
        out = torch.relu(out)
        out = self.dropout(out)
        if self.has_residual:
            out = out + residual
        return out


class MVAD_V2(nn.Module):
    """Lightweight DS-Conv1D model for multi-class VAD.

    Classifies each frame as silence (0), single-speaker (1), or
    overlap (2).

    Args:
        n_mels: Number of input mel features.
        hidden_ch: Hidden channel dimension.
        kernel_size: Depthwise conv kernel size.
        n_blocks: Number of DS-Conv blocks.
        num_classes: Number of output classes.
        dropout: Dropout probability.
    """

    def __init__(self, n_mels=40, hidden_ch=64, kernel_size=15,
                 n_blocks=5, num_classes=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_mels, hidden_ch, 1, bias=False),
            nn.BatchNorm1d(hidden_ch),
        )
        blocks = []
        for i in range(n_blocks):
            out_ch = hidden_ch if i < n_blocks - 1 else hidden_ch // 2
            blocks.append(DSConvBlock(hidden_ch, out_ch, kernel_size, dropout))
            hidden_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.output_head = nn.Conv1d(hidden_ch, num_classes, 1)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Tensor of shape (B, n_mels, T).

        Returns:
            Tensor of shape (B, num_classes, T) -- logits.
        """
        out = self.input_proj(x)
        out = torch.relu(out)
        for block in self.blocks:
            out = block(out)
        return self.output_head(out)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_mvad_model(model_path, device=None):
    """Load MVAD_V2 checkpoint.

    Args:
        model_path: Path to .pt checkpoint file.
        device: Torch device string. If None, uses cuda if available.

    Returns:
        Tuple of (model, feat_mean, feat_std, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(model_path, map_location="cpu")
    config = ckpt["config"]

    model = MVAD_V2(
        n_mels=config["n_mels"],
        hidden_ch=config["hidden_ch"],
        kernel_size=config["kernel_size"],
        n_blocks=config["n_blocks"],
        num_classes=config["num_classes"],
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    feat_mean = np.asarray(ckpt["standardisation"]["mean"], dtype=np.float32)
    feat_std = np.asarray(ckpt["standardisation"]["std"], dtype=np.float32)

    return model, feat_mean, feat_std, device


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def mvad_predict(audio, sr, model, feat_mean, feat_std, device):
    """Run MVAD_V2 inference on an audio signal.

    Args:
        audio: np.ndarray of shape (n_samples,), float32.
        sr: Sample rate of audio.
        model: MVAD_V2 model instance (eval mode).
        feat_mean: np.ndarray of shape (n_mels,).
        feat_std: np.ndarray of shape (n_mels,).
        device: Torch device string.

    Returns:
        np.ndarray of shape (T,), int32, values in {0, 1, 2}.
          Frame rate is 100 Hz.
    """
    log_mel = _compute_log_mel(audio, sr)  # (T, n_mels)

    # Standardise
    log_mel = (log_mel - feat_mean) / feat_std

    # Model expects (B, n_mels, T)
    x = torch.from_numpy(log_mel.T[np.newaxis, :, :]).float().to(device)
    logits = model(x)  # (1, num_classes, T)
    labels = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

    return labels


# ---------------------------------------------------------------------------
# Post-filters
# ---------------------------------------------------------------------------

def sliding_window_filter(labels, hop_sec=0.01, window_sec=1.0,
                          overlap_thresh=0.7, single_thresh=0.3):
    """Apply centred sliding-window majority filter.

    For each frame, examine a window of surrounding frames. If overlap
    fraction >= overlap_thresh, label as overlap. Else if single fraction
    >= single_thresh, label as single. Else silence.

    Args:
        labels: np.ndarray of shape (T,), int32.
        hop_sec: Frame hop in seconds (default 0.01 for 100Hz).
        window_sec: Window duration in seconds.
        overlap_thresh: Fraction threshold for overlap decision.
        single_thresh: Fraction threshold for single-speaker decision.

    Returns:
        np.ndarray of shape (T,), int32, filtered labels.
    """
    half_win = int(round(window_sec / hop_sec / 2))
    n_frames = len(labels)
    filtered = np.zeros(n_frames, dtype=np.int32)

    for i in range(n_frames):
        start = max(0, i - half_win)
        end = min(n_frames, i + half_win + 1)
        window = labels[start:end]
        n_window = len(window)

        n_overlap = np.sum(window == CLASS_OVERLAP)
        n_single = np.sum(window == CLASS_SINGLE)

        if n_overlap / n_window >= overlap_thresh:
            filtered[i] = CLASS_OVERLAP
        elif n_single / n_window >= single_thresh:
            filtered[i] = CLASS_SINGLE
        else:
            filtered[i] = CLASS_SILENCE

    return filtered


def hold_filter(labels, hop_sec=0.01, hold_ms=750):
    """Apply causal hold/sustain filter.

    Once a class is active, it persists for at least hold_ms even if
    subsequent frames disagree. This smooths short dropouts.

    Args:
        labels: np.ndarray of shape (T,), int32.
        hop_sec: Frame hop in seconds.
        hold_ms: Hold duration in milliseconds.

    Returns:
        np.ndarray of shape (T,), int32, filtered labels.
    """
    hold_frames = int(round(hold_ms / 1000.0 / hop_sec))
    n_frames = len(labels)
    filtered = np.zeros(n_frames, dtype=np.int32)

    current_class = labels[0] if n_frames > 0 else CLASS_SILENCE
    hold_counter = hold_frames

    for i in range(n_frames):
        if labels[i] == current_class:
            hold_counter = hold_frames
            filtered[i] = current_class
        elif hold_counter > 0:
            hold_counter -= 1
            filtered[i] = current_class
        else:
            current_class = labels[i]
            hold_counter = hold_frames
            filtered[i] = current_class

    return filtered


# ---------------------------------------------------------------------------
# High-level functions
# ---------------------------------------------------------------------------

def mvad_predict_filtered(audio, sr, model, feat_mean, feat_std, device):
    """Run MVAD inference with post-filters applied.

    Convenience wrapper that chains mvad_predict -> sliding_window_filter
    -> hold_filter.

    Args:
        audio: np.ndarray of shape (n_samples,), float32.
        sr: Sample rate of audio.
        model: MVAD_V2 model instance.
        feat_mean: np.ndarray of shape (n_mels,).
        feat_std: np.ndarray of shape (n_mels,).
        device: Torch device string.

    Returns:
        np.ndarray of shape (T,), int32, post-filtered frame labels.
    """
    labels = mvad_predict(audio, sr, model, feat_mean, feat_std, device)
    labels = sliding_window_filter(labels)
    labels = hold_filter(labels)
    return labels


def mvad_check_overlap(audio_chunk, sr, model, feat_mean, feat_std, device,
                       threshold=0.5):
    """Check whether an audio chunk should be skipped due to overlap/silence.

    Runs MVAD inference + post-filters and returns True if the fraction
    of single-speaker frames is below threshold (i.e. the chunk is
    unreliable for embedding extraction).

    Args:
        audio_chunk: np.ndarray of shape (n_samples,), float32.
        sr: Sample rate of audio_chunk.
        model: MVAD_V2 model instance.
        feat_mean: np.ndarray of shape (n_mels,).
        feat_std: np.ndarray of shape (n_mels,).
        device: Torch device string.
        threshold: Minimum fraction of single-speaker frames required
          to process this chunk. Default 0.5.

    Returns:
        True if the chunk should be skipped (too few single-speaker
          frames), False if it should be processed.
    """
    labels = mvad_predict_filtered(
        audio_chunk, sr, model, feat_mean, feat_std, device)

    n_total = len(labels)
    if n_total == 0:
        return True

    n_single = np.sum(labels == CLASS_SINGLE)
    return (n_single / n_total) < threshold
