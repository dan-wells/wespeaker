"""Streaming speaker diarization pipeline.

Processes audio files in a causal sliding-window fashion, producing
speaker labels incrementally. Supports both ONNX and PyTorch embedding
models, with online clustering or enrolled-speaker identification for
speaker assignment.
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

import argparse
import json
import os
import sys
import time

import kaldiio
import numpy as np
import silero_vad
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from wespeaker.cli.speaker import load_model_pt
from wespeaker.diar.extract_emb import init_session
from wespeaker.diar.identify import identify
from wespeaker.diar.make_rttm import merge_segments


SAMPLE_RATE = 16000


class RawDefaultsFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter):
    pass


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
            self.model = load_model_pt(source)
            self.model.to(torch.device(device))
        else:
            raise ValueError("Unknown backend: %s" % backend)

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


class OnlineClusterer:
    """Online speaker clustering via running centroid comparison.

    Maintains L2-normalized cluster centroids and assigns each new
    embedding to the most similar centroid, or creates a new cluster
    when similarity falls below a threshold.

    Args:
      threshold: Min cosine similarity to assign to an existing cluster.
      max_speakers: Maximum number of clusters to create.
    """

    def __init__(self, threshold=0.5, max_speakers=20):
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.centroids = []
        self.counts = []

    def reset(self):
        """Clear all cluster state."""
        self.centroids = []
        self.counts = []

    def assign(self, embedding):
        """Assign an embedding to a cluster or create a new one.

        Args:
          embedding: np.ndarray of shape (emb_dim,).

        Returns:
          Tuple of (label, confidence) where label is an integer
            cluster ID and confidence is cosine similarity to the
            assigned centroid (1.0 for newly created clusters).
        """
        emb_norm = embedding / np.linalg.norm(embedding)

        if len(self.centroids) == 0:
            self.centroids.append(emb_norm.copy())
            self.counts.append(1)
            return 0, 1.0

        centroid_matrix = np.stack(self.centroids)
        scores = centroid_matrix @ emb_norm
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= self.threshold:
            count = self.counts[best_idx]
            raw = self.centroids[best_idx] * count + emb_norm
            self.centroids[best_idx] = raw / np.linalg.norm(raw)
            self.counts[best_idx] = count + 1
            return best_idx, best_score
        elif len(self.centroids) < self.max_speakers:
            idx = len(self.centroids)
            self.centroids.append(emb_norm.copy())
            self.counts.append(1)
            return idx, 1.0
        else:
            return best_idx, best_score


class SpeakerIdentifier:
    """Speaker identification via cosine similarity to enrolled embeddings.

    Loads enrolled speaker embeddings once, then scores each incoming
    embedding against the gallery to find the best match.

    Args:
      enrol_scp: Path to enrollment embedding scp file.
      metadata_dir: Optional directory with per-meeting metadata.json
        files to filter the gallery to meeting-specific speakers.
    """

    def __init__(self, enrol_scp, metadata_dir=None):
        self.enrol_dict = kaldiio.load_scp(enrol_scp)
        self.metadata_dir = metadata_dir
        self.gallery_embeddings = None
        self.speaker_ids = None

    def setup_for_file(self, utt_id=None):
        """Load gallery embeddings, optionally filtered by metadata.

        Args:
          utt_id: Utterance ID used to locate per-meeting metadata.
        """
        if self.metadata_dir is not None and utt_id is not None:
            metadata_path = os.path.join(self.metadata_dir, utt_id,
                                         "metadata.json")
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.speaker_ids = [spk["id"] for spk in metadata["speakers"]]
        else:
            self.speaker_ids = list(self.enrol_dict.keys())

        self.gallery_embeddings = np.stack(
            [self.enrol_dict[spk_id] for spk_id in self.speaker_ids])

    def assign(self, embedding):
        """Assign an embedding to the closest enrolled speaker.

        Args:
          embedding: np.ndarray of shape (emb_dim,).

        Returns:
          Tuple of (label, confidence) where label is a speaker ID
            string and confidence is cosine similarity.
        """
        labels, scores = identify(
            embedding[np.newaxis, :],
            self.gallery_embeddings,
            self.speaker_ids)
        return labels[0], float(scores[0])


def compute_fbank_for_chunk(audio_chunk):
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


def process_file(audio, utt_id, emb_model, assigner, vad_model, args):
    """Run streaming diarization on a single audio file.

    Slides a window of --window-secs over the audio with --stride-secs
    steps. For each window, optionally runs VAD, extracts fbank
    features and an embedding, and assigns a speaker label.

    Args:
      audio: np.ndarray of shape (n_samples,), float32, 16kHz.
      utt_id: Utterance identifier string.
      emb_model: EmbeddingModel instance.
      assigner: OnlineClusterer or SpeakerIdentifier instance.
      vad_model: Silero VAD model, or None if VAD is disabled.
      args: Parsed CLI arguments.

    Returns:
      List of (start_sec, end_sec, label, confidence) tuples for
        speech chunks that passed the assignment threshold.
    """
    num_samples = len(audio)
    window_samples = int(args.window_secs * SAMPLE_RATE)
    stride_samples = int(args.stride_secs * SAMPLE_RATE)
    window_frames = int(args.window_secs * 1000) // 10

    results = []
    pos = 0
    while pos < num_samples:
        t0 = time.monotonic()

        window_end = min(pos + stride_samples, num_samples)
        window_start = max(0, window_end - window_samples)
        chunk = audio[window_start:window_end]

        start_sec = window_start / SAMPLE_RATE
        end_sec = window_end / SAMPLE_RATE

        # Skip if chunk is shorter than one stride
        if len(chunk) < stride_samples:
            break

        # Skip if waiting for full buffer and chunk is short
        if args.wait_full_buffer and len(chunk) < window_samples:
            pos += stride_samples
            continue

        # VAD gate
        if vad_model is not None:
            vad_model.reset_states()
            chunk_tensor = torch.from_numpy(chunk).float()
            timestamps = silero_vad.get_speech_timestamps(
                chunk_tensor, vad_model,
                sampling_rate=SAMPLE_RATE,
                threshold=args.vad_threshold)
            if not timestamps:
                if not args.quiet:
                    print("%8.3f %8.3f  --" % (start_sec, end_sec))
                    sys.stdout.flush()
                pos += stride_samples
                if args.real_time:
                    elapsed = time.monotonic() - t0
                    time.sleep(max(0, args.stride_secs - elapsed))
                continue

        # Compute fbank features and apply per-window CMN
        fbank = compute_fbank_for_chunk(chunk)
        if args.subseg_cmn:
            fbank = fbank - fbank.mean(axis=0)

        # Pad short chunks to expected window length (matches offline
        # np.resize behavior in extract_emb.subsegment)
        if fbank.shape[0] < window_frames:
            fbank = np.resize(fbank, (window_frames, fbank.shape[1]))

        # Extract embedding and assign speaker
        embedding = emb_model.extract(fbank)
        label, confidence = assigner.assign(embedding)

        # Apply assignment threshold
        if (args.assign_threshold is not None
                and confidence < args.assign_threshold):
            if not args.quiet:
                print("%8.3f %8.3f  --" % (start_sec, end_sec))
                sys.stdout.flush()
        else:
            results.append((start_sec, end_sec, label, confidence))
            if not args.quiet:
                print("%8.3f %8.3f  %s  %.2f"
                      % (start_sec, end_sec, label, confidence))
                sys.stdout.flush()

        pos += stride_samples
        if args.real_time:
            elapsed = time.monotonic() - t0
            time.sleep(max(0, args.stride_secs - elapsed))

    return results


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RawDefaultsFormatter)

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--wav", help="single wav file path")
    input_group.add_argument("--wav-scp",
                             help="wav.scp for sequential processing")

    # Model
    parser.add_argument("--model", required=True,
                        help="path to .onnx file or PyTorch model directory")
    parser.add_argument("--backend", default=None,
                        choices=["onnx", "pytorch"],
                        help="model backend (auto-detected from --model)")
    parser.add_argument("--device", default="cuda",
                        choices=["cpu", "cuda"],
                        help="inference device")

    # Speaker assignment
    parser.add_argument("--assign", default="cluster",
                        choices=["cluster", "identify"],
                        help="speaker assignment strategy")
    parser.add_argument("--cluster-threshold", type=float, default=0.5,
                        help="min cosine similarity to assign to existing "
                             "cluster (clustering mode only)")
    parser.add_argument("--max-speakers", type=int, default=20,
                        help="max clusters for online clustering")
    parser.add_argument("--assign-threshold", type=float, default=None,
                        help="min confidence to emit a speaker label")
    parser.add_argument("--enrol-scp", default=None,
                        help="enrollment embedding scp "
                             "(required for --assign identify)")
    parser.add_argument("--metadata-dir", default=None,
                        help="per-meeting metadata directory for filtering "
                             "enrolled speakers")

    # Windowing
    parser.add_argument("--window-secs", type=float, default=1.5,
                        help="sliding window duration in seconds")
    parser.add_argument("--stride-secs", type=float, default=0.5,
                        help="stride between windows in seconds")
    parser.add_argument("--no-subseg-cmn", dest="subseg_cmn",
                        action="store_false", default=True,
                        help="disable per-window cepstral mean normalization")
    parser.add_argument("--wait-full-buffer", action="store_true",
                        help="wait for full window before first inference")

    # VAD
    parser.add_argument("--no-vad", dest="vad", action="store_false",
                        default=True, help="disable streaming Silero VAD")
    parser.add_argument("--vad-threshold", type=float, default=0.5,
                        help="speech probability threshold for VAD")

    # Output
    parser.add_argument("--output", required=True,
                        help="output labels file path")
    parser.add_argument("--output-rttm", default=None,
                        help="optional RTTM output path")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-chunk stdout output")

    # Pacing
    parser.add_argument("--real-time", action="store_true",
                        help="sleep between chunks to simulate real-time "
                             "pacing")

    args = parser.parse_args()

    if args.assign == "identify" and args.enrol_scp is None:
        parser.error("--enrol-scp is required when --assign identify")

    return args


def main():
    args = get_args()

    emb_model = EmbeddingModel(args.model, args.device, args.backend)

    if args.assign == "cluster":
        assigner = OnlineClusterer(args.cluster_threshold, args.max_speakers)
    else:
        assigner = SpeakerIdentifier(args.enrol_scp, args.metadata_dir)

    vad_model = None
    if args.vad:
        vad_model = silero_vad.load_silero_vad()

    # Build file list from single wav or wav.scp
    if args.wav is not None:
        utt_id = os.path.splitext(os.path.basename(args.wav))[0]
        file_list = [(utt_id, args.wav)]
    else:
        file_list = []
        with open(args.wav_scp) as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                file_list.append((parts[0], parts[1]))

    all_results = {}
    for utt_id, wav_path in file_list:
        #if not args.quiet:
        print("--- %s ---" % utt_id)

        audio = load_audio(wav_path)

        if args.assign == "cluster":
            assigner.reset()
        else:
            assigner.setup_for_file(utt_id)

        results = process_file(
            audio, utt_id, emb_model, assigner, vad_model, args)
        all_results[utt_id] = results

    # Write labels file
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        for utt_id, results in all_results.items():
            for start, end, label, conf in results:
                f.write("%s %.3f %.3f %s %.4f\n"
                        % (utt_id, start, end, label, conf))

    # Write RTTM if requested
    if args.output_rttm is not None:
        rttm_dir = os.path.dirname(args.output_rttm)
        if rttm_dir:
            os.makedirs(rttm_dir, exist_ok=True)

        utt_to_subseg_labels = {}
        for utt_id, results in all_results.items():
            utt_to_subseg_labels[utt_id] = [
                (start, end, str(label))
                for start, end, label, _ in results]

        merged = merge_segments(utt_to_subseg_labels)
        with open(args.output_rttm, "w") as f:
            for utt, begin, end, label in merged:
                f.write("SPEAKER %s 1 %.3f %.3f <NA> <NA> %s <NA> <NA>\n"
                        % (utt, begin, end - begin, label))


if __name__ == "__main__":
    main()
