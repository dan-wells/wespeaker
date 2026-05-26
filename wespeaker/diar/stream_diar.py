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

# Optional: only needed for --audio-device/--list-devices.
# OSError covers Linux where the package is installed but libportaudio is not.
try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

from wespeaker.diar.identify import identify
from wespeaker.diar.make_rttm import merge_segments
from wespeaker.diar.mvad import load_mvad_model, mvad_check_overlap
from wespeaker.utils.audio import SAMPLE_RATE, compute_fbank, load_audio
from wespeaker.utils.embedding import EmbeddingModel


class AudioRingBuffer:
    """Fixed-size circular buffer for streaming audio samples.

    Pre-allocates a numpy array of the given capacity and supports
    efficient append and retrieval of the most recent samples without
    per-stride allocation.

    Args:
      capacity: Maximum number of samples to store.
    """

    def __init__(self, capacity):
        self.buffer = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.write_pos = 0

    def append(self, data):
        """Append samples to the buffer, overwriting oldest if full."""
        n = len(data)
        if n >= self.capacity:
            self.buffer[:] = data[-self.capacity:]
            self.write_pos += n
            return
        start = self.write_pos % self.capacity
        end = start + n
        if end <= self.capacity:
            self.buffer[start:end] = data
        else:
            first = self.capacity - start
            self.buffer[start:] = data[:first]
            self.buffer[:n - first] = data[first:]
        self.write_pos += n

    def get_last_n(self, n):
        """Return the last n samples as a contiguous array."""
        available = min(n, self.write_pos, self.capacity)
        end = self.write_pos % self.capacity
        if available <= end:
            return self.buffer[end - available:end].copy()
        else:
            return np.concatenate(
                [self.buffer[self.capacity - (available - end):],
                 self.buffer[:end]])

    @property
    def total_samples(self):
        """Total number of samples written since creation."""
        return self.write_pos


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
      speakers: Optional list of speaker IDs to use from enrol_scp.
    """

    def __init__(self, enrol_scp, metadata_dir=None, speakers=None):
        self.enrol_dict = kaldiio.load_scp(enrol_scp)
        self.metadata_dir = metadata_dir
        self.speakers = speakers
        self.gallery_embeddings = None
        self.speaker_ids = None

    def setup_for_file(self, utt_id=None):
        """Load gallery embeddings, optionally filtered by metadata.

        Args:
          utt_id: Utterance ID used to locate per-meeting metadata.
        """
        if self.speakers is not None:
            self.speaker_ids = self.speakers
        elif self.metadata_dir is not None and utt_id is not None:
            metadata_path = os.path.join(self.metadata_dir, utt_id, "metadata.json")
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



def process_chunk(chunk, start_sec, end_sec, emb_model, assigner,
                  vad_model, args, window_frames):
    """Process a single audio chunk: VAD, fbank, embedding, assignment.

    Args:
      chunk: np.ndarray of shape (n_samples,), float32, 16kHz.
      start_sec: Start time of this chunk in seconds.
      end_sec: End time of this chunk in seconds.
      emb_model: EmbeddingModel instance.
      assigner: OnlineClusterer or SpeakerIdentifier instance.
      vad_model: Silero VAD model, MVAD state tuple, or None.
      args: Parsed CLI arguments.
      window_frames: Expected number of fbank frames for a full window.

    Returns:
      Tuple of (start_sec, end_sec, label, confidence) if speech is
        detected and above threshold, else None.
    """
    # VAD gate
    if vad_model is not None and args.vad_mode == "mvad":
        mvad_model, feat_mean, feat_std, mvad_device = vad_model
        skip = mvad_check_overlap(
            chunk, SAMPLE_RATE, mvad_model, feat_mean, feat_std,
            mvad_device, threshold=args.mvad_overlap_threshold)
        if skip:
            if not args.quiet:
                print("{:8.3f} {:8.3f}  --".format(start_sec, end_sec))
                sys.stdout.flush()
            return None
    elif vad_model is not None:
        vad_model.reset_states()
        chunk_tensor = torch.from_numpy(chunk).float()
        timestamps = silero_vad.get_speech_timestamps(
            chunk_tensor, vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=args.vad_threshold)
        if not timestamps:
            if not args.quiet:
                print("{:8.3f} {:8.3f}  --".format(start_sec, end_sec))
                sys.stdout.flush()
            return None

    # Compute fbank features and apply per-window CMN
    fbank = compute_fbank(chunk)
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
    if (args.assign_threshold is not None and confidence < args.assign_threshold):
        if not args.quiet:
            print("{:8.3f} {:8.3f}  unknown".format(start_sec, end_sec))
            sys.stdout.flush()
        return None

    if not args.quiet:
        print("{:8.3f} {:8.3f}  {}  {:.2f}".format(
            start_sec, end_sec, label, confidence))
        sys.stdout.flush()
    return (start_sec, end_sec, label, confidence)


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
      vad_model: Silero VAD model, MVAD state tuple, or None.
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

        result = process_chunk(
            chunk, start_sec, end_sec,
            emb_model, assigner, vad_model, args, window_frames)
        if result is not None:
            results.append(result)

        pos += stride_samples
        if args.real_time:
            elapsed = time.monotonic() - t0
            time.sleep(max(0, args.stride_secs - elapsed))

    return results


def process_device_stream(utt_id, emb_model, assigner, vad_model, args):
    """Run streaming diarization on live audio from a device input.

    Opens an audio input stream, reads chunks of stride_secs duration,
    and processes each through the same VAD/fbank/embedding/assignment
    pipeline as file-based mode. Runs until interrupted with Ctrl+C.

    Args:
      utt_id: Utterance identifier string.
      emb_model: EmbeddingModel instance.
      assigner: OnlineClusterer or SpeakerIdentifier instance.
      vad_model: Silero VAD model, MVAD state tuple, or None.
      args: Parsed CLI arguments.

    Returns:
      List of (start_sec, end_sec, label, confidence) tuples.
    """
    window_samples = int(args.window_secs * SAMPLE_RATE)
    stride_samples = int(args.stride_secs * SAMPLE_RATE)
    window_frames = int(args.window_secs * 1000) // 10

    results = []
    ring_buf = AudioRingBuffer(window_samples)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=stride_samples,
        device=args.audio_device)

    try:
        stream.start()
        if not args.quiet:
            print("--- {} (device {}) ---".format(utt_id, args.audio_device))
            print("Listening... press Ctrl+C to stop.")
            sys.stdout.flush()

        while True:
            data, overflowed = stream.read(stride_samples)
            if overflowed:
                print("[WARNING] audio input overflowed", file=sys.stderr)

            ring_buf.append(data[:, 0])

            # Determine how much audio is available
            available = min(ring_buf.total_samples, window_samples)
            chunk = ring_buf.get_last_n(available)

            end_sec = ring_buf.total_samples / SAMPLE_RATE
            start_sec = (ring_buf.total_samples - available) / SAMPLE_RATE

            if available < stride_samples:
                continue
            if args.wait_full_buffer and available < window_samples:
                continue

            result = process_chunk(
                chunk, start_sec, end_sec,
                emb_model, assigner, vad_model, args, window_frames)
            if result is not None:
                results.append(result)

    except KeyboardInterrupt:
        if not args.quiet:
            print("\nStopping...")
    finally:
        stream.stop()
        stream.close()

    return results


class RawDefaultsFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter):
    pass


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RawDefaultsFormatter)

    # Input
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--wav", help="single wav file path")
    input_group.add_argument("--wav-scp", help="wav.scp for sequential processing")
    input_group.add_argument("--audio-device", type=int, default=None, metavar="DEVICE_ID",
                             help="audio input device index for live capture")
    parser.add_argument("--list-devices", action="store_true",
                        help="list available audio devices and exit")
    parser.add_argument("--utt-id", default="stream",
                        help="utterance ID for device input mode")

    # Model
    parser.add_argument("--model", default=None,
                        help="path to .onnx file or PyTorch model directory")
    parser.add_argument("--backend", default=None, choices=["onnx", "pytorch"],
                        help="model backend (auto-detected from --model)")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                        help="inference device")

    # Speaker assignment
    parser.add_argument("--assign", default="cluster", choices=["cluster", "identify"],
                        help="speaker assignment strategy")
    parser.add_argument("--cluster-threshold", type=float, default=0.5,
                        help="min cosine similarity to assign to existing "
                             "cluster (clustering mode only)")
    parser.add_argument("--max-speakers", type=int, default=20,
                        help="max clusters for online clustering")
    parser.add_argument("--assign-threshold", type=float, default=None,
                        help="min confidence to emit a speaker label")
    parser.add_argument("--enrol-scp", default=None,
                        help="enrollment embedding scp (required for identification mode)")
    parser.add_argument("--metadata-dir", default=None,
                        help="per-meeting metadata directory for filtering "
                             "enrolled speakers")
    parser.add_argument("--speakers", nargs="+", default=None,
                        help="speaker IDs to use from enrol-scp (identification mode only)")

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
    parser.add_argument("--vad-mode", default="silero",
                        choices=["silero", "mvad"],
                        help="VAD backend: silero (binary speech/silence) "
                             "or mvad (multi-class with overlap detection)")
    parser.add_argument("--no-vad", dest="vad", action="store_false", default=True,
                        help="disable VAD entirely")
    parser.add_argument("--vad-threshold", type=float, default=0.5,
                        help="speech probability threshold for Silero VAD")
    parser.add_argument("--mvad-model", default=None,
                        help="path to MVAD_V2 checkpoint "
                             "(required when --vad-mode mvad)")
    parser.add_argument("--mvad-overlap-threshold", type=float, default=0.5,
                        help="min fraction of single-speaker frames to "
                             "process a window (MVAD mode)")

    # Output
    parser.add_argument("--output", default=None,
                        help="output labels file path")
    parser.add_argument("--output-rttm", default=None,
                        help="optional RTTM output path")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-chunk stdout output")

    # Pacing
    parser.add_argument("--real-time", action="store_true",
                        help="sleep between chunks to simulate real-time pacing")

    args = parser.parse_args()

    if (args.list_devices or args.audio_device is not None) and sd is None:
        parser.error(
            "sounddevice is required for device input -- "
            "pip install sounddevice")

    if args.list_devices:
        return args

    if args.wav is None and args.wav_scp is None and args.audio_device is None:
        parser.error("one of --wav, --wav-scp, or --audio-device is required")
    if args.model is None:
        parser.error("--model is required")
    if args.assign == "identify" and args.enrol_scp is None:
        parser.error("--enrol-scp is required when --assign identify")
    if args.vad and args.vad_mode == "mvad" and args.mvad_model is None:
        parser.error("--mvad-model is required when --vad-mode mvad")

    return args


def main():
    args = get_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    emb_model = EmbeddingModel(args.model, args.device, args.backend)

    if args.assign == "cluster":
        assigner = OnlineClusterer(args.cluster_threshold, args.max_speakers)
    else:
        assigner = SpeakerIdentifier(args.enrol_scp, args.metadata_dir, args.speakers)

    vad_model = None
    if args.vad:
        if args.vad_mode == "mvad":
            vad_model = load_mvad_model(args.mvad_model, args.device)
        else:
            vad_model = silero_vad.load_silero_vad()

    all_results = {}

    if args.audio_device is not None:
        utt_id = args.utt_id

        if args.assign == "cluster":
            assigner.reset()
        else:
            assigner.setup_for_file(utt_id)

        results = process_device_stream(
            utt_id, emb_model, assigner, vad_model, args)
        all_results[utt_id] = results
    else:
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

        for utt_id, wav_path in file_list:
            print("--- {} ---".format(utt_id))

            audio = load_audio(wav_path)

            if args.assign == "cluster":
                assigner.reset()
            else:
                assigner.setup_for_file(utt_id)

            results = process_file(
                audio, utt_id, emb_model, assigner, vad_model, args)
            all_results[utt_id] = results

    # Write labels file
    if args.output is not None:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            for utt_id, results in all_results.items():
                for start, end, label, conf in results:
                    f.write("{} {:.3f} {:.3f} {} {:.4f}\n".format(
                        utt_id, start, end, label, conf))

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
                f.write("SPEAKER {} 1 {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>\n".format(
                    utt, begin, end - begin, label))


if __name__ == "__main__":
    main()
