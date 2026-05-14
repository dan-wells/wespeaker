# Copyright (c) 2024 Dan Wells
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

"""DOA-based speaker diarization using SRP-PHAT.

Estimates direction-of-arrival for subsegments of multichannel audio,
then assigns speakers either by matching to known positions (supervised)
or by clustering DOA angles (unsupervised).

The default clustering threshold of 15 degrees represents the angular resolution
of an 8-mic linear array with 4cm spacing, where:

  lambda = 343 m/s / 4000 Hz = 0.086 m  (wavelength at 4 kHz, where speech energy is strong)
  D = (n_mics - 1) * spacing = 7 * 0.04 m = 0.28 m  (array aperture)
  half-power (-3 dB) beamwidth points are where sinc(x) = 1 / sqrt(2), at x = +/- 0.443
  theta_3dB = (2 * 0.443 * lambda) / D = 0.27 radians = 15.6 degrees
"""

import argparse
import concurrent.futures
import json
import os
from collections import OrderedDict

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from wespeaker.utils.utils import validate_path


def read_wav_scp(scp_file):
    """Read wav.scp into an ordered dict of utt -> path."""
    wav_dict = OrderedDict()
    with open(scp_file) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            wav_dict[parts[0]] = parts[1]
    return wav_dict


def read_segments(segments_file):
    """Read segments file grouped by utterance.

    Args:
      segments_file: Path to segments file with format:
        <seg_id> <utt_id> <begin_s> <end_s>

    Returns:
      OrderedDict mapping utt_id to list of (seg_id, begin_s, end_s).
    """
    utt_segments = OrderedDict()
    with open(segments_file) as f:
        for line in f:
            seg_id, utt, begin, end = line.strip().split()
            if utt not in utt_segments:
                utt_segments[utt] = []
            utt_segments[utt].append((seg_id, float(begin), float(end)))
    return utt_segments


def subsegment(seg_id, seg_begin_s, seg_end_s, window_fs, period_fs,
               frame_shift):
    """Generate subsegment IDs matching extract_emb.py windowing logic.

    Args:
      seg_id: Segment ID string (e.g. "meeting_0000-00001230-00004560").
      seg_begin_s: Segment start time in seconds.
      seg_end_s: Segment end time in seconds.
      window_fs: Window size in frames.
      period_fs: Stride in frames.
      frame_shift: Frame shift in ms.

    Returns:
      List of (subseg_id, subseg_begin_s, subseg_end_s) tuples.
    """
    seg_begin_ms = int(seg_begin_s * 1000)
    seg_end_ms = int(seg_end_s * 1000)
    seg_length = (seg_end_ms - seg_begin_ms) // frame_shift

    subsegs = []
    if seg_length <= window_fs:
        subseg_id = "{}-{:08d}-{:08d}".format(seg_id, 0, seg_length)
        subseg_begin_s = seg_begin_s
        subseg_end_s = seg_end_s
        subsegs.append((subseg_id, subseg_begin_s, subseg_end_s))
    else:
        max_subseg_begin = seg_length - window_fs + period_fs
        for subseg_begin in range(0, max_subseg_begin, period_fs):
            subseg_end = min(subseg_begin + window_fs, seg_length)
            subseg_id = "{}-{:08d}-{:08d}".format(
                seg_id, subseg_begin, subseg_end)
            subseg_begin_s = seg_begin_s + subseg_begin * frame_shift / 1000.0
            subseg_end_s = seg_begin_s + subseg_end * frame_shift / 1000.0
            subsegs.append((subseg_id, subseg_begin_s, subseg_end_s))
    return subsegs


def build_mic_array(n_mics, spacing, mic_center_y):
    """Build linear microphone array positions for pyroomacoustics.

    Array is along the y-axis (matching room.py conventions).

    Args:
      n_mics: Number of microphone elements.
      spacing: Distance between adjacent elements in meters.
      mic_center_y: Y-coordinate of the array center.

    Returns:
      np.ndarray of shape (2, n_mics) -- 2D mic positions (x, y).
    """
    array_length = (n_mics - 1) * spacing
    mic_positions = np.zeros((2, n_mics))
    for i in range(n_mics):
        mic_positions[0, i] = 0.0  # x = 0 (relative coordinates)
        mic_positions[1, i] = (mic_center_y
                               - array_length / 2.0
                               + i * spacing)
    return mic_positions


def estimate_doa(multichannel_chunk, sample_rate, mic_positions):
    """Estimate dominant azimuth via SRP-PHAT.

    Args:
      multichannel_chunk: np.ndarray of shape (n_mics, n_samples).
      sample_rate: Audio sample rate.
      mic_positions: np.ndarray of shape (2, n_mics).

    Returns:
      Estimated azimuth in degrees (0-360).
    """
    doa = pra.doa.SRP(mic_positions, sample_rate,
                      nfft=512, num_src=1, dim=2)
    # SRP expects (n_mics, n_freq, n_frames) -- use STFT
    stft = pra.transform.stft.analysis(
        multichannel_chunk.T, L=512, hop=256).transpose(2, 1, 0)
    doa.locate_sources(stft)
    azimuth_rad = doa.azimuth_recon[0]
    return np.degrees(azimuth_rad) % 360


def compute_reference_azimuths(speaker_positions, mic_center):
    """Compute azimuth from mic center to each speaker position.

    Azimuth is measured in the x-y plane from the positive x-axis.

    Args:
      speaker_positions: List of [x, y, z] positions.
      mic_center: [x, y, z] mic array center position.

    Returns:
      List of azimuth angles in degrees (0-360).
    """
    azimuths = []
    for pos in speaker_positions:
        dx = pos[0] - mic_center[0]
        dy = pos[1] - mic_center[1]
        azimuth = np.degrees(np.arctan2(dy, dx)) % 360
        azimuths.append(azimuth)
    return azimuths


def angular_distance(a, b):
    """Compute minimum angular distance between two angles in degrees."""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def assign_supervised(doa_estimates, reference_azimuths, speaker_ids):
    """Assign speakers by closest reference azimuth.

    Args:
      doa_estimates: List of estimated azimuth angles (degrees).
      reference_azimuths: List of reference azimuth angles (degrees).
      speaker_ids: List of speaker ID strings.

    Returns:
      List of speaker ID strings, one per subsegment.
    """
    labels = []
    for doa in doa_estimates:
        distances = [angular_distance(doa, ref) for ref in reference_azimuths]
        labels.append(speaker_ids[np.argmin(distances)])
    return labels


def assign_unsupervised(doa_estimates, cluster_threshold):
    """Assign pseudo-labels by clustering DOA angles.

    Uses agglomerative clustering with a precomputed angular distance
    matrix and a distance threshold to determine the number of clusters.

    Args:
      doa_estimates: List of estimated azimuth angles (degrees).
      cluster_threshold: Distance threshold in degrees for clustering.

    Returns:
      List of integer pseudo-labels.
    """
    n = len(doa_estimates)
    if n <= 1:
        return list(range(n))

    # Build pairwise angular distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = angular_distance(doa_estimates[i], doa_estimates[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=cluster_threshold)
    labels = clustering.fit_predict(dist_matrix)
    return labels.tolist()


def compute_doa_timeline(wav_path, rttm_path, metadata_path,
                         n_mics=8, mic_spacing=0.04,
                         window_secs=1.5, period_secs=0.75,
                         frame_shift=10, min_overlap=0.1):
    """Compute per-subsegment DOA estimates labelled by ground-truth speaker.

    Runs SRP-PHAT on fixed-length subsegments across the full audio file,
    then labels each estimate by the speaker with the most temporal overlap
    according to the reference RTTM.

    Args:
      wav_path: Path to multichannel wav file.
      rttm_path: Path to reference RTTM file for this meeting.
      metadata_path: Path to meeting metadata.json (for mic_center).
      n_mics: Number of microphone elements.
      mic_spacing: Spacing between adjacent mics in meters.
      window_secs: Subsegment window duration in seconds.
      period_secs: Subsegment stride in seconds.
      frame_shift: Frame shift in ms.
      min_overlap: Minimum overlap (seconds) with a speaker to assign
        a label. Subsegments below this threshold are labelled None.

    Returns:
      Dict with keys:
        - 'doa_by_speaker': dict mapping speaker_id to list of
          (midpoint_s, doa_degrees) tuples.
        - 'ref_azimuths': dict mapping speaker_id to reference azimuth
          in degrees.
    """
    with open(metadata_path) as mf:
        metadata = json.load(mf)

    audio, sr = sf.read(wav_path, always_2d=True)
    audio = audio.T

    mic_center = metadata['mic_center']
    mic_positions = build_mic_array(n_mics, mic_spacing, mic_center[1])

    # Parse RTTM to get per-speaker intervals
    speaker_intervals = {}
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            spk = parts[7]
            start = float(parts[3])
            dur = float(parts[4])
            speaker_intervals.setdefault(spk, []).append(
                (start, start + dur))

    # Generate subsegments over the full file
    window_fs = int(window_secs * 1000) // frame_shift
    period_fs = int(period_secs * 1000) // frame_shift
    duration_s = audio.shape[1] / sr
    subsegs = subsegment('utt', 0.0, duration_s,
                         window_fs, period_fs, frame_shift)

    # Estimate DOA per subsegment
    doa_by_speaker = {}
    for _, begin_s, end_s in subsegs:
        # Find dominant speaker
        best_spk = None
        best_overlap = 0.0
        for spk, intervals in speaker_intervals.items():
            overlap = sum(
                max(0, min(end_s, ie) - max(begin_s, ib))
                for ib, ie in intervals)
            if overlap > best_overlap:
                best_overlap = overlap
                best_spk = spk
        if best_spk is None or best_overlap < min_overlap:
            continue

        begin_sample = int(begin_s * sr)
        end_sample = int(end_s * sr)
        chunk = audio[:, begin_sample:end_sample]
        if chunk.shape[1] < 512:
            pad_width = 512 - chunk.shape[1]
            chunk = np.pad(
                chunk, ((0, 0), (0, pad_width)), mode='constant')

        az = estimate_doa(chunk, sr, mic_positions)
        midpoint = (begin_s + end_s) / 2.0
        doa_by_speaker.setdefault(best_spk, []).append((midpoint, az))

    # Reference azimuths
    ref_azimuths = {}
    speaker_positions = [spk['position'] for spk in metadata['speakers']]
    speaker_ids = [spk['id'] for spk in metadata['speakers']]
    azimuths = compute_reference_azimuths(speaker_positions, mic_center)
    for spk_id, az in zip(speaker_ids, azimuths):
        ref_azimuths[spk_id] = az

    return {'doa_by_speaker': doa_by_speaker, 'ref_azimuths': ref_azimuths}


def process_utterance(utt, wav_path, segments, metadata_dir, mode,
                      sample_rate, n_mics, mic_spacing,
                      window_fs, period_fs, frame_shift,
                      cluster_threshold):
    """Process one utterance: load audio, estimate DOA, assign labels.

    Args:
      utt: Utterance ID string.
      wav_path: Path to multichannel wav file.
      segments: List of (seg_id, begin_s, end_s) tuples.
      metadata_dir: Path to metadata directory (or None).
      mode: 'supervised' or 'unsupervised'.
      sample_rate: Expected audio sample rate.
      n_mics: Number of microphone elements.
      mic_spacing: Spacing between adjacent mics in meters.
      window_fs: Window size in frames.
      period_fs: Stride in frames.
      frame_shift: Frame shift in ms.
      cluster_threshold: Angular distance threshold for clustering.

    Returns:
      List of (subseg_id, label) tuples, or empty list if no subsegments.
    """
    # Load multichannel audio
    audio, sr = sf.read(wav_path, always_2d=True)
    assert sr == sample_rate, (
        f"Sample rate mismatch: {sr} != {sample_rate}")
    # audio shape: (n_samples, n_channels) -> (n_channels, n_samples)
    audio = audio.T

    # Build mic array positions for DOA estimation
    mic_center_y = 0.0
    metadata = None
    if metadata_dir is not None:
        metadata_path = os.path.join(metadata_dir, utt, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path) as mf:
                metadata = json.load(mf)
            mic_center_y = metadata['mic_center'][1]

    mic_positions = build_mic_array(n_mics, mic_spacing, mic_center_y)

    # Generate subsegments and estimate DOA for each
    all_subseg_ids = []
    all_doa_estimates = []

    for seg_id, seg_begin, seg_end in segments:
        subsegs = subsegment(
            seg_id, seg_begin, seg_end,
            window_fs, period_fs, frame_shift)

        for subseg_id, subseg_begin_s, subseg_end_s in subsegs:
            # Extract multichannel chunk
            begin_sample = int(subseg_begin_s * sample_rate)
            end_sample = int(subseg_end_s * sample_rate)
            chunk = audio[:, begin_sample:end_sample]

            # Skip chunks that are too short for STFT
            if chunk.shape[1] < 512:
                # Pad with zeros
                pad_width = 512 - chunk.shape[1]
                chunk = np.pad(
                    chunk, ((0, 0), (0, pad_width)), mode='constant')

            azimuth = estimate_doa(chunk, sample_rate, mic_positions)
            all_subseg_ids.append(subseg_id)
            all_doa_estimates.append(azimuth)

    if len(all_subseg_ids) == 0:
        return []

    # Assign labels
    if mode == 'supervised':
        speaker_positions = [
            spk['position'] for spk in metadata['speakers']]
        speaker_ids = [spk['id'] for spk in metadata['speakers']]
        mic_center = metadata['mic_center']

        reference_azimuths = compute_reference_azimuths(
            speaker_positions, mic_center)
        labels = assign_supervised(
            all_doa_estimates, reference_azimuths, speaker_ids)
    else:
        labels = assign_unsupervised(
            all_doa_estimates, cluster_threshold)

    return list(zip(all_subseg_ids, labels))


class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter):
    pass


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RawDefaultsHelpFormatter)
    parser.add_argument('--wav-scp', required=True,
                        help='multichannel wav.scp')
    parser.add_argument('--segments', required=True,
                        help='VAD segments file')
    parser.add_argument('--metadata-dir', default=None,
                        help='directory with per-meeting metadata.json '
                             '(required for supervised mode)')
    parser.add_argument('--output', required=True,
                        help='output label file')
    parser.add_argument('--mode', default='supervised',
                        choices=['supervised', 'unsupervised'],
                        help='assignment mode')
    parser.add_argument('--window-secs', type=float, default=1.5,
                        help='subsegment window in seconds')
    parser.add_argument('--period-secs', type=float, default=0.75,
                        help='subsegment stride in seconds')
    parser.add_argument('--frame-shift', type=int, default=10,
                        help='frame shift in ms')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='audio sample rate')
    parser.add_argument('--n-mics', type=int, default=8,
                        help='number of microphone elements')
    parser.add_argument('--mic-spacing', type=float, default=0.04,
                        help='spacing between adjacent mics in meters')
    parser.add_argument('--cluster-threshold', type=float, default=15.0,
                        help='angular distance threshold in degrees '
                             '(unsupervised mode)')
    parser.add_argument('--n-workers', type=int, default=1,
                        help='number of parallel workers for DOA estimation '
                             '(1 = sequential)')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.mode == 'supervised' and args.metadata_dir is None:
        raise ValueError("--metadata-dir is required for supervised mode")

    # Prevent nested parallelism from NumPy/BLAS in worker processes
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Transform duration to frame number
    window_fs = int(args.window_secs * 1000) // args.frame_shift
    period_fs = int(args.period_secs * 1000) // args.frame_shift

    wav_dict = read_wav_scp(args.wav_scp)
    utt_segments = read_segments(args.segments)

    # Build list of utterances to process
    utt_work = []
    for utt, segments in utt_segments.items():
        if utt in wav_dict:
            utt_work.append((utt, wav_dict[utt], segments))

    validate_path(args.output)
    with open(args.output, 'w') as f:
        if args.n_workers <= 1:
            # Sequential processing
            for utt, wav_path, segments in tqdm(utt_work):
                results = process_utterance(
                    utt, wav_path, segments, args.metadata_dir, args.mode,
                    args.sample_rate, args.n_mics, args.mic_spacing,
                    window_fs, period_fs, args.frame_shift,
                    args.cluster_threshold)
                for subseg_id, label in results:
                    print(subseg_id, label, file=f)
        else:
            # Parallel processing
            results_by_utt = {}
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=args.n_workers) as executor:
                futures = {}
                for utt, wav_path, segments in utt_work:
                    future = executor.submit(
                        process_utterance,
                        utt, wav_path, segments, args.metadata_dir,
                        args.mode, args.sample_rate, args.n_mics,
                        args.mic_spacing, window_fs, period_fs,
                        args.frame_shift, args.cluster_threshold)
                    futures[future] = utt

                for future in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures)):
                    utt = futures[future]
                    results_by_utt[utt] = future.result()

            # Write in original utterance order
            for utt, wav_path, segments in utt_work:
                for subseg_id, label in results_by_utt[utt]:
                    print(subseg_id, label, file=f)


if __name__ == '__main__':
    main()
