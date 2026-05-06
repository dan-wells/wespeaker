# Copyright (c) 2026 Dan Wells
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diarization visualization module.

Provides functions for plotting diarization results (hypothesis and
reference RTTMs) as per-speaker timeline tracks, with optional audio
RMS envelope and speaker identity mapping.

Designed for notebook use::

    from wespeaker.diar.plot_diar import plot_diarization
    fig, axes = plot_diarization("hyp.rttm", ref_rttm="ref.rttm",
                                 wav_path="mixed.wav")
"""

import argparse
from collections import OrderedDict

import kaldiio
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
import torchaudio

from wespeaker.diar.make_oracle_sad import read_rttm

# Default colour palette (tab10, colourblind-friendly)
_DEFAULT_COLOURS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]
_MONOCHROME_COLOURS = [
    '#222222', '#555555', '#888888', '#AAAAAA', '#444444',
    '#666666', '#999999', '#BBBBBB', '#333333', '#777777',
]

# Layout constants
_TRACK_SPACING = 1.0
_REF_OFFSET = 0.1
_HYP_OFFSET = -0.1
_SEGMENT_LINEWIDTH = 6
_REF_BG_LINEWIDTH = 12
_REF_BG_ALPHA = 0.2
_REF_HOLLOW_LINEWIDTH = 1.5
_REF_LINE_LINEWIDTH = 1.5


def map_speakers_by_overlap(ref_segments, hyp_segments):
    """Map hypothesis speaker labels to reference labels by temporal overlap.

    Builds an overlap matrix between all (ref_speaker, hyp_speaker) pairs
    and solves the optimal one-to-one assignment using the Hungarian
    algorithm.

    Args:
      ref_segments: List of (start, end, speaker_label) tuples from
        reference RTTM.
      hyp_segments: List of (start, end, speaker_label) tuples from
        hypothesis RTTM.

    Returns:
      Dict mapping hyp_speaker_label -> ref_speaker_label. Hypothesis
      speakers without a match (when n_hyp > n_ref) retain their
      original labels.
    """
    ref_speakers = sorted(set(seg[2] for seg in ref_segments))
    hyp_speakers = sorted(set(seg[2] for seg in hyp_segments))

    # Group segments by speaker
    ref_by_spk = {spk: [(s, e) for s, e, sp in ref_segments if sp == spk]
                  for spk in ref_speakers}
    hyp_by_spk = {spk: [(s, e) for s, e, sp in hyp_segments if sp == spk]
                  for spk in hyp_speakers}

    # Build overlap matrix
    overlap = np.zeros((len(ref_speakers), len(hyp_speakers)))
    for i, rspk in enumerate(ref_speakers):
        for j, hspk in enumerate(hyp_speakers):
            overlap[i, j] = _compute_overlap(ref_by_spk[rspk],
                                             hyp_by_spk[hspk])

    # Hungarian assignment (maximise overlap -> minimise negative)
    row_ind, col_ind = linear_sum_assignment(-overlap)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if overlap[r, c] > 0:
            mapping[hyp_speakers[c]] = ref_speakers[r]

    # Unmapped hyp speakers keep their original label
    for hspk in hyp_speakers:
        if hspk not in mapping:
            mapping[hspk] = hspk

    return mapping


def _compute_overlap(segs_a, segs_b):
    """Compute total temporal overlap between two sorted segment lists.

    Args:
      segs_a: Sorted list of (start, end) tuples.
      segs_b: Sorted list of (start, end) tuples.

    Returns:
      Total overlap duration in seconds.
    """
    total = 0.0
    i, j = 0, 0
    while i < len(segs_a) and j < len(segs_b):
        start_a, end_a = segs_a[i]
        start_b, end_b = segs_b[j]
        overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
        total += overlap
        if end_a <= end_b:
            i += 1
        else:
            j += 1
    return total


def map_speakers_by_embeddings(emb_scp, hyp_segments, enrol_scp):
    """Map hypothesis speaker labels to enrolled speakers via embeddings.

    Loads sub-segment embeddings from the diarization pipeline, computes
    a centroid for each hypothesis cluster, then scores centroids against
    enrolled speaker embeddings using cosine similarity.

    Args:
      emb_scp: Path to the emb.scp file from the diarization pipeline
        (sub-segment embeddings).
      hyp_segments: List of (start, end, speaker_label) tuples from
        hypothesis RTTM.
      enrol_scp: Path to enrolment embedding .scp file, keyed by
        speaker ID (one embedding per speaker). Produced by the
        meeting-sim enroll step or any speaker embedding extraction
        pipeline.

    Returns:
      Dict mapping hyp_speaker_label -> enrolled_speaker_id (best match).
    """
    # Load enrolment embeddings
    enrol_ids = []
    enrol_embs = []
    for spk_id, emb in kaldiio.load_scp_sequential(enrol_scp):
        enrol_ids.append(spk_id)
        enrol_embs.append(emb.astype(np.float32))
    enrol_embs = np.stack(enrol_embs)

    # Load diarization sub-segment embeddings
    emb_dict = OrderedDict()
    for subseg_id, emb in kaldiio.load_scp_sequential(emb_scp):
        emb_dict[subseg_id] = emb.astype(np.float32)

    # Parse sub-segment IDs to get time midpoints and assign to hyp speakers
    hyp_speakers = sorted(set(seg[2] for seg in hyp_segments))
    cluster_embs = {spk: [] for spk in hyp_speakers}

    for subseg_id, emb in emb_dict.items():
        midpoint = _subseg_midpoint(subseg_id)
        if midpoint is None:
            continue
        # Find which hyp segment contains this midpoint
        spk = _find_speaker_at_time(hyp_segments, midpoint)
        if spk is not None:
            cluster_embs[spk].append(emb)

    # Compute L2-normalised centroids
    centroids = {}
    for spk, embs in cluster_embs.items():
        if len(embs) == 0:
            continue
        centroid = np.mean(np.stack(embs), axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids[spk] = centroid

    if len(centroids) == 0:
        return {spk: spk for spk in hyp_speakers}

    # Cosine similarity: (n_clusters, n_enrol)
    centroid_ids = sorted(centroids.keys())
    centroid_matrix = np.stack([centroids[cid] for cid in centroid_ids])
    sim_matrix = cosine_similarity(centroid_matrix, enrol_embs)

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[centroid_ids[r]] = enrol_ids[c]

    # Unmapped speakers keep original label
    for spk in hyp_speakers:
        if spk not in mapping:
            mapping[spk] = spk

    return mapping


def _subseg_midpoint(subseg_id, frame_shift=10):
    """Parse a sub-segment ID and return its time midpoint in seconds.

    Sub-segment ID format:
      {utt}-{begin_ms}-{end_ms}-{begin_frames}-{end_frames}

    Args:
      subseg_id: Sub-segment identifier string.
      frame_shift: Frame shift in milliseconds.

    Returns:
      Midpoint time in seconds, or None if parsing fails.
    """
    parts = subseg_id.split('-')
    if len(parts) < 5:
        return None
    try:
        begin_ms = int(parts[-4])
        begin_frames = int(parts[-2])
        end_frames = int(parts[-1])
        mid_frames = (begin_frames + end_frames) / 2.0
        midpoint = (begin_ms + mid_frames * frame_shift) / 1000.0
        return midpoint
    except (ValueError, IndexError):
        return None


def _find_speaker_at_time(segments, time):
    """Find which speaker is active at a given time.

    Args:
      segments: List of (start, end, speaker_label) tuples, sorted by
        start time.
      time: Time in seconds to query.

    Returns:
      Speaker label string, or None if no segment contains this time.
    """
    for start, end, spk in segments:
        if start <= time <= end:
            return spk
        if start > time:
            break
    return None


def _compute_rms_envelope(wav_path, start_time=None, end_time=None,
                          window_ms=50, hop_ms=25):
    """Compute windowed RMS envelope of an audio file.

    Args:
      wav_path: Path to the audio file.
      start_time: Optional start time in seconds (load from this point).
      end_time: Optional end time in seconds (load up to this point).
      window_ms: RMS window length in milliseconds.
      hop_ms: Hop between windows in milliseconds.

    Returns:
      Tuple of (times, rms) where times is a numpy array of window
      centre times in seconds and rms is the corresponding RMS values.
    """
    info = torchaudio.info(wav_path)
    sr = info.sample_rate

    frame_offset = 0
    num_frames = -1
    if start_time is not None:
        frame_offset = int(start_time * sr)
    if end_time is not None:
        end_frame = int(end_time * sr)
        num_frames = end_frame - frame_offset

    waveform, sr = torchaudio.load(wav_path, frame_offset=frame_offset,
                                   num_frames=num_frames)
    # Mix to mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    signal = waveform.squeeze(0).numpy()

    window_samples = int(window_ms * sr / 1000)
    hop_samples = int(hop_ms * sr / 1000)

    n_frames = max(0, (len(signal) - window_samples) // hop_samples + 1)
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_samples
        frame = signal[start:start + window_samples]
        rms[i] = np.sqrt(np.mean(frame ** 2))

    time_offset = start_time if start_time is not None else 0.0
    times = time_offset + (np.arange(n_frames) * hop_samples
                           + window_samples / 2.0) / sr

    return times, rms


def plot_diarization(hyp_rttm, ref_rttm=None, wav_path=None,
                     enrol_scp=None, emb_scp=None,
                     utt_id=None,
                     start_time=None, end_time=None,
                     show_overlap=False, ref_style='field',
                     monochrome=False, figsize=None, ax=None):
    """Plot diarization results with optional reference comparison.

    Displays each speaker on a horizontal track with time on the x-axis.
    If both ref_rttm and hyp_rttm are provided, maps hypothesis speakers
    to reference speakers and shows both as offset bars within each track.

    Speaker mapping strategy:
      - If both enrol_scp and emb_scp are provided, uses embedding-based
        cosine similarity matching against enrolled speaker embeddings.
      - Otherwise (default when ref_rttm is given), uses temporal overlap
        with the Hungarian algorithm.

    Args:
      hyp_rttm: Path to the hypothesis RTTM file.
      ref_rttm: Optional path to the reference RTTM file.
      wav_path: Optional path to audio file for RMS envelope subplot.
        Ignored if ax is provided.
      enrol_scp: Optional path to enrolment embedding .scp file, keyed
        by speaker ID (one embedding per speaker).
      emb_scp: Optional path to diarization sub-segment embedding .scp
        file. Required for embedding-based speaker mapping.
      utt_id: Which utterance to plot if RTTM contains multiple.
        Defaults to the first utterance found.
      start_time: Optional start of time range to display (seconds).
      end_time: Optional end of time range to display (seconds).
      show_overlap: If True, draw vertical dashed lines at the
        boundaries of overlap regions (where 2+ speakers are active
        simultaneously in the reference). Uses ref_rttm if available,
        otherwise hyp_rttm.
      ref_style: How to render reference segments. One of:
        - "background": Wide faint bar behind the hypothesis (default).
          Missed speech is visible as faint bar with no solid overlay.
        - "hollow": Hollow rectangle outline around reference segments.
        - "line": Thin line slightly offset below the hypothesis track.
      monochrome: If True, use greyscale colours instead of colour cycle.
      figsize: Optional tuple (width, height) for the figure.
      ax: Optional matplotlib Axes to plot on. If provided, wav_path is
        ignored and diarization is drawn on this axes.

    Returns:
      Tuple of (fig, axes) where axes is a single Axes or ndarray of
      Axes. If ax was provided, returns (ax.figure, ax).
    """
    colours = _MONOCHROME_COLOURS if monochrome else _DEFAULT_COLOURS

    # Parse hypothesis RTTM
    hyp_data = read_rttm(hyp_rttm, with_labels=True)
    if utt_id is None:
        utt_id = next(iter(hyp_data))
    hyp_segments = hyp_data.get(utt_id, [])

    # Parse reference RTTM if provided
    ref_segments = None
    if ref_rttm is not None:
        ref_data = read_rttm(ref_rttm, with_labels=True)
        ref_segments = ref_data.get(utt_id, [])

    # Speaker mapping
    mapping = None
    if ref_segments is not None:
        if emb_scp is not None and enrol_scp is not None:
            mapping = map_speakers_by_embeddings(
                emb_scp, hyp_segments, enrol_scp)
        else:
            mapping = map_speakers_by_overlap(ref_segments, hyp_segments)

    # Determine speaker order: use reference speakers as base, add any
    # unmapped hyp speakers at the end
    if ref_segments is not None:
        ref_speakers = sorted(set(seg[2] for seg in ref_segments))
        hyp_speaker_labels = set(seg[2] for seg in hyp_segments)
        mapped_labels = set()
        if mapping:
            mapped_labels = set(mapping.values())
        # Extra speakers from hyp that didn't map to any ref speaker
        extra_speakers = sorted(
            spk for spk in hyp_speaker_labels
            if mapping and mapping.get(spk, spk) not in set(ref_speakers))
        all_speakers = ref_speakers + extra_speakers
    else:
        all_speakers = sorted(set(seg[2] for seg in hyp_segments))

    # Build speaker -> track index
    spk_to_track = {spk: i for i, spk in enumerate(all_speakers)}
    n_tracks = len(all_speakers)

    # Create figure
    if ax is not None:
        diar_ax = ax
        fig = ax.figure
        axes_out = ax
    else:
        show_rms = wav_path is not None
        if figsize is None:
            height = max(3, n_tracks * 0.6 + (2 if show_rms else 0))
            figsize = (12, height)

        if show_rms:
            fig, (rms_ax, diar_ax) = plt.subplots(
                2, 1, figsize=figsize, sharex=True,
                constrained_layout=True,
                gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.08})
            axes_out = np.array([rms_ax, diar_ax])
        else:
            fig, diar_ax = plt.subplots(1, 1, figsize=figsize,
                                        constrained_layout=True)
            axes_out = diar_ax

    # Plot RMS envelope
    if wav_path is not None and ax is None:
        times, rms = _compute_rms_envelope(wav_path, start_time, end_time)
        rms_ax.plot(times, rms, color='#333333', linewidth=0.5)
        rms_ax.set_ylabel('RMS')
        rms_ax.set_xlim(
            start_time if start_time is not None else times[0] if len(times) else 0,
            end_time if end_time is not None else times[-1] if len(times) else 1)
        plt.setp(rms_ax.get_xticklabels(), visible=False)

    # Filter and clip segments to time range
    hyp_filtered = _filter_segments(hyp_segments, start_time, end_time)
    ref_filtered = None
    if ref_segments is not None:
        ref_filtered = _filter_segments(ref_segments, start_time, end_time)

    # Plot reference segments
    if ref_filtered is not None:
        for start, end, spk in ref_filtered:
            if spk not in spk_to_track:
                continue
            track = spk_to_track[spk]
            colour = colours[track % len(colours)]
            if ref_style == 'field':
                y = track * _TRACK_SPACING
                diar_ax.hlines(y, start, end, colors=colour,
                               linewidth=_REF_BG_LINEWIDTH,
                               linestyles='solid', alpha=_REF_BG_ALPHA,
                               zorder=1)
            elif ref_style == 'hollow':
                y_bottom = track * _TRACK_SPACING - 0.2
                height = 0.4
                rect = plt.Rectangle(
                    (start, y_bottom), end - start, height,
                    linewidth=_REF_HOLLOW_LINEWIDTH,
                    edgecolor=colour, facecolor='none',
                    linestyle='solid', alpha=0.7, zorder=1)
                diar_ax.add_patch(rect)
            elif ref_style == 'line':
                y = track * _TRACK_SPACING + _REF_OFFSET
                diar_ax.hlines(y, start, end, colors=colour,
                               linewidth=_REF_LINE_LINEWIDTH,
                               linestyles='solid', alpha=0.6,
                               zorder=1)

    # Plot hypothesis segments
    for start, end, spk in hyp_filtered:
        if mapping:
            mapped_spk = mapping.get(spk, spk)
        else:
            mapped_spk = spk
        if mapped_spk not in spk_to_track:
            continue
        track = spk_to_track[mapped_spk]
        if ref_style == 'line' and ref_segments is not None:
            y = track * _TRACK_SPACING + _HYP_OFFSET
        else:
            y = track * _TRACK_SPACING
        colour = colours[track % len(colours)]
        diar_ax.hlines(y, start, end, colors=colour,
                       linewidth=_SEGMENT_LINEWIDTH,
                       linestyles='solid', zorder=2)

    # Overlap region shading
    if show_overlap:
        overlap_source = ref_segments if ref_segments is not None else hyp_segments
        overlap_regions = _find_overlap_regions(overlap_source)
        for ov_start, ov_end in overlap_regions:
            if start_time is not None and ov_end <= start_time:
                continue
            if end_time is not None and ov_start >= end_time:
                continue
            diar_ax.axvspan(ov_start, ov_end, color='#000000',
                            alpha=0.06, zorder=0)

    # Formatting
    diar_ax.set_yticks([i * _TRACK_SPACING for i in range(n_tracks)])
    diar_ax.set_yticklabels(all_speakers)
    diar_ax.set_xlabel('Time (s)')
    diar_ax.set_ylim(-0.5, (n_tracks - 1) * _TRACK_SPACING + 0.5)
    diar_ax.invert_yaxis()

    # Set x-axis limits
    if start_time is not None or end_time is not None:
        xlim_left = start_time if start_time is not None else 0
        all_ends = [seg[1] for seg in hyp_segments]
        if ref_segments:
            all_ends += [seg[1] for seg in ref_segments]
        xlim_right = end_time if end_time is not None else max(all_ends)
        diar_ax.set_xlim(xlim_left, xlim_right)

    # Legend
    legend_handles = []
    legend_handles.append(mlines.Line2D(
        [], [], color='#888888', linewidth=_SEGMENT_LINEWIDTH,
        solid_capstyle='butt', label='Hypothesis'))
    if ref_segments is not None:
        if ref_style == 'field':
            legend_handles.append(mlines.Line2D(
                [], [], color='#888888', linewidth=_REF_BG_LINEWIDTH,
                alpha=_REF_BG_ALPHA, solid_capstyle='butt',
                label='Reference'))
        elif ref_style == 'hollow':
            legend_handles.append(mpatches.Patch(
                facecolor='none', edgecolor='#888888',
                linewidth=_REF_HOLLOW_LINEWIDTH, label='Reference'))
        elif ref_style == 'line':
            legend_handles.append(mlines.Line2D(
                [], [], color='#888888', linewidth=_REF_LINE_LINEWIDTH,
                alpha=0.6, solid_capstyle='butt', label='Reference'))
    diar_ax.legend(handles=legend_handles, loc='upper right', fontsize='small')

    return fig, axes_out


def _find_overlap_regions(segments):
    """Find time regions where 2+ speakers are active simultaneously.

    Args:
      segments: List of (start, end, speaker_label) tuples.

    Returns:
      List of (start, end) tuples representing overlap regions.
    """
    # Collect all boundaries as events
    events = []
    for start, end, spk in segments:
        events.append((start, 1))
        events.append((end, -1))
    events.sort()

    overlaps = []
    active = 0
    overlap_start = None
    for time, delta in events:
        active += delta
        if active >= 2 and overlap_start is None:
            overlap_start = time
        elif active < 2 and overlap_start is not None:
            overlaps.append((overlap_start, time))
            overlap_start = None

    return overlaps


def _filter_segments(segments, start_time, end_time):
    """Filter and clip segments to a time range.

    Args:
      segments: List of (start, end, speaker) tuples.
      start_time: Optional start boundary (seconds).
      end_time: Optional end boundary (seconds).

    Returns:
      Filtered list with segment boundaries clipped to the range.
    """
    filtered = []
    for seg_start, seg_end, spk in segments:
        if start_time is not None and seg_end <= start_time:
            continue
        if end_time is not None and seg_start >= end_time:
            continue
        clipped_start = max(seg_start, start_time) if start_time else seg_start
        clipped_end = min(seg_end, end_time) if end_time else seg_end
        filtered.append((clipped_start, clipped_end, spk))
    return filtered


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot diarization results')
    parser.add_argument('--hyp-rttm', required=True,
                        help='Path to hypothesis RTTM file')
    parser.add_argument('--ref-rttm', default=None,
                        help='Path to reference RTTM file')
    parser.add_argument('--wav', default=None,
                        help='Path to audio file for RMS envelope')
    parser.add_argument('--enrol-scp', default=None,
                        help='Path to enrolment embedding .scp file '
                             '(keyed by speaker ID)')
    parser.add_argument('--emb-scp', default=None,
                        help='Path to diarization sub-segment embedding '
                             '.scp file')
    parser.add_argument('--utt-id', default=None,
                        help='Utterance ID to plot (default: first)')
    parser.add_argument('--start', type=float, default=None,
                        help='Start time in seconds')
    parser.add_argument('--end', type=float, default=None,
                        help='End time in seconds')
    parser.add_argument('--ref-style', default='field',
                        choices=['field', 'hollow', 'line'],
                        help='Reference rendering style')
    parser.add_argument('--show-overlap', action='store_true',
                        help='Draw vertical lines at overlap boundaries')
    parser.add_argument('--monochrome', action='store_true',
                        help='Use greyscale colours')
    parser.add_argument('--output', default=None,
                        help='Save figure to path (e.g. plot.png)')
    parser.add_argument('--figsize', nargs=2, type=float, default=None,
                        help='Figure size as width height')
    args = parser.parse_args()

    figsize = tuple(args.figsize) if args.figsize else None
    fig, axes = plot_diarization(
        hyp_rttm=args.hyp_rttm,
        ref_rttm=args.ref_rttm,
        wav_path=args.wav,
        enrol_scp=args.enrol_scp,
        emb_scp=args.emb_scp,
        utt_id=args.utt_id,
        start_time=args.start,
        end_time=args.end,
        show_overlap=args.show_overlap,
        ref_style=args.ref_style,
        monochrome=args.monochrome,
        figsize=figsize,
    )

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print("Saved: {}".format(args.output))
    else:
        plt.show()
