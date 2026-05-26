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

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torchaudio

from wespeaker.diar.make_oracle_sad import read_rttm
from wespeaker.diar.map_speakers import (
    map_speakers_by_overlap,
    map_speakers_by_embeddings,
)

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
                     doa_timeline=None,
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
      doa_timeline: Optional dict as returned by
        beamform_diar.compute_doa_timeline(), with keys
        'doa_by_speaker' and 'ref_azimuths'. If provided, adds a DOA
        timeline subplot below the diarization tracks. Ignored if ax
        is provided.
      monochrome: If True, use greyscale colours instead of colour cycle.
      figsize: Optional tuple (width, height) for the figure.
      ax: Optional matplotlib Axes to plot on. If provided, wav_path
        and doa_timeline are ignored and diarization is drawn on this
        axes only.

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
        ref_speakers = list(dict.fromkeys(seg[2] for seg in ref_segments))
        # Extra speakers from hyp that didn't map to any ref speaker
        extra_speakers = [
            spk for spk in dict.fromkeys(seg[2] for seg in hyp_segments)
            if mapping and mapping.get(spk, spk) not in set(ref_speakers)]
        all_speakers = ref_speakers + extra_speakers
    else:
        all_speakers = list(dict.fromkeys(seg[2] for seg in hyp_segments))

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
        show_doa = doa_timeline is not None
        if figsize is None:
            height = max(3, n_tracks * 0.6
                         + (2 if show_rms else 0)
                         + (2.5 if show_doa else 0))
            figsize = (12, height)

        # Build subplot grid
        n_subplots = 1 + int(show_rms) + int(show_doa)
        ratios = []
        if show_rms:
            ratios.append(1)
        ratios.append(3)
        if show_doa:
            ratios.append(2)

        if n_subplots > 1:
            fig, axes_arr = plt.subplots(
                n_subplots, 1, figsize=figsize, sharex=True,
                constrained_layout=True,
                gridspec_kw={'height_ratios': ratios, 'hspace': 0.08})
            idx = 0
            if show_rms:
                rms_ax = axes_arr[idx]
                idx += 1
            diar_ax = axes_arr[idx]
            idx += 1
            if show_doa:
                doa_ax = axes_arr[idx]
            axes_out = axes_arr
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
    diar_ax.set_ylim(-0.5, (n_tracks - 1) * _TRACK_SPACING + 0.5)
    diar_ax.invert_yaxis()
    # Only show x-label on the bottom-most subplot
    if doa_timeline is not None and ax is None:
        plt.setp(diar_ax.get_xticklabels(), visible=False)
    else:
        diar_ax.set_xlabel('Time (s)')

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

    # Plot DOA timeline
    if doa_timeline is not None and ax is None:
        plot_doa_timeline(
            doa_timeline['doa_by_speaker'],
            ref_azimuths=doa_timeline.get('ref_azimuths'),
            ax=doa_ax)

    return fig, axes_out


def _wrap_angle(deg):
    """Wrap angle to [-180, 180) range (0 = broadside / +x axis)."""
    return ((deg + 180) % 360) - 180


def plot_doa_timeline(doa_by_speaker, ref_azimuths=None, ax=None,
                      figsize=(12, 4)):
    """Plot DOA estimates over time, coloured by ground-truth speaker.

    The y-axis is wrapped so that 0 degrees (broadside, positive x-axis)
    is at the centre, with the range [-180, 180]. This maps naturally
    onto the room layout where the mic array faces into the room.

    Args:
      doa_by_speaker: Dict mapping speaker_id to list of
        (midpoint_s, doa_degrees) tuples. Angles in [0, 360).
      ref_azimuths: Optional dict mapping speaker_id to reference
        azimuth in degrees. Drawn as horizontal dashed lines.
      ax: Optional matplotlib Axes. If None, creates a new figure.
      figsize: Figure size tuple (width, height).

    Returns:
      Tuple of (fig, ax).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    colours = _DEFAULT_COLOURS
    speakers = sorted(doa_by_speaker.keys())

    for i, spk in enumerate(speakers):
        colour = colours[i % len(colours)]
        points = doa_by_speaker[spk]
        times = [p[0] for p in points]
        doas = [_wrap_angle(p[1]) for p in points]
        ax.scatter(times, doas, c=colour, s=8, alpha=0.6, label=spk)

        if ref_azimuths is not None and spk in ref_azimuths:
            ax.axhline(_wrap_angle(ref_azimuths[spk]), color=colour,
                       ls='--', lw=1.5, alpha=0.7)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('DOA (degrees)')
    ax.set_ylim(-180, 180)
    ax.axhline(0, color='#cccccc', lw=0.5, zorder=0)
    ax.legend(loc='upper right', fontsize='small')

    return fig, ax


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
