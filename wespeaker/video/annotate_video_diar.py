# Copyright (c) 2026 Dan Wells
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

"""Annotate a video with a scrolling diarization timeline strip.

Reads a video file and hypothesis RTTM, renders a per-speaker timeline
that scrolls with playback, and produces an output video with audio.
"""

import argparse
import logging
import os
import shutil
import subprocess
import tempfile

import cv2
from tqdm import tqdm
import numpy as np

from wespeaker.diar.make_oracle_sad import read_rttm
from wespeaker.diar.map_speakers import map_speakers_by_overlap
from wespeaker.diar.plot_diar import (
    _DEFAULT_COLOURS,
    _filter_segments,
    _find_overlap_regions,
)
from wespeaker.utils.file_utils import read_scp
from wespeaker.video.faces_3ddfa import FaceLandmarks

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

_STRIP_THEMES = {
    'dark': {
        'bg': (30, 30, 30),
        'cursor': (255, 255, 255),
        'label': (255, 255, 255),
        'overlap': (80, 80, 80),
    },
    'light': {
        'bg': (240, 240, 240),
        'cursor': (60, 60, 60),
        'label': (40, 40, 40),
        'overlap': (200, 200, 200),
    },
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_THICKNESS = 1
_FONT_SCALE_REF = 0.45
_FONT_HEIGHT_REF = 720  # set readable font height based on 720p video


def compute_font_scale(frame_height):
    """Compute font scale proportional to frame height.

    Args:
        frame_height: Video frame height in pixels.

    Returns:
        Font scale suitable for cv2.putText.
    """
    return _FONT_SCALE_REF * frame_height / _FONT_HEIGHT_REF


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Annotate a video with a scrolling diarization strip.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video', required=True,
                        help='Input video file path')
    parser.add_argument('--audio', default=None,
                        help='Audio file (extracted from video if omitted)')
    parser.add_argument('--audio-offset', type=float, default=0.0,
                        help='Audio offset in seconds. Positive: audio '
                             'starts before video (trims from audio start). '
                             'Negative: video starts before audio (pads '
                             'silence at audio start).')
    parser.add_argument('--hyp-rttm', required=True,
                        help='Hypothesis RTTM file')
    parser.add_argument('--ref-rttm', default=None,
                        help='Reference RTTM file')
    parser.add_argument('--output', required=True,
                        help='Output video file path')
    parser.add_argument('--spk2spk', default=None,
                        help='Speaker label mapping file '
                             '(lines: orig_label mapped_label)')
    parser.add_argument('--utt-id', default=None,
                        help='Utterance ID to use (default: first in RTTM)')
    parser.add_argument('--window-width', type=float, default=30.0,
                        help='Sliding window width in seconds')
    parser.add_argument('--cursor-position', type=float, default=0.25,
                        help='Cursor x-position as fraction of width (0-1)')
    parser.add_argument('--mode', default='extend',
                        choices=['extend', 'overlay'],
                        help='Layout mode: extend canvas or overlay on video')
    parser.add_argument('--overlay-ratio', type=float, default=0.15,
                        help='Fraction of video height for overlay strip')
    parser.add_argument('--overlay-alpha', type=float, default=0.6,
                        help='Blending alpha for overlay mode (0-1)')
    parser.add_argument('--strip-height', type=int, default=120,
                        help='Strip height in pixels (extend mode)')
    parser.add_argument('--ref-style', default='hollow',
                        choices=['field', 'hollow', 'line'],
                        help='Reference segment rendering style')
    parser.add_argument('--show-overlap', action='store_true',
                        help='Shade overlap regions')
    parser.add_argument('--strip-bg', default=None,
                        choices=['dark', 'light'],
                        help='Strip background colour (default: light)')
    parser.add_argument('--faces', default=None,
                        help='3DDFA_V2 landmark CSV file for face bounding '
                             'boxes')
    parser.add_argument('--faces-orig-size', default=None,
                        help='Original video resolution as WxH (e.g. '
                             '1920x1080) for rescaling face coordinates')
    parser.add_argument('--faces-buffer', type=float, default=5.0,
                        help='Temporal buffer duration in seconds for '
                             'filtering spurious face detections (0 to '
                             'disable)')
    parser.add_argument('--faces-min-presence', type=float, default=0.5,
                        help='Minimum fraction of buffer frames a face '
                             'must appear in to be displayed')
    parser.add_argument('--faces-active-only', action='store_true',
                        help='Only draw face bounding boxes for active '
                             'speakers (hides inactive faces entirely)')
    parser.add_argument('--faces-enrol-threshold', type=float, default=None,
                        help='Cumulative speech duration (seconds) before a '
                             'speaker is considered enrolled. Until enrolled, '
                             'active faces are drawn transparently. '
                             '(default: disabled)')
    parser.add_argument('--face2spk', default=None,
                        help='Face-to-speaker mapping file '
                             '(lines: face_idx speaker_label)')
    parser.add_argument('--codec', default='mp4v',
                        help='FourCC codec for video writer')
    parser.add_argument('--reencode', action='store_true', default=False,
                        help='Re-encode video with libx264 during mux '
                             '(smaller output, slower)')
    return parser.parse_args()


def hex_to_bgr(hex_colour):
    """Convert a hex colour string to a BGR tuple for OpenCV.

    Args:
        hex_colour: Colour string like '#1f77b4'.

    Returns:
        Tuple of (B, G, R) integers.
    """
    hex_colour = hex_colour.lstrip('#')
    r = int(hex_colour[0:2], 16)
    g = int(hex_colour[2:4], 16)
    b = int(hex_colour[4:6], 16)
    return (b, g, r)


def extract_audio_from_video(video_path, output_audio_path):
    """Extract audio track from a video file using ffmpeg.

    Args:
        video_path: Path to the input video.
        output_audio_path: Path to write the extracted audio (wav).

    Raises:
        RuntimeError: If ffmpeg returns a non-zero exit code.
    """
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000',
        output_audio_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg audio extraction failed:\n{}".format(
                result.stderr.decode('utf-8', errors='replace')))


def mux_audio_video(video_path, audio_path, output_path, audio_offset=0.0,
                    reencode=False):
    """Combine a silent video with an audio file using ffmpeg.

    Args:
        video_path: Path to the silent annotated video.
        audio_path: Path to the audio file.
        output_path: Path for the final output video.
        audio_offset: Offset in seconds. Positive: audio started before
            video, seek this far into audio. Negative: video started
            before audio, pad silence at audio start.
        reencode: If True, re-encode video with libx264 (smaller output).

    Raises:
        RuntimeError: If ffmpeg returns a non-zero exit code.
    """
    cmd = ['ffmpeg', '-y', '-i', video_path]
    if audio_offset > 0.0:
        # Audio starts before video: seek into audio to align
        cmd += ['-ss', str(audio_offset)]
    elif audio_offset < 0.0:
        # Video starts before audio: delay audio with silence
        cmd += ['-itsoffset', str(-audio_offset)]
    cmd += ['-i', audio_path]
    if reencode:
        cmd += ['-c:v', 'libx264', '-crf', '23', '-preset', 'medium']
    else:
        cmd += ['-c:v', 'copy']
    cmd += ['-c:a', 'aac', '-shortest', output_path]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg muxing failed:\n{}".format(
                result.stderr.decode('utf-8', errors='replace')))


def _get_audio_duration(audio_path):
    """Get duration of an audio file in seconds using ffprobe.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Duration in seconds, or None if it cannot be determined.
    """
    cmd = [
        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except (ValueError, TypeError):
        return None


def build_speaker_layout(hyp_segments, ref_segments=None, mapping=None):
    """Determine speaker ordering and track assignments.

    Args:
        hyp_segments: List of (start, end, speaker) tuples.
        ref_segments: Optional list of (start, end, speaker) tuples.
        mapping: Optional dict mapping hyp_speaker -> ref_speaker.

    Returns:
        Tuple of (all_speakers, spk_to_track) where all_speakers is
        an ordered list and spk_to_track maps speaker -> track index.
    """
    if ref_segments is not None:
        ref_speakers = list(dict.fromkeys(seg[2] for seg in ref_segments))
        extra_speakers = [
            spk for spk in dict.fromkeys(seg[2] for seg in hyp_segments)
            if mapping and mapping.get(spk, spk) not in set(ref_speakers)]
        all_speakers = ref_speakers + extra_speakers
    else:
        all_speakers = list(dict.fromkeys(seg[2] for seg in hyp_segments))

    spk_to_track = {spk: i for i, spk in enumerate(all_speakers)}
    return all_speakers, spk_to_track


def time_to_x(t, window_start, window_end, strip_width):
    """Convert a timestamp to an x-pixel coordinate.

    Args:
        t: Time value in seconds.
        window_start: Left edge of the visible window (seconds).
        window_end: Right edge of the visible window (seconds).
        strip_width: Width of the strip in pixels.

    Returns:
        Integer x-coordinate.
    """
    if window_end == window_start:
        return 0
    frac = (t - window_start) / (window_end - window_start)
    return int(frac * strip_width)


def render_diarization_strip(timestamp, window_width, cursor_position,
                             strip_width, strip_height, hyp_segments,
                             all_speakers, spk_to_track, colours,
                             display_names=None,
                             ref_segments=None, mapping=None,
                             ref_style='hollow', show_overlap=False,
                             theme=None, font_scale=None):
    """Render the diarization timeline strip for a single frame.

    Args:
        timestamp: Current playback time in seconds.
        window_width: Duration of the visible window in seconds.
        cursor_position: Fraction (0-1) of strip width for the cursor.
        strip_width: Width of the strip in pixels.
        strip_height: Height of the strip in pixels.
        hyp_segments: Full list of (start, end, speaker) tuples.
        all_speakers: Ordered list of speaker labels.
        spk_to_track: Dict mapping speaker label to track index.
        colours: List of BGR colour tuples.
        display_names: Optional dict mapping speaker label to display name.
        ref_segments: Optional full list of reference segment tuples.
        mapping: Optional dict mapping hyp_speaker -> ref_speaker.
        ref_style: One of 'field', 'hollow', 'line'.
        show_overlap: Whether to shade overlap regions.
        theme: Dict with keys 'bg', 'cursor', 'label', 'overlap' mapping
            to BGR tuples. Defaults to _STRIP_THEMES['dark'].
        font_scale: Font scale for labels. If None, uses _FONT_SCALE_REF.

    Returns:
        numpy.ndarray of shape (strip_height, strip_width, 3) dtype uint8.
    """
    if theme is None:
        theme = _STRIP_THEMES['dark']
    if font_scale is None:
        font_scale = _FONT_SCALE_REF
    bg_colour = theme['bg']
    cursor_colour = theme['cursor']
    label_colour = theme['label']
    overlap_colour = theme['overlap']

    strip = np.full((strip_height, strip_width, 3), bg_colour,
                    dtype=np.uint8)

    window_start = timestamp - cursor_position * window_width
    window_end = timestamp + (1.0 - cursor_position) * window_width

    n_tracks = len(all_speakers)
    if n_tracks == 0:
        return strip

    track_height = strip_height / n_tracks
    bar_height = max(2, int(track_height * 0.5))

    # Draw reference segments
    if ref_segments is not None:
        ref_filtered = _filter_segments(ref_segments, window_start, window_end)
        for seg_start, seg_end, spk in ref_filtered:
            if spk not in spk_to_track:
                continue
            track = spk_to_track[spk]
            colour = colours[track % len(colours)]
            x1 = max(0, time_to_x(seg_start, window_start, window_end,
                                   strip_width))
            x2 = min(strip_width, time_to_x(seg_end, window_start,
                                            window_end, strip_width))
            x2 = max(x2, x1 + 1)
            y_centre = int((track + 0.5) * track_height)

            if ref_style == 'field':
                y1 = y_centre - bar_height
                y2 = y_centre + bar_height
                overlay = strip[y1:y2, x1:x2].copy()
                cv2.rectangle(strip, (x1, y1), (x2, y2), colour, cv2.FILLED)
                cv2.addWeighted(strip[y1:y2, x1:x2], 0.25, overlay, 0.75,
                                0, strip[y1:y2, x1:x2])
            elif ref_style == 'hollow':
                y1 = y_centre - bar_height // 2
                y2 = y_centre + bar_height // 2
                cv2.rectangle(strip, (x1, y1), (x2, y2), colour, 1)
            elif ref_style == 'line':
                y_line = y_centre - bar_height // 2 - 2
                cv2.line(strip, (x1, y_line), (x2, y_line), colour, 1)

    # Draw hypothesis segments
    hyp_filtered = _filter_segments(hyp_segments, window_start, window_end)
    for seg_start, seg_end, spk in hyp_filtered:
        if mapping:
            mapped_spk = mapping.get(spk, spk)
        else:
            mapped_spk = spk
        if mapped_spk not in spk_to_track:
            continue
        track = spk_to_track[mapped_spk]
        colour = colours[track % len(colours)]
        x1 = max(0, time_to_x(seg_start, window_start, window_end,
                               strip_width))
        x2 = min(strip_width, time_to_x(seg_end, window_start, window_end,
                                        strip_width))
        x2 = max(x2, x1 + 1)
        y_centre = int((track + 0.5) * track_height)
        y1 = y_centre - bar_height // 2
        y2 = y_centre + bar_height // 2
        cv2.rectangle(strip, (x1, y1), (x2, y2), colour, cv2.FILLED)

    # Overlap shading
    if show_overlap:
        overlap_source = (ref_segments if ref_segments is not None
                          else hyp_segments)
        overlap_regions = _find_overlap_regions(overlap_source)
        for ov_start, ov_end in overlap_regions:
            if ov_end <= window_start or ov_start >= window_end:
                continue
            ox1 = max(0, time_to_x(ov_start, window_start, window_end,
                                   strip_width))
            ox2 = min(strip_width, time_to_x(ov_end, window_start,
                                             window_end, strip_width))
            overlay = strip[:, ox1:ox2].copy()
            cv2.rectangle(strip, (ox1, 0), (ox2, strip_height),
                          overlap_colour, cv2.FILLED)
            cv2.addWeighted(strip[:, ox1:ox2], 0.3, overlay, 0.7,
                            0, strip[:, ox1:ox2])

    return strip


def draw_strip_overlays(target, strip_y_offset, strip_width, strip_height,
                        cursor_position, all_speakers, spk_to_track,
                        display_names=None, font_scale=None, theme=None):
    """Draw cursor and speaker labels onto the composited output frame.

    This is drawn after compositing so that text is never affected by
    overlay alpha blending.

    Args:
        target: Output frame (modified in place).
        strip_y_offset: Y pixel offset where the strip region starts.
        strip_width: Width of the strip in pixels.
        strip_height: Height of the strip in pixels.
        cursor_position: Fraction (0-1) of strip width for the cursor.
        all_speakers: Ordered list of speaker labels.
        spk_to_track: Dict mapping speaker label to track index.
        display_names: Optional dict mapping speaker label to display name.
        font_scale: Font scale for labels. If None, uses _FONT_SCALE_REF.
        theme: Dict with keys 'bg', 'cursor', 'label'. Defaults to dark.
    """
    if theme is None:
        theme = _STRIP_THEMES['dark']
    if font_scale is None:
        font_scale = _FONT_SCALE_REF
    bg_colour = theme['bg']
    cursor_colour = theme['cursor']
    label_colour = theme['label']

    n_tracks = len(all_speakers)
    if n_tracks == 0:
        return
    track_height = strip_height / n_tracks

    cursor_x = int(cursor_position * strip_width)
    cv2.line(target, (cursor_x, strip_y_offset),
             (cursor_x, strip_y_offset + strip_height - 1),
             cursor_colour, 1)

    label_margin = 8
    for spk in all_speakers:
        track = spk_to_track[spk]
        if display_names:
            label = display_names.get(spk, spk)
        else:
            label = spk
        y_centre = int(strip_y_offset + (track + 0.5) * track_height)
        (text_w, text_h), _ = cv2.getTextSize(
            label, _FONT, font_scale, _FONT_THICKNESS)
        text_x = cursor_x - text_w - label_margin
        text_y = y_centre + text_h // 2
        if text_x >= 0:
            cv2.putText(target, label, (text_x, text_y), _FONT,
                        font_scale, bg_colour, _FONT_THICKNESS + 1,
                        cv2.LINE_AA)
            cv2.putText(target, label, (text_x, text_y), _FONT,
                        font_scale, label_colour, _FONT_THICKNESS,
                        cv2.LINE_AA)


def composite_frame(frame, strip, mode, overlay_alpha=0.6):
    """Attach the diarization strip to a video frame.

    Args:
        frame: Video frame as numpy array (H, W, 3).
        strip: Rendered strip as numpy array (strip_H, W, 3).
        mode: 'extend' or 'overlay'.
        overlay_alpha: Blending factor for overlay mode.

    Returns:
        Composite frame as numpy array.
    """
    if mode == 'extend':
        return np.vstack([frame, strip])
    else:
        h = strip.shape[0]
        region = frame[-h:, :, :]
        blended = cv2.addWeighted(strip, overlay_alpha, region,
                                  1.0 - overlay_alpha, 0)
        output = frame.copy()
        output[-h:, :, :] = blended
        return output


def annotate_video(args):
    """Main pipeline: read video, render annotation, write output.

    Args:
        args: Parsed argparse.Namespace.
    """
    # Validate ffmpeg
    if shutil.which('ffmpeg') is None:
        raise RuntimeError("ffmpeg not found on PATH")

    if not os.path.isfile(args.video):
        raise FileNotFoundError(
            "Video file not found: {}".format(args.video))
    if not os.path.isfile(args.hyp_rttm):
        raise FileNotFoundError(
            "Hypothesis RTTM not found: {}".format(args.hyp_rttm))

    # Read RTTM(s)
    hyp_data = read_rttm(args.hyp_rttm, with_labels=True)
    utt_id = args.utt_id
    if utt_id is None:
        utt_id = next(iter(hyp_data))
    if utt_id not in hyp_data:
        raise ValueError(
            "utt_id '{}' not found in hypothesis RTTM. "
            "Available: {}".format(utt_id, list(hyp_data.keys())))
    hyp_segments = hyp_data[utt_id]

    ref_segments = None
    mapping = None
    if args.ref_rttm is not None:
        if not os.path.isfile(args.ref_rttm):
            raise FileNotFoundError(
                "Reference RTTM not found: {}".format(args.ref_rttm))
        ref_data = read_rttm(args.ref_rttm, with_labels=True)
        ref_segments = ref_data.get(utt_id, [])
        if ref_segments:
            mapping = map_speakers_by_overlap(ref_segments, hyp_segments)

    # Shift RTTM timestamps to align with video timeline
    if args.audio_offset != 0.0:
        hyp_segments = [(s - args.audio_offset, e - args.audio_offset, spk)
                        for s, e, spk in hyp_segments]
        if ref_segments is not None:
            ref_segments = [(s - args.audio_offset, e - args.audio_offset, spk)
                           for s, e, spk in ref_segments]

    # Speaker layout
    all_speakers, spk_to_track = build_speaker_layout(
        hyp_segments, ref_segments, mapping)

    if len(all_speakers) > 10:
        logger.warning("Large number of speakers (%d) -- tracks will be thin",
                       len(all_speakers))

    # Display name mapping
    display_names = None
    if args.spk2spk is not None:
        if not os.path.isfile(args.spk2spk):
            raise FileNotFoundError(
                "spk2spk file not found: {}".format(args.spk2spk))
        display_names = dict(read_scp(args.spk2spk))

    # Colour palette and strip theme
    colours = [hex_to_bgr(c) for c in _DEFAULT_COLOURS]
    strip_bg = args.strip_bg
    if strip_bg is None:
        strip_bg = 'dark'
    theme = _STRIP_THEMES[strip_bg]

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open video: {}".format(args.video))

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info("Video: %dx%d @ %.2f fps, %d frames",
                frame_width, frame_height, fps, total_frames)

    font_scale = compute_font_scale(frame_height)

    # Load face landmarks if provided
    face_landmarks = None
    face2spk = None
    spk2face = None
    if args.faces is not None:
        if not os.path.isfile(args.faces):
            raise FileNotFoundError(
                "Faces CSV not found: {}".format(args.faces))

        face_colors = None
        face_labels = None
        if args.face2spk is not None:
            if not os.path.isfile(args.face2spk):
                raise FileNotFoundError(
                    "face2spk file not found: {}".format(args.face2spk))
            face2spk = dict(read_scp(args.face2spk))
            spk2face = {spk: fi for fi, spk in face2spk.items()}
            face_colors = {}
            face_labels = {}
            for face_idx_str, spk in face2spk.items():
                if spk in spk_to_track:
                    track = spk_to_track[spk]
                    face_colors[face_idx_str] = colours[
                        track % len(colours)]
                label = spk
                if display_names and spk in display_names:
                    label = display_names[spk]
                face_labels[face_idx_str] = label

            # Warn about unmapped speakers
            mapped_spks = set(face2spk.values())
            for spk in all_speakers:
                if spk not in mapped_spks:
                    logger.warning(
                        "Speaker '%s' in RTTM has no face mapping", spk)

        face_scale = None
        if args.faces_orig_size is not None:
            orig_w, orig_h = [int(x) for x in
                              args.faces_orig_size.split('x')]
            face_scale = (frame_width / orig_w, frame_height / orig_h)

        face_landmarks = FaceLandmarks(
            args.faces, fps, frame_height, frame_width,
            face_colors=face_colors, face_labels=face_labels,
            font_scale=0.8*font_scale, scale=face_scale,
            buffer_duration=args.faces_buffer,
            min_presence=args.faces_min_presence)

        # Warn about unmapped faces
        if face2spk is not None:
            for fi in face_landmarks.face_indices:
                if fi not in face2spk:
                    logger.warning(
                        "Face %s in CSV has no speaker mapping", fi)

    # Determine strip dimensions
    if args.mode == 'overlay':
        strip_height = int(frame_height * args.overlay_ratio)
        out_height = frame_height
    else:
        strip_height = args.strip_height
        out_height = frame_height + strip_height
    strip_width = frame_width

    # Check audio/video duration mismatch (only for separate audio files)
    if args.audio is not None:
        if not os.path.isfile(args.audio):
            raise FileNotFoundError(
                "Audio file not found: {}".format(args.audio))
        audio_duration = _get_audio_duration(args.audio)
        video_duration = total_frames / fps
        if audio_duration is not None:
            effective_audio = audio_duration - args.audio_offset
            diff = effective_audio - video_duration
            if abs(diff) > 0.1:
                logger.warning(
                    "Audio/video duration mismatch: audio %.2fs (effective "
                    "%.2fs with offset %.2fs), video %.2fs, difference %.2fs",
                    audio_duration, effective_audio, args.audio_offset,
                    video_duration, diff)

    # Temp file for silent video
    tmp_dir = tempfile.mkdtemp()
    tmp_video = os.path.join(tmp_dir, 'silent.mp4')

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(tmp_video, fourcc, fps,
                             (frame_width, out_height))
    if not writer.isOpened():
        raise RuntimeError(
            "Cannot open video writer with codec '{}'".format(args.codec))

    # Precompute enrolment threshold crossings per speaker
    enrol_threshold = args.faces_enrol_threshold
    enrol_frame = {}
    if enrol_threshold is not None and enrol_threshold > 0:
        spk_segments = {}
        for start, end, spk in hyp_segments:
            spk_segments.setdefault(spk, []).append((start, end))
        for spk, segs in spk_segments.items():
            cumulative = 0.0
            for start, end in sorted(segs):
                cumulative += end - start
                if cumulative >= enrol_threshold:
                    enrol_time = end - (cumulative - enrol_threshold)
                    enrol_frame[spk] = int(enrol_time * fps)
                    break

    for frame_idx in tqdm(range(total_frames), desc='Rendering',
                           unit='frame'):
        ret, frame = cap.read()
        if not ret:
            break

        if face_landmarks is not None:
            active_faces = None
            enrolled_faces = None
            if spk2face is not None:
                timestamp = frame_idx / fps
                active_spks = set(
                    spk for start, end, spk in hyp_segments
                    if start <= timestamp < end)
                active_faces = set(
                    spk2face[spk] for spk in active_spks
                    if spk in spk2face)
                if enrol_frame:
                    enrolled_faces = set(
                        spk2face[spk] for spk in all_speakers
                        if spk in spk2face
                        and spk in enrol_frame
                        and frame_idx >= enrol_frame[spk])
            face_landmarks.draw_bboxes(frame, frame_idx, active_faces,
                                       hide_inactive=args.faces_active_only,
                                       enrolled_faces=enrolled_faces)

        timestamp = frame_idx / fps

        strip = render_diarization_strip(
            timestamp=timestamp,
            window_width=args.window_width,
            cursor_position=args.cursor_position,
            strip_width=strip_width,
            strip_height=strip_height,
            hyp_segments=hyp_segments,
            all_speakers=all_speakers,
            spk_to_track=spk_to_track,
            colours=colours,
            display_names=display_names,
            ref_segments=ref_segments,
            mapping=mapping,
            ref_style=args.ref_style,
            show_overlap=args.show_overlap,
            theme=theme,
            font_scale=font_scale,
        )

        output_frame = composite_frame(frame, strip, args.mode,
                                       args.overlay_alpha)
        strip_y_offset = output_frame.shape[0] - strip_height
        draw_strip_overlays(
            output_frame, strip_y_offset, strip_width, strip_height,
            args.cursor_position, all_speakers, spk_to_track,
            display_names=display_names, font_scale=font_scale,
            theme=theme)
        writer.write(output_frame)

    cap.release()
    writer.release()

    # Handle audio
    audio_path = args.audio
    if audio_path is None:
        tmp_audio = os.path.join(tmp_dir, 'audio.wav')
        logger.info("Extracting audio from video...")
        extract_audio_from_video(args.video, tmp_audio)
        audio_path = tmp_audio

    # Mux
    logger.info("Muxing audio and video...")
    mux_audio_video(tmp_video, audio_path, args.output,
                    audio_offset=args.audio_offset,
                    reencode=args.reencode)
    logger.info("Output written: %s", args.output)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    """Entry point."""
    args = get_args()
    annotate_video(args)


if __name__ == '__main__':
    main()
