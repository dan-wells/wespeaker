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

"""Load and draw face bounding boxes from 3DDFA_V2 landmark CSV files.

Adapted from 3DDFA_V2 (https://github.com/cleardusk/3DDFA_V2),
MIT License, Copyright (c) 2020 Jianzhu Guo.
"""

import csv
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Colours for distinguishing faces (BGR format).
# Adapted from 3DDFA_V2 multiface_distance_render.py.
FACE_COLORS_BGR = [
    (228, 119, 31),   # Blue
    (14, 127, 255),   # Orange
    (44, 160, 44),    # Green
    (40, 39, 214),    # Red
    (189, 103, 148),  # Purple
    (75, 86, 140),    # Brown
    (194, 119, 227),  # Pink
    (127, 127, 127),  # Gray
    (34, 189, 188),   # Olive
    (207, 190, 23),   # Cyan
]


class FaceLandmarks(object):
    """Load and draw face bounding boxes from 3DDFA_V2 landmark CSV dumps.

    Precomputes per-frame, per-face landmark data at construction time
    for efficient per-frame lookups during video processing.

    Args:
        csv_path: Path to the 3DDFA_V2 mouth_position CSV file.
        fps: Video frame rate (used to map CSV timestamps to frame indices).
        frame_height: Video frame height in pixels (for clamping).
        frame_width: Video frame width in pixels (for clamping).
        padding_ratio: Fractional padding to add around the keypoint bbox.
        line_width: Line width for drawn rectangles.
        face_colors: Optional dict mapping face_idx (str) to BGR colour
            tuple. If None, uses FACE_COLORS_BGR by index.
        face_labels: Optional dict mapping face_idx (str) to display
            label. If None, uses "Face {idx}".
        font_scale: Font scale for label text. If None, defaults to 0.5.
        scale: Optional (sx, sy) tuple to rescale CSV coordinates. Used
            when the video has been resized from the original resolution.
        buffer_duration: Duration in seconds of the temporal buffer for
            filtering spurious detections. 0 disables filtering.
        min_presence: Minimum fraction of buffer frames a face must
            appear in to be displayed.
        causal: If True, buffer looks backward from current frame. If
            False, buffer is centered on the current frame.
    """

    def __init__(self, csv_path, fps, frame_height, frame_width,
                 padding_ratio=0.35, line_width=2, face_colors=None,
                 face_labels=None, font_scale=None, scale=None,
                 buffer_duration=5.0, min_presence=0.5, causal=True):
        self._fps = fps
        self._frame_height = frame_height
        self._frame_width = frame_width
        self._padding_ratio = padding_ratio
        self._line_width = line_width
        self._face_colors = face_colors
        self._face_labels = face_labels
        self._font_scale = font_scale if font_scale is not None else 0.5
        self._scale = scale
        self._buffer_frames = int(round(buffer_duration * fps))
        self._min_presence = min_presence
        self._causal = causal
        self._data = {}
        self._load_csv(csv_path)

    def _load_csv(self, csv_path):
        """Parse CSV and build per-frame, per-face landmark index.

        Populates self._data as:
            {frame_idx: {face_idx: np.ndarray of shape (N, 3)}}
        """
        raw = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                seconds = float(row['seconds'])
                face_idx = int(row['face_idx'])
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])

                frame_idx = int(round(seconds * self._fps))

                if frame_idx not in raw:
                    raw[frame_idx] = {}
                if face_idx not in raw[frame_idx]:
                    raw[frame_idx][face_idx] = []
                raw[frame_idx][face_idx].append((x, y, z))

        for frame_idx, faces in raw.items():
            self._data[frame_idx] = {}
            for face_idx, points in faces.items():
                self._data[frame_idx][face_idx] = np.array(
                    points, dtype=np.float32)

        if self._scale is not None:
            sx, sy = self._scale
            for faces in self._data.values():
                for pts in faces.values():
                    pts[:, 0] *= sx
                    pts[:, 1] *= sy

        self._face_indices = set(
            str(fi) for faces in self._data.values() for fi in faces.keys())

        self._presence = {}
        for frame_idx, faces in self._data.items():
            for face_idx in faces:
                if face_idx not in self._presence:
                    self._presence[face_idx] = set()
                self._presence[face_idx].add(frame_idx)

        n_frames = len(self._data)
        logger.info("Loaded face landmarks: %d frames, %d unique faces",
                    n_frames, len(self._face_indices))

    @property
    def face_indices(self):
        """Set of face index strings present in the loaded CSV."""
        return self._face_indices

    def _is_face_visible(self, face_idx, frame_idx):
        """Check if a face has sufficient presence in the buffer window."""
        if self._buffer_frames == 0:
            return True
        if self._causal:
            win_start = max(0, frame_idx - self._buffer_frames + 1)
            win_end = frame_idx
        else:
            half = self._buffer_frames // 2
            win_start = max(0, frame_idx - half)
            win_end = frame_idx + (self._buffer_frames - half - 1)
        window_size = win_end - win_start + 1
        presence = self._presence.get(face_idx, set())
        count = len(presence & set(range(win_start, win_end + 1)))
        return count >= self._min_presence * window_size

    def get_bboxes(self, frame_idx):
        """Get face bounding boxes for a given frame.

        Args:
            frame_idx: Zero-based frame index.

        Returns:
            List of (face_idx, x1, y1, x2, y2) tuples with integer
            pixel coordinates, or empty list if no data for this frame.
        """
        faces = self._data.get(frame_idx, {})
        if not faces:
            return []

        bboxes = []
        for face_idx, points in faces.items():
            if not self._is_face_visible(face_idx, frame_idx):
                continue

            x_min = float(points[:, 0].min())
            x_max = float(points[:, 0].max())
            y_min = float(points[:, 1].min())
            y_max = float(points[:, 1].max())

            w = x_max - x_min
            h = y_max - y_min
            pad_w = w * self._padding_ratio
            pad_h = h * self._padding_ratio

            x1 = int(max(0, x_min - pad_w))
            y1 = int(max(0, y_min - pad_h))
            x2 = int(min(self._frame_width, x_max + pad_w))
            y2 = int(min(self._frame_height, y_max + pad_h))

            bboxes.append((face_idx, x1, y1, x2, y2))

        return bboxes

    def draw_bboxes(self, frame, frame_idx, active_faces=None):
        """Draw face bounding boxes on a video frame.

        Args:
            frame: BGR image (numpy array), modified in place.
            frame_idx: Zero-based frame index.
            active_faces: Optional set of face_idx strings that are
                currently speaking. If None, all boxes are drawn at full
                opacity. If provided, inactive/unmapped faces are drawn
                with alpha transparency.

        Returns:
            The frame (same reference, modified in place).
        """
        bboxes = self.get_bboxes(frame_idx)
        for face_idx, x1, y1, x2, y2 in bboxes:
            face_key = str(face_idx)

            if self._face_colors and face_key in self._face_colors:
                color = self._face_colors[face_key]
            else:
                color = (128, 128, 128)

            is_active = (face_key in active_faces
                         or active_faces is None)  # fall back to highlighting all

            if self._face_labels and face_key in self._face_labels:
                label = self._face_labels[face_key]
            else:
                label = "Face {}".format(face_idx)

            target = frame if is_active else frame.copy()

            cv2.rectangle(target, (x1, y1), (x2, y2), color,
                          self._line_width)
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self._font_scale, 1)
            tag_h = th + baseline + 4
            tag_x = x1 - self._line_width // 2
            tag_y1 = max(0, y1 - tag_h)
            tag_y2 = tag_y1 + tag_h
            cv2.rectangle(target, (tag_x, tag_y1), (tag_x + tw + 4, tag_y2),
                          color, cv2.FILLED)
            cv2.putText(target, label, (tag_x + 2, tag_y2 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, self._font_scale,
                        (0, 0, 0), 1, cv2.LINE_AA)

            if not is_active:
                cv2.addWeighted(target, 0.35, frame, 0.65, 0, frame)

        return frame
