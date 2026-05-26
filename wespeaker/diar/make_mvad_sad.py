"""Generate Kaldi-style segments file using MVAD_V2 multi-class VAD.

Produces segments containing only single-speaker regions (overlap and
silence are excluded). Output format matches make_system_sad.py:
  {utt}-{begin_ms:08d}-{end_ms:08d} {utt} {begin_sec:.3f} {end_sec:.3f}
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

from wespeaker.diar.mvad import (
    CLASS_SINGLE,
    load_mvad_model,
    mvad_predict_filtered,
)
from wespeaker.utils.audio import SAMPLE_RATE, load_audio
from wespeaker.utils.file_utils import read_scp


class RawDefaultsFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter):
    pass


def extract_single_speaker_segments(labels, hop_sec=0.01, min_duration=0.0):
    """Extract contiguous single-speaker regions from frame labels.

    Args:
        labels: np.ndarray of shape (T,), int32, with class labels.
        hop_sec: Frame hop in seconds (0.01 for 100Hz).
        min_duration: Minimum segment duration in seconds.

    Returns:
        List of (begin_sec, end_sec) tuples.
    """
    segments = []
    n_frames = len(labels)
    in_segment = False
    seg_start = 0

    for i in range(n_frames):
        if labels[i] == CLASS_SINGLE and not in_segment:
            seg_start = i
            in_segment = True
        elif labels[i] != CLASS_SINGLE and in_segment:
            begin = seg_start * hop_sec
            end = i * hop_sec
            if end - begin >= min_duration:
                segments.append((begin, end))
            in_segment = False

    # Handle segment running to end of file
    if in_segment:
        begin = seg_start * hop_sec
        end = n_frames * hop_sec
        if end - begin >= min_duration:
            segments.append((begin, end))

    return segments


def process_file(utt, wav_path, model, feat_mean, feat_std, device,
                 min_duration):
    """Run MVAD on a single file and return formatted segments string.

    Args:
        utt: Utterance ID.
        wav_path: Path to audio file.
        model: MVAD_V2 model instance.
        feat_mean: Feature mean array.
        feat_std: Feature std array.
        device: Torch device string.
        min_duration: Minimum segment duration in seconds.

    Returns:
        String of formatted segment lines.
    """
    audio = load_audio(wav_path)
    labels = mvad_predict_filtered(
        audio, SAMPLE_RATE, model, feat_mean, feat_std, device)

    segments = extract_single_speaker_segments(
        labels, hop_sec=0.01, min_duration=min_duration)

    result = ""
    for begin, end in segments:
        begin_ms = int(begin * 1000)
        end_ms = int(end * 1000)
        result += "{}-{:08d}-{:08d} {} {:.3f} {:.3f}\n".format(
            utt, begin_ms, end_ms, utt, begin, end)

    return result


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RawDefaultsFormatter)
    parser.add_argument("--scp", required=True, help="wav.scp file")
    parser.add_argument("--model", required=True,
                        help="path to MVAD_V2 checkpoint")
    parser.add_argument("--min-duration", type=float, default=0.25,
                        help="minimum segment duration in seconds")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                        help="inference device")
    return parser.parse_args()


def main():
    args = get_args()

    model, feat_mean, feat_std, device = load_mvad_model(
        args.model, args.device)

    utt_wav_pairs = read_scp(args.scp)

    for utt, wav_path in utt_wav_pairs:
        output = process_file(
            utt, wav_path, model, feat_mean, feat_std, device,
            args.min_duration)
        print(output, end="")


if __name__ == "__main__":
    main()
