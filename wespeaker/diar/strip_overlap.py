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

"""Strip overlapping speech regions from a reference RTTM.

Finds all time intervals where two or more speakers are simultaneously
active and removes those intervals from every speaker segment, outputting
a modified RTTM that contains only single-speaker regions.

When this stripped RTTM is used as the reference for md-eval.pl scoring:
  - Hypothesis speech in overlap regions counts as false alarm (penalised).
  - No hypothesis speech in overlap regions counts as correct.

This is more informative than md-eval.pl -1 (which ignores overlap
entirely): it equals -1 when a threshold suppresses all overlap
assignments, and additionally penalises any overlap leakage.
"""

import argparse
import sys

from wespeaker.diar.make_oracle_sad import read_rttm


def find_overlap_intervals(segments):
    """Find all time intervals where two or more speakers are active.

    Args:
      segments: List of (begin, end, speaker) tuples for one utterance,
        sorted by begin time.

    Returns:
      List of (begin, end) tuples covering all overlap regions, merged
        and sorted.
    """
    events = []
    for begin, end, _ in segments:
        events.append((begin, +1))
        events.append((end, -1))
    # Sort by time; break ties by putting -1 (end) before +1 (begin) so
    # that a speaker ending exactly when another begins is not counted as
    # overlap.
    events.sort(key=lambda x: (x[0], x[1]))

    overlap_intervals = []
    count = 0
    overlap_start = None
    for time, delta in events:
        prev_count = count
        count += delta
        if prev_count < 2 and count >= 2:
            overlap_start = time
        elif prev_count >= 2 and count < 2:
            overlap_intervals.append((overlap_start, time))
            overlap_start = None

    return overlap_intervals


def subtract_intervals(begin, end, overlap_intervals):
    """Subtract a list of intervals from a segment, returning sub-segments.

    Args:
      begin: Segment start time.
      end: Segment end time.
      overlap_intervals: Sorted list of (ovlp_begin, ovlp_end) to remove.

    Returns:
      List of (begin, end) tuples for the remaining portions of the
        segment. Empty list if the segment is fully covered.
    """
    result = []
    cursor = begin
    for ovlp_begin, ovlp_end in overlap_intervals:
        if ovlp_end <= cursor:
            continue
        if ovlp_begin >= end:
            break
        if ovlp_begin > cursor:
            result.append((cursor, ovlp_begin))
        cursor = max(cursor, ovlp_end)
    if cursor < end:
        result.append((cursor, end))
    return result


class RawDefaultsFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter):
    pass


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RawDefaultsFormatter)
    parser.add_argument('--rttm', required=True,
                        help='input reference RTTM file')
    parser.add_argument('--output', default='-',
                        help='output RTTM file path (- for stdout)')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    utt_to_segments = read_rttm(args.rttm, with_labels=True)

    rttm_line = 'SPEAKER {} 1 {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>'
    out = open(args.output, 'w') if args.output != '-' else sys.stdout
    try:
        for utt, segments in utt_to_segments.items():
            overlap_intervals = find_overlap_intervals(segments)
            for begin, end, speaker in segments:
                sub_segs = subtract_intervals(begin, end, overlap_intervals)
                for sub_begin, sub_end in sub_segs:
                    print(rttm_line.format(
                        utt, sub_begin, sub_end - sub_begin, speaker),
                        file=out)
    finally:
        if args.output != '-':
            out.close()


if __name__ == '__main__':
    main()
