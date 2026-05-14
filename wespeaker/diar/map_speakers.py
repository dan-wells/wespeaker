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

"""Speaker label mapping for diarization output.

Maps anonymous integer cluster labels in a hypothesis RTTM to named speaker
IDs using one of two strategies:

  - Overlap-based (--ref-rttm): matches hypothesis clusters to reference
    speakers by maximising temporal overlap (Hungarian algorithm).
  - Embedding-based (--enrol-scp + --emb-scp): scores hypothesis cluster
    centroids against enrolled speaker embeddings by cosine similarity
    (Hungarian algorithm). Takes precedence when --enrol-scp is provided.

Usage::

    python3 wespeaker/diar/map_speakers.py \\
        --hyp-rttm hyp.rttm \\
        --output mapped.rttm \\
        --ref-rttm ref.rttm

    python3 wespeaker/diar/map_speakers.py \\
        --hyp-rttm hyp.rttm \\
        --output mapped.rttm \\
        --enrol-scp enrol.scp \\
        --emb-scp emb.scp
"""

import argparse
from collections import OrderedDict

import kaldiio
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

from wespeaker.diar.make_oracle_sad import read_rttm


class RawDefaultsFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter):
    pass


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
    enrol_ids, enrol_embs = _load_enrol_scp(enrol_scp)
    emb_dict = _load_emb_scp(emb_scp)
    return _map_speakers_by_emb_dict(emb_dict, hyp_segments,
                                     enrol_ids, enrol_embs)


def _load_enrol_scp(enrol_scp):
    """Load enrolment embeddings from a .scp file.

    Args:
      enrol_scp: Path to the .scp file keyed by speaker ID.

    Returns:
      Tuple of (enrol_ids, enrol_embs) where enrol_ids is a list of
      speaker ID strings and enrol_embs is a numpy array of shape
      (n_speakers, emb_dim).
    """
    enrol_ids = []
    enrol_embs = []
    for spk_id, emb in kaldiio.load_scp_sequential(enrol_scp):
        enrol_ids.append(spk_id)
        enrol_embs.append(emb.astype(np.float32))
    return enrol_ids, np.stack(enrol_embs)


def _load_emb_scp(emb_scp):
    """Load sub-segment embeddings from a .scp file.

    Args:
      emb_scp: Path to the .scp file.

    Returns:
      OrderedDict mapping subseg_id to embedding array (float32).
    """
    emb_dict = OrderedDict()
    for subseg_id, emb in kaldiio.load_scp_sequential(emb_scp):
        emb_dict[subseg_id] = emb.astype(np.float32)
    return emb_dict


def _map_speakers_by_emb_dict(emb_dict, hyp_segments, enrol_ids, enrol_embs):
    """Map hypothesis speakers to enrolled speakers given pre-loaded embeddings.

    Args:
      emb_dict: OrderedDict mapping subseg_id to embedding array.
      hyp_segments: List of (start, end, speaker_label) tuples from
        hypothesis RTTM.
      enrol_ids: List of enrolled speaker ID strings.
      enrol_embs: Numpy array of shape (n_enrol, emb_dim) with stacked
        enrolment embeddings.

    Returns:
      Dict mapping hyp_speaker_label -> enrolled_speaker_id (best match).
    """
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RawDefaultsFormatter)
    parser.add_argument('--hyp-rttm', required=True,
                        help='Input hypothesis RTTM file')
    parser.add_argument('--output', required=True,
                        help='Output RTTM with mapped speaker labels')
    parser.add_argument('--ref-rttm', default=None,
                        help='Reference RTTM for overlap-based mapping')
    parser.add_argument('--enrol-scp', default=None,
                        help='Enrolment embedding .scp (enables '
                             'embedding-based mapping)')
    parser.add_argument('--emb-scp', default=None,
                        help='Diarization sub-segment embedding .scp '
                             '(required with --enrol-scp)')
    parser.add_argument('--channel', type=int, default=1,
                        help='Channel field value in output RTTM')
    args = parser.parse_args()

    if args.enrol_scp is None and args.ref_rttm is None:
        parser.error('one of --enrol-scp or --ref-rttm is required')
    if args.enrol_scp is not None and args.emb_scp is None:
        parser.error('--emb-scp is required when --enrol-scp is provided')

    hyp_data = read_rttm(args.hyp_rttm, with_labels=True)
    rttm_fmt = "SPEAKER {} {} {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>"
    channel = args.channel

    if args.enrol_scp is not None:
        # Load enrolment embeddings once
        enrol_ids, enrol_embs = _load_enrol_scp(args.enrol_scp)

        # Load all sub-seg embeddings once, grouped by utt prefix
        all_emb = _load_emb_scp(args.emb_scp)
        utt_emb_dicts = {}
        for subseg_id, emb in all_emb.items():
            parts = subseg_id.split('-')
            utt = '-'.join(parts[:-4])
            if utt not in utt_emb_dicts:
                utt_emb_dicts[utt] = OrderedDict()
            utt_emb_dicts[utt][subseg_id] = emb

        with open(args.output, 'w') as out_f:
            for utt, hyp_segs in hyp_data.items():
                utt_emb = utt_emb_dicts.get(utt, OrderedDict())
                mapping = _map_speakers_by_emb_dict(
                    utt_emb, hyp_segs, enrol_ids, enrol_embs)
                for start, end, spk in hyp_segs:
                    mapped_spk = mapping.get(spk, spk)
                    out_f.write(rttm_fmt.format(
                        utt, channel, start, end - start, mapped_spk) + '\n')
    else:
        ref_data = read_rttm(args.ref_rttm, with_labels=True)

        with open(args.output, 'w') as out_f:
            for utt, hyp_segs in hyp_data.items():
                ref_segs = ref_data.get(utt, [])
                if ref_segs:
                    mapping = map_speakers_by_overlap(ref_segs, hyp_segs)
                else:
                    mapping = {seg[2]: seg[2] for seg in hyp_segs}
                for start, end, spk in hyp_segs:
                    mapped_spk = mapping.get(spk, spk)
                    out_f.write(rttm_fmt.format(
                        utt, channel, start, end - start, mapped_spk) + '\n')


if __name__ == '__main__':
    main()
