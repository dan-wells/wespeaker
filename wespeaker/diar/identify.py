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

"""Speaker identification via cosine similarity to enrolled embeddings.

For each meeting, loads enrolled speaker embeddings and assigns each
subsegment embedding to the speaker with highest cosine similarity.
"""

import argparse
import json
import os
from collections import OrderedDict

import kaldiio
import numpy as np

from wespeaker.utils.utils import validate_path


def read_emb(scp):
    """Read embeddings grouped by utterance.

    Args:
      scp: Path to embedding scp file.

    Returns:
      Tuple of (subsegs_list, embeddings_list) where each element
        corresponds to one utterance.
    """
    emb_dict = OrderedDict()
    for sub_seg_id, emb in kaldiio.load_scp_sequential(scp):
        utt = sub_seg_id.split('-')[0]
        if utt not in emb_dict:
            emb_dict[utt] = {'sub_seg': [], 'embs': []}
        emb_dict[utt]['sub_seg'].append(sub_seg_id)
        emb_dict[utt]['embs'].append(emb)

    subsegs_list = []
    embeddings_list = []
    for utt, utt_emb_dict in emb_dict.items():
        subsegs_list.append(utt_emb_dict['sub_seg'])
        embeddings_list.append(np.stack(utt_emb_dict['embs']))
    return subsegs_list, embeddings_list


def identify(embeddings, gallery_embeddings, speaker_ids):
    """Assign each embedding to the closest enrolled speaker.

    Args:
      embeddings: np.ndarray of shape (n_subsegs, emb_dim).
      gallery_embeddings: np.ndarray of shape (n_speakers, emb_dim).
      speaker_ids: List of speaker ID strings, parallel to
        gallery_embeddings rows.

    Returns:
      Tuple of (labels, max_scores) where labels is a list of speaker
        ID strings and max_scores is an np.ndarray of shape
        (n_subsegs,) with the max cosine similarity for each
        assignment.
    """
    # L2-normalize
    emb_norm = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True)
    gal_norm = gallery_embeddings / np.linalg.norm(
        gallery_embeddings, axis=1, keepdims=True)

    # Cosine similarity matrix: (n_subsegs, n_speakers)
    scores = emb_norm @ gal_norm.T
    best_indices = np.argmax(scores, axis=1)
    max_scores = np.max(scores, axis=1)
    labels = [speaker_ids[i] for i in best_indices]
    return labels, max_scores


class RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter):
    pass


def get_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RawDefaultsHelpFormatter)
    parser.add_argument('--scp', required=True,
                        help='subsegment embedding scp')
    parser.add_argument('--enrol-scp', required=True,
                        help='enrollment embedding scp (speaker_id -> emb)')
    parser.add_argument('--metadata-dir', required=True,
                        help='directory containing per-meeting metadata.json')
    parser.add_argument('--output', required=True,
                        help='output label file')
    parser.add_argument('--threshold', type=float, default=None,
                        help='min cosine similarity for assignment; '
                             'subsegments below this are omitted')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Load all enrollment embeddings (random-access)
    enrol_dict = kaldiio.load_scp(args.enrol_scp)

    # Read subsegment embeddings grouped by meeting
    subsegs_list, embeddings_list = read_emb(args.scp)

    validate_path(args.output)
    with open(args.output, 'w') as f:
        for subsegs, embeddings in zip(subsegs_list, embeddings_list):
            # Determine meeting ID from first subseg
            utt = subsegs[0].split('-')[0]

            # Load metadata to get speaker list for this meeting
            metadata_path = os.path.join(args.metadata_dir, utt,
                                         'metadata.json')
            with open(metadata_path) as mf:
                metadata = json.load(mf)

            speaker_ids = [spk['id'] for spk in metadata['speakers']]

            # Build gallery matrix from enrolled embeddings
            gallery_embeddings = np.stack(
                [enrol_dict[spk_id] for spk_id in speaker_ids])

            # Identify speakers
            labels, max_scores = identify(
                embeddings, gallery_embeddings, speaker_ids)

            for subseg, label, score in zip(subsegs, labels, max_scores):
                if args.threshold is not None and score < args.threshold:
                    continue
                print(subseg, label, file=f)


if __name__ == '__main__':
    main()
