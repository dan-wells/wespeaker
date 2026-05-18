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

"""Compute per-speaker mean embeddings for speaker enrolment.

Produces an enrol.ark and enrol.scp suitable for use with
  wespeaker.diar.stream_diar --assign identify --enrol-scp enrol.scp
"""

import argparse
import logging
import os
import sys
from collections import defaultdict

import kaldiio
import numpy as np
from tqdm import tqdm

from wespeaker.utils.file_utils import read_scp
from wespeaker.utils.utils import validate_path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(
        description='Compute per-speaker mean embeddings for enrolment.')
    parser.add_argument('--wav-scp', required=True,
                        help='wav.scp mapping utt_id to audio path')
    parser.add_argument('--utt2spk', required=True,
                        help='utt2spk mapping utt_id to speaker_id')
    parser.add_argument('--output', required=True,
                        help='output path prefix (writes <output>.ark and <output>.scp)')
    parser.add_argument('--embedding-scp', default=None,
                        help='pre-computed embedding scp (skips extraction)')
    parser.add_argument('--model', default=None,
                        help='path to ONNX model (.onnx) or PyTorch model directory')
    parser.add_argument('--backend', default='auto',
                        choices=['auto', 'onnx', 'pytorch'],
                        help='model backend (default: auto-detect from path)')
    parser.add_argument('--device', default='cpu',
                        help='inference device: cpu or cuda')
    parser.add_argument('--no-subseg-cmn', dest='subseg_cmn',
                        action='store_false', default=True,
                        help='disable per-utterance cepstral mean normalisation')
    parser.add_argument('--max-enroll-utts', type=int, default=None,
                        help='max utterances per speaker (subsample if more)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for subsampling')
    return parser.parse_args()


def extract_embedding(wav_path, emb_model, load_audio, compute_fbank, subseg_cmn=True):
    """Extract a single embedding from an audio file.

    Args:
        wav_path: Path to audio file.
        emb_model: EmbeddingModel instance.
        load_audio: Function to load audio as float32 numpy array.
        compute_fbank: Function to compute fbank features.
        subseg_cmn: Whether to apply cepstral mean normalisation.

    Returns:
        np.ndarray of shape (emb_dim,), or None on failure.
    """
    try:
        audio = load_audio(wav_path)
        fbank = compute_fbank(audio)
        if subseg_cmn:
            fbank = fbank - fbank.mean(axis=0)
        return emb_model.extract(fbank)
    except Exception as e:
        logger.warning("Failed to extract embedding for %s: %s", wav_path, e)
        return None


def main():
    args = get_args()

    if args.embedding_scp is None and args.model is None:
        logger.error("Must provide either --embedding-scp or --model")
        sys.exit(1)

    wav_dict = dict(read_scp(args.wav_scp))
    utt2spk = dict(read_scp(args.utt2spk))

    spk_to_utts = defaultdict(list)
    for utt_id, spk_id in utt2spk.items():
        if utt_id in wav_dict:
            spk_to_utts[spk_id].append(utt_id)

    rng = np.random.default_rng(args.seed)
    if args.max_enroll_utts is not None:
        for spk_id in spk_to_utts:
            utts = spk_to_utts[spk_id]
            if len(utts) > args.max_enroll_utts:
                spk_to_utts[spk_id] = rng.choice(
                    utts, args.max_enroll_utts, replace=False).tolist()

    if args.embedding_scp is not None:
        logger.info("Loading pre-computed embeddings from %s",
                    args.embedding_scp)
        emb_dict = kaldiio.load_scp(args.embedding_scp)

        def get_embedding(utt_id):
            if utt_id in emb_dict:
                return emb_dict[utt_id]
            logger.warning("No embedding for utt %s", utt_id)
            return None
    else:
        from wespeaker.utils.audio import compute_fbank, load_audio
        from wespeaker.utils.embedding import EmbeddingModel

        backend = None if args.backend == 'auto' else args.backend
        logger.info("Loading model from %s (backend=%s, device=%s)",
                    args.model, args.backend, args.device)
        emb_model = EmbeddingModel(args.model, device=args.device, backend=backend)

        def get_embedding(utt_id):
            return extract_embedding(
                wav_dict[utt_id], emb_model, load_audio, compute_fbank, args.subseg_cmn)

    ark_path = os.path.abspath(args.output + ".ark")
    scp_path = os.path.abspath(args.output + ".scp")
    validate_path(ark_path)

    n_speakers = 0
    with kaldiio.WriteHelper('ark,scp:' + ark_path + "," + scp_path) as writer:
        for spk_id in tqdm(sorted(spk_to_utts.keys()), desc="Enrolling"):
            utts = spk_to_utts[spk_id]
            embeddings = []
            for utt_id in utts:
                emb = get_embedding(utt_id)
                if emb is not None:
                    embeddings.append(np.asarray(emb, dtype=np.float32))

            if len(embeddings) == 0:
                logger.warning("No valid embeddings for speaker %s, skipping", spk_id)
                continue

            mean_emb = np.mean(np.stack(embeddings), axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            writer(spk_id, mean_emb.astype(np.float32))
            n_speakers += 1

    logger.info("Enrolled %d speakers -> %s", n_speakers, scp_path)


if __name__ == '__main__':
    main()
