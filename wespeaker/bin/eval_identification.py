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

"""Speaker identification evaluation harness.

Supports two modes:
  1. End-to-end: provide wav.scp + utt2spk + pretrained model, embeddings
     are extracted on the fly and trials are generated automatically.
  2. Pre-extracted: provide embedding_scp + utt2spk + trials JSON, trials
     are evaluated directly against pre-extracted embeddings.
"""

import argparse
import json
import os
import sys

import kaldiio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from wespeaker.cli.speaker import load_model
from wespeaker.utils.file_utils import read_scp
from wespeaker.utils.identification_trials import (
    generate_trials, load_trials, load_utt2spk, save_trials)
from wespeaker.utils.identification_metrics import (
    compute_identification_metrics, confusion_matrix)


def load_embeddings_from_scp(scp_path):
    """Load embeddings from a Kaldi scp file.

    Args:
        scp_path: Path to embedding scp file.

    Returns:
        dict: Mapping from utt_id to numpy embedding array.
    """
    emb_dict = {}
    for utt, emb in kaldiio.load_scp_sequential(scp_path):
        emb_dict[utt] = emb
    return emb_dict


def extract_embeddings_from_model(model, wav_scp_path):
    """Extract embeddings for all utterances using a Speaker model.

    Args:
        model: A wespeaker.cli.speaker.Speaker instance.
        wav_scp_path: Path to wav.scp file.

    Returns:
        dict: Mapping from utt_id to numpy embedding array.
    """
    emb_dict = {}
    wav_list = read_scp(wav_scp_path)
    for utt_id, wav_path in tqdm(wav_list, desc="Extracting embeddings"):
        embedding = model.extract_embedding(wav_path)
        if embedding is not None:
            emb_dict[utt_id] = embedding.detach().numpy()
        else:
            print("Warning: failed to extract embedding for {}".format(utt_id),
                  file=sys.stderr)
    return emb_dict


def build_gallery_embedding(enroll_utts, emb_dict, strategy="mean"):
    """Build a single gallery embedding for a speaker from enrollment utts.

    Args:
        enroll_utts: List of utt_ids for enrollment.
        emb_dict: Dict mapping utt_id to numpy embedding.
        strategy: "mean" to average embeddings, "individual" to return all.

    Returns:
        numpy array of shape (D,) if strategy=="mean", or (K, D) if
        strategy=="individual". Returns None if no valid embeddings found.
    """
    embeddings = []
    for utt_id in enroll_utts:
        if utt_id in emb_dict:
            embeddings.append(emb_dict[utt_id])
    if len(embeddings) == 0:
        return None
    embeddings = np.stack(embeddings)
    if strategy == "mean":
        return np.mean(embeddings, axis=0)
    return embeddings


def run_trial(trial, emb_dict, enroll_strategy="mean"):
    """Run a single identification trial.

    Args:
        trial: Trial dict with "probe", "gallery", "target_in_gallery" keys.
        emb_dict: Dict mapping utt_id to numpy embedding.
        enroll_strategy: "mean" or "max_score".

    Returns:
        dict: Prediction result with keys:
          - "trial_id" (int)
          - "true_speaker" (str)
          - "target_in_gallery" (bool)
          - "ranked_speakers" (list of str): gallery speakers sorted by
            descending similarity
          - "scores" (dict): mapping speaker_id -> similarity score
    """
    probe_utt = trial["probe"]["utt_id"]
    true_speaker = trial["probe"]["speaker_id"]

    if probe_utt not in emb_dict:
        return None

    probe_emb = emb_dict[probe_utt].reshape(1, -1)

    speaker_scores = {}
    for entry in trial["gallery"]:
        spk_id = entry["speaker_id"]
        if enroll_strategy == "mean":
            gallery_emb = build_gallery_embedding(
                entry["enroll_utts"], emb_dict, strategy="mean")
            if gallery_emb is None:
                continue
            score = cosine_similarity(
                probe_emb, gallery_emb.reshape(1, -1))[0][0]
        elif enroll_strategy == "max_score":
            gallery_embs = build_gallery_embedding(
                entry["enroll_utts"], emb_dict, strategy="individual")
            if gallery_embs is None:
                continue
            scores = cosine_similarity(probe_emb, gallery_embs)[0]
            score = float(np.max(scores))
        else:
            raise ValueError(
                "Unknown enroll_strategy: {}".format(enroll_strategy))
        speaker_scores[spk_id] = float(score)

    if len(speaker_scores) == 0:
        return None

    ranked = sorted(speaker_scores.keys(),
                    key=lambda s: speaker_scores[s], reverse=True)

    return {
        "trial_id": trial["trial_id"],
        "true_speaker": true_speaker,
        "target_in_gallery": trial["target_in_gallery"],
        "ranked_speakers": ranked,
        "scores": speaker_scores,
    }


def run_evaluation(trials, emb_dict, enroll_strategy="mean"):
    """Run all identification trials and compute metrics.

    Args:
        trials: List of trial dicts.
        emb_dict: Dict mapping utt_id to numpy embedding.
        enroll_strategy: "mean" or "max_score".

    Returns:
        tuple: (predictions, metrics) where predictions is a list of
          prediction dicts and metrics is a summary dict.
    """
    predictions = []
    skipped = 0
    for trial in tqdm(trials, desc="Running trials"):
        result = run_trial(trial, emb_dict, enroll_strategy)
        if result is None:
            skipped += 1
            continue
        predictions.append(result)

    if skipped > 0:
        print("Warning: skipped {} trials due to missing embeddings".format(
            skipped), file=sys.stderr)

    metrics = compute_identification_metrics(predictions)
    return predictions, metrics


def get_args():
    parser = argparse.ArgumentParser(
        description='Speaker identification evaluation')

    # Mode 1: end-to-end with model
    parser.add_argument('--wav_scp', default=None,
                        help='Path to wav.scp (triggers embedding extraction)')
    parser.add_argument('--pretrain', default=None,
                        help='Pretrained model name or directory')
    parser.add_argument('--device', default='cpu',
                        help='Device for model inference (cpu, cuda, cuda:0)')

    # Mode 2: pre-extracted embeddings
    parser.add_argument('--embedding_scp', default=None,
                        help='Path to pre-extracted embedding scp file')

    # Common
    parser.add_argument('--utt2spk', required=True,
                        help='Path to utt2spk file')
    parser.add_argument('--trials', default=None,
                        help='Path to pre-generated trials JSON. If not '
                             'provided, trials are generated from utt2spk.')
    parser.add_argument('--num_speakers', type=int, default=10,
                        help='Number of gallery speakers per trial')
    parser.add_argument('--num_enroll_utts', type=int, default=3,
                        help='Enrollment utterances per gallery speaker')
    parser.add_argument('--num_trials', type=int, default=500,
                        help='Number of trials to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for trial generation')
    parser.add_argument('--enroll_strategy', default='mean',
                        choices=['mean', 'max_score'],
                        help='How to combine enrollment embeddings')
    parser.add_argument('--output_file', default=None,
                        help='Path to write JSON results')
    parser.add_argument('--save_trials', default=None,
                        help='Path to save generated trials JSON (optional)')

    return parser.parse_args()


def main():
    args = get_args()

    # Load or extract embeddings
    if args.embedding_scp is not None:
        print("Loading pre-extracted embeddings from {}".format(
            args.embedding_scp))
        emb_dict = load_embeddings_from_scp(args.embedding_scp)
    elif args.wav_scp is not None and args.pretrain is not None:
        print("Loading model: {}".format(args.pretrain))
        model = load_model(args.pretrain)
        model.set_device(args.device)
        emb_dict = extract_embeddings_from_model(model, args.wav_scp)
    else:
        print("Error: provide either --embedding_scp or "
              "both --wav_scp and --pretrain", file=sys.stderr)
        sys.exit(1)

    print("Loaded {} embeddings".format(len(emb_dict)))

    # Load or generate trials
    if args.trials is not None:
        print("Loading trials from {}".format(args.trials))
        trials = load_trials(args.trials)
    else:
        print("Generating trials from utt2spk")
        spk2utts = load_utt2spk(args.utt2spk)
        trials = generate_trials(
            spk2utts,
            num_speakers=args.num_speakers,
            num_enroll_utts=args.num_enroll_utts,
            num_trials=args.num_trials,
            seed=args.seed,
        )
        if args.save_trials is not None:
            save_trials(trials, args.save_trials)
            print("Saved trials -> {}".format(args.save_trials))

    print("Running {} trials with enroll_strategy={}".format(
        len(trials), args.enroll_strategy))

    # Run evaluation
    predictions, metrics = run_evaluation(
        trials, emb_dict, enroll_strategy=args.enroll_strategy)

    # Print results
    print("\n=== Speaker Identification Results ===")
    print("Trials: {} total, {} closed-set, {} open-set".format(
        metrics["num_trials"], metrics["num_closed_set"],
        metrics["num_open_set"]))
    for k, acc in sorted(metrics["top_k_accuracy"].items()):
        print("Top-{} accuracy: {:.2f}%".format(k, acc * 100))

    # Save results
    if args.output_file is not None:
        output = {
            "metrics": metrics,
            "predictions": predictions,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)),
                     exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as fout:
            json.dump(output, fout, indent=2)
        print("Results saved -> {}".format(args.output_file))


if __name__ == '__main__':
    main()
