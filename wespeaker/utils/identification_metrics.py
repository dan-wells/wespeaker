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

import numpy as np
from collections import defaultdict


def top_k_accuracy(predictions, k=1):
    """Compute top-k accuracy from identification trial predictions.

    Args:
        predictions: List of dicts, each containing:
          - "true_speaker" (str): ground-truth speaker ID
          - "ranked_speakers" (list of str): gallery speakers ranked by
            descending similarity score
          - "target_in_gallery" (bool): whether this is a closed-set trial
        k: Number of top ranks to consider.

    Returns:
        float: Fraction of closed-set trials where the true speaker
          appears in the top-k ranked speakers.
    """
    correct = 0
    total = 0
    for pred in predictions:
        if not pred["target_in_gallery"]:
            continue
        total += 1
        if pred["true_speaker"] in pred["ranked_speakers"][:k]:
            correct += 1
    if total == 0:
        return 0.0
    return correct / total


def confusion_matrix(predictions, speaker_ids=None):
    """Build a confusion matrix from identification predictions.

    Only includes closed-set trials (target_in_gallery == True).

    Args:
        predictions: List of prediction dicts (see top_k_accuracy).
        speaker_ids: Optional ordered list of speaker IDs for row/column
          ordering. If None, sorted unique speakers from predictions are
          used.

    Returns:
        tuple: (matrix, speaker_ids) where matrix is a numpy array of
          shape (N, N) with counts, and speaker_ids is the ordered list
          of speaker labels for rows (true) and columns (predicted).
    """
    if speaker_ids is None:
        spk_set = set()
        for pred in predictions:
            if pred["target_in_gallery"]:
                spk_set.add(pred["true_speaker"])
                spk_set.add(pred["ranked_speakers"][0])
        speaker_ids = sorted(spk_set)

    spk_to_idx = {spk: i for i, spk in enumerate(speaker_ids)}
    n = len(speaker_ids)
    matrix = np.zeros((n, n), dtype=np.int64)

    for pred in predictions:
        if not pred["target_in_gallery"]:
            continue
        true_spk = pred["true_speaker"]
        pred_spk = pred["ranked_speakers"][0]
        if true_spk in spk_to_idx and pred_spk in spk_to_idx:
            matrix[spk_to_idx[true_spk], spk_to_idx[pred_spk]] += 1

    return matrix, speaker_ids


def compute_identification_metrics(predictions, top_k_values=(1, 3, 5)):
    """Compute a summary of identification metrics.

    Args:
        predictions: List of prediction dicts (see top_k_accuracy).
        top_k_values: Tuple of k values for top-k accuracy.

    Returns:
        dict: Metrics summary with keys:
          - "num_trials": total number of trials
          - "num_closed_set": number of closed-set trials
          - "num_open_set": number of open-set trials
          - "top_k_accuracy": dict mapping k -> accuracy (closed-set only)
    """
    num_closed = sum(1 for p in predictions if p["target_in_gallery"])
    num_open = len(predictions) - num_closed

    accuracies = {}
    for k in top_k_values:
        accuracies[k] = top_k_accuracy(predictions, k=k)

    return {
        "num_trials": len(predictions),
        "num_closed_set": num_closed,
        "num_open_set": num_open,
        "top_k_accuracy": accuracies,
    }
