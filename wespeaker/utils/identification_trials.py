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

import argparse
import json
import random
from collections import defaultdict

from wespeaker.utils.file_utils import read_scp


def load_utt2spk(utt2spk_path):
    """Load utt2spk file and return spk2utts mapping.

    Args:
        utt2spk_path: Path to utt2spk file (format: <utt_id> <speaker_id>).

    Returns:
        dict: Mapping from speaker_id to list of utt_ids.
    """
    spk2utts = defaultdict(list)
    for utt_id, spk_id in read_scp(utt2spk_path):
        spk2utts[spk_id].append(utt_id)
    return dict(spk2utts)


def generate_base_trials(spk2utts, num_trials, max_num_speakers,
                         max_num_enroll_utts, seed=42,
                         open_set_fraction=0.0):
    """Generate base trials with fixed probe and distractor pool.

    Each base trial fixes the target speaker, probe utterance, and a pool
    of distractor speakers large enough to support any gallery size up to
    max_num_speakers. A separate RNG stream is used for enrollment
    utterance selection, so the base trials are stable across different
    num_enroll_utts values.

    Args:
        spk2utts: Dict mapping speaker_id to list of utt_ids.
        num_trials: Number of trials to generate.
        max_num_speakers: Maximum gallery size that will be used. The
          distractor pool for each trial will contain at least this many
          speakers (including the target).
        max_num_enroll_utts: Maximum enrollment utterances per speaker
          that will be used. Speakers need at least this many + 1 utts.
        seed: Random seed for reproducibility.
        open_set_fraction: Fraction of trials where the target speaker is
          absent from the gallery (for open-set evaluation).

    Returns:
        list: List of base trial dicts, each containing:
          - trial_id (int)
          - probe (dict): {"utt_id": str, "speaker_id": str}
          - speaker_pool (list of str): ordered list of speaker_ids,
            with target speaker first (for closed-set) or absent (open-set)
          - target_in_gallery (bool)
    """
    rng = random.Random(seed)

    # Filter to speakers with enough utterances
    min_utts = max_num_enroll_utts + 1
    eligible_spks = {
        spk: utts for spk, utts in spk2utts.items()
        if len(utts) >= min_utts
    }
    eligible_spk_ids = sorted(eligible_spks.keys())

    if len(eligible_spk_ids) < max_num_speakers:
        raise ValueError(
            "Not enough eligible speakers: need {} but only {} speakers have "
            ">= {} utterances".format(
                max_num_speakers, len(eligible_spk_ids), min_utts))

    if open_set_fraction > 0 and len(eligible_spk_ids) < max_num_speakers + 1:
        raise ValueError(
            "Open-set trials need at least {} eligible speakers but only {} "
            "are available".format(max_num_speakers + 1, len(eligible_spk_ids)))

    base_trials = []
    for trial_id in range(num_trials):
        is_open_set = rng.random() < open_set_fraction

        if is_open_set:
            # Sample pool + 1 extra speaker as the probe-only target
            sampled = rng.sample(eligible_spk_ids, max_num_speakers + 1)
            speaker_pool = sampled[:max_num_speakers]
            target_spk = sampled[max_num_speakers]
            # Probe can be any utterance from target (not in any gallery)
            probe_utt = rng.choice(eligible_spks[target_spk])
        else:
            speaker_pool = rng.sample(eligible_spk_ids, max_num_speakers)
            target_spk = speaker_pool[0]
            # Reserve one utterance as probe (pick from all utts, we'll
            # exclude it from enrollment later)
            probe_utt = rng.choice(eligible_spks[target_spk])

        base_trials.append({
            "trial_id": trial_id,
            "probe": {
                "utt_id": probe_utt,
                "speaker_id": target_spk,
            },
            "speaker_pool": speaker_pool,
            "target_in_gallery": not is_open_set,
        })

    return base_trials


def realize_trials(base_trials, spk2utts, num_speakers, num_enroll_utts,
                   seed=42):
    """Build concrete trials from base trials for a specific condition.

    Takes base trials (with stable probes and speaker pools) and selects
    the gallery subset and enrollment utterances for a given condition.

    Args:
        base_trials: List of base trial dicts from generate_base_trials().
        spk2utts: Dict mapping speaker_id to list of utt_ids.
        num_speakers: Number of speakers in the gallery for this condition.
        num_enroll_utts: Number of enrollment utterances per speaker.
        seed: Random seed for enrollment utterance selection. Using a
          different seed from the base trial seed ensures the enrollment
          sampling is independent.

    Returns:
        list: List of trial dicts in the standard format (trial_id, probe,
          gallery, target_in_gallery).
    """
    rng = random.Random(seed)

    trials = []
    for base in base_trials:
        pool = base["speaker_pool"]
        target_spk = base["probe"]["speaker_id"]
        probe_utt = base["probe"]["utt_id"]
        is_closed_set = base["target_in_gallery"]

        # Select gallery speakers: take the first num_speakers from the
        # pool. For closed-set trials, the target is always pool[0], so
        # it's always included.
        gallery_spks = pool[:num_speakers]

        # Build gallery with enrollment utterances
        gallery = []
        for spk in gallery_spks:
            available = [u for u in spk2utts[spk] if u != probe_utt]
            enroll_utts = rng.sample(available, min(num_enroll_utts,
                                                    len(available)))
            gallery.append({
                "speaker_id": spk,
                "enroll_utts": enroll_utts,
            })

        trials.append({
            "trial_id": base["trial_id"],
            "probe": base["probe"],
            "gallery": gallery,
            "target_in_gallery": is_closed_set,
        })

    return trials


def generate_trials(spk2utts, num_speakers, num_enroll_utts, num_trials,
                    seed=42, open_set_fraction=0.0):
    """Generate speaker identification trials (single-condition convenience).

    Generates base trials and realizes them for a single condition. For
    multi-condition sweeps, use generate_base_trials() + realize_trials()
    directly.

    Args:
        spk2utts: Dict mapping speaker_id to list of utt_ids.
        num_speakers: Number of speakers in each trial's gallery.
        num_enroll_utts: Number of enrollment utterances per gallery speaker.
        num_trials: Number of trials to generate.
        seed: Random seed for reproducibility.
        open_set_fraction: Fraction of trials where the target speaker is
          absent from the gallery (for open-set evaluation).

    Returns:
        list: List of trial dicts, each containing:
          - trial_id (int)
          - probe (dict): {"utt_id": str, "speaker_id": str}
          - gallery (list of dicts):
            [{"speaker_id": str, "enroll_utts": [str, ...]}, ...]
          - target_in_gallery (bool)
    """
    base_trials = generate_base_trials(
        spk2utts, num_trials,
        max_num_speakers=num_speakers,
        max_num_enroll_utts=num_enroll_utts,
        seed=seed,
        open_set_fraction=open_set_fraction,
    )
    return realize_trials(base_trials, spk2utts, num_speakers,
                          num_enroll_utts, seed=seed + 1)


def save_trials(trials, output_path):
    """Save trials to a JSON file.

    Args:
        trials: List of trial dicts (base or realized).
        output_path: Path to write JSON output.
    """
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(trials, fout, indent=2)


def load_trials(trials_path):
    """Load trials from a JSON file.

    Args:
        trials_path: Path to JSON trials file.

    Returns:
        list: List of trial dicts.
    """
    with open(trials_path, 'r', encoding='utf-8') as fin:
        return json.load(fin)


def get_args():
    parser = argparse.ArgumentParser(
        description='Generate speaker identification trials')
    parser.add_argument('--utt2spk', required=True,
                        help='Path to utt2spk file')
    parser.add_argument('--num_speakers', type=int, required=True,
                        help='Number of speakers in each trial gallery')
    parser.add_argument('--num_enroll_utts', type=int, required=True,
                        help='Number of enrollment utterances per speaker')
    parser.add_argument('--num_trials', type=int, required=True,
                        help='Number of trials to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--open_set_fraction', type=float, default=0.0,
                        help='Fraction of trials with target absent from '
                             'gallery (0.0 for closed-set only)')
    parser.add_argument('--output', required=True,
                        help='Output JSON file path')
    return parser.parse_args()


def main():
    args = get_args()
    spk2utts = load_utt2spk(args.utt2spk)
    trials = generate_trials(
        spk2utts,
        num_speakers=args.num_speakers,
        num_enroll_utts=args.num_enroll_utts,
        num_trials=args.num_trials,
        seed=args.seed,
        open_set_fraction=args.open_set_fraction,
    )
    save_trials(trials, args.output)
    print("Generated {} trials -> {}".format(len(trials), args.output))


if __name__ == '__main__':
    main()
