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

"""Segment-level speaker identification evaluation.

Evaluates 2-speaker identification by slicing utterances into enrolment
(prefix) and probe (suffix or random) segments of specified durations,
sweeping both independently.
"""

import argparse
import itertools
import json
import os
import random
import sys

import numpy as np
import torch
import torchaudio
from silero_vad import get_speech_timestamps
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from wespeaker.cli.speaker import load_model
from wespeaker.utils.file_utils import read_scp
from wespeaker.utils.identification_metrics import confusion_matrix, top_k_accuracy


def apply_vad_to_pcm(pcm, sample_rate, vad_model):
    """Apply Silero VAD and return speech-only PCM.

    Args:
        pcm: Tensor of shape (1, num_samples) or (num_samples,).
        sample_rate: Sample rate of pcm.
        vad_model: Silero VAD model instance.

    Returns:
        Tensor of shape (1, num_speech_samples), or None if all silence.
    """
    if pcm.dim() == 1:
        pcm = pcm.unsqueeze(0)
    wav = pcm
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    vad_sample_rate = 16000
    if sample_rate != vad_sample_rate:
        transform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=vad_sample_rate)
        vad_wav = transform(wav)
    else:
        vad_wav = wav

    segments = get_speech_timestamps(vad_wav, vad_model, return_seconds=True)
    if len(segments) == 0:
        return None

    pcm_total = torch.Tensor()
    for segment in segments:
        start = int(segment['start'] * sample_rate)
        end = int(segment['end'] * sample_rate)
        pcm_total = torch.cat([pcm_total, pcm[0, start:end]], 0)

    return pcm_total.unsqueeze(0)


def load_and_prepare_audio(wav_scp, utt2spk, model, do_vad,
                           sample_rate=16000):
    """Load audio, apply VAD if needed, select 1 utt/speaker.

    No duration filtering is applied here -- callers filter per-condition.

    Args:
        wav_scp: Path to wav.scp file.
        utt2spk: Path to utt2spk file.
        model: Speaker model instance (used for VAD model access).
        do_vad: Whether to apply VAD before slicing.
        sample_rate: Target sample rate.

    Returns:
        dict: {spk_id: (utt_id, pcm_tensor, duration_seconds)}
          pcm_tensor is shape (1, num_samples).
    """
    wav_list = read_scp(wav_scp)
    utt2spk_list = read_scp(utt2spk)

    # Group utterances by speaker
    spk_to_utts = {}
    for utt, spk in utt2spk_list:
        if spk not in spk_to_utts:
            spk_to_utts[spk] = []
        spk_to_utts[spk].append(utt)

    wav_paths = {utt: path for utt, path in wav_list}

    # Load and process all utterances, pick longest per speaker
    audio_dict = {}
    for spk, utts in tqdm(spk_to_utts.items(), desc="Loading audio"):
        best_utt = None
        best_pcm = None
        best_dur = 0.0

        for utt in utts:
            if utt not in wav_paths:
                print("Warning: {} not in wav.scp, skipping".format(utt),
                      file=sys.stderr)
                continue

            pcm, sr = torchaudio.load(wav_paths[utt])
            if sr != sample_rate:
                pcm = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=sample_rate)(pcm)

            if do_vad:
                pcm = apply_vad_to_pcm(pcm, sample_rate, model.vad)
                if pcm is None:
                    print("Warning: {} is all silence after VAD, "
                          "skipping".format(utt), file=sys.stderr)
                    continue

            dur = pcm.size(1) / sample_rate
            if dur > best_dur:
                best_utt = utt
                best_pcm = pcm
                best_dur = dur

        if best_pcm is not None:
            audio_dict[spk] = (best_utt, best_pcm, best_dur)

    return audio_dict


def filter_speakers_by_duration(audio_dict, min_duration):
    """Return the subset of audio_dict whose utterances meet min_duration.

    Args:
        audio_dict: {spk_id: (utt_id, pcm_tensor, duration_seconds)}.
        min_duration: Minimum duration in seconds.

    Returns:
        dict: Filtered subset of audio_dict.
    """
    return {spk: val for spk, val in audio_dict.items()
            if val[2] >= min_duration}


def generate_speaker_pairs(speakers, num_pairs=None, seed=42):
    """Generate speaker pairs, exhaustive or sampled.

    Args:
        speakers: List of speaker IDs.
        num_pairs: If None, all C(N,2) pairs. Otherwise sample this many.
        seed: Random seed for sampling.

    Returns:
        list of (spk_a, spk_b) tuples.
    """
    all_pairs = list(itertools.combinations(sorted(speakers), 2))
    if num_pairs is None or num_pairs >= len(all_pairs):
        return all_pairs
    rng = random.Random(seed)
    return rng.sample(all_pairs, num_pairs)


def slice_audio(pcm, sample_rate, enrol_dur, probe_dur, guard_band=0.5,
                probe_mode="end", rng=None):
    """Slice utterance into enrolment prefix and probe segment.

    Args:
        pcm: Tensor of shape (1, num_samples).
        sample_rate: Sample rate.
        enrol_dur: Enrolment duration in seconds.
        probe_dur: Probe duration in seconds.
        guard_band: Gap between enrolment and probe in seconds.
        probe_mode: "end" (last K seconds) or "random" (random from remainder).
        rng: random.Random instance, required if probe_mode=="random".

    Returns:
        tuple: (enrol_pcm, probe_pcm) as 1D tensors.
    """
    pcm_1d = pcm.squeeze(0)
    total_samples = pcm_1d.size(0)
    enrol_samples = int(enrol_dur * sample_rate)
    probe_samples = int(probe_dur * sample_rate)
    guard_samples = int(guard_band * sample_rate)

    enrol_pcm = pcm_1d[:enrol_samples]
    remainder_start = enrol_samples + guard_samples

    if probe_mode == "end":
        probe_start = total_samples - probe_samples
        # Ensure probe doesn't overlap with enrol + guard band
        probe_start = max(probe_start, remainder_start)
        probe_pcm = pcm_1d[probe_start:probe_start + probe_samples]
    elif probe_mode == "random":
        max_start = total_samples - probe_samples
        probe_start = rng.randint(remainder_start, max_start)
        probe_pcm = pcm_1d[probe_start:probe_start + probe_samples]
    else:
        raise ValueError("Unknown probe_mode: {}".format(probe_mode))

    return enrol_pcm, probe_pcm


def extract_embedding_from_slice(model, pcm_slice, sample_rate):
    """Extract embedding from a PCM slice.

    The model's VAD should be disabled before calling this.

    Args:
        model: Speaker model instance.
        pcm_slice: 1D tensor of audio samples.
        sample_rate: Sample rate.

    Returns:
        numpy array of shape (D,), or None on failure.
    """
    pcm_2d = pcm_slice.unsqueeze(0)
    embedding = model.extract_embedding_from_pcm(pcm_2d, sample_rate)
    if embedding is None:
        return None
    return embedding.detach().numpy()


def run_condition(audio_dict, model, enrol_dur, probe_dur, guard_band,
                  probe_mode, sample_rate, num_pairs=None,
                  num_repetitions=1, seed=42,
                  enrol_cache=None, probe_cache=None):
    """Run all 2-speaker trials for one (enrol_dur, probe_dur) condition.

    Filters speakers by the duration requirement for this specific condition,
    then generates pairs and runs trials.

    Args:
        audio_dict: {spk_id: (utt_id, pcm_tensor, duration)}.
        model: Speaker model instance (VAD disabled).
        enrol_dur: Enrolment duration in seconds.
        probe_dur: Probe duration in seconds.
        guard_band: Guard band in seconds.
        probe_mode: "end" or "random".
        sample_rate: Sample rate.
        num_pairs: Number of pairs to sample (None = exhaustive).
        num_repetitions: Number of repetitions per pair.
        seed: Random seed.
        enrol_cache: Dict for caching enrolment embeddings, keyed by
          (spk_id, enrol_dur).
        probe_cache: Dict for caching probe embeddings (end mode only),
          keyed by (spk_id, enrol_dur, probe_dur).

    Returns:
        dict with keys: enrol_dur, probe_dur, accuracy, num_decisions,
          num_speakers, num_pairs, predictions, confusion_matrix, speaker_ids.
    """
    if enrol_cache is None:
        enrol_cache = {}
    if probe_cache is None:
        probe_cache = {}

    min_duration = enrol_dur + guard_band + probe_dur
    cond_audio = filter_speakers_by_duration(audio_dict, min_duration)
    speakers = sorted(cond_audio.keys())
    pairs = generate_speaker_pairs(speakers, num_pairs, seed)

    rng = random.Random(seed)
    predictions = []

    for spk_a, spk_b in pairs:
        # Get or compute enrolment embeddings
        for spk in (spk_a, spk_b):
            cache_key = (spk, enrol_dur)
            if cache_key not in enrol_cache:
                pcm = cond_audio[spk][1]
                enrol_pcm, _ = slice_audio(
                    pcm, sample_rate, enrol_dur, probe_dur,
                    guard_band, probe_mode="end")
                emb = extract_embedding_from_slice(model, enrol_pcm,
                                                   sample_rate)
                enrol_cache[cache_key] = emb

        enrol_a = enrol_cache[(spk_a, enrol_dur)]
        enrol_b = enrol_cache[(spk_b, enrol_dur)]
        if enrol_a is None or enrol_b is None:
            continue

        gallery = np.stack([enrol_a, enrol_b])  # shape (2, D)
        gallery_ids = [spk_a, spk_b]

        for rep in range(num_repetitions):
            for probe_spk in (spk_a, spk_b):
                # Get or compute probe embedding
                if probe_mode == "end":
                    pcache_key = (probe_spk, enrol_dur, probe_dur)
                    if pcache_key not in probe_cache:
                        pcm = cond_audio[probe_spk][1]
                        _, probe_pcm = slice_audio(
                            pcm, sample_rate, enrol_dur, probe_dur,
                            guard_band, probe_mode="end")
                        emb = extract_embedding_from_slice(
                            model, probe_pcm, sample_rate)
                        probe_cache[pcache_key] = emb
                    probe_emb = probe_cache[pcache_key]
                else:
                    pcm = cond_audio[probe_spk][1]
                    _, probe_pcm = slice_audio(
                        pcm, sample_rate, enrol_dur, probe_dur,
                        guard_band, probe_mode="random", rng=rng)
                    probe_emb = extract_embedding_from_slice(
                        model, probe_pcm, sample_rate)

                if probe_emb is None:
                    continue

                # Score probe against gallery
                scores = cosine_similarity(
                    probe_emb.reshape(1, -1), gallery)[0]
                ranked_idx = np.argsort(scores)[::-1]
                ranked_speakers = [gallery_ids[i] for i in ranked_idx]

                predictions.append({
                    "true_speaker": probe_spk,
                    "ranked_speakers": ranked_speakers,
                    "target_in_gallery": True,
                    "pair": [spk_a, spk_b],
                    "repetition": rep,
                })

    accuracy = top_k_accuracy(predictions, k=1)

    # Compute confusion matrix over this condition's speakers
    cm, speaker_ids = confusion_matrix(predictions, speaker_ids=speakers)

    return {
        "enrol_dur": enrol_dur,
        "probe_dur": probe_dur,
        "accuracy": accuracy,
        "num_decisions": len(predictions),
        "num_speakers": len(speakers),
        "num_pairs": len(pairs),
        "predictions": predictions,
        "confusion_matrix": cm.tolist(),
        "speaker_ids": speaker_ids,
    }


def print_confused_pairs(aggregated_cm, speaker_ids, top_n=10):
    """Print the most confused speaker pairs from an aggregated confusion matrix.

    Args:
        aggregated_cm: numpy array of shape (N, N).
        speaker_ids: List of speaker IDs matching matrix rows/columns.
        top_n: Number of confused pairs to print.
    """
    errors = []
    for i in range(len(speaker_ids)):
        for j in range(len(speaker_ids)):
            if i != j and aggregated_cm[i, j] > 0:
                total_decisions = int(aggregated_cm[i].sum())
                errors.append((speaker_ids[i], speaker_ids[j],
                               int(aggregated_cm[i, j]), total_decisions))

    errors.sort(key=lambda x: x[2], reverse=True)
    if errors:
        print("\nMost confused speaker pairs:")
        for true_spk, pred_spk, count, total in errors[:top_n]:
            print("  {} -> {}: {} errors (of {} decisions)".format(
                true_spk, pred_spk, count, total))


def print_results(results):
    """Print formatted summary from a results dict (as loaded from JSON output).

    Args:
        results: Dict with keys "config", "conditions", "aggregated",
          "matched_subset".
    """
    config = results["config"]
    conditions = results["conditions"]
    aggregated = results["aggregated"]

    enrol_durs_sorted = sorted(config["enrol_durs"])
    probe_durs_sorted = sorted(config["probe_durs"])

    print("\n=== Segment-Level Speaker Identification ===")
    print("Speakers loaded: {}".format(config["num_speakers_loaded"]))
    print("VAD: {}, Probe mode: {}".format(
        "enabled" if config["apply_vad"] else "disabled",
        config["probe_mode"]))

    # Accuracy table
    print("\nAccuracy (%) by condition [speakers / pairs]:")
    header = "{:>12s}".format("")
    for pd in probe_durs_sorted:
        header += "  {:>14s}".format("probe={:.1f}s".format(pd))
    print(header)

    results_by_condition = {}
    for r in conditions:
        results_by_condition[(r["enrol_dur"], r["probe_dur"])] = r

    for ed in enrol_durs_sorted:
        row = "enrol={:<5s}".format("{:.1f}s".format(ed))
        for pd in probe_durs_sorted:
            r = results_by_condition[(ed, pd)]
            cell = "{:.1f} [{}/{}]".format(
                r["accuracy"] * 100, r["num_speakers"], r["num_pairs"])
            row += "  {:>14s}".format(cell)
        print(row)

    total_decisions = sum(r["num_decisions"] for r in conditions)
    print("\nOverall accuracy: {:.1f}% ({} decisions)".format(
        aggregated["overall_accuracy"] * 100, total_decisions))

    # Matched subset
    matched = results.get("matched_subset", {})
    if matched.get("num_speakers", 0) >= 2:
        print("\nMatched-subset accuracy ({} speakers in all conditions):".format(
            matched["num_speakers"]))

        if "per_condition" in matched:
            header = "{:>12s}".format("")
            for pd in probe_durs_sorted:
                header += "  {:>10s}".format("probe={:.1f}s".format(pd))
            print(header)

            matched_by_cond = {}
            for entry in matched["per_condition"]:
                matched_by_cond[(entry["enrol_dur"], entry["probe_dur"])] = entry

            for ed in enrol_durs_sorted:
                row = "enrol={:<5s}".format("{:.1f}s".format(ed))
                for pd in probe_durs_sorted:
                    acc = matched_by_cond[(ed, pd)]["accuracy"] * 100
                    row += "  {:>10.1f}".format(acc)
                print(row)

        if "overall_accuracy" in matched:
            print("\n  Overall (matched): {:.1f}% ({} decisions)".format(
                matched["overall_accuracy"] * 100,
                matched.get("num_decisions", 0)))

    # Confused pairs
    cm = np.array(aggregated["confusion_matrix"], dtype=np.int64)
    print_confused_pairs(cm, aggregated["speaker_ids"], top_n=20)


def get_args():
    parser = argparse.ArgumentParser(
        description='Segment-level speaker identification evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--wav_scp', default=None,
                        help='Path to wav.scp (required for evaluation)')
    parser.add_argument('--utt2spk', default=None,
                        help='Path to utt2spk file (required for evaluation)')
    parser.add_argument('--pretrain', default=None,
                        help='Pretrained model name or directory '
                             '(required for evaluation)')
    parser.add_argument('--device', default='cpu',
                        help='Device for model inference (cpu, cuda, cuda:0)')
    parser.add_argument('--apply_vad', action='store_true',
                        help='Enable VAD before slicing (off by default)')
    parser.add_argument('--enrol_durs', nargs='+', type=float,
                        default=[3, 5, 10],
                        help='Enrolment durations in seconds')
    parser.add_argument('--probe_durs', nargs='+', type=float,
                        default=[1, 2, 5],
                        help='Probe durations in seconds')
    parser.add_argument('--guard_band', type=float, default=0.5,
                        help='Guard band between enrolment and probe (seconds)')
    parser.add_argument('--probe_mode', default='end',
                        choices=['end', 'random'],
                        help='Probe selection mode')
    parser.add_argument('--num_repetitions', type=int, default=1,
                        help='Repetitions per pair (for random probe mode)')
    parser.add_argument('--num_pairs', type=int, default=None,
                        help='Number of speaker pairs to sample (default: all)')
    parser.add_argument('--fixed_speakers', action='store_true',
                        help='Use the same speaker set across all conditions '
                             '(filter by max(enrol_durs) + max(probe_durs) + guard_band)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_file', default=None,
                        help='Path to write JSON results')
    parser.add_argument('--results_file', default=None,
                        help='Path to existing JSON results to display '
                             '(skips evaluation, just prints summary)')

    return parser.parse_args()


def main():
    args = get_args()

    # Display-only mode: load existing results and print
    if args.results_file is not None:
        with open(args.results_file, 'r', encoding='utf-8') as inf:
            results = json.load(inf)
        print_results(results)
        return

    # Validate required args for evaluation mode
    if not all([args.wav_scp, args.utt2spk, args.pretrain]):
        print("Error: --wav_scp, --utt2spk, and --pretrain are required "
              "for evaluation", file=sys.stderr)
        sys.exit(1)

    sample_rate = 16000

    # Load model
    print("Loading model: {}".format(args.pretrain))
    model = load_model(args.pretrain)
    model.set_device(args.device)

    # Load and prepare audio (no duration filtering yet)
    print("Loading audio (VAD: {})...".format(
        "enabled" if args.apply_vad else "disabled"))
    audio_dict = load_and_prepare_audio(
        args.wav_scp, args.utt2spk, model, args.apply_vad, sample_rate)

    num_original = len(audio_dict)

    # If --fixed_speakers, pre-filter to the strictest duration requirement
    if args.fixed_speakers:
        global_min = (max(args.enrol_durs) + args.guard_band
                      + max(args.probe_durs))
        audio_dict = filter_speakers_by_duration(audio_dict, global_min)
        print("Fixed speaker mode: {} speakers meet {:.1f}s requirement "
              "(from {} loaded)".format(
                  len(audio_dict), global_min, num_original))

    if len(audio_dict) < 2:
        print("Error: fewer than 2 speakers with usable audio",
              file=sys.stderr)
        sys.exit(1)

    # Disable VAD for slice-level extraction (already applied during loading)
    model.set_vad(False)

    # Run condition sweep
    enrol_cache = {}
    probe_cache = {}
    condition_results = []

    for enrol_dur in sorted(args.enrol_durs):
        for probe_dur in sorted(args.probe_durs):
            result = run_condition(
                audio_dict, model, enrol_dur, probe_dur,
                args.guard_band, args.probe_mode, sample_rate,
                args.num_pairs, args.num_repetitions, args.seed,
                enrol_cache, probe_cache)
            condition_results.append(result)

    # Build output dict
    all_predictions = [p for r in condition_results for p in r["predictions"]]
    overall_accuracy = top_k_accuracy(all_predictions, k=1)

    # Aggregated confusion matrix (over all speakers that appear anywhere)
    all_speaker_ids = sorted(audio_dict.keys())
    n_spk = len(all_speaker_ids)
    spk_to_idx = {spk: i for i, spk in enumerate(all_speaker_ids)}
    aggregated_cm = np.zeros((n_spk, n_spk), dtype=np.int64)
    for p in all_predictions:
        true_idx = spk_to_idx[p["true_speaker"]]
        pred_idx = spk_to_idx[p["ranked_speakers"][0]]
        aggregated_cm[true_idx, pred_idx] += 1

    # Matched-subset: speakers present in ALL conditions
    all_condition_speakers = [set(r["speaker_ids"]) for r in condition_results]
    matched_speakers = sorted(set.intersection(*all_condition_speakers))
    matched_per_condition = []
    matched_overall_acc = 0.0
    matched_overall_decisions = 0

    if len(matched_speakers) >= 2:
        matched_set = set(matched_speakers)
        results_by_condition = {(r["enrol_dur"], r["probe_dur"]): r
                                for r in condition_results}
        for ed in sorted(args.enrol_durs):
            for pd in sorted(args.probe_durs):
                r = results_by_condition[(ed, pd)]
                matched_preds = [
                    p for p in r["predictions"]
                    if p["pair"][0] in matched_set
                    and p["pair"][1] in matched_set]
                acc = top_k_accuracy(matched_preds, k=1)
                matched_per_condition.append({
                    "enrol_dur": ed,
                    "probe_dur": pd,
                    "accuracy": acc,
                    "num_decisions": len(matched_preds),
                })

        all_matched_preds = [
            p for p in all_predictions
            if p["pair"][0] in matched_set and p["pair"][1] in matched_set]
        matched_overall_acc = top_k_accuracy(all_matched_preds, k=1)
        matched_overall_decisions = len(all_matched_preds)

    conditions_output = []
    for r in condition_results:
        conditions_output.append({
            "enrol_dur": r["enrol_dur"],
            "probe_dur": r["probe_dur"],
            "accuracy": r["accuracy"],
            "num_decisions": r["num_decisions"],
            "num_speakers": r["num_speakers"],
            "num_pairs": r["num_pairs"],
            "confusion_matrix": r["confusion_matrix"],
            "speaker_ids": r["speaker_ids"],
        })

    output = {
        "config": {
            "pretrain": args.pretrain,
            "apply_vad": args.apply_vad,
            "enrol_durs": args.enrol_durs,
            "probe_durs": args.probe_durs,
            "guard_band": args.guard_band,
            "probe_mode": args.probe_mode,
            "num_repetitions": args.num_repetitions,
            "num_speakers_loaded": num_original,
            "fixed_speakers": args.fixed_speakers,
            "seed": args.seed,
        },
        "conditions": conditions_output,
        "aggregated": {
            "confusion_matrix": aggregated_cm.tolist(),
            "speaker_ids": all_speaker_ids,
            "overall_accuracy": overall_accuracy,
        },
        "matched_subset": {
            "speaker_ids": matched_speakers,
            "num_speakers": len(matched_speakers),
            "per_condition": matched_per_condition,
            "overall_accuracy": matched_overall_acc,
            "num_decisions": matched_overall_decisions,
        },
    }

    # Print results
    print_results(output)

    # Save JSON results
    if args.output_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)),
                    exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as fout:
            json.dump(output, fout, indent=2)
        print("\nResults saved -> {}".format(args.output_file))


if __name__ == '__main__':
    main()
