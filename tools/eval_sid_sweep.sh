#!/usr/bin/env bash
# Copyright (c) 2026 Dan Wells
#
# Sweep speaker identification evaluation across gallery sizes and
# enrollment conditions. Generates base trials once, then realizes and
# evaluates for each (num_speakers, num_enroll_utts, enroll_strategy)
# combination.
#
# Usage:
#   bash tools/eval_sid_sweep.sh \
#     --embedding_scp exp/embeddings/xvector.scp \
#     --utt2spk data/utt2spk \
#     --output_dir exp/sid_sweep
#
# Or with end-to-end extraction:
#   bash tools/eval_sid_sweep.sh \
#     --wav_scp data/wav.scp \
#     --utt2spk data/utt2spk \
#     --pretrain campplus \
#     --device cuda \
#     --output_dir exp/sid_sweep

set -euo pipefail

# Defaults
embedding_scp=""
wav_scp=""
utt2spk=""
pretrain=""
device="cpu"
output_dir="exp/sid_sweep"
num_trials=500
seed=42
gallery_sizes="5 10 20"
enroll_counts="1 3 5"
strategies="mean max_score"

# Parse arguments
. tools/parse_options.sh || {
  # Fallback if parse_options.sh is not available
  while [ $# -gt 0 ]; do
    case "$1" in
      --embedding_scp) embedding_scp="$2"; shift 2 ;;
      --wav_scp) wav_scp="$2"; shift 2 ;;
      --utt2spk) utt2spk="$2"; shift 2 ;;
      --pretrain) pretrain="$2"; shift 2 ;;
      --device) device="$2"; shift 2 ;;
      --output_dir) output_dir="$2"; shift 2 ;;
      --num_trials) num_trials="$2"; shift 2 ;;
      --seed) seed="$2"; shift 2 ;;
      --gallery_sizes) gallery_sizes="$2"; shift 2 ;;
      --enroll_counts) enroll_counts="$2"; shift 2 ;;
      --strategies) strategies="$2"; shift 2 ;;
      *) echo "Unknown option: $1"; exit 1 ;;
    esac
  done
}

if [ -z "$utt2spk" ]; then
  echo "Error: --utt2spk is required"
  exit 1
fi

if [ -z "$embedding_scp" ] && { [ -z "$wav_scp" ] || [ -z "$pretrain" ]; }; then
  echo "Error: provide --embedding_scp or both --wav_scp and --pretrain"
  exit 1
fi

mkdir -p "$output_dir/trials" "$output_dir/results"

# Find the maximum gallery size and enrollment count for base trial generation
max_nspk=0
for nspk in $gallery_sizes; do
  [ "$nspk" -gt "$max_nspk" ] && max_nspk=$nspk
done
max_nenroll=0
for nenroll in $enroll_counts; do
  [ "$nenroll" -gt "$max_nenroll" ] && max_nenroll=$nenroll
done

echo "=== Speaker Identification Grid Sweep ==="
echo "Gallery sizes: $gallery_sizes"
echo "Enrollment counts: $enroll_counts"
echo "Strategies: $strategies"
echo "Num trials: $num_trials"
echo "Max gallery: $max_nspk, Max enrollment: $max_nenroll"
echo ""

# Step 1: Generate base trials once
base_trials="$output_dir/trials/base_trials.json"
echo "Generating base trials -> $base_trials"
python -c "
from wespeaker.utils.identification_trials import (
    load_utt2spk, generate_base_trials, save_trials)

spk2utts = load_utt2spk('${utt2spk}')
base = generate_base_trials(
    spk2utts,
    num_trials=${num_trials},
    max_num_speakers=${max_nspk},
    max_num_enroll_utts=${max_nenroll},
    seed=${seed},
)
save_trials(base, '${base_trials}')
print('Generated {} base trials'.format(len(base)))
"

# Step 2: Realize trials for each condition and run evaluation
# Build the embedding argument
if [ -n "$embedding_scp" ]; then
  emb_arg="--embedding_scp $embedding_scp"
else
  emb_arg="--wav_scp $wav_scp --pretrain $pretrain --device $device"
fi

for nspk in $gallery_sizes; do
  for nenroll in $enroll_counts; do
    # Realize trials for this condition
    trial_file="$output_dir/trials/nspk${nspk}_nenroll${nenroll}.json"
    python -c "
from wespeaker.utils.identification_trials import (
    load_utt2spk, load_trials, realize_trials, save_trials)

spk2utts = load_utt2spk('${utt2spk}')
base = load_trials('${base_trials}')
trials = realize_trials(base, spk2utts,
                        num_speakers=${nspk},
                        num_enroll_utts=${nenroll},
                        seed=${seed} + 1)
save_trials(trials, '${trial_file}')
print('Realized {} trials for nspk={}, nenroll={}'.format(
    len(trials), ${nspk}, ${nenroll}))
"

    for strategy in $strategies; do
      result_file="$output_dir/results/nspk${nspk}_nenroll${nenroll}_${strategy}.json"
      echo "Running: nspk=$nspk nenroll=$nenroll strategy=$strategy"

      python wespeaker/bin/eval_identification.py \
        $emb_arg \
        --utt2spk "$utt2spk" \
        --trials "$trial_file" \
        --enroll_strategy "$strategy" \
        --output_file "$result_file"

      echo ""
    done
  done
done

echo "=== Sweep complete ==="
echo "Results in: $output_dir/results/"
echo ""
echo "Run the summary script:"
echo "  python tools/summarize_sid_sweep.py --results_dir $output_dir/results"
