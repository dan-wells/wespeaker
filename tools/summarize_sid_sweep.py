#!/usr/bin/env python3
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

"""Summarize results from a speaker identification grid sweep.

Reads result JSON files produced by eval_sid_sweep.sh and prints a
summary table. Expects filenames matching the pattern:
  nspk{N}_nenroll{K}_{strategy}.json
"""

import argparse
import glob
import json
import os
import re


def parse_filename(filename):
    """Extract condition parameters from a result filename.

    Args:
        filename: Basename like 'nspk10_nenroll3_mean.json'.

    Returns:
        tuple: (num_speakers, num_enroll_utts, strategy) or None if the
          filename doesn't match the expected pattern.
    """
    match = re.match(r'nspk(\d+)_nenroll(\d+)_(\w+)\.json$', filename)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), match.group(3)


def load_results(results_dir):
    """Load all result files from a directory.

    Args:
        results_dir: Path to directory containing result JSON files.

    Returns:
        list: List of dicts with keys: num_speakers, num_enroll_utts,
          strategy, metrics.
    """
    results = []
    pattern = os.path.join(results_dir, 'nspk*_nenroll*_*.json')
    for path in sorted(glob.glob(pattern)):
        parsed = parse_filename(os.path.basename(path))
        if parsed is None:
            continue
        num_speakers, num_enroll_utts, strategy = parsed
        with open(path, 'r', encoding='utf-8') as fin:
            data = json.load(fin)
        results.append({
            'num_speakers': num_speakers,
            'num_enroll_utts': num_enroll_utts,
            'strategy': strategy,
            'metrics': data['metrics'],
        })
    return results


def print_table(results, top_k=1):
    """Print a summary table for a given top-k accuracy.

    Rows are (num_speakers, strategy), columns are num_enroll_utts.

    Args:
        results: List of result dicts from load_results().
        top_k: Which top-k accuracy to display.
    """
    # Collect unique values
    enroll_counts = sorted(set(r['num_enroll_utts'] for r in results))
    row_keys = sorted(
        set((r['num_speakers'], r['strategy']) for r in results))

    # Build lookup
    lookup = {}
    for r in results:
        key = (r['num_speakers'], r['strategy'], r['num_enroll_utts'])
        acc = r['metrics']['top_k_accuracy'].get(str(top_k),
              r['metrics']['top_k_accuracy'].get(top_k, None))
        if acc is not None:
            lookup[key] = acc

    # Print header
    header_cols = ['gallery', 'strategy'] + [
        'enroll={}'.format(k) for k in enroll_counts]
    col_widths = [max(8, len(c)) for c in header_cols]
    header = '  '.join(c.rjust(w) for c, w in zip(header_cols, col_widths))
    print(header)
    print('-' * len(header))

    # Print rows
    for nspk, strategy in row_keys:
        row = [str(nspk), strategy]
        for nenroll in enroll_counts:
            acc = lookup.get((nspk, strategy, nenroll))
            if acc is not None:
                row.append('{:.1f}%'.format(acc * 100))
            else:
                row.append('-')
        line = '  '.join(c.rjust(w) for c, w in zip(row, col_widths))
        print(line)


def print_full_summary(results):
    """Print summary tables for top-1, top-3, and top-5 accuracy.

    Args:
        results: List of result dicts from load_results().
    """
    # Determine which top-k values are available
    all_k_values = set()
    for r in results:
        all_k_values.update(r['metrics']['top_k_accuracy'].keys())
    # Normalize to ints
    k_values = sorted(int(k) for k in all_k_values)

    for k in k_values:
        print('=== Top-{} Accuracy ==='.format(k))
        print()
        print_table(results, top_k=k)
        print()


def save_csv(results, output_path):
    """Save results as a CSV file.

    Args:
        results: List of result dicts from load_results().
        output_path: Path to write CSV.
    """
    all_k_values = set()
    for r in results:
        all_k_values.update(r['metrics']['top_k_accuracy'].keys())
    k_values = sorted(int(k) for k in all_k_values)

    header = ['num_speakers', 'num_enroll_utts', 'strategy']
    header += ['top_{}_accuracy'.format(k) for k in k_values]

    with open(output_path, 'w', encoding='utf-8') as fout:
        fout.write(','.join(header) + '\n')
        for r in sorted(results, key=lambda x: (
                x['num_speakers'], x['num_enroll_utts'], x['strategy'])):
            row = [str(r['num_speakers']), str(r['num_enroll_utts']),
                   r['strategy']]
            for k in k_values:
                acc = r['metrics']['top_k_accuracy'].get(
                    str(k), r['metrics']['top_k_accuracy'].get(k, ''))
                if acc != '':
                    row.append('{:.4f}'.format(acc))
                else:
                    row.append('')
            fout.write(','.join(row) + '\n')


def get_args():
    parser = argparse.ArgumentParser(
        description='Summarize speaker identification sweep results')
    parser.add_argument('--results_dir', required=True,
                        help='Directory containing result JSON files')
    parser.add_argument('--csv', default=None,
                        help='Optional path to save CSV output')
    return parser.parse_args()


def main():
    args = get_args()
    results = load_results(args.results_dir)
    if len(results) == 0:
        print("No result files found in {}".format(args.results_dir))
        return
    print("Loaded {} result files\n".format(len(results)))
    print_full_summary(results)
    if args.csv is not None:
        save_csv(results, args.csv)
        print("CSV saved -> {}".format(args.csv))


if __name__ == '__main__':
    main()
