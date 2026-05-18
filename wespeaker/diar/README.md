# Streaming Speaker Diarization

## Overview

`stream_diar.py` implements a causal, sliding-window speaker diarization pipeline. It processes audio incrementally -- producing speaker labels as it goes -- rather than requiring the full recording up front.

A window of `--window-secs` (default 1.5s) slides forward by `--stride-secs` (default 0.5s). Each window is passed through:

1. VAD gating (Silero VAD, optional)
2. Fbank feature extraction (80-dim, Kaldi-compatible)
3. Per-window cepstral mean normalisation
4. Speaker embedding extraction (ONNX or PyTorch model)
5. Speaker assignment (identification or clustering)

This contrasts with the offline diarization pipeline (through `spectral_clusterer.py`, `umap_clusterer.py`) which requires all embeddings before computing a global clustering solution.


## Prerequisites

- **Speaker embedding model**: either an ONNX file (`.onnx`) or a PyTorch model directory (containing `config.yaml` + checkpoint).
- **For identification mode**: an `enrol.scp` file mapping speaker IDs to pre-computed mean embeddings. See [Producing Enrolment Data](#producing-enrolment-data) below.
- **Silero VAD**: installed as a dependency (`silero-vad` package). Used by default; disable with `--no-vad`.
- **For live device input**: the `sounddevice` package. On Linux this also requires the system `libportaudio2` library; macOS and Windows wheels bundle it. Not needed for file-based processing.


## Quick Start

```bash
# Cluster speakers in a single recording
python -m wespeaker.diar.stream_diar \
    --model /path/to/model.onnx \
    --wav recording.wav \
    --assign cluster \
    --output labels.txt
```


## Input Modes

### Live device input (`--audio-device`)

Stream from a sound input device in real time. Use `--list-devices` to find the device ID.

```bash
# List available devices
python -m wespeaker.diar.stream_diar --list-devices

# Stream from device 0
python -m wespeaker.diar.stream_diar \
    --model model.onnx \
    --audio-device 0 \
    --assign cluster
```

Press Ctrl+C to stop. Results are written on exit if `--output` is given.

### Single file (`--wav`)

Process one audio file. The utterance ID is derived from the filename.

```bash
python -m wespeaker.diar.stream_diar \
    --model model.onnx \
    --wav meeting.wav \
    --output meeting.labels
```

### Batch processing (`--wav-scp`)

Process multiple files listed in a Kaldi-format `wav.scp`:

```bash
python -m wespeaker.diar.stream_diar \
    --model model.onnx \
    --wav-scp data/wav.scp \
    --assign identify \
    --enrol-scp data/enrol.scp \
    --output labels.txt \
    --output-rttm output.rttm
```


## Speaker Assignment

### Speaker Identification (`--assign identify`)

Assigns each window to the most similar enrolled speaker by cosine similarity. Requires pre-enrolled speaker embeddings.

```bash
python -m wespeaker.diar.stream_diar \
    --model model.onnx \
    --wav meeting.wav \
    --assign identify \
    --enrol-scp enrol.scp \
    --output labels.txt
```

Key parameters:

- `--enrol-scp`: path to enrolment scp file (required)
- `--speakers SPK1 SPK2 ...`: restrict identification to a subset of enrolled speakers
- `--metadata-dir DIR`: per-meeting metadata directory containing `{utt_id}/metadata.json` with a `speakers` list, used to automatically select the relevant speaker subset
- `--assign-threshold`: minimum cosine similarity to emit a label (windows below this are discarded)

#### Producing enrolment data

Use `tools/enroll_speakers.py` to compute per-speaker mean embeddings:

```bash
python tools/enroll_speakers.py \
    --wav-scp data/enrol_wav.scp \
    --utt2spk data/utt2spk \
    --model /path/to/model.onnx \
    --output data/enrol
```

This produces `data/enrol.ark` and `data/enrol.scp`. The scp file maps speaker IDs to their L2-normalised mean embeddings in Kaldi ark format:

```
speaker_a /path/to/enrol.ark:7
speaker_b /path/to/enrol.ark:792
speaker_c /path/to/enrol.ark:1577
```

If you already have per-utterance embeddings (e.g. from the offline pipeline's `extract_emb.py`), you can skip model inference:

```bash
python tools/enroll_speakers.py \
    --wav-scp data/enrol_wav.scp \
    --utt2spk data/utt2spk \
    --embedding-scp embeddings/emb.scp \
    --output data/enrol
```

Use `--max-enroll-utts N` to cap the number of utterances per speaker (randomly subsampled).

**Important**: use the same embedding model for enrolment and diarization. Embeddings from different models are not compatible.

### Online Clustering (`--assign cluster`)

Maintains running speaker centroids and assigns each window to the most similar centroid, or creates a new cluster if similarity is below threshold. No enrolment data is needed.

```bash
python -m wespeaker.diar.stream_diar \
    --model model.onnx \
    --wav meeting.wav \
    --assign cluster \
    --cluster-threshold 0.55 \
    --output labels.txt
```

Key parameters:

- `--cluster-threshold` (default 0.5): minimum cosine similarity to assign to an existing cluster. Lower values merge more aggressively; higher values produce more clusters.
- `--max-speakers` (default 20): maximum number of clusters to create. Once reached, all new embeddings are assigned to the nearest existing centroid regardless of threshold.


## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--window-secs` | 1.5 | Window duration for embedding extraction |
| `--stride-secs` | 0.5 | Stride between windows (controls time resolution) |
| `--no-subseg-cmn` | off | Disable per-window cepstral mean normalisation |
| `--wait-full-buffer` | off | Skip inference until a full window is available |
| `--vad-threshold` | 0.5 | Silero VAD speech probability threshold |
| `--no-vad` | off | Disable VAD entirely |
| `--assign-threshold` | None | Discard windows with confidence below this |
| `--real-time` | off | Pace file processing to simulate real-time input |
| `--quiet` | off | Suppress per-window stdout output |

**Tuning guidance**:

- Shorter `--stride-secs` gives finer time resolution at the cost of more compute per second of audio.
- For identification, `--assign-threshold` can filter out non-speech or uncertain regions that VAD alone misses.
- For clustering, if speakers are being over-split, lower `--cluster-threshold`; if different speakers are being merged, raise it.


## Output Formats

### Labels file (`--output`)

One line per labelled window:

```
utt_id start_sec end_sec speaker_label confidence
```

Example:

```
meeting001 0.000 1.500 speaker_a 0.8234
meeting001 0.500 2.000 speaker_a 0.7891
meeting001 1.000 2.500 speaker_b 0.6512
```

Confidence scores are the cosine similarities between each chunk and the enrolment embedding or cluster centroid of its assigned speaker.

### RTTM (`--output-rttm`)

Standard NIST RTTM format. Adjacent windows with the same label are
merged into contiguous segments:

```
SPEAKER meeting001 1 0.000 2.000 <NA> <NA> speaker_a <NA> <NA>
SPEAKER meeting001 1 1.000 1.500 <NA> <NA> speaker_b <NA> <NA>
```
