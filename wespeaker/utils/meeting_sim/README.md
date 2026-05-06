# meeting_sim -- Multi-Party Meeting Simulation

Generates synthetic multi-speaker meeting audio by concatenating single-speaker utterances and convolving with room impulse responses (RIRs) from a simulated shoebox room.
Produces mixed audio files with ground-truth RTTM annotations suitable for training or evaluating speaker diarization and verification systems.

The pipeline has two main stages:

1. **Enrolment** -- build a speaker pool from audio files and pre-extracted (or freshly computed) speaker embeddings.
   Speakers are clustered by vocal similarity so that later selection can control how confusable the meeting participants are.

2. **Generation** -- batch-produce meetings by selecting speakers, constructing a turn sequence with overlap, simulating room acoustics, and mixing everything down to WAV + RTTM.


## Command-Line Interface

The CLI is exposed via `python-fire` on the `MeetingSimulator` class:

```
# From the installed package:
pip install -e .
wespeaker-meeting-sim <subcommand> [options]

# Or run the module directly:
python -m wespeaker.utils.meeting_sim <subcommand> [options]
```

### `enroll` -- Build a Speaker Pool

```
wespeaker-meeting-sim enroll \
    --wav_scp <wav.scp> \
    --utt2spk <utt2spk> \
    --output_dir <pool_dir> \
    [--embedding_scp <embedding.scp>] \
    [--model_dir <wespeaker_model_dir>] \
    [--device cpu|cuda:0] \
    [--config <config.yaml>] \
    [--similarity_thresh 0.75] \
    [--max_enrol_utts 20] \
    [--precompute_vad] \
    [--n_workers 8] \
    [--seed 42]
```

Reads Kaldi-style `wav.scp` and `utt2spk` files.
Provide `--embedding_scp` to load pre-extracted embeddings from ark/scp format, or `--model_dir` to extract them on the fly with a wespeaker model.
Only embeddings needed for enrolment are loaded (random-access from the ark file based on `max_enrol_utts` subsampling), so this is fast even with large embedding archives.

Computes pairwise cosine similarities and clusters speakers with agglomerative clustering at `similarity_thresh`.

Parameters default to values from the config file (`speakers.similarity_thresh`, `speakers.max_enrol_utts`, `batch.seed`, `batch.n_workers`).
CLI flags override config values.

Use `--max_enrol_utts` to cap the number of utterances used per speaker when computing the mean embedding.
If omitted, all available utterances are used.

Pass `--precompute_vad` to run Silero VAD on all utterances and cache the speech boundaries in the pool.
This makes subsequent batch generation significantly faster (no per-turn VAD inference at generation time).
Use `--n_workers` to parallelise VAD computation across processes.
Without `--precompute_vad`, VAD runs on the fly during generation.

Outputs `pool.pkl` and `similarity_matrix.npy` in `output_dir`.

### `generate` -- Batch-Generate Meetings

```
wespeaker-meeting-sim generate \
    --pool_dir <pool_dir> \
    --output_dir <output_dir> \
    [--n_meetings 10] \
    [--config <config.yaml>] \
    [--n_workers 1] \
    [--seed 42] \
```

Loads a speaker pool and YAML config (defaults to `default_config.yaml`), then generates `n_meetings` meetings.

Each meeting is written to `<output_dir>/meetings/meeting_NNNN/` with:
- `mixed.wav` -- beamformed mono mix (always)
- `multichannel.wav` -- per-mic signals (if `output.channels` includes multichannel)
- `meeting.rttm` -- ground-truth speaker timing
- `metadata.json` -- room geometry, speaker info, similarity scores
- `per_speaker/` -- optional reverberant/dry per-speaker WAVs

### `select` -- Debug: Speaker Selection

```
wespeaker-meeting-sim select \
    --pool_dir <pool_dir> \
    [--n_speakers 3] \
    [--similarity_mode similar-close-subgroup] \
    [--similar_subgroup_size 2] \
    [--seed 42]
```

Prints the selected speaker group and pairwise similarities without generating audio.

### `turns` -- Debug: Turn Sequence

```
wespeaker-meeting-sim turns \
    --pool_dir <pool_dir> \
    [--n_speakers 3] \
    [--duration 300] \
    [--overlap_ratio 0.15] \
    [--seed 42]
```

Prints the turn timeline with onset/offset times and overlap annotations.

### `room` -- Debug: Room Geometry

```
wespeaker-meeting-sim room \
    [--config <config.yaml>] \
    [--seed 42]
```

Creates a room from config, computes RIRs, and prints geometry statistics.


## Configuration

All parameters are controlled via a YAML config file (`default_config.yaml` provides the defaults).
Key sections:

### Top-level

| Key              | Default | Description                                         |
|------------------|---------|-----------------------------------------------------|
| `resample_rate`  | 16000   | Sample rate in Hz for all audio processing          |

### `speakers`

| Key                    | Default                  | Description                                          |
|------------------------|--------------------------|------------------------------------------------------|
| `n_speakers`           | 3                        | Number of participants per meeting                   |
| `max_enrol_utts`       | 3                        | Max utterances per speaker for mean embedding (null = all) |
| `similarity_mode`      | similar-close-subgroup   | Speaker selection strategy (see below)               |
| `similar_subgroup_size`| 2                        | Size of the similar subgroup (subgroup modes)        |
| `fallback_strategy`    | dissimilar               | How to fill remaining slots: `random` or `dissimilar`|
| `similarity_thresh`    | 0.75                     | Cosine-sim threshold for "same cluster"              |
| `dissimilarity_thresh` | 0.55                     | Max similarity for "dissimilar" selections           |
| `unique_across_meetings`| false                   | Draw without replacement across meetings (`random` mode, `n_workers: 1` only) |

Similarity modes (`similarity_mode` values):

- `random` -- no constraints.
- `all-similar` -- all speakers drawn from one similarity cluster.
- `all-dissimilar` -- each speaker from a different cluster, all pairwise similarities below `dissimilarity_thresh`.
- `similar-close-subgroup` -- a subgroup of similar speakers placed at positions with the smallest angular separation from the mic array (hardest to distinguish by beamforming); remainder filled per `fallback_strategy`.
- `similar-distant-subgroup` -- similar subgroup at positions with the largest angular separation.

### `meeting`

| Key                        | Default                  | Description                                          |
|----------------------------|--------------------------|------------------------------------------------------|
| `duration_s`               | 300.0                    | Target meeting length in seconds                     |
| `overlap_ratio`            | 0.15                     | Target fraction of meeting time with overlapping speech |
| `random_jump_rate`         | 0.3                      | Probability of jumping to a random speaker instead of round-robin |
| `min_overlap_anchor_dur_s` | 1.0                      | Minimum turn duration (s) to be eligible for receiving overlaps |
| `allow_multi_overlap`      | true                     | Whether long anchors can receive multiple overlapping interjections (one per 4s of anchor duration) |
| `pause_range`              | [0.2, 1.5]              | Inter-turn pause range [min_s, max_s]                |
| `overlap_weights`          | bc:0.3 comp:0.3 coll:0.2 sim:0.2 | Relative weights for overlap event types |
| `segment_durations`        | (see below)              | Weighted duration bins for turn lengths              |
| `exchange_bursts`          | (see below)              | Rapid back-and-forth exchange episodes               |

`segment_durations` is a list of bins, each with a `range` ([min_s, max_s]) and a `weight`.
Weights are normalised internally.
Default:

```yaml
segment_durations:
  - range: [1.0, 3.0]    # short
    weight: 0.3
  - range: [5.0, 10.0]   # medium
    weight: 0.4
  - range: [12.0, 20.0]  # long
    weight: 0.3
```

`exchange_bursts` injects contiguous runs of rapid turn-switching between a speaker pair, useful for stress-testing diarization on fast exchanges without biasing the whole meeting toward short segments.
Set to `null` to disable.
Default:

```yaml
exchange_bursts:
  n_bursts: 2               # episodes per meeting
  n_turns: [4, 8]           # turns per burst (uniform range)
  segment_range: [0.5, 2.0] # segment duration within bursts (seconds)
  pause_range: [0.05, 0.3]  # inter-turn pause within bursts (seconds)
  target_speakers: "similar-subgroup"  # or "random-pair"
```

When `target_speakers` is `"similar-subgroup"`, the burst alternates between the speakers placed at the similar-subgroup positions (i.e. the pair that is both vocally similar and spatially close/distant depending on mode).
`"random-pair"` picks any two speakers.

Overlap types (distributed according to `overlap_weights`):

| Type            | Duration   | Energy     | Description                            |
|-----------------|------------|------------|----------------------------------------|
| `backchannel`   | 0.5--1.5 s | 0.3--0.6x  | Short interjection mid-turn            |
| `competitive`   | 1.0--3.0 s | 0.8--1.0x  | Next speaker starts before current ends|
| `collaborative` | 0.3--1.0 s | ramp 1->0.3| Finishing another's sentence           |
| `simultaneous`  | 2.0--5.0 s | 1.0x       | Full-energy parallel speech            |

### `room`

| Key                      | Default           | Description                                     |
|--------------------------|-------------------|-------------------------------------------------|
| `length`                 | 4.0               | Room x-dimension in meters                      |
| `width`                  | 3.0               | Room y-dimension in meters                      |
| `height`                 | 2.4               | Room z-dimension in meters                      |
| `rt60`                   | 0.4               | Reverberation time (T60) in seconds             |
| `speaker_position_jitter`| 0.2               | Half-width of uniform position jitter (meters)  |
| `speaker_positions`      | [[1.5,1.0,1.3], [2.5,1.0,1.3], [2.0,2.0,1.3]] | Fixed [x,y,z] positions per slot; must supply at least `n_speakers` entries |

### `array`

| Key       | Default | Description                              |
|-----------|---------|------------------------------------------|
| `n_mics`  | 8       | Number of microphone elements            |
| `spacing` | 0.04    | Inter-element spacing in meters (4 cm)   |

### `output`

| Key              | Default | Description                                         |
|------------------|---------|-----------------------------------------------------|
| `channels`       | [mono]  | List of output formats: `mono` (beamformed), `stereo` (2-mic ~17 cm baseline), `multichannel` (all array mics). E.g. `[mono, stereo]`. |
| `save_reverberant`| false  | Write per-speaker reverberant WAVs                  |
| `save_dry`       | false   | Write per-speaker anechoic WAVs                     |

### `noise`

| Key       | Default | Description                                   |
|-----------|---------|-----------------------------------------------|
| `enabled` | false   | Whether to add background noise               |
| `wav_path`| null    | Path to noise WAV file                        |
| `snr_db`  | 30.0    | Target signal-to-noise ratio in dB            |

### `batch`

| Key          | Default | Description                                      |
|--------------|---------|--------------------------------------------------|
| `n_meetings` | 10      | Number of meetings to generate                   |
| `seed`       | 42      | Base random seed (used for enrolment subsampling and generation; meeting i uses seed + i) |
| `n_workers`  | 1       | Parallel workers (meeting generation and VAD pre-computation) |
| `vary`       | (see below) | Per-meeting parameter jitter specifications  |

The `batch.vary` block maps dotted config paths to jitter specs.
Numeric values specify the half-width of uniform jitter around the base value: e.g. `room.length: 0.3` means each meeting samples room length from U(4.7, 5.3).
List values cycle round-robin across meetings: e.g. `speakers.similarity_mode: [random, all-similar]` alternates modes.

Default vary specs:

```yaml
vary:
  room.length: 0.3
  room.width: 0.3
  room.rt60: 0.1
  room.speaker_position_jitter: 0.05
  meeting.overlap_ratio: 0.05
```


## Python API

The module exports `MeetingSimulator` from the top-level `__init__.py`.
The classes and functions below are useful for programmatic interaction.

### `wespeaker.utils.meeting_sim.room`

```python
from wespeaker.utils.meeting_sim.room import (
    RoomConfig, ArrayConfig, MeetingRoom, create_room, compute_rirs, beamform,
    plot_room_topdown, select_stereo_mics,
)
```

**`RoomConfig(length=4.0, width=3.0, height=2.4, rt60=0.4)`**
  Shoebox room dimensions and reverberation time.

**`ArrayConfig(n_mics=8, spacing=0.04)`**
  Linear microphone array geometry.

**`MeetingRoom`**
  Returned by `create_room`.
  Attributes:
  - `room` -- `pyroomacoustics.ShoeBox` instance
  - `mic_center` -- np.ndarray, 3D mic array center position
  - `speaker_positions` -- list of np.ndarray, 3D position per speaker
  - `room_config`, `array_config` -- the configs used

**`create_room(room_cfg, array_cfg, n_speakers=3, ...)`**
  Build a room with mic array and speaker positions.
  Returns `MeetingRoom`.
  Accepts `speaker_position_jitter` (meters), `configured_positions` (list of [x,y,z]), and a numpy `rng`.

**`compute_rirs(meeting_room, sample_rate=16000)`**
  Returns np.ndarray of shape `(n_speakers, n_mics, rir_length)`.

**`beamform(multichannel, meeting_room, sample_rate=16000)`**
  Delay-and-sum beamforming steered toward the mean speaker position.
  Input: `(n_mics, n_samples)`.
  Output: `(n_samples,)`.

**`plot_room_topdown(meeting_room)`**
  Return a `matplotlib.figure.Figure` with a bird's-eye (x-y) view of the room, showing the mic array (blue squares) and speaker positions (red circles).
  Useful for quickly verifying room geometry.
  Call `fig.savefig(path)` or `plt.show()` to display.

**`select_stereo_mics(array_cfg)`**
  Returns `(left_idx, right_idx)` -- the mic pair whose separation is closest to 17 cm, most-centered in the array, ties broken toward mic 0.
  Used internally when `'stereo'` is in `output_channels`.

### `wespeaker.utils.meeting_sim.speakers`

```python
from wespeaker.utils.meeting_sim.speakers import (
    SpeakerInfo, SpeakerPool,
    build_speaker_pool, cluster_speakers, compute_vad_segments,
    select_speakers, find_subgroup_positions, save_pool, load_pool,
)
```

**`SpeakerInfo(speaker_id, wav_paths, embedding=None, vad_segments=None)`**
  Single speaker: ID, list of audio paths, mean embedding vector, and optional cached VAD segments (dict of wav_path -> list of (start_sample, end_sample) tuples).

**`SpeakerPool`**
  Attributes:
  - `speakers` -- dict of speaker_id -> SpeakerInfo
  - `speaker_ids` -- ordered list (defines similarity matrix indexing)
  - `similarity_matrix` -- np.ndarray, shape `(N, N)`, values in [0, 1]
  - `clusters` -- dict of cluster_id -> list of speaker_ids
  - `speaker_to_cluster` -- dict of speaker_id -> cluster_id

**`build_speaker_pool(wav_scp, utt2spk, embedding_scp=None, model_dir=None, device='cpu', max_enrol_utts=None, seed=42)`**
  Construct pool from Kaldi-format files.
  Returns `SpeakerPool` with similarity matrix computed.
  `max_enrol_utts` caps utterances per speaker for mean embedding (None = use all).
  When `embedding_scp` is provided, only the needed utterances are loaded via random access.

**`cluster_speakers(pool, similarity_thresh=0.75)`**
  Agglomerative clustering (average linkage) on the similarity matrix.
  Modifies and returns the pool.

**`compute_vad_segments(pool, sample_rate=16000, min_dur=0.1, n_workers=1)`**
  Run Silero VAD on all utterances and cache segments in each `SpeakerInfo.vad_segments`.
  Parallelisable via `n_workers`.

**`select_speakers(pool, n_speakers, similarity_mode, ...)`**
  Select a speaker group per the given mode.
  Returns list of `SpeakerInfo` ordered by position index.

**`save_pool(pool, output_dir)` / `load_pool(pool_dir)`**
  Persist/restore via pickle + npy.

### `wespeaker.utils.meeting_sim.turns`

```python
from wespeaker.utils.meeting_sim.turns import (
    Segment, SegmentChunk, Turn, OverlapSpec,
    build_turn_sequence, get_speech_segments, select_segment,
)
```

**`Segment(speaker_id, wav_path=None, start_sample=0, end_sample=0, chunks=None, pause_samples=0)`**
  One or more audio chunks from a speaker.
  For multi-chunk segments, chunks are concatenated with `pause_samples` silence between them.
  `.duration_samples` gives total length.

**`Turn(segment, meeting_onset_sample, overlap=None)`**
  A segment placed on the meeting timeline.
  `overlap` is an optional `OverlapSpec`.

**`OverlapSpec(overlap_type, duration_s, energy_scale)`**
  Describes how an overlapping turn is scaled.

**`build_turn_sequence(speakers, meeting_duration_s, overlap_ratio, overlap_weights=None, segment_durations=None, random_jump_rate=0.3, exchange_bursts=None, subgroup_indices=None, min_overlap_anchor_dur_s=1.0, allow_multi_overlap=True, pause_range=None, sample_rate=16000, rng=None)`**
  Build a full meeting turn list.
  Returns `list[Turn]` sorted by onset.

**`get_speech_segments(wav_path, sample_rate=16000, min_dur=0.3)`**
  Run Silero VAD on an audio file.
  Returns list of `(start_sample, end_sample)` tuples.

**`select_segment(speaker, target_dur_s, sample_rate, rng, ...)`**
  Pick an audio segment from a speaker's recordings, with VAD-based trimming and optional multi-utterance concatenation.

### `wespeaker.utils.meeting_sim.mixer`

```python
from wespeaker.utils.meeting_sim.mixer import (
    MixConfig, mix_meeting, write_meeting_outputs,
    generate_rttm, write_rttm, add_background_noise,
)
```

**`MixConfig(sample_rate=16000, output_channels=None, noise_wav=None, noise_snr_db=30.0, save_reverberant=False, save_dry=False)`**
  Controls mixing behaviour and which outputs to produce.
  `output_channels` is a list of `'mono'`, `'stereo'`, and/or `'multichannel'`; a bare string is also accepted. Defaults to `['mono']`.

**`mix_meeting(turns, meeting_room, rirs, mix_cfg, speaker_index_map, rng)`**
  Core mixing function.
  Convolves each turn with its speaker's RIR and sums into multichannel buffers.
  Returns a dict with keys `'mono'`, `'multichannel'` (optional), `'reverberant'` (optional), `'dry'` (optional), `'rttm'`.

**`write_meeting_outputs(result, output_dir, meeting_id, mix_cfg)`**
  Write all outputs (WAVs, RTTM, per-speaker files) to disk.

**`generate_rttm(turns, sample_rate)`**
  Extract `(speaker_id, start_s, duration_s)` tuples from turns.

**`add_background_noise(signal, noise_path, snr_db, rng)`**
  Mix noise at a target SNR.


## Design Decisions

### Speaker pool and similarity

Speaker selection draws independently per meeting by default, so speakers may appear in multiple meetings within a batch.
For the `random` mode, the expected number of unique speakers follows birthday-problem statistics: with a pool of N speakers and M meetings of K speakers each, expect roughly `N * (1 - ((N-K)/N)^M)` unique speakers -- e.g. ~55 from a 110-speaker pool over 25 3-speaker meetings, or ~96 over 75 meetings.
Other modes are more constrained: `all-similar` can only draw from clusters with >= K members, limiting the reachable pool.
Set `speakers.unique_across_meetings: true` to cycle through the pool without replacement in `random` mode (sequential generation only).

Speakers are represented by their mean L2-normalised embedding.
Pairwise similarity uses the wespeaker convention `(cosine + 1) / 2`, mapping raw cosine [-1, 1] into [0, 1].
This matches the score domain used in verification thresholds elsewhere in wespeaker.

Agglomerative clustering (average linkage, distance = 1 - similarity) groups speakers so that later selection can request "speakers from the same cluster" or "from different clusters" without recomputing similarities at selection time.
The threshold is deliberately configurable -- tighter thresholds produce smaller clusters which may fail the "find N similar speakers" constraint, so the pipeline auto-relaxes in 0.05 decrements.

### Subgroup placement and angular separation

The `similar-close-subgroup` mode is designed to create maximally challenging diarization scenarios: a pair of speakers whose embeddings are already similar are placed at room positions that also have the smallest angular separation as seen from the mic array.
This means beamforming provides minimal spatial discrimination between them.
Conversely, `similar-distant-subgroup` places similar speakers far apart angularly, testing whether the diarization system can exploit spatial cues to compensate for embedding confusion.

Angular separation is computed as the angle between direction vectors from `mic_center` to each speaker position.
When `mic_center` is not available (e.g. in tests), the code falls back to Euclidean distance between positions.

### Turn construction

Turn-taking uses a single-pass timeline generator that interleaves normal turns, exchange bursts, and overlap events.
At each step the generator either enters an exchange burst (if a trigger point is reached) or emits a normal "anchor" turn and optionally attaches one or more overlaps to it.

**Normal turns**: Segment lengths are drawn from a mixture of short (1--3 s), medium (5--10 s), and long (12--20 s) durations.
Speaker transitions use a weighted round-robin with 30% probability of a random jump to avoid lock-step alternation.

**Overlap insertion**: After emitting an anchor turn (minimum 1.0 s), an overlap budget of `overlap_ratio * duration_s` seconds is spent by probabilistically attaching overlaps.
Long anchors (>4 s) can receive multiple interjections.
The insertion probability decreases as budget is consumed, spreading overlaps across the meeting.
Overlap types include backchannel/simultaneous (placed within the anchor's time span from a different speaker) and competitive/collaborative (the next turn starts before the anchor ends).
Same-speaker self-overlaps are prevented by construction: interjectors are always different from the anchor speaker, and a per-anchor exclusion set prevents the same speaker from receiving multiple overlaps on a single anchor.

**Exchange bursts**: Rapid-fire alternation between a chosen speaker pair, inserted at evenly-spaced trigger points.
No overlaps are placed on or around burst turns.

### Segment selection and VAD trimming

Source utterances often have leading/trailing silence that would create unnatural dead air in the mix.
Silero VAD detects speech boundaries and the segment is trimmed to the first/last speech frame plus a short buffer (default 80 ms).
This preserves natural onset/offset coarticulation while removing recording-artefact silence.

VAD segments can be pre-computed during enrolment (`--precompute_vad`) and cached in the speaker pool.
At generation time, cached boundaries are looked up directly -- no VAD inference needed.
If the pool has no cached VAD (enrolment ran without `--precompute_vad`), VAD is computed on the fly per utterance as a fallback.

For long turns that exceed a single source utterance, the pipeline concatenates multiple utterances from the same speaker with short randomised pauses (50--200 ms).
A `used_paths` set per speaker avoids reusing the same utterance within a meeting when the pool is large enough.

### Room acoustics

The room uses pyroomacoustics' shoebox model with the image-source method.
RT60 is converted to wall absorption via `inverse_sabine`.
The mic array is placed at x=0.1 (near one wall), facing into the room.
The default layout has two speakers close together on one side and one far away -- this asymmetry is deliberate, creating a near-field/far-field contrast that interacts with the subgroup placement logic.

RIRs are computed once per meeting and stored as a 3D tensor `(n_speakers, n_mics, rir_length)`.
Each turn's audio is convolved with its speaker's per-mic RIR via FFT convolution, then summed into the multichannel buffer.
Beamforming (delay-and-sum toward the mean speaker position) produces the mono output.

### Overlap energy scaling

Different overlap types model distinct conversational phenomena:

- **Backchannel** (low energy, short) -- listener feedback like "uh-huh" that doesn't interrupt the floor-holder.
- **Competitive** (near-full energy, medium) -- a speaker attempts to take the floor; achieved by pulling the next turn's onset earlier so it overlaps the current turn's tail.
- **Collaborative** (ramped energy, short) -- completing another's sentence; the energy ramps down to signal deference.
- **Simultaneous** (full energy, long) -- extended parallel speech from an independent speaker.

Fade-in envelopes (10 ms) prevent clicks at overlap onsets.

### Per-meeting parameter jitter

To produce a diverse dataset from a single config, the `batch.vary` mechanism applies per-meeting jitter.
Each meeting gets an independent numpy Generator seeded from `base_seed + meeting_index`, ensuring reproducibility while varying room size, RT60, overlap ratio, and speaker positions across the batch.
List-valued vary specs cycle deterministically (round-robin by meeting index) to guarantee balanced coverage of categorical conditions.

### Reproducibility

All randomness flows through explicit `numpy.random.Generator` instances seeded from `base_seed + meeting_idx`.
This means a given (config, seed, meeting_index) triple always produces the same output regardless of batch size or parallelism level.


## Dependencies

- `pyroomacoustics` -- room simulation and RIR computation
- `silero-vad` (via `silero_vad`) -- speech activity detection for segment trimming
- `torchaudio` -- audio I/O and resampling
- `soundfile` -- WAV output
- `kaldiio` -- reading pre-extracted embeddings from ark/scp
- `scipy` -- FFT convolution and hierarchical clustering
- `numpy`
- `tqdm` -- progress bars
- `python-fire` -- CLI
- `pyyaml` -- config loading
