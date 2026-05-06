"""Turn-taking construction for meeting simulation.

Builds temporal layouts of multi-speaker meetings with configurable
overlap patterns and VAD-informed segmentation.
"""

import numpy as np
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import torchaudio


# Lazy-loaded VAD model (shared across calls)
_vad_model = None


def _get_vad_model():
    """Load the Silero VAD model (cached after first call)."""
    global _vad_model
    if _vad_model is None:
        _vad_model = load_silero_vad()
    return _vad_model


class SegmentChunk:
    """A single chunk of audio from a source file.

    Attributes:
      wav_path: Path to the source audio file.
      start_sample: Start position within the source file (samples).
      end_sample: End position within the source file (samples).
    """

    def __init__(self, wav_path, start_sample, end_sample):
        self.wav_path = wav_path
        self.start_sample = start_sample
        self.end_sample = end_sample

    @property
    def duration_samples(self):
        return self.end_sample - self.start_sample


class Segment:
    """A segment of audio from one or more source files.

    Supports single-utterance segments (backward compatible) and
    multi-utterance concatenation for longer turns.

    Attributes:
      speaker_id: ID of the speaker.
      chunks: List of SegmentChunk objects to concatenate.
      pause_samples: Number of silence samples to insert between
        chunks when concatenating (ignored for single-chunk segments).
      wav_path: Path to source file (first chunk, for compatibility).
      start_sample: Start position in first chunk (for compatibility).
      end_sample: End position in first chunk (for compatibility).
    """

    def __init__(self, speaker_id, wav_path=None, start_sample=0,
                 end_sample=0, chunks=None, pause_samples=0):
        self.speaker_id = speaker_id
        if chunks is not None:
            self.chunks = chunks
        else:
            self.chunks = [SegmentChunk(wav_path, start_sample, end_sample)]
        self.pause_samples = pause_samples

    @property
    def wav_path(self):
        return self.chunks[0].wav_path

    @property
    def start_sample(self):
        return self.chunks[0].start_sample

    @property
    def end_sample(self):
        return self.chunks[0].end_sample

    @property
    def duration_samples(self):
        total = sum(c.duration_samples for c in self.chunks)
        if len(self.chunks) > 1:
            total += self.pause_samples * (len(self.chunks) - 1)
        return total


class OverlapSpec:
    """Specification for an overlap event.

    Attributes:
      type: One of 'backchannel', 'competitive', 'collaborative',
        'simultaneous'.
      duration_s: Duration of the overlap in seconds.
      energy_scale: Energy scaling factor for the overlapping speaker.
        For 'collaborative' type this is 'ramp' indicating a linear
        ramp from 1.0 to 0.3.
    """

    def __init__(self, overlap_type, duration_s, energy_scale):
        self.type = overlap_type
        self.duration_s = duration_s
        self.energy_scale = energy_scale


class Turn:
    """A turn in the meeting timeline.

    Attributes:
      segment: The audio Segment for this turn.
      meeting_onset_sample: Position in the output timeline (samples).
      overlap: Optional OverlapSpec if this turn overlaps with another.
      is_burst: Whether this turn is part of an exchange burst.
    """

    def __init__(self, segment, meeting_onset_sample, overlap=None,
                 is_burst=False):
        self.segment = segment
        self.meeting_onset_sample = meeting_onset_sample
        self.overlap = overlap
        self.is_burst = is_burst


def get_speech_segments(wav_path, sample_rate=16000, min_dur=0.3):
    """Find speech segments in an audio file using Silero VAD.

    Returns boundaries aligned to silence gaps, suitable for cutting
    without splitting words.

    Args:
      wav_path: Path to audio file.
      sample_rate: Target sample rate for VAD processing.
      min_dur: Minimum segment duration in seconds.

    Returns:
      List of (start_sample, end_sample) tuples at the target
        sample_rate.
    """
    vad_model = _get_vad_model()

    wav = read_audio(wav_path)
    # read_audio returns 16kHz mono tensor
    timestamps = get_speech_timestamps(
        wav, vad_model, return_seconds=False, sampling_rate=16000)

    # Convert to target sample rate if different
    rate_ratio = sample_rate / 16000.0
    segments = []
    for ts in timestamps:
        start = int(ts['start'] * rate_ratio)
        end = int(ts['end'] * rate_ratio)
        duration = (end - start) / sample_rate
        if duration >= min_dur:
            segments.append((start, end))

    return segments


def select_segment(speaker, target_dur_s, sample_rate=16000, rng=None,
                   trim_silence=True, silence_buffer_s=0.08,
                   used_paths=None):
    """Select an audio segment from a speaker's recordings.

    For short targets, picks a single utterance and extracts a VAD-
    aligned region. For longer targets, concatenates multiple
    utterances with short pauses between them.

    Each utterance is trimmed to remove leading/trailing silence
    beyond a short buffer, based on VAD boundaries. Uses pre-computed
    VAD segments from speaker.vad_segments when available.

    Args:
      speaker: SpeakerInfo object with wav_paths and optional
        vad_segments cache.
      target_dur_s: Target segment duration in seconds.
      sample_rate: Sample rate for output segment boundaries.
      rng: numpy random Generator.
      trim_silence: Whether to trim leading/trailing silence using
        VAD boundaries.
      silence_buffer_s: Silence buffer to keep at utterance edges
        (seconds). Only used when trim_silence=True.
      used_paths: Optional set of wav_paths already used for this
        speaker in the meeting. Selected paths will be added to
        this set. If all paths are exhausted, falls back to reuse.

    Returns:
      Segment object, or None if no suitable segment found.
    """
    if rng is None:
        rng = np.random.default_rng()

    cached_vad = speaker.vad_segments if speaker.vad_segments else None

    # Shuffle utterance order, preferring unused paths
    wav_paths = list(speaker.wav_paths)
    rng.shuffle(wav_paths)
    if used_paths is not None:
        unused = [p for p in wav_paths if p not in used_paths]
        used = [p for p in wav_paths if p in used_paths]
        wav_paths = unused + used

    target_samples = int(target_dur_s * sample_rate)
    buffer_samples = int(silence_buffer_s * sample_rate)
    pause_samples = int(rng.uniform(0.05, 0.2) * sample_rate)

    # Process utterances lazily, stopping as soon as we have enough
    selected_chunks = []
    accumulated_samples = 0
    single_match = None

    for wav_path in wav_paths:
        chunk = _make_trimmed_chunk(
            wav_path, sample_rate, buffer_samples, trim_silence,
            cached_vad)
        if chunk is None:
            continue

        chunk_dur = chunk.duration_samples

        # Check for a single-utterance match (only if we haven't
        # already committed to concatenation)
        if (single_match is None and not selected_chunks
                and target_samples * 0.7 <= chunk_dur
                <= target_samples * 1.3):
            single_match = chunk

        # Accumulate chunks for concatenation
        if accumulated_samples < target_samples:
            selected_chunks.append(chunk)
            accumulated_samples += chunk_dur
            if len(selected_chunks) > 1:
                accumulated_samples += pause_samples

        # Stop once we have a single match or enough accumulated material
        if single_match is not None or accumulated_samples >= target_samples:
            break

    # Prefer single-utterance match if found
    if single_match is not None:
        if used_paths is not None:
            used_paths.add(single_match.wav_path)
        return Segment(speaker.speaker_id, single_match.wav_path,
                       single_match.start_sample, single_match.end_sample)

    if not selected_chunks:
        return None

    # If we only got one chunk, find a sub-segment if it's too long
    if len(selected_chunks) == 1:
        chunk = selected_chunks[0]
        if used_paths is not None:
            used_paths.add(chunk.wav_path)
        if chunk.duration_samples > target_samples * 1.3:
            if cached_vad is not None and chunk.wav_path in cached_vad:
                vad_segs = cached_vad[chunk.wav_path]
            else:
                vad_segs = get_speech_segments(
                    chunk.wav_path, sample_rate, min_dur=0.1)
            if vad_segs:
                best = _find_best_subsegment(
                    vad_segs, target_samples, sample_rate, rng)
                if best is not None:
                    return Segment(speaker.speaker_id, chunk.wav_path,
                                   best[0], best[1])
        return Segment(speaker.speaker_id, chunk.wav_path,
                       chunk.start_sample, chunk.end_sample)

    # Multi-chunk segment
    if used_paths is not None:
        for chunk in selected_chunks:
            used_paths.add(chunk.wav_path)
    return Segment(speaker.speaker_id,
                   chunks=selected_chunks, pause_samples=pause_samples)


def _make_trimmed_chunk(wav_path, sample_rate, buffer_samples,
                        trim_silence, cached_vad=None):
    """Create a SegmentChunk with silence-trimmed boundaries.

    Uses VAD to find speech boundaries, then adds a short buffer
    on each side. If cached_vad is provided, uses pre-computed
    segments instead of running VAD.

    Args:
      wav_path: Path to audio file.
      sample_rate: Target sample rate.
      buffer_samples: Silence buffer to keep at edges (samples).
      trim_silence: Whether to apply trimming.
      cached_vad: Optional dict mapping wav_path -> list of
        (start_sample, end_sample) tuples.

    Returns:
      SegmentChunk, or None if no speech found.
    """
    if not trim_silence:
        info = torchaudio.info(wav_path)
        n_samples = info.num_frames
        if info.sample_rate != sample_rate:
            n_samples = int(n_samples * sample_rate / info.sample_rate)
        return SegmentChunk(wav_path, 0, n_samples)

    if cached_vad is not None and wav_path in cached_vad:
        vad_segments = cached_vad[wav_path]
    else:
        vad_segments = get_speech_segments(wav_path, sample_rate, min_dur=0.1)
    if not vad_segments:
        return None

    # Trim to first/last speech boundary with buffer
    start = max(0, vad_segments[0][0] - buffer_samples)
    end = vad_segments[-1][1] + buffer_samples

    # Clamp end to file length
    info = torchaudio.info(wav_path)
    n_samples = info.num_frames
    if info.sample_rate != sample_rate:
        n_samples = int(n_samples * sample_rate / info.sample_rate)
    end = min(end, n_samples)

    if end <= start:
        return None

    return SegmentChunk(wav_path, start, end)


def _find_best_subsegment(vad_segments, target_samples, sample_rate, rng):
    """Find a contiguous range of VAD segments closest to target duration.

    Prefers cutting at VAD boundaries (silence gaps) to avoid mid-word
    cuts.

    Args:
      vad_segments: List of (start, end) tuples from VAD.
      target_samples: Target duration in samples.
      sample_rate: Sample rate.
      rng: numpy random Generator.

    Returns:
      (start_sample, end_sample) tuple, or None.
    """
    n = len(vad_segments)
    if n == 0:
        return None

    # Find all possible contiguous sub-ranges and their durations
    candidates = []
    for i in range(n):
        cumulative_end = vad_segments[i][0]
        for j in range(i, n):
            cumulative_end = vad_segments[j][1]
            duration = cumulative_end - vad_segments[i][0]
            if duration >= target_samples * 0.7:
                candidates.append((vad_segments[i][0], cumulative_end,
                                   abs(duration - target_samples)))
            if duration > target_samples * 1.3:
                break

    if not candidates:
        # Use the longest available range
        start = vad_segments[0][0]
        end = vad_segments[-1][1]
        return (start, end)

    # Sort by closeness to target, pick randomly from top candidates
    candidates.sort(key=lambda x: x[2])
    top_k = min(3, len(candidates))
    chosen = candidates[rng.integers(top_k)]
    return (chosen[0], chosen[1])


def generate_overlap_spec(overlap_type, rng=None):
    """Sample parameters for an overlap event.

    Args:
      overlap_type: One of 'backchannel', 'competitive',
        'collaborative', 'simultaneous'.
      rng: numpy random Generator.

    Returns:
      OverlapSpec with sampled duration and energy scale.
    """
    if rng is None:
        rng = np.random.default_rng()

    if overlap_type == 'backchannel':
        duration = rng.uniform(0.5, 1.5)
        energy_scale = rng.uniform(0.3, 0.6)
    elif overlap_type == 'competitive':
        duration = rng.uniform(1.0, 3.0)
        energy_scale = rng.uniform(0.8, 1.0)
    elif overlap_type == 'collaborative':
        duration = rng.uniform(0.3, 1.0)
        energy_scale = 'ramp'  # linear ramp 1.0 -> 0.3
    elif overlap_type == 'simultaneous':
        duration = rng.uniform(2.0, 5.0)
        energy_scale = 1.0
    else:
        raise ValueError("Unknown overlap type: %s" % overlap_type)

    return OverlapSpec(overlap_type, duration, energy_scale)


def build_turn_sequence(speakers, meeting_duration_s=300.0,
                        overlap_ratio=0.15, overlap_weights=None,
                        segment_durations=None, random_jump_rate=0.3,
                        exchange_bursts=None, subgroup_indices=None,
                        min_overlap_anchor_dur_s=1.0,
                        allow_multi_overlap=True, pause_range=None,
                        sample_rate=16000, rng=None):
    """Build a full meeting turn sequence with overlap patterns.

    Constructs a timeline of turns from the given speakers, with
    natural turn-taking and configurable overlap. Uses a single-pass
    approach where overlaps are placed inline as the timeline is
    built, preventing same-speaker self-overlaps by construction.

    Args:
      speakers: List of SpeakerInfo objects.
      meeting_duration_s: Target meeting duration in seconds.
      overlap_ratio: Target fraction of meeting time with overlapping
        speech (0.0 to 1.0).
      overlap_weights: Dict mapping overlap type names to relative
        weights. Defaults to equal weights for all 4 types.
      segment_durations: List of dicts with 'range' ([min_s, max_s])
        and 'weight' keys defining segment duration bins. If None,
        uses default short/medium/long bins.
      random_jump_rate: Probability of jumping to a random speaker
        instead of advancing sequentially in the round-robin (0.0
        to 1.0).
      exchange_bursts: Optional config dict for rapid-exchange
        episodes. Keys: 'n_bursts' (int), 'n_turns' ([min, max]),
        'segment_range' ([min_s, max_s]), 'pause_range' ([min_s,
        max_s]), 'target_speakers' ('similar-subgroup' or
        'random-pair').
      subgroup_indices: List of speaker indices that form the similar
        subgroup (used when target_speakers='similar-subgroup').
      min_overlap_anchor_dur_s: Minimum anchor turn duration (seconds)
        to be eligible for receiving overlaps. Shorter turns are
        never overlapped.
      allow_multi_overlap: Whether long anchors can receive multiple
        overlapping interjections. When True, the maximum number of
        overlaps per anchor is max(1, int(anchor_dur_s / 4.0)).
        When False, at most one overlap per anchor.
      pause_range: Inter-turn pause range [min_s, max_s]. Defaults
        to [0.2, 1.5].
      sample_rate: Sample rate for all audio.
      rng: numpy random Generator.

    Returns:
      List of Turn objects ordered by meeting_onset_sample.
    """
    if rng is None:
        rng = np.random.default_rng()

    if overlap_weights is None:
        overlap_weights = {
            'backchannel': 0.3,
            'competitive': 0.3,
            'collaborative': 0.2,
            'simultaneous': 0.2,
        }

    if segment_durations is None:
        segment_durations = [
            {'range': [1.0, 3.0], 'weight': 0.3},
            {'range': [5.0, 10.0], 'weight': 0.4},
            {'range': [12.0, 20.0], 'weight': 0.3},
        ]

    if pause_range is None:
        pause_range = [0.2, 1.5]

    meeting_duration_samples = int(meeting_duration_s * sample_rate)
    overlap_budget_s = overlap_ratio * meeting_duration_s

    # Track used utterances per speaker to avoid repetition
    used_paths = {spk.speaker_id: set() for spk in speakers}

    turns = _generate_timeline(
        speakers, meeting_duration_samples, overlap_budget_s,
        overlap_weights, segment_durations, random_jump_rate,
        exchange_bursts, subgroup_indices, min_overlap_anchor_dur_s,
        allow_multi_overlap, pause_range, sample_rate, rng, used_paths)

    # Sort by onset time
    turns.sort(key=lambda t: t.meeting_onset_sample)
    return turns


def _generate_timeline(speakers, meeting_duration_samples, overlap_budget_s,
                       overlap_weights, segment_durations, random_jump_rate,
                       exchange_bursts, subgroup_indices,
                       min_overlap_anchor_dur_s, allow_multi_overlap,
                       pause_range, sample_rate, rng, used_paths):
    """Generate the full meeting timeline in a single pass.

    Interleaves normal turns, exchange bursts, and overlap events.
    Overlaps are placed relative to the most recently emitted turn
    (the "anchor"), which prevents same-speaker self-overlaps by
    construction.

    Args:
      speakers: List of SpeakerInfo.
      meeting_duration_samples: Total meeting length in samples.
      overlap_budget_s: Total overlap time budget in seconds.
      overlap_weights: Dict of overlap type -> weight.
      segment_durations: List of dicts with 'range' and 'weight' keys.
      random_jump_rate: Probability of jumping to a random speaker
        instead of advancing sequentially.
      exchange_bursts: Config dict for burst episodes, or None.
      subgroup_indices: Speaker indices forming the similar subgroup,
        or None.
      min_overlap_anchor_dur_s: Minimum anchor duration to be eligible
        for overlaps.
      allow_multi_overlap: Whether long anchors can receive multiple
        overlaps.
      pause_range: Inter-turn pause range [min_s, max_s].
      sample_rate: Sample rate.
      rng: numpy random Generator.
      used_paths: Dict mapping speaker_id -> set of used wav_paths.

    Returns:
      List of Turn objects.
    """
    # Parse segment duration bins
    bins = [(b['range'][0], b['range'][1]) for b in segment_durations]
    weights = np.array([b['weight'] for b in segment_durations])
    seg_probs = weights / weights.sum()

    # Parse overlap type weights
    overlap_types = list(overlap_weights.keys())
    ov_weights = np.array([overlap_weights[t] for t in overlap_types])
    ov_probs = ov_weights / ov_weights.sum()

    # Plan burst trigger points (evenly spaced through the timeline)
    n_bursts = 0
    burst_pair = None
    burst_triggers = []
    if exchange_bursts is not None:
        n_bursts = exchange_bursts.get('n_bursts', 0)
    if n_bursts > 0:
        burst_pair = _resolve_burst_pair(
            speakers, exchange_bursts, subgroup_indices, rng)
        spacing = meeting_duration_samples // (n_bursts + 1)
        for i in range(n_bursts):
            centre = spacing * (i + 1)
            jitter = rng.integers(-spacing // 4, spacing // 4 + 1)
            trigger = max(0, centre + jitter)
            burst_triggers.append(trigger)
        burst_triggers.sort()

    turns = []
    current_pos = 0
    n_speakers = len(speakers)
    speaker_idx = 0
    next_burst = 0
    remaining_budget_s = overlap_budget_s

    while current_pos < meeting_duration_samples:
        # 1. Check whether we should enter burst mode
        if (next_burst < len(burst_triggers)
                and current_pos >= burst_triggers[next_burst]):
            burst_turns, current_pos = _generate_burst(
                burst_pair, exchange_bursts, current_pos,
                sample_rate, rng, used_paths)
            turns.extend(burst_turns)
            next_burst += 1
            continue

        # 2. Emit a normal turn (the "anchor")
        bin_idx = rng.choice(len(bins), p=seg_probs)
        target_dur = rng.uniform(bins[bin_idx][0], bins[bin_idx][1])

        if rng.random() >= random_jump_rate:
            speaker_idx = (speaker_idx + 1) % n_speakers
        else:
            other_indices = [i for i in range(n_speakers)
                            if i != speaker_idx]
            speaker_idx = rng.choice(other_indices)

        speaker = speakers[speaker_idx]
        segment = select_segment(speaker, target_dur, sample_rate, rng,
                                 used_paths=used_paths[speaker.speaker_id])
        if segment is None:
            continue

        anchor = Turn(segment, current_pos)
        turns.append(anchor)

        # 3. Advance past the anchor
        pause_s = rng.uniform(pause_range[0], pause_range[1])
        pause_samples = int(pause_s * sample_rate)
        current_pos += segment.duration_samples + pause_samples

        # 4. Possibly attach overlap(s) to this anchor
        anchor_dur_s = anchor.segment.duration_samples / sample_rate
        if (anchor_dur_s < min_overlap_anchor_dur_s
                or remaining_budget_s <= 0):
            continue

        anchor_speaker_id = anchor.segment.speaker_id
        other_speakers = [s for s in speakers
                          if s.speaker_id != anchor_speaker_id]
        if not other_speakers:
            continue

        if allow_multi_overlap:
            max_overlaps = max(1, int(anchor_dur_s / 4.0))
        else:
            max_overlaps = 1
        used_interjectors = set()
        for _ in range(max_overlaps):
            if remaining_budget_s <= 0:
                break
            insert_prob = remaining_budget_s / max(1.0, overlap_budget_s)
            if rng.random() >= insert_prob:
                break

            overlap_type = rng.choice(overlap_types, p=ov_probs)
            overlap_spec = generate_overlap_spec(overlap_type, rng)
            if overlap_spec.duration_s > remaining_budget_s:
                overlap_spec = OverlapSpec(
                    overlap_spec.type, remaining_budget_s,
                    overlap_spec.energy_scale)

            # Select an eligible speaker for the overlap
            eligible = [s for s in other_speakers
                        if s.speaker_id not in used_interjectors]
            if not eligible:
                break
            ov_speaker = rng.choice(eligible)

            if overlap_type in ('backchannel', 'simultaneous'):
                # Place interjection within the anchor's time span
                ov_segment = select_segment(
                    ov_speaker, overlap_spec.duration_s, sample_rate,
                    rng, used_paths=used_paths[ov_speaker.speaker_id])
                if ov_segment is None:
                    break

                margin = int(0.5 * sample_rate)
                anchor_dur_samples = anchor.segment.duration_samples
                onset_offset = rng.integers(
                    margin,
                    max(margin + 1, anchor_dur_samples - margin))
                onset = anchor.meeting_onset_sample + onset_offset
                turns.append(Turn(ov_segment, onset, overlap_spec))
                used_interjectors.add(ov_speaker.speaker_id)

                # Ensure current_pos is past the interjection's end
                interjection_end = onset + ov_segment.duration_samples
                if interjection_end > current_pos:
                    current_pos = interjection_end + int(
                        rng.uniform(0.2, 0.5) * sample_rate)

                remaining_budget_s -= overlap_spec.duration_s

            elif overlap_type in ('competitive', 'collaborative'):
                # Emit the next turn overlapping with anchor's tail
                ov_segment = select_segment(
                    ov_speaker, target_dur, sample_rate, rng,
                    used_paths=used_paths[ov_speaker.speaker_id])
                if ov_segment is None:
                    break

                overlap_samples = int(
                    overlap_spec.duration_s * sample_rate)
                anchor_end = (anchor.meeting_onset_sample
                              + anchor.segment.duration_samples)
                onset = anchor_end - overlap_samples
                if onset <= anchor.meeting_onset_sample:
                    break
                turns.append(Turn(ov_segment, onset, overlap_spec))

                # Advance current_pos past both anchor and overlap turn
                post_pause_s = rng.uniform(
                    pause_range[0], pause_range[1])
                overlap_end = onset + ov_segment.duration_samples
                current_pos = (max(anchor_end, overlap_end)
                               + int(post_pause_s * sample_rate))

                remaining_budget_s -= overlap_spec.duration_s
                # Competitive/collaborative transitions to new turn
                break

    return turns


def _resolve_burst_pair(speakers, burst_cfg, subgroup_indices, rng):
    """Determine the speaker pair for exchange bursts.

    Args:
      speakers: List of SpeakerInfo.
      burst_cfg: Exchange burst config dict.
      subgroup_indices: Similar-subgroup position indices, or None.
      rng: numpy random Generator.

    Returns:
      Tuple of two SpeakerInfo objects.
    """
    target_mode = burst_cfg.get('target_speakers', 'similar-subgroup')
    if (target_mode == 'similar-subgroup' and subgroup_indices is not None
            and len(subgroup_indices) >= 2):
        pair_indices = list(subgroup_indices[:2])
    else:
        pair_indices = rng.choice(
            len(speakers), size=2, replace=False).tolist()
    return (speakers[pair_indices[0]], speakers[pair_indices[1]])


def _generate_burst(burst_pair, burst_cfg, current_pos, sample_rate,
                    rng, used_paths):
    """Generate a single exchange burst episode.

    Args:
      burst_pair: Tuple of two SpeakerInfo objects.
      burst_cfg: Exchange burst config dict.
      current_pos: Current position in the timeline (samples).
      sample_rate: Sample rate.
      rng: numpy random Generator.
      used_paths: Dict mapping speaker_id -> set of used wav_paths.

    Returns:
      Tuple of (list of Turn objects, updated current_pos).
    """
    n_turns_range = burst_cfg.get('n_turns', [4, 8])
    seg_range = burst_cfg.get('segment_range', [0.5, 2.0])
    pause_range = burst_cfg.get('pause_range', [0.05, 0.3])

    n_burst_turns = rng.integers(n_turns_range[0], n_turns_range[1] + 1)

    turns = []
    for t in range(n_burst_turns):
        speaker = burst_pair[t % 2]
        target_dur = rng.uniform(seg_range[0], seg_range[1])

        segment = select_segment(
            speaker, target_dur, sample_rate, rng,
            used_paths=used_paths[speaker.speaker_id])
        if segment is None:
            continue

        turns.append(Turn(segment, current_pos, is_burst=True))
        pause_s = rng.uniform(pause_range[0], pause_range[1])
        current_pos += segment.duration_samples + int(
            pause_s * sample_rate)

    return turns, current_pos


