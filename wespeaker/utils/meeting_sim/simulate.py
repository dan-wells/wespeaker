"""CLI orchestrator for multi-party meeting simulation.

Provides fire-based subcommands for speaker enrolment, meeting
generation, and individual pipeline steps for debugging.

Usage:
  python -m wespeaker.utils.meeting_sim.simulate enroll ...
  python -m wespeaker.utils.meeting_sim.simulate generate ...
  python -m wespeaker.utils.meeting_sim.simulate select ...
"""

import copy
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import fire
import numpy as np
import yaml

from wespeaker.utils.cli import validate_fire_args
from wespeaker.utils.meeting_sim.speakers import (
    build_speaker_pool, cluster_speakers, compute_vad_segments,
    select_speakers, find_subgroup_positions,
    save_pool, load_pool,
)
from wespeaker.utils.meeting_sim.turns import build_turn_sequence
from wespeaker.utils.meeting_sim.room import (
    RoomConfig, ArrayConfig, create_room, compute_rirs,
)
from wespeaker.utils.meeting_sim.mixer import (
    MixConfig, mix_meeting, write_meeting_outputs,
)


logger = logging.getLogger('meeting_sim')


def _setup_logging(level='INFO'):
    """Configure logging for the simulation pipeline."""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def _load_config(config_path=None):
    """Load configuration from YAML file or use defaults.

    Args:
      config_path: Path to YAML config file. If None, loads the
        default config bundled with the package.

    Returns:
      Dict of configuration values.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), 'default_config.yaml')

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def _apply_jitter(cfg, vary_spec, rng, meeting_idx=0):
    """Apply uniform jitter to config parameters.

    Args:
      cfg: Nested config dict.
      vary_spec: Dict mapping dotted parameter paths to jitter
        specifications. For numeric parameters, the value is a
        half-width for uniform jitter: e.g. {'room.length': 0.3}
        means sample from U(default - 0.3, default + 0.3). For
        categorical parameters, the value is a list of options
        that will be cycled through round-robin across meetings:
        e.g. {'meeting.similarity_mode': ['random', 'all-similar']}.
      rng: numpy random Generator.
      meeting_idx: Index of the current meeting (used for
        round-robin cycling of list-valued vary specs).

    Returns:
      Modified cfg dict with jittered values.
    """
    jittered = copy.deepcopy(cfg)

    for path, spec in vary_spec.items():
        parts = path.split('.')
        # Navigate to the parent dict
        d = jittered
        valid = True
        for part in parts[:-1]:
            if part not in d:
                valid = False
                break
            d = d[part]
        if not valid or parts[-1] not in d:
            logger.warning("Jitter path '%s' not found in config, skipping.",
                           path)
            continue

        current = d[parts[-1]]
        if isinstance(spec, list):
            # Round-robin cycling through the list of options
            d[parts[-1]] = spec[meeting_idx % len(spec)]
        elif isinstance(current, (int, float)):
            new_val = current + rng.uniform(-spec, spec)
            # Keep same type
            if isinstance(current, int):
                new_val = int(round(new_val))
            d[parts[-1]] = new_val

    return jittered


def _generate_single_meeting(meeting_idx, pool, cfg, base_seed,
                             used_speaker_ids=None):
    """Generate a single meeting (called per-meeting in the batch loop).

    Args:
      meeting_idx: Integer index of this meeting.
      pool: SpeakerPool instance.
      cfg: Resolved config dict.
      base_seed: Base random seed.
      used_speaker_ids: Optional set of speaker IDs already used in
        this batch (for without-replacement selection in random mode).
        Selected speakers are added to this set in-place.

    Returns:
      Tuple of (meeting_idx, meeting_id, metadata_dict).
    """
    rng = np.random.default_rng(base_seed + meeting_idx)

    # Apply per-meeting jitter
    vary_spec = cfg.get('batch', {}).get('vary', {})
    meeting_cfg = _apply_jitter(cfg, vary_spec, rng, meeting_idx)

    meeting_id = 'meeting_%04d' % meeting_idx

    # Create room first (need positions for speaker selection)
    speakers_cfg = meeting_cfg['speakers']
    n_speakers = speakers_cfg['n_speakers']
    room_cfg = RoomConfig(
        length=meeting_cfg['room']['length'],
        width=meeting_cfg['room']['width'],
        height=meeting_cfg['room']['height'],
        rt60=meeting_cfg['room']['rt60'],
    )
    array_cfg = ArrayConfig(
        n_mics=meeting_cfg['array']['n_mics'],
        spacing=meeting_cfg['array']['spacing'],
    )
    speaker_position_jitter = meeting_cfg['room'].get('speaker_position_jitter', 0.0)
    sample_rate = meeting_cfg.get('resample_rate', 16000)

    # Use configured positions if provided, otherwise use defaults
    configured_positions = meeting_cfg['room'].get('speaker_positions')

    meeting_room = create_room(
        room_cfg, array_cfg, n_speakers,
        speaker_position_jitter=speaker_position_jitter,
        configured_positions=configured_positions,
        rng=rng, sample_rate=sample_rate,
    )

    # Select speakers (using room positions for subgroup placement)
    similarity_mode = speakers_cfg.get('similarity_mode', 'random')
    speakers = select_speakers(
        pool,
        n_speakers=n_speakers,
        similarity_mode=similarity_mode,
        similar_subgroup_size=speakers_cfg.get(
            'similar_subgroup_size', 2),
        fallback_strategy=speakers_cfg.get(
            'fallback_strategy', 'random'),
        similarity_thresh=speakers_cfg.get('similarity_thresh', 0.75),
        dissimilarity_thresh=speakers_cfg.get(
            'dissimilarity_thresh', 0.55),
        speaker_positions=meeting_room.speaker_positions,
        mic_center=meeting_room.mic_center,
        rng=rng,
        exclude_ids=used_speaker_ids,
    )

    # Track selected speakers for without-replacement batch selection
    # Only accumulate when mode is random (exclude_ids is ignored for
    # other modes, so accumulating their speakers would waste pool space).
    if used_speaker_ids is not None and similarity_mode == 'random':
        for spk in speakers:
            used_speaker_ids.add(spk.speaker_id)

    # Build speaker index map (order in which they're placed in room)
    speaker_index_map = {
        spk.speaker_id: i for i, spk in enumerate(speakers)}

    # Compute subgroup indices for exchange bursts
    subgroup_size = speakers_cfg.get('similar_subgroup_size', 2)
    use_distant = (similarity_mode == 'similar-distant-subgroup')
    if similarity_mode in ('similar-close-subgroup',
                           'similar-distant-subgroup'):
        subgroup_indices = find_subgroup_positions(
            meeting_room.speaker_positions, subgroup_size,
            use_distant, meeting_room.mic_center)
    else:
        subgroup_indices = None

    # Build turn sequence
    meeting_section = meeting_cfg['meeting']
    turns = build_turn_sequence(
        speakers,
        meeting_duration_s=meeting_section['duration_s'],
        overlap_ratio=meeting_section['overlap_ratio'],
        overlap_weights=meeting_section.get('overlap_weights'),
        segment_durations=meeting_section.get('segment_durations'),
        random_jump_rate=meeting_section.get('random_jump_rate', 0.3),
        exchange_bursts=meeting_section.get('exchange_bursts'),
        subgroup_indices=subgroup_indices,
        min_overlap_anchor_dur_s=meeting_section.get(
            'min_overlap_anchor_dur_s', 1.0),
        allow_multi_overlap=meeting_section.get(
            'allow_multi_overlap', True),
        pause_range=meeting_section.get('pause_range'),
        sample_rate=sample_rate,
        rng=rng,
    )

    # Compute RIRs
    rirs = compute_rirs(meeting_room, sample_rate)

    # Mix meeting
    output_cfg = meeting_cfg.get('output', {})
    noise_cfg = meeting_cfg.get('noise', {})
    mix_cfg = MixConfig(
        sample_rate=sample_rate,
        output_channels=output_cfg.get('channels', ['mono']),
        noise_wav=(noise_cfg.get('wav_path')
                   if noise_cfg.get('enabled') else None),
        noise_snr_db=noise_cfg.get('snr_db', 30.0),
        save_reverberant=output_cfg.get('save_reverberant', False),
        save_dry=output_cfg.get('save_dry', False),
    )

    result = mix_meeting(
        turns, meeting_room, rirs, mix_cfg, speaker_index_map, rng)

    # Compute pairwise similarities between selected speakers
    id_to_idx = {sid: i for i, sid in enumerate(pool.speaker_ids)}
    pairwise_similarities = {}
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            idx_i = id_to_idx[speakers[i].speaker_id]
            idx_j = id_to_idx[speakers[j].speaker_id]
            sim = float(pool.similarity_matrix[idx_i, idx_j])
            pairwise_similarities['%s-%s' % (
                speakers[i].speaker_id, speakers[j].speaker_id)] = sim

    # Compute actual overlap and speech ratios from the mixed output
    rttm = result['rttm']
    duration_s = result['n_samples'] / sample_rate
    actual_overlap_s = _compute_overlap_duration(rttm)
    actual_overlap_ratio = actual_overlap_s / duration_s if duration_s > 0 else 0
    speech_s = _compute_speech_duration(rttm)
    silence_ratio = 1.0 - (speech_s / duration_s) if duration_s > 0 else 0

    # Compute exchange burst statistics
    burst_stats = _compute_burst_stats(turns, sample_rate)

    # Build metadata
    metadata = {
        'meeting_id': meeting_id,
        'n_speakers': n_speakers,
        'speakers': [
            {
                'id': spk.speaker_id,
                'cluster': pool.speaker_to_cluster.get(spk.speaker_id),
                'position': meeting_room.speaker_positions[i].tolist(),
            }
            for i, spk in enumerate(speakers)
        ],
        'similarity_mode': similarity_mode,
        'pairwise_similarities': pairwise_similarities,
        'room': {
            'length': room_cfg.length,
            'width': room_cfg.width,
            'height': room_cfg.height,
            'rt60': room_cfg.rt60,
        },
        'mic_center': meeting_room.mic_center.tolist(),
        'n_turns': len(turns),
        'target_overlap_ratio': meeting_section['overlap_ratio'],
        'actual_overlap_ratio': round(actual_overlap_ratio, 4),
        'silence_ratio': round(silence_ratio, 4),
        'bursts': burst_stats,
        'duration_s': duration_s,
    }

    return meeting_idx, meeting_id, result, metadata, mix_cfg


class MeetingSimulator:
    """Multi-party meeting simulation pipeline.

    Subcommands:
      enroll   -- Build speaker pool from audio files.
      generate -- Batch-generate simulated meetings.
      select   -- Debug: select and print a speaker group.
      turns    -- Debug: build and print a turn sequence.
      room     -- Debug: create a room and print RIR stats.
    """

    def enroll(self, wav_scp, utt2spk, output_dir, embedding_scp=None,
              model_dir=None, device='cpu', config=None,
              similarity_thresh=None, max_enroll_utts=None,
              precompute_vad=False, n_workers=None, seed=None):
        """Build speaker pool from audio files.

        Loads or extracts embeddings, computes pairwise similarities,
        clusters speakers, and optionally pre-computes VAD segments.
        Provide embedding_scp for pre-extracted embeddings, or model_dir
        to extract fresh embeddings.

        Args:
          wav_scp: Path to wav.scp file.
          utt2spk: Path to utt2spk file.
          output_dir: Directory to save the speaker pool.
          embedding_scp: Path to pre-extracted embedding .scp file.
          model_dir: Path to pretrained wespeaker model directory.
          device: Device for embedding extraction.
          config: Path to YAML config file (uses default if not given).
          similarity_thresh: Threshold for clustering similar speakers.
          max_enroll_utts: Maximum utterances per speaker for mean embedding.
          precompute_vad: Run VAD on all utterances and cache the segments in the pool.
          n_workers: Number of parallel workers for VAD computation.
          seed: Random seed. Overrides config batch.seed.
        """
        _setup_logging()

        cfg = _load_config(config)
        speakers_cfg = cfg.get('speakers', {})
        batch_cfg = cfg.get('batch', {})

        if similarity_thresh is None:
            similarity_thresh = speakers_cfg.get('similarity_thresh', 0.75)
        if max_enroll_utts is None:
            max_enroll_utts = speakers_cfg.get('max_enroll_utts')
        if n_workers is None:
            n_workers = batch_cfg.get('n_workers', 1)
        if seed is None:
            seed = batch_cfg.get('seed', 42)

        logger.info("Building speaker pool from %s", wav_scp)
        pool = build_speaker_pool(
            wav_scp, utt2spk,
            embedding_scp=embedding_scp,
            model_dir=model_dir,
            device=device,
            max_enroll_utts=max_enroll_utts,
            seed=seed,
        )

        logger.info("Clustering speakers (similarity_thresh=%.2f)",
                    similarity_thresh)
        pool = cluster_speakers(pool, similarity_thresh)

        if precompute_vad:
            sample_rate = cfg.get('resample_rate', 16000)
            pool = compute_vad_segments(
                pool, sample_rate=sample_rate, n_workers=n_workers)

        save_pool(pool, output_dir)
        logger.info("Done. Pool saved to %s", output_dir)

    def generate(self, pool_dir, output_dir, n_meetings=None, config=None,
                 n_workers=1, seed=None):
        """Batch-generate N simulated meetings.

        Loads a speaker pool and config, then generates meetings with
        parameter variation.

        Args:
          pool_dir: Directory containing the speaker pool (from enroll).
          output_dir: Directory to write meeting outputs.
          n_meetings: Number of meetings to generate.
          config: Path to YAML config file (uses default if not given).
          n_workers: Number of parallel workers.
          seed: Random seed.
        """
        _setup_logging()

        cfg = _load_config(config)

        if n_meetings is not None:
            cfg.setdefault('batch', {})['n_meetings'] = n_meetings
        if seed is not None:
            cfg.setdefault('batch', {})['seed'] = seed

        batch_cfg = cfg.get('batch', {})
        n_meetings = batch_cfg.get('n_meetings', 10)
        base_seed = batch_cfg.get('seed', 42)

        logger.info("Loading speaker pool from %s", pool_dir)
        pool = load_pool(pool_dir)

        logger.info("Generating %d meetings (seed=%d, workers=%d)",
                    n_meetings, base_seed, n_workers)

        os.makedirs(output_dir, exist_ok=True)

        # Save resolved config
        config_out_path = os.path.join(output_dir, 'config.yaml')
        with open(config_out_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

        all_metadata = []

        # Without-replacement speaker selection (random mode only)
        speakers_cfg = cfg.get('speakers', {})
        unique_across = speakers_cfg.get('unique_across_meetings', False)
        use_unique = (
            unique_across
            and speakers_cfg.get('similarity_mode', 'random') == 'random'
        )

        if n_workers <= 1:
            # Sequential generation
            used_speaker_ids = set() if use_unique else None
            n_spk = speakers_cfg.get('n_speakers', 3)
            for i in range(n_meetings):
                # Reset used set when pool is exhausted
                if (used_speaker_ids is not None
                        and len(used_speaker_ids)
                        > len(pool.speaker_ids) - n_spk):
                    logger.info(
                        "Speaker pool exhausted (%d/%d used), resetting.",
                        len(used_speaker_ids), len(pool.speaker_ids))
                    used_speaker_ids.clear()
                idx, mid, result, metadata, mix_cfg = (
                    _generate_single_meeting(
                        i, pool, cfg, base_seed, used_speaker_ids))
                meeting_dir = os.path.join(output_dir, 'meetings', mid)
                write_meeting_outputs(result, meeting_dir, mid, mix_cfg)
                _write_metadata(metadata, meeting_dir)
                all_metadata.append(metadata)
                logger.info("Generated %s (%.1f s)",
                            mid, metadata['duration_s'])
        else:
            # Parallel generation (unique_across_meetings not supported)
            if use_unique:
                logger.warning(
                    "unique_across_meetings requires n_workers=1; "
                    "ignoring with %d workers.", n_workers)
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {}
                for i in range(n_meetings):
                    future = executor.submit(
                        _generate_single_meeting, i, pool, cfg, base_seed)
                    futures[future] = i

                for future in as_completed(futures):
                    idx, mid, result, metadata, mix_cfg = future.result()
                    meeting_dir = os.path.join(output_dir, 'meetings', mid)
                    write_meeting_outputs(result, meeting_dir, mid, mix_cfg)
                    _write_metadata(metadata, meeting_dir)
                    all_metadata.append(metadata)
                    logger.info("Generated %s (%.1f s)",
                                mid, metadata['duration_s'])

        _log_batch_summary(all_metadata)
        logger.info("Done. %d meetings written to %s",
                    n_meetings, output_dir)

    def select(self, pool_dir, n_speakers=3,
               similarity_mode='similar-close-subgroup',
               similar_subgroup_size=2, fallback_strategy='random',
               similarity_thresh=0.75, dissimilarity_thresh=0.55,
               config=None, seed=42):
        """Debug: select a speaker group and print details.

        Args:
          pool_dir: Directory containing the speaker pool.
          n_speakers: Number of speakers to select.
          similarity_mode: Selection mode (random, all-similar,
            all-dissimilar, similar-close-subgroup,
            similar-distant-subgroup).
          similar_subgroup_size: Size of similar subgroup.
          fallback_strategy: 'random' or 'dissimilar' for remainder.
          similarity_thresh: Threshold for similar grouping.
          dissimilarity_thresh: Threshold for dissimilar selection.
          config: Path to YAML config (for speaker positions).
          seed: Random seed.
        """
        _setup_logging()

        cfg = _load_config(config)
        speaker_positions = cfg['room'].get('speaker_positions')
        if speaker_positions is not None:
            speaker_positions = [np.array(p) for p in speaker_positions]

        # Compute mic center from config (same logic as create_room)
        room_width = cfg['room'].get('width', 4.0)
        mic_center = np.array([0.1, room_width / 2.0, 1.5])

        pool = load_pool(pool_dir)
        rng = np.random.default_rng(seed)
        speakers = select_speakers(
            pool, n_speakers, similarity_mode,
            similar_subgroup_size=similar_subgroup_size,
            fallback_strategy=fallback_strategy,
            similarity_thresh=similarity_thresh,
            dissimilarity_thresh=dissimilarity_thresh,
            speaker_positions=speaker_positions,
            mic_center=mic_center,
            rng=rng)

        print("Selected speakers:")
        for i, spk in enumerate(speakers):
            cluster = pool.speaker_to_cluster.get(spk.speaker_id, '?')
            print("  [%d] %s (cluster %s, %d utterances)" % (
                i, spk.speaker_id, cluster, len(spk.wav_paths)))

        # Print pairwise similarities
        print("\nPairwise similarities:")
        id_to_idx = {sid: i for i, sid in enumerate(pool.speaker_ids)}
        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                idx_i = id_to_idx[speakers[i].speaker_id]
                idx_j = id_to_idx[speakers[j].speaker_id]
                sim = pool.similarity_matrix[idx_i, idx_j]
                print("  (%d, %d) %s - %s: %.4f" % (
                    i, j, speakers[i].speaker_id,
                    speakers[j].speaker_id, sim))

    def turns(self, pool_dir, n_speakers=3,
              similarity_mode='random',
              duration=300.0, overlap_ratio=0.15, config=None, seed=42):
        """Debug: build and print a turn sequence.

        Args:
          pool_dir: Directory containing the speaker pool.
          n_speakers: Number of speakers.
          similarity_mode: Speaker selection mode.
          duration: Meeting duration in seconds.
          overlap_ratio: Target overlap ratio.
          config: Path to YAML config (for segment_durations,
            exchange_bursts, random_jump_rate).
          seed: Random seed.
        """
        _setup_logging()

        cfg = _load_config(config)
        meeting_cfg = cfg.get('meeting', {})
        sample_rate = cfg.get('resample_rate', 16000)

        pool = load_pool(pool_dir)
        rng = np.random.default_rng(seed)
        speakers = select_speakers(
            pool, n_speakers, similarity_mode, rng=rng)

        turn_list = build_turn_sequence(
            speakers, duration, overlap_ratio,
            overlap_weights=meeting_cfg.get('overlap_weights'),
            segment_durations=meeting_cfg.get('segment_durations'),
            random_jump_rate=meeting_cfg.get('random_jump_rate', 0.3),
            exchange_bursts=meeting_cfg.get('exchange_bursts'),
            min_overlap_anchor_dur_s=meeting_cfg.get(
                'min_overlap_anchor_dur_s', 1.0),
            allow_multi_overlap=meeting_cfg.get(
                'allow_multi_overlap', True),
            pause_range=meeting_cfg.get('pause_range'),
            sample_rate=sample_rate,
            rng=rng)

        print("Turn sequence (%d turns, %.1f s target):" % (
            len(turn_list), duration))
        for i, turn in enumerate(turn_list):
            onset_s = turn.meeting_onset_sample / sample_rate
            dur_s = turn.segment.duration_samples / sample_rate
            overlap_str = ""
            if turn.overlap:
                overlap_str = " [%s, %.2fs]" % (
                    turn.overlap.type, turn.overlap.duration_s)
            print("  %3d  %.2f - %.2f s  %s%s" % (
                i, onset_s, onset_s + dur_s,
                turn.segment.speaker_id, overlap_str))

    def room(self, config=None, seed=42):
        """Debug: create a room and print RIR statistics.

        Args:
          config: Path to YAML config file.
          seed: Random seed.
        """
        _setup_logging()

        cfg = _load_config(config)
        rng = np.random.default_rng(seed)

        room_cfg = RoomConfig(
            length=cfg['room']['length'],
            width=cfg['room']['width'],
            height=cfg['room']['height'],
            rt60=cfg['room']['rt60'],
        )
        array_cfg = ArrayConfig(
            n_mics=cfg['array']['n_mics'],
            spacing=cfg['array']['spacing'],
        )

        configured_positions = cfg['room'].get('speaker_positions')
        meeting_room = create_room(
            room_cfg, array_cfg,
            configured_positions=configured_positions, rng=rng)
        rirs = compute_rirs(meeting_room)

        print("Room: %.1f x %.1f x %.1f m, RT60=%.2f s" % (
            room_cfg.length, room_cfg.width, room_cfg.height, room_cfg.rt60))
        print("Mic array: %d elements, %.3f m spacing" % (
            array_cfg.n_mics, array_cfg.spacing))
        print("RIR shape: %s (%.3f s max)" % (
            rirs.shape, rirs.shape[2] / 16000.0))

        for i, pos in enumerate(meeting_room.speaker_positions):
            print("  Speaker %d: (%.2f, %.2f, %.2f)" % (
                i, pos[0], pos[1], pos[2]))


def _compute_overlap_duration(rttm):
    """Compute total duration of overlapping speech from RTTM entries.

    Args:
      rttm: List of (speaker_id, start_s, duration_s) tuples.

    Returns:
      Total overlap duration in seconds.
    """
    if not rttm:
        return 0.0

    # Build list of (start, end) intervals
    intervals = [(start, start + dur) for _, start, dur in rttm]
    intervals.sort()

    # Sweep line: count time where >= 2 speakers are active
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    events.sort()

    overlap_s = 0.0
    active = 0
    prev_time = 0.0
    for time, delta in events:
        if active >= 2:
            overlap_s += time - prev_time
        active += delta
        prev_time = time

    return overlap_s


def _compute_burst_stats(turns, sample_rate):
    """Compute exchange burst statistics from a turn list.

    Identifies contiguous runs of burst-flagged turns and computes
    per-burst turn count and duration.

    Args:
      turns: List of Turn objects (sorted by onset).
      sample_rate: Sample rate for time conversion.

    Returns:
      Dict with 'n_bursts', 'turns_per_burst' (list), and
        'duration_per_burst_s' (list).
    """
    bursts = []
    current_burst = []
    for turn in turns:
        if turn.is_burst:
            current_burst.append(turn)
        else:
            if current_burst:
                bursts.append(current_burst)
                current_burst = []
    if current_burst:
        bursts.append(current_burst)

    turns_per_burst = [len(b) for b in bursts]
    duration_per_burst_s = []
    for burst in bursts:
        onset = burst[0].meeting_onset_sample
        last = burst[-1]
        end = last.meeting_onset_sample + last.segment.duration_samples
        duration_per_burst_s.append((end - onset) / sample_rate)

    return {
        'n_bursts': len(bursts),
        'turns_per_burst': turns_per_burst,
        'duration_per_burst_s': duration_per_burst_s,
    }


def _compute_speech_duration(rttm):
    """Compute total duration of speech (union of all intervals).

    Args:
      rttm: List of (speaker_id, start_s, duration_s) tuples.

    Returns:
      Total speech duration in seconds (overlapping regions counted
        once).
    """
    if not rttm:
        return 0.0

    intervals = sorted((start, start + dur) for _, start, dur in rttm)

    # Merge overlapping intervals
    merged_start, merged_end = intervals[0]
    total = 0.0
    for start, end in intervals[1:]:
        if start <= merged_end:
            merged_end = max(merged_end, end)
        else:
            total += merged_end - merged_start
            merged_start, merged_end = start, end
    total += merged_end - merged_start

    return total


def _log_batch_summary(all_metadata):
    """Log summary statistics for a completed batch of meetings.

    Args:
      all_metadata: List of metadata dicts, one per meeting.
    """
    n = len(all_metadata)
    if n == 0:
        return

    durations = [m['duration_s'] for m in all_metadata]
    target_overlaps = [m['target_overlap_ratio'] for m in all_metadata]
    actual_overlaps = [m['actual_overlap_ratio'] for m in all_metadata]
    silence_ratios = [m['silence_ratio'] for m in all_metadata]
    n_turns = [m['n_turns'] for m in all_metadata]

    logger.info("=" * 60)
    logger.info("Batch summary (%d meetings)", n)
    logger.info("-" * 60)

    # Duration spread
    logger.info(
        "Duration (s):   min=%.1f  max=%.1f  mean=%.1f  std=%.1f",
        np.min(durations), np.max(durations),
        np.mean(durations), np.std(durations))

    # Turns per meeting
    logger.info(
        "Turns/meeting:  min=%d  max=%d  mean=%.1f",
        np.min(n_turns), np.max(n_turns), np.mean(n_turns))

    # Overlap ratio: target vs realised
    logger.info(
        "Overlap ratio:  target=%.3f  realised: min=%.3f  max=%.3f  "
        "mean=%.3f",
        np.mean(target_overlaps), np.min(actual_overlaps),
        np.max(actual_overlaps), np.mean(actual_overlaps))

    # Silence ratio
    logger.info(
        "Silence ratio:  min=%.3f  max=%.3f  mean=%.3f",
        np.min(silence_ratios), np.max(silence_ratios),
        np.mean(silence_ratios))

    # Exchange burst statistics
    all_burst_turns = []
    all_burst_durations = []
    n_bursts_per_meeting = []
    for m in all_metadata:
        burst_info = m.get('bursts', {})
        n_bursts_per_meeting.append(burst_info.get('n_bursts', 0))
        all_burst_turns.extend(burst_info.get('turns_per_burst', []))
        all_burst_durations.extend(
            burst_info.get('duration_per_burst_s', []))
    total_bursts = sum(n_bursts_per_meeting)
    if total_bursts > 0:
        logger.info(
            "Bursts:         %d total (%.1f/meeting)  "
            "turns/burst: mean=%.1f  dur/burst: mean=%.1f s",
            total_bursts, np.mean(n_bursts_per_meeting),
            np.mean(all_burst_turns), np.mean(all_burst_durations))
    else:
        logger.info("Bursts:         none")

    # Similarity mode distribution
    mode_counts = {}
    for m in all_metadata:
        mode = m['similarity_mode']
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    mode_str = "  ".join(
        "%s=%d" % (mode, count)
        for mode, count in sorted(mode_counts.items()))
    logger.info("Similarity modes:  %s", mode_str)

    # Unique speakers used across the batch
    all_speaker_ids = set()
    for m in all_metadata:
        for spk in m['speakers']:
            all_speaker_ids.add(spk['id'])
    logger.info("Unique speakers used:  %d", len(all_speaker_ids))

    logger.info("=" * 60)


def _write_metadata(metadata, output_dir):
    """Write meeting metadata to JSON file.

    Args:
      metadata: Dict of metadata.
      output_dir: Directory to write into.
    """
    path = os.path.join(output_dir, 'metadata.json')
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    validate_fire_args(MeetingSimulator)
    fire.Fire(MeetingSimulator)


if __name__ == '__main__':
    main()
