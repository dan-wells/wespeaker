"""Speaker pool management for meeting simulation.

Handles embedding extraction/loading, pairwise similarity computation,
agglomerative clustering of speakers by vocal similarity, and
constraint-based speaker group selection.
"""

import logging
import os
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

import kaldiio
import numpy as np
import torchaudio
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tqdm import tqdm

from wespeaker.utils.file_utils import read_scp
from wespeaker.utils.meeting_sim.turns import get_speech_segments


logger = logging.getLogger('meeting_sim.speakers')


class SpeakerInfo:
    """Information about a single speaker in the pool.

    Attributes:
      speaker_id: Unique identifier for the speaker.
      wav_paths: List of paths to the speaker's audio files.
      embedding: Mean embedding vector (np.ndarray), or None if not yet
        computed.
      vad_segments: Dict mapping wav_path -> list of (start_sample,
        end_sample) tuples from VAD. None if not yet computed.
      rms: Mean RMS energy of the speaker's utterances (float), or
        None if not yet computed.
    """

    def __init__(self, speaker_id, wav_paths, embedding=None,
                 vad_segments=None, rms=None):
        self.speaker_id = speaker_id
        self.wav_paths = wav_paths
        self.embedding = embedding
        self.vad_segments = vad_segments if vad_segments is not None else {}
        self.rms = rms


class SpeakerPool:
    """Pool of speakers with similarity and clustering information.

    Attributes:
      speakers: Dict mapping speaker_id -> SpeakerInfo.
      speaker_ids: Ordered list of speaker IDs (defines matrix indexing).
      similarity_matrix: Symmetric matrix of pairwise cosine similarities,
        shape (n_speakers, n_speakers), values in [0, 1].
      clusters: Dict mapping cluster_id (int) -> list of speaker_ids.
      speaker_to_cluster: Dict mapping speaker_id -> cluster_id.
    """

    def __init__(self, speakers, speaker_ids, similarity_matrix=None,
                 clusters=None, speaker_to_cluster=None):
        self.speakers = speakers
        self.speaker_ids = speaker_ids
        self.similarity_matrix = similarity_matrix
        self.clusters = clusters if clusters is not None else {}
        self.speaker_to_cluster = (speaker_to_cluster
                                   if speaker_to_cluster is not None else {})


def build_speaker_pool(wav_scp, utt2spk, embedding_scp=None, model_dir=None,
                       device='cpu', max_enroll_utts=None, seed=42):
    """Build a speaker pool from wav.scp and utt2spk files.

    Reads audio paths and speaker assignments, then either loads
    pre-extracted embeddings from an ark/scp file or extracts them using
    a wespeaker model. Computes mean embedding per speaker.

    Args:
      wav_scp: Path to wav.scp file (format: utt_id /path/to/wav).
      utt2spk: Path to utt2spk file (format: utt_id spk_id).
      embedding_scp: Optional path to embedding .scp file with
        pre-extracted embeddings in ark/scp format. If provided,
        model_dir is not needed.
      model_dir: Path to pretrained wespeaker model directory. Required
        if embedding_scp is not provided.
      device: Device for embedding extraction ('cpu' or 'cuda:N').
      max_enroll_utts: Maximum number of utterances to use per speaker
        for mean embedding computation. If None, all utterances are
        used. When set, a random subset of this size is sampled per
        speaker (deterministically using seed).
      seed: Random seed for utterance subsampling.

    Returns:
      SpeakerPool with embeddings and similarity matrix computed.

    Raises:
      ValueError: If neither embedding_scp nor model_dir is provided.
    """
    if embedding_scp is None and model_dir is None:
        raise ValueError(
            "Must provide either embedding_scp or model_dir.")

    # Read wav.scp and utt2spk
    wav_entries = dict(read_scp(wav_scp))
    utt2spk_entries = dict(read_scp(utt2spk))

    # Group utterances by speaker
    spk_to_utts = {}
    for utt_id, spk_id in utt2spk_entries.items():
        if utt_id not in wav_entries:
            continue
        if spk_id not in spk_to_utts:
            spk_to_utts[spk_id] = []
        spk_to_utts[spk_id].append(utt_id)

    # Subsample utterances per speaker before loading embeddings
    rng = np.random.default_rng(seed)
    speaker_ids = sorted(spk_to_utts.keys())
    spk_enrol_utts = {}
    for spk_id in speaker_ids:
        utts = spk_to_utts[spk_id]
        if max_enroll_utts is not None and len(utts) > max_enroll_utts:
            indices = rng.choice(
                len(utts), size=max_enroll_utts, replace=False)
            spk_enrol_utts[spk_id] = [utts[i] for i in sorted(indices)]
        else:
            spk_enrol_utts[spk_id] = utts

    # Load or extract embeddings (only those needed for enrolment)
    needed_utts = set(
        u for utts in spk_enrol_utts.values() for u in utts)
    if embedding_scp is not None:
        utt_embeddings = _load_embeddings_from_scp(
            embedding_scp, needed_utts)
    else:
        needed_wav_entries = {
            u: wav_entries[u] for u in needed_utts if u in wav_entries}
        utt_embeddings = _extract_embeddings(
            needed_wav_entries, model_dir, device)

    # Build SpeakerInfo objects with mean embeddings
    speakers = {}
    for spk_id in speaker_ids:
        utts = spk_to_utts[spk_id]
        wav_paths = [wav_entries[u] for u in utts]

        # Compute mean embedding from enrolment utterances
        emb_list = []
        for utt_id in spk_enrol_utts[spk_id]:
            if utt_id in utt_embeddings:
                emb_list.append(utt_embeddings[utt_id])
        if len(emb_list) == 0:
            logger.warning(
                "No embeddings found for speaker %s, skipping.", spk_id)
            continue

        mean_embedding = np.mean(np.stack(emb_list), axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
        speakers[spk_id] = SpeakerInfo(spk_id, wav_paths, mean_embedding)

    # Rebuild speaker_ids to only include speakers with embeddings
    speaker_ids = [s for s in speaker_ids if s in speakers]

    # Compute pairwise similarity matrix
    similarity_matrix = _compute_similarity_matrix(speakers, speaker_ids)

    pool = SpeakerPool(speakers, speaker_ids, similarity_matrix)
    return pool


def _load_embeddings_from_scp(embedding_scp, needed_utts=None):
    """Load pre-extracted embeddings from a Kaldi ark/scp file.

    Uses random access to load only the requested utterances when
    needed_utts is provided, avoiding a full sequential scan.

    Args:
      embedding_scp: Path to the .scp file.
      needed_utts: Optional set of utterance IDs to load. If None,
        all embeddings are loaded sequentially.

    Returns:
      Dict mapping utt_id -> np.ndarray.
    """
    if needed_utts is not None:
        # Random access: only read the embeddings we need
        scp_dict = kaldiio.load_scp(embedding_scp)
        utt_embeddings = {}
        for utt_id in tqdm(needed_utts, desc="Loading embeddings"):
            if utt_id in scp_dict:
                utt_embeddings[utt_id] = scp_dict[utt_id].astype(
                    np.float32)
        return utt_embeddings
    else:
        # Sequential fallback (loads everything)
        utt_embeddings = {}
        for utt_id, embedding in tqdm(
                kaldiio.load_scp_sequential(embedding_scp),
                desc="Loading embeddings"):
            utt_embeddings[utt_id] = embedding.astype(np.float32)
        return utt_embeddings


def _extract_embeddings(wav_entries, model_dir, device):
    """Extract embeddings for all utterances using a wespeaker model.

    Args:
      wav_entries: Dict mapping utt_id -> wav_path.
      model_dir: Path to pretrained model directory.
      device: Device string.

    Returns:
      Dict mapping utt_id -> np.ndarray.
    """
    # Deferred import: Speaker pulls in the full model infrastructure,
    # only needed when extracting embeddings (not when using pre-extracted).
    from wespeaker.cli.speaker import Speaker

    speaker_model = Speaker(model_dir)
    speaker_model.set_device(device)

    utt_embeddings = {}
    for utt_id, wav_path in tqdm(wav_entries.items(),
                                  desc="Extracting embeddings",
                                  total=len(wav_entries)):
        embedding = speaker_model.extract_embedding(wav_path)
        if embedding is not None:
            utt_embeddings[utt_id] = embedding.detach().numpy()
    return utt_embeddings


def _compute_similarity_matrix(speakers, speaker_ids):
    """Compute pairwise cosine similarity matrix.

    Similarities are normalized to [0, 1] range following the wespeaker
    convention: (cosine + 1) / 2.

    Args:
      speakers: Dict mapping speaker_id -> SpeakerInfo (with embeddings).
      speaker_ids: Ordered list of speaker IDs.

    Returns:
      np.ndarray of shape (n_speakers, n_speakers).
    """
    n = len(speaker_ids)
    logger.info("Computing pairwise similarities (%d speakers)", n)
    matrix = np.ones((n, n), dtype=np.float32)
    for i in range(n):
        emb_i = speakers[speaker_ids[i]].embedding
        for j in range(i + 1, n):
            emb_j = speakers[speaker_ids[j]].embedding
            cosine = np.dot(emb_i, emb_j)
            similarity = (cosine + 1.0) / 2.0
            matrix[i, j] = similarity
            matrix[j, i] = similarity
    return matrix


def cluster_speakers(pool, similarity_thresh=0.75):
    """Cluster speakers by vocal similarity using agglomerative clustering.

    Uses cosine distance (1 - similarity) with average linkage. Speakers
    within the same cluster have average pairwise similarity above
    similarity_thresh.

    Args:
      pool: SpeakerPool with similarity_matrix computed.
      similarity_thresh: Minimum similarity for speakers to be in the
        same cluster. Higher values produce smaller, tighter clusters.

    Returns:
      The same SpeakerPool with clusters and speaker_to_cluster
        populated.
    """
    n = len(pool.speaker_ids)
    if n < 2:
        pool.clusters = {0: list(pool.speaker_ids)}
        pool.speaker_to_cluster = {sid: 0 for sid in pool.speaker_ids}
        return pool

    # Convert similarity matrix to condensed distance matrix
    distance_matrix = 1.0 - pool.similarity_matrix
    np.fill_diagonal(distance_matrix, 0.0)
    condensed = squareform(distance_matrix, checks=False)

    # Agglomerative clustering with average linkage
    linkage_matrix = linkage(condensed, method='average')

    # Cut at distance threshold (1 - similarity_thresh)
    distance_thresh = 1.0 - similarity_thresh
    labels = fcluster(linkage_matrix, t=distance_thresh,
                      criterion='distance')

    # Build cluster index
    clusters = {}
    speaker_to_cluster = {}
    for i, label in enumerate(labels):
        cluster_id = int(label)
        speaker_id = pool.speaker_ids[i]
        speaker_to_cluster[speaker_id] = cluster_id
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(speaker_id)

    pool.clusters = clusters
    pool.speaker_to_cluster = speaker_to_cluster

    cluster_sizes = [len(v) for v in clusters.values()]
    size_counts = Counter(cluster_sizes)
    size_summary = "\n".join(
        "  size %d: %d cluster%s" % (s, c, "s" if c > 1 else "")
        for s, c in sorted(size_counts.items()))
    logger.info(
        "Clustered %d speakers into %d clusters "
        "(similarity_thresh=%.2f):\n%s",
        n, len(clusters), similarity_thresh, size_summary)

    return pool


def compute_vad_segments(pool, sample_rate=16000, min_dur=0.1,
                         n_workers=1):
    """Pre-compute VAD segments for all utterances in the pool.

    Runs Silero VAD on every wav file and caches the speech boundaries
    in each SpeakerInfo's vad_segments dict. This avoids repeated VAD
    calls during meeting generation.

    Args:
      pool: SpeakerPool with speakers populated.
      sample_rate: Target sample rate for VAD boundary alignment.
      min_dur: Minimum segment duration in seconds.
      n_workers: Number of parallel workers for VAD computation.

    Returns:
      The same SpeakerPool with vad_segments populated.
    """
    all_paths = []
    for spk_id in pool.speaker_ids:
        for wav_path in pool.speakers[spk_id].wav_paths:
            all_paths.append((spk_id, wav_path))

    logger.info("Computing VAD segments for %d utterances (workers=%d)",
                len(all_paths), n_workers)

    if n_workers <= 1:
        for spk_id, wav_path in tqdm(all_paths, desc="Computing VAD"):
            segments = get_speech_segments(wav_path, sample_rate, min_dur)
            pool.speakers[spk_id].vad_segments[wav_path] = segments
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _vad_worker, wav_path, sample_rate, min_dur
                ): (spk_id, wav_path)
                for spk_id, wav_path in all_paths
            }
            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc="Computing VAD"):
                spk_id, wav_path = futures[future]
                pool.speakers[spk_id].vad_segments[wav_path] = (
                    future.result())

    # Remove utterances with no detected speech
    speakers_to_drop = []
    for spk_id in pool.speaker_ids:
        spk = pool.speakers[spk_id]
        empty_paths = [p for p in spk.wav_paths
                       if not spk.vad_segments.get(p)]
        if empty_paths:
            logger.warning(
                "Speaker %s: %d/%d utterances have no detected speech "
                "and will be excluded.", spk_id, len(empty_paths),
                len(spk.wav_paths))
            spk.wav_paths = [p for p in spk.wav_paths
                             if p not in set(empty_paths)]
            for p in empty_paths:
                del spk.vad_segments[p]
        if not spk.wav_paths:
            logger.warning(
                "Speaker %s has no utterances with detectable speech, "
                "dropping from pool.", spk_id)
            speakers_to_drop.append(spk_id)

    if speakers_to_drop:
        keep_indices = [i for i, sid in enumerate(pool.speaker_ids)
                        if sid not in speakers_to_drop]
        pool.speaker_ids = [pool.speaker_ids[i] for i in keep_indices]
        for spk_id in speakers_to_drop:
            del pool.speakers[spk_id]
            pool.speaker_to_cluster.pop(spk_id, None)
        if pool.similarity_matrix is not None:
            pool.similarity_matrix = (
                pool.similarity_matrix[np.ix_(keep_indices, keep_indices)])
        # Rebuild clusters from remaining speakers
        if pool.clusters:
            pool.clusters = {}
            for spk_id, cluster_id in pool.speaker_to_cluster.items():
                pool.clusters.setdefault(cluster_id, []).append(spk_id)

    return pool


def _vad_worker(wav_path, sample_rate, min_dur):
    """Worker function for parallel VAD computation.

    Each worker process lazily loads its own VAD model instance.

    Args:
      wav_path: Path to audio file.
      sample_rate: Target sample rate.
      min_dur: Minimum segment duration in seconds.

    Returns:
      List of (start_sample, end_sample) tuples.
    """
    # Import here because ProcessPoolExecutor workers are separate
    # processes that don't inherit the parent's module state. Each
    # worker needs its own import to trigger the lazy VAD model
    # initialisation inside get_speech_segments.
    from wespeaker.utils.meeting_sim.turns import get_speech_segments
    return get_speech_segments(wav_path, sample_rate, min_dur)


def compute_speaker_rms(pool, sample_rate=16000, n_workers=1):
    """Compute mean RMS energy per speaker across all utterances.

    For each utterance, RMS is computed over speech-active regions if
    VAD segments are available in the pool, otherwise over the whole
    file. The per-speaker mean RMS is stored in SpeakerInfo.rms.

    Args:
      pool: SpeakerPool with speakers populated. If vad_segments are
        available, they are used to restrict RMS to speech regions.
      sample_rate: Target sample rate for audio loading.
      n_workers: Number of parallel workers.

    Returns:
      The same SpeakerPool with rms values populated.
    """
    all_paths = []
    for spk_id in pool.speaker_ids:
        spk = pool.speakers[spk_id]
        for wav_path in spk.wav_paths:
            vad_segs = spk.vad_segments.get(wav_path)
            all_paths.append((spk_id, wav_path, vad_segs))

    logger.info("Computing speaker RMS for %d utterances (workers=%d)",
                len(all_paths), n_workers)

    # Collect per-utterance RMS values keyed by speaker
    spk_rms_values = {spk_id: [] for spk_id in pool.speaker_ids}

    if n_workers <= 1:
        for spk_id, wav_path, vad_segs in tqdm(all_paths,
                                                desc="Computing RMS"):
            rms = _rms_worker(wav_path, sample_rate, vad_segs)
            if rms is not None:
                spk_rms_values[spk_id].append(rms)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _rms_worker, wav_path, sample_rate, vad_segs
                ): spk_id
                for spk_id, wav_path, vad_segs in all_paths
            }
            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc="Computing RMS"):
                spk_id = futures[future]
                rms = future.result()
                if rms is not None:
                    spk_rms_values[spk_id].append(rms)

    for spk_id in pool.speaker_ids:
        values = spk_rms_values[spk_id]
        if values:
            pool.speakers[spk_id].rms = float(np.mean(values))
        else:
            logger.warning("Speaker %s: no valid RMS values.", spk_id)

    return pool


def _rms_worker(wav_path, sample_rate, vad_segments=None):
    """Compute RMS energy for a single utterance.

    Args:
      wav_path: Path to audio file.
      sample_rate: Target sample rate in Hz.
      vad_segments: Optional list of (start_sample, end_sample) tuples.
        If provided, RMS is computed only over these regions.

    Returns:
      Float RMS value, or None if the file cannot be loaded or has
        no audio samples.
    """
    try:
        waveform, sr = torchaudio.load(wav_path)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", wav_path, exc)
        return None

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    audio = waveform.squeeze(0).numpy().astype(np.float64)

    if vad_segments:
        parts = []
        for start, end in vad_segments:
            end = min(end, len(audio))
            if start < end:
                parts.append(audio[start:end])
        if parts:
            audio = np.concatenate(parts)
        # else fall through to use whole file

    if len(audio) == 0:
        return None

    return float(np.sqrt(np.mean(audio ** 2)))


def select_speakers(pool, n_speakers=3, similarity_mode='random',
                    similar_subgroup_size=2, fallback_strategy='random',
                    similarity_thresh=0.75, dissimilarity_thresh=0.55,
                    speaker_positions=None, mic_center=None, rng=None,
                    exclude_ids=None):
    """Select a group of speakers based on similarity mode.

    Modes:
      - random: No constraints, pick randomly.
      - all-similar: All speakers from the same cluster.
      - all-dissimilar: Each speaker from a different cluster, with
          pairwise similarity below dissimilarity_thresh.
      - similar-close-subgroup: A subgroup of similar speakers placed
          at positions with the smallest angular separation from the
          mic array (hardest to distinguish via beamforming); remainder
          selected per fallback_strategy.
      - similar-distant-subgroup: A subgroup of similar speakers placed
          at positions with the largest angular separation from the mic
          array; remainder per fallback_strategy.

    Args:
      pool: SpeakerPool with clusters assigned.
      n_speakers: Number of speakers to select.
      similarity_mode: One of the modes listed above.
      similar_subgroup_size: Number of speakers in the similar subgroup
        (for subgroup modes).
      fallback_strategy: How to select remaining speakers after the
        subgroup. 'random' picks freely, 'dissimilar' ensures remaining
        speakers have similarity below dissimilarity_thresh with the
        subgroup.
      similarity_thresh: Used for re-clustering if needed.
      dissimilarity_thresh: Max similarity for "dissimilar" selections.
      speaker_positions: List of 3D position arrays (one per speaker
        slot). Required for subgroup modes to determine close/distant
        positions. If None, subgroup modes fall back to random position
        assignment.
      mic_center: 3D position of the microphone array center. Used to
        compute angular separation between speaker positions. If None,
        falls back to Euclidean distance between positions.
      rng: numpy random Generator instance.
      exclude_ids: Optional set of speaker IDs to exclude from
        selection. If exclusion leaves fewer than n_speakers candidates,
        falls back to the full pool.

    Returns:
      List of SpeakerInfo objects for the selected speakers, ordered
        by position index (i.e. result[i] goes to speaker_positions[i]).

    Raises:
      ValueError: If the mode is unrecognized or selection fails.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Ensure clustering is done
    if not pool.clusters:
        pool = cluster_speakers(pool, similarity_thresh)

    valid_modes = {'random', 'all-similar', 'all-dissimilar',
                   'similar-close-subgroup', 'similar-distant-subgroup'}
    if similarity_mode not in valid_modes:
        raise ValueError(
            "Unknown similarity_mode '%s'. Must be one of: %s"
            % (similarity_mode, ', '.join(sorted(valid_modes))))

    if similarity_mode == 'random':
        return _select_random(pool, n_speakers, rng, exclude_ids)
    elif similarity_mode == 'all-similar':
        return _select_all_similar(pool, n_speakers, similarity_thresh, rng)
    elif similarity_mode == 'all-dissimilar':
        return _select_all_dissimilar(
            pool, n_speakers, dissimilarity_thresh, rng)
    elif similarity_mode in ('similar-close-subgroup',
                             'similar-distant-subgroup'):
        use_distant = (similarity_mode == 'similar-distant-subgroup')
        return _select_subgroup(
            pool, n_speakers, similar_subgroup_size, use_distant,
            fallback_strategy, dissimilarity_thresh, speaker_positions,
            mic_center, rng)


def _select_random(pool, n_speakers, rng, exclude_ids=None):
    """Select speakers randomly without constraints.

    Args:
      pool: SpeakerPool.
      n_speakers: Number of speakers to select.
      rng: numpy random Generator.
      exclude_ids: Optional set of speaker IDs to exclude. Falls back
        to the full pool if exclusion leaves too few candidates.
    """
    candidates = pool.speaker_ids
    if exclude_ids:
        filtered = [sid for sid in pool.speaker_ids
                    if sid not in exclude_ids]
        if len(filtered) >= n_speakers:
            candidates = filtered

    chosen_ids = rng.choice(
        candidates, size=n_speakers, replace=False).tolist()
    logger.info("Selected speakers (random): %s", chosen_ids)
    return [pool.speakers[sid] for sid in chosen_ids]


def _select_all_similar(pool, n_speakers, similarity_thresh, rng):
    """Select all speakers from the same cluster.

    If no single cluster is large enough, relaxes the threshold
    incrementally until one is found.
    """
    thresh = similarity_thresh
    for _ in range(10):  # max relaxation steps
        # Find clusters large enough
        valid_clusters = [
            cid for cid, members in pool.clusters.items()
            if len(members) >= n_speakers]

        if valid_clusters:
            cluster_id = rng.choice(valid_clusters)
            members = pool.clusters[cluster_id]
            chosen_ids = rng.choice(
                members, size=n_speakers, replace=False).tolist()
            logger.info(
                "Selected speakers (all-similar, cluster %d, "
                "thresh=%.2f): %s", cluster_id, thresh, chosen_ids)
            return [pool.speakers[sid] for sid in chosen_ids]

        # Relax threshold and re-cluster
        thresh -= 0.05
        logger.info("No cluster large enough, relaxing threshold to %.2f",
                    thresh)
        pool = cluster_speakers(pool, thresh)

    raise ValueError(
        "Could not find %d similar speakers even after relaxing "
        "threshold to %.2f." % (n_speakers, thresh))


def _select_all_dissimilar(pool, n_speakers, dissimilarity_thresh, rng):
    """Select speakers from different clusters with low pairwise similarity.

    Picks one speaker per cluster, verifying all pairwise similarities
    are below dissimilarity_thresh.
    """
    id_to_idx = {sid: i for i, sid in enumerate(pool.speaker_ids)}

    # Get clusters with at least one member, shuffled
    cluster_ids = list(pool.clusters.keys())
    rng.shuffle(cluster_ids)

    if len(cluster_ids) < n_speakers:
        raise ValueError(
            "Only %d clusters available but need %d dissimilar speakers."
            % (len(cluster_ids), n_speakers))

    max_attempts = 500
    for _ in range(max_attempts):
        rng.shuffle(cluster_ids)
        selected = []

        for cid in cluster_ids:
            if len(selected) >= n_speakers:
                break

            members = pool.clusters[cid]
            rng.shuffle(members)

            for candidate in members:
                cand_idx = id_to_idx[candidate]
                # Check dissimilarity with all already-selected
                ok = True
                for prev_id in selected:
                    prev_idx = id_to_idx[prev_id]
                    if (pool.similarity_matrix[cand_idx, prev_idx]
                            > dissimilarity_thresh):
                        ok = False
                        break
                if ok:
                    selected.append(candidate)
                    break

        if len(selected) >= n_speakers:
            chosen_ids = selected[:n_speakers]
            logger.info("Selected speakers (all-dissimilar): %s", chosen_ids)
            return [pool.speakers[sid] for sid in chosen_ids]

    raise ValueError(
        "Could not find %d dissimilar speakers (thresh=%.2f) after %d "
        "attempts." % (n_speakers, dissimilarity_thresh, max_attempts))


def _select_subgroup(pool, n_speakers, subgroup_size, use_distant,
                     fallback_strategy, dissimilarity_thresh,
                     speaker_positions, mic_center, rng):
    """Select a similar subgroup and assign to positions based on angular separation.

    Args:
      pool: SpeakerPool.
      n_speakers: Total speakers to select.
      subgroup_size: Number in the similar subgroup.
      use_distant: If True, place similar speakers at positions with
        largest angular separation from the mic. If False, place at
        positions with smallest angular separation.
      fallback_strategy: 'random' or 'dissimilar' for remainder.
      dissimilarity_thresh: Max similarity for dissimilar fallback.
      speaker_positions: List of 3D arrays (physical positions).
      mic_center: 3D position of mic array center, or None.
      rng: numpy random Generator.

    Returns:
      List of SpeakerInfo ordered by position index.
    """
    if subgroup_size > n_speakers:
        subgroup_size = n_speakers
    if subgroup_size < 2:
        # No meaningful subgroup, just select randomly
        return _select_random(pool, n_speakers, rng)

    id_to_idx = {sid: i for i, sid in enumerate(pool.speaker_ids)}

    # Determine which position indices get the similar subgroup
    subgroup_positions = find_subgroup_positions(
        speaker_positions, subgroup_size, use_distant, mic_center)
    remainder_positions = [
        i for i in range(n_speakers) if i not in subgroup_positions]

    # Select similar subgroup from a cluster
    subgroup_speakers = _pick_from_cluster(
        pool, subgroup_size, rng)
    if subgroup_speakers is None:
        logger.warning(
            "Could not find a cluster with %d members, falling back to "
            "closest pair.", subgroup_size)
        subgroup_speakers = _pick_closest_pair(pool, subgroup_size, rng)

    # Select remainder speakers
    remainder_speakers = _pick_remainder(
        pool, n_speakers - subgroup_size, subgroup_speakers,
        fallback_strategy, dissimilarity_thresh, id_to_idx, rng)

    # Assemble result ordered by position
    result = [None] * n_speakers
    for i, pos_idx in enumerate(subgroup_positions):
        result[pos_idx] = pool.speakers[subgroup_speakers[i]]
    for i, pos_idx in enumerate(remainder_positions):
        result[pos_idx] = pool.speakers[remainder_speakers[i]]

    chosen_ids = [s.speaker_id for s in result]
    logger.info(
        "Selected speakers (subgroup at positions %s): %s",
        subgroup_positions, chosen_ids)
    return result


def find_subgroup_positions(speaker_positions, subgroup_size, use_distant,
                             mic_center):
    """Find position indices for the subgroup based on angular separation.

    Uses the angle between direction vectors from the microphone array
    center to each speaker position. Small angular separation means
    speakers are hard to distinguish via beamforming.

    Falls back to Euclidean distance if mic_center is not provided.

    Args:
      speaker_positions: List of 3D position arrays, or None.
      subgroup_size: Number of positions to pick.
      use_distant: If True, pick positions with largest total angular
        separation. If False, pick positions with smallest.
      mic_center: 3D position of the mic array center, or None.

    Returns:
      List of position indices for the subgroup.
    """
    if speaker_positions is None or len(speaker_positions) < 2:
        # No spatial info, use first N positions
        return list(range(subgroup_size))

    n = len(speaker_positions)
    positions = np.array(speaker_positions)

    best_score = None
    best_indices = None

    for indices in combinations(range(n), subgroup_size):
        total_separation = 0.0
        for i, j in combinations(indices, 2):
            total_separation += _position_separation(
                positions[i], positions[j], mic_center)

        if use_distant:
            if best_score is None or total_separation > best_score:
                best_score = total_separation
                best_indices = indices
        else:
            if best_score is None or total_separation < best_score:
                best_score = total_separation
                best_indices = indices

    return list(best_indices)


def _position_separation(pos_a, pos_b, mic_center):
    """Compute the separation between two speaker positions.

    If mic_center is provided, computes the angular separation (in
    radians) as seen from the microphone array. Otherwise falls back
    to Euclidean distance.

    Args:
      pos_a: 3D position of speaker A.
      pos_b: 3D position of speaker B.
      mic_center: 3D mic array center, or None.

    Returns:
      Separation measure (radians if angular, meters if Euclidean).
    """
    if mic_center is None:
        return np.linalg.norm(pos_a - pos_b)

    # Direction vectors from mic to each speaker
    dir_a = pos_a - mic_center
    dir_b = pos_b - mic_center

    norm_a = np.linalg.norm(dir_a)
    norm_b = np.linalg.norm(dir_b)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0

    # Angle between the two direction vectors
    cos_angle = np.dot(dir_a, dir_b) / (norm_a * norm_b)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def _pick_from_cluster(pool, size, rng):
    """Pick `size` speakers from a single cluster.

    Chooses a random cluster that is large enough.

    Returns:
      List of speaker_ids, or None if no cluster is large enough.
    """
    valid_clusters = [
        cid for cid, members in pool.clusters.items()
        if len(members) >= size]

    if not valid_clusters:
        return None

    cluster_id = rng.choice(valid_clusters)
    members = list(pool.clusters[cluster_id])
    chosen = rng.choice(members, size=size, replace=False).tolist()
    return chosen


def _pick_closest_pair(pool, size, rng):
    """Fallback: pick the `size` most similar speakers regardless of clusters.

    Used when no single cluster is large enough.

    Returns:
      List of speaker_ids.
    """
    n = len(pool.speaker_ids)
    # Find the pair with highest similarity and extend greedily
    flat_idx = np.argsort(pool.similarity_matrix.ravel())[::-1]

    selected = set()
    for idx in flat_idx:
        i, j = idx // n, idx % n
        if i == j:
            continue
        if i not in selected and j not in selected:
            selected.add(i)
            selected.add(j)
            break
        elif len(selected) < size:
            selected.add(i)
            selected.add(j)
        if len(selected) >= size:
            break

    chosen = [pool.speaker_ids[i] for i in list(selected)[:size]]

    # If still not enough, pad with random speakers
    remaining = set(range(n)) - selected
    while len(chosen) < size and remaining:
        idx = rng.choice(list(remaining))
        remaining.remove(idx)
        chosen.append(pool.speaker_ids[idx])

    return chosen


def _pick_remainder(pool, n_remainder, subgroup_ids, fallback_strategy,
                    dissimilarity_thresh, id_to_idx, rng):
    """Pick remainder speakers after the similar subgroup.

    Args:
      pool: SpeakerPool.
      n_remainder: Number of speakers to pick.
      subgroup_ids: Speaker IDs already selected for the subgroup.
      fallback_strategy: 'random' or 'dissimilar'.
      dissimilarity_thresh: Max similarity for 'dissimilar' strategy.
      id_to_idx: Dict mapping speaker_id -> matrix index.
      rng: numpy random Generator.

    Returns:
      List of speaker_ids.
    """
    excluded = set(subgroup_ids)
    available = [sid for sid in pool.speaker_ids if sid not in excluded]

    if n_remainder <= 0:
        return []

    if fallback_strategy == 'random':
        chosen = rng.choice(available, size=n_remainder,
                            replace=False).tolist()
        return chosen

    # fallback_strategy == 'dissimilar'
    # Pick speakers that are below dissimilarity_thresh with all subgroup
    # members
    selected = []
    rng.shuffle(available)

    for candidate in available:
        if len(selected) >= n_remainder:
            break
        cand_idx = id_to_idx[candidate]
        ok = True
        # Check against subgroup
        for sub_id in subgroup_ids:
            sub_idx = id_to_idx[sub_id]
            if (pool.similarity_matrix[cand_idx, sub_idx]
                    > dissimilarity_thresh):
                ok = False
                break
        # Check against already-selected remainder
        if ok:
            for prev_id in selected:
                prev_idx = id_to_idx[prev_id]
                if (pool.similarity_matrix[cand_idx, prev_idx]
                        > dissimilarity_thresh):
                    ok = False
                    break
        if ok:
            selected.append(candidate)

    if len(selected) < n_remainder:
        # Couldn't satisfy all constraints, pad with least-similar available
        remaining = [s for s in available if s not in selected]
        # Sort by max similarity to subgroup (ascending = most dissimilar first)
        def max_sim_to_subgroup(sid):
            idx = id_to_idx[sid]
            return max(pool.similarity_matrix[idx, id_to_idx[sub]]
                       for sub in subgroup_ids)
        remaining.sort(key=max_sim_to_subgroup)
        for sid in remaining:
            if len(selected) >= n_remainder:
                break
            selected.append(sid)

    return selected


def save_pool(pool, output_dir):
    """Save a SpeakerPool to disk.

    Saves:
      - pool.pkl: Full SpeakerPool object.
      - similarity_matrix.npy: Similarity matrix for inspection.
      - enrol.ark + enrol.scp: Per-speaker mean embeddings in Kaldi
        ark/scp format, keyed by speaker ID.

    Args:
      pool: SpeakerPool to save.
      output_dir: Directory to write files into.
    """
    os.makedirs(output_dir, exist_ok=True)

    pool_path = os.path.join(output_dir, 'pool.pkl')
    with open(pool_path, 'wb') as f:
        pickle.dump(pool, f)

    matrix_path = os.path.join(output_dir, 'similarity_matrix.npy')
    np.save(matrix_path, pool.similarity_matrix)

    export_enrol_scp(pool, output_dir)

    logger.info("Saved speaker pool to %s (%d speakers)",
                output_dir, len(pool.speaker_ids))


def export_enrol_scp(pool, output_dir):
    """Export per-speaker mean embeddings as a Kaldi ark/scp file.

    Writes enrol.ark and enrol.scp to output_dir, keyed by speaker ID.
    This provides a standard format for enrolment embeddings that can
    be used independently of the pool.pkl for speaker identification
    and diarization visualization.

    Args:
      pool: SpeakerPool with speakers that have embeddings set.
      output_dir: Directory to write enrol.ark and enrol.scp into.
    """
    os.makedirs(output_dir, exist_ok=True)
    ark_path = os.path.join(os.path.abspath(output_dir), 'enrol.ark')
    scp_path = os.path.join(os.path.abspath(output_dir), 'enrol.scp')

    with kaldiio.WriteHelper('ark,scp:' + ark_path + ',' + scp_path) as writer:
        for spk_id in pool.speaker_ids:
            emb = pool.speakers[spk_id].embedding
            if emb is not None:
                writer(spk_id, emb.astype(np.float32))


def load_pool(pool_dir):
    """Load a SpeakerPool from disk.

    Args:
      pool_dir: Directory containing pool.pkl.

    Returns:
      SpeakerPool instance.
    """
    pool_path = os.path.join(pool_dir, 'pool.pkl')
    with open(pool_path, 'rb') as f:
        pool = pickle.load(f)
    return pool
