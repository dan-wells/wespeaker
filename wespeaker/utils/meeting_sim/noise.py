"""Noise generation for meeting simulation.

Supports three noise modes:
  - inside: point source placed inside the room, convolved with room RIR.
  - outside: point source with low-pass wall transmission filter before
    RIR convolution.
  - diffuse: spatially correlated noise across the mic array using the
    sinc coherence model.
"""

import glob
import logging
import os

import numpy as np
import torchaudio
from scipy.signal import butter, sosfilt, fftconvolve


logger = logging.getLogger('meeting_sim.noise')

SPEED_OF_SOUND = 343.0  # m/s


class NoiseConfig:
    """Configuration for noise generation.

    Attributes:
      enabled: Whether noise is enabled.
      source: Noise source type ('inside', 'outside', or 'diffuse').
      snr_db: Target signal-to-noise ratio in dB.
      audio: Path to noise audio. Can be a single WAV file, a directory
        of WAV files, or a text file listing WAV paths (one per line).
      fill_mode: How to handle noise files shorter than the meeting.
        'repeat' tiles a single randomly-selected file. 'concatenate'
        joins shuffled files with crossfade. 'concatenate-gaps' joins
        shuffled files with random silence gaps (5-20 s).
      wall_cutoff_hz: Butterworth cutoff frequency for outside mode.
      wall_filter_order: Butterworth filter order for outside mode.
      unique_across_meetings: If true, avoid reusing noise files
        across meetings in a batch (sequential generation only).
    """

    def __init__(self, enabled=False, source='inside', snr_db=20.0,
                 audio=None, fill_mode='repeat', wall_cutoff_hz=1200,
                 wall_filter_order=2, unique_across_meetings=False):
        self.enabled = enabled
        self.source = source
        self.snr_db = snr_db
        self.audio = audio
        self.fill_mode = fill_mode
        self.wall_cutoff_hz = wall_cutoff_hz
        self.wall_filter_order = wall_filter_order
        self.unique_across_meetings = unique_across_meetings

    @classmethod
    def from_dict(cls, cfg_dict):
        """Build NoiseConfig from the noise: section of the YAML config.

        Args:
          cfg_dict: Dict from the 'noise' key of the config.

        Returns:
          NoiseConfig instance.
        """
        if cfg_dict is None:
            return cls()
        return cls(
            enabled=cfg_dict.get('enabled', False),
            source=cfg_dict.get('source', 'inside'),
            snr_db=cfg_dict.get('snr_db', 20.0),
            audio=cfg_dict.get('audio'),
            fill_mode=cfg_dict.get('fill_mode', 'repeat'),
            wall_cutoff_hz=cfg_dict.get('wall_cutoff_hz', 1200),
            wall_filter_order=cfg_dict.get('wall_filter_order', 2),
            unique_across_meetings=cfg_dict.get(
                'unique_across_meetings', False),
        )


def _resolve_audio_paths(audio_path):
    """Resolve the audio config field to a list of WAV file paths.

    Accepts:
      - A single .wav file path.
      - A directory (globs for *.wav inside it).
      - A text file (reads lines as paths, strips whitespace/blanks).

    Args:
      audio_path: String path.

    Returns:
      Sorted list of absolute file paths.

    Raises:
      ValueError: If no WAV files are found.
    """
    if os.path.isdir(audio_path):
        paths = sorted(glob.glob(os.path.join(audio_path, '*.wav')))
    elif audio_path.lower().endswith('.wav'):
        paths = [audio_path]
    else:
        # Treat as a text file listing paths
        with open(audio_path, 'r') as f:
            paths = [line.strip() for line in f if line.strip()]
    if not paths:
        raise ValueError(
            "No WAV files found from audio path: %s" % audio_path)
    return paths


def _load_mono_wav(wav_path, target_sample_rate):
    """Load a WAV file as mono float64 at the target sample rate.

    Args:
      wav_path: Path to WAV file.
      target_sample_rate: Desired sample rate in Hz.

    Returns:
      np.ndarray of mono audio samples (float64).
    """
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze(0).numpy().astype(np.float64)


def load_noise_audio(noise_cfg, duration_samples, sample_rate, rng,
                     used_paths=None):
    """Load and assemble noise audio to cover the meeting duration.

    Args:
      noise_cfg: NoiseConfig instance.
      duration_samples: Required output length in samples.
      sample_rate: Sample rate in Hz.
      rng: numpy random Generator.
      used_paths: Optional set of file paths already used in prior
        meetings (for unique_across_meetings). Consumed paths are
        added to this set in-place.

    Returns:
      Tuple of (audio, consumed_paths) where audio is np.ndarray of
        mono noise (float64, length duration_samples) and
        consumed_paths is a list of file paths used, in load order.
    """
    all_paths = _resolve_audio_paths(noise_cfg.audio)

    # Filter out already-used paths when tracking across meetings
    if used_paths is not None:
        available = [p for p in all_paths if p not in used_paths]
        if not available:
            logger.warning(
                "All %d noise files used; resetting pool.",
                len(all_paths))
            used_paths.clear()
            available = all_paths
    else:
        available = all_paths

    if noise_cfg.fill_mode == 'repeat':
        audio, consumed = _fill_repeat(
            available, duration_samples, sample_rate, rng)
    elif noise_cfg.fill_mode == 'concatenate':
        audio, consumed = _fill_concatenate(
            available, duration_samples, sample_rate, rng, use_gaps=False)
    elif noise_cfg.fill_mode == 'concatenate-gaps':
        audio, consumed = _fill_concatenate(
            available, duration_samples, sample_rate, rng, use_gaps=True)
    else:
        raise ValueError("Unknown fill_mode: %s" % noise_cfg.fill_mode)

    if used_paths is not None:
        used_paths.update(consumed)

    return audio, consumed


def _fill_repeat(paths, duration_samples, sample_rate, rng):
    """Fill duration by tiling a single randomly-selected file.

    Picks a random file, loads it, tiles to cover the duration, then
    applies a random start offset to avoid always starting at the same
    point.

    Args:
      paths: List of WAV file paths.
      duration_samples: Required output length.
      sample_rate: Sample rate in Hz.
      rng: numpy random Generator.

    Returns:
      Tuple of (audio, consumed_paths) where audio is np.ndarray of
        length duration_samples (float64) and consumed_paths is a list
        of file paths that were loaded.
    """
    chosen = paths[rng.integers(len(paths))]
    audio = _load_mono_wav(chosen, sample_rate)

    if len(audio) == 0:
        return np.zeros(duration_samples, dtype=np.float64), [chosen]

    # Tile to cover duration
    while len(audio) < duration_samples:
        audio = np.concatenate([audio, audio])

    # Random start offset
    max_offset = len(audio) - duration_samples
    if max_offset > 0:
        offset = rng.integers(0, max_offset)
        audio = audio[offset:offset + duration_samples]
    else:
        audio = audio[:duration_samples]

    return audio, [chosen]


def _fill_concatenate(paths, duration_samples, sample_rate, rng,
                      use_gaps=False):
    """Fill duration by concatenating shuffled files.

    Files are shuffled and concatenated. If use_gaps is False, a 10 ms
    crossfade is applied at splice points. If use_gaps is True, random
    silence gaps (5-20 s) are inserted between files. Loops over the
    file list if all files are exhausted before the duration is filled.

    Args:
      paths: List of WAV file paths.
      duration_samples: Required output length.
      sample_rate: Sample rate in Hz.
      rng: numpy random Generator.
      use_gaps: If True, insert silence gaps instead of crossfading.

    Returns:
      Tuple of (audio, consumed_paths) where audio is np.ndarray of
        length duration_samples (float64) and consumed_paths is a list
        of file paths in the order they were loaded (unique).
    """
    crossfade_samples = int(0.010 * sample_rate)  # 10 ms
    gap_range_s = (5.0, 20.0)

    shuffled = list(paths)
    rng.shuffle(shuffled)
    file_idx = 0
    consumed = []
    consumed_set = set()

    result = np.zeros(0, dtype=np.float64)

    while len(result) < duration_samples:
        current_path = shuffled[file_idx]
        audio = _load_mono_wav(current_path, sample_rate)
        if current_path not in consumed_set:
            consumed.append(current_path)
            consumed_set.add(current_path)
        file_idx = (file_idx + 1) % len(shuffled)
        # Re-shuffle when wrapping around
        if file_idx == 0:
            rng.shuffle(shuffled)

        if len(audio) == 0:
            continue

        if len(result) == 0:
            result = audio
        elif use_gaps:
            gap_s = rng.uniform(*gap_range_s)
            gap_samples = int(gap_s * sample_rate)
            gap = np.zeros(gap_samples, dtype=np.float64)
            result = np.concatenate([result, gap, audio])
        else:
            # Crossfade
            n_fade = min(crossfade_samples, len(result), len(audio))
            if n_fade > 0:
                fade_out = np.linspace(1.0, 0.0, n_fade)
                fade_in = np.linspace(0.0, 1.0, n_fade)
                result[-n_fade:] *= fade_out
                audio_copy = audio.copy()
                audio_copy[:n_fade] *= fade_in
                result[-n_fade:] += audio_copy[:n_fade]
                result = np.concatenate([result, audio_copy[n_fade:]])
            else:
                result = np.concatenate([result, audio])

    return result[:duration_samples], consumed


def pick_noise_position(room_cfg, source_mode, rng):
    """Pick a random noise source position near a room wall.

    Selects one of three walls (far-x, near-y, far-y) -- excludes the
    mic wall (near-x, where the array sits). Places the source at a
    random location along the chosen wall.

    For 'inside' mode, the source is 0.2-0.5 m from the wall surface.
    For 'outside' mode, the source is 0.05-0.15 m from the wall surface.

    Args:
      room_cfg: RoomConfig instance.
      source_mode: 'inside' or 'outside'.
      rng: numpy random Generator.

    Returns:
      np.ndarray of shape (3,) with the source position.
    """
    if source_mode == 'inside':
        dist_min, dist_max = 0.2, 0.5
    else:
        dist_min, dist_max = 0.05, 0.15

    margin = 0.3  # margin from perpendicular walls
    z_range = (1.0, 2.0)  # height range for noise source

    # Three candidate walls
    # wall 0: far-x (x = room_cfg.length)
    # wall 1: near-y (y = 0)
    # wall 2: far-y (y = room_cfg.width)
    wall = rng.integers(3)
    dist = rng.uniform(dist_min, dist_max)
    z_pos = rng.uniform(*z_range)

    if wall == 0:
        # Far-x wall
        x_pos = room_cfg.length - dist
        y_pos = rng.uniform(margin, room_cfg.width - margin)
    elif wall == 1:
        # Near-y wall
        x_pos = rng.uniform(margin, room_cfg.length - margin)
        y_pos = dist
    else:
        # Far-y wall
        x_pos = rng.uniform(margin, room_cfg.length - margin)
        y_pos = room_cfg.width - dist

    return np.array([x_pos, y_pos, z_pos])


def apply_wall_filter(audio, cutoff_hz, order, sample_rate):
    """Apply a Butterworth low-pass filter to simulate wall transmission.

    Args:
      audio: np.ndarray of mono audio samples.
      cutoff_hz: Cutoff frequency in Hz.
      order: Filter order.
      sample_rate: Sample rate in Hz.

    Returns:
      Filtered audio (new array).
    """
    sos = butter(order, cutoff_hz, btype='low', fs=sample_rate, output='sos')
    return sosfilt(sos, audio)


def generate_diffuse_noise(mono_noise, array_cfg, sample_rate, rng):
    """Generate spatially correlated multichannel noise from mono input.

    Uses the sinc coherence model for a diffuse field:
      gamma(f, d) = sinc(2 * pi * f * d / c)
    where d is the inter-mic distance.

    Implemented via overlap-add with precomputed Cholesky factors of the
    coherence matrix at each frequency bin.

    Args:
      mono_noise: np.ndarray of mono noise audio (1D, float64).
      array_cfg: ArrayConfig instance (n_mics, spacing).
      sample_rate: Sample rate in Hz.
      rng: numpy random Generator.

    Returns:
      np.ndarray of shape (n_mics, n_samples) with spatially correlated
        noise.
    """
    n_mics = array_cfg.n_mics
    spacing = array_cfg.spacing
    n_samples = len(mono_noise)
    fft_size = 4096
    hop_size = fft_size // 2
    n_bins = fft_size // 2 + 1

    # Precompute Cholesky factors for all frequency bins
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)  # (n_bins,)

    # Build coherence matrices: (n_bins, n_mics, n_mics)
    # For a linear array, d_ij = |i - j| * spacing
    mic_indices = np.arange(n_mics)
    distances = np.abs(mic_indices[:, None] - mic_indices[None, :]) * spacing
    # distances shape: (n_mics, n_mics)

    # sinc(2*pi*f*d/c) -- numpy sinc is sin(pi*x)/(pi*x), so we need
    # sinc(2*f*d/c) to get sin(2*pi*f*d/c)/(2*pi*f*d/c)
    # i.e. np.sinc(x) = sin(pi*x)/(pi*x), we want sin(2*pi*f*d/c)/(2*pi*f*d/c)
    # which is np.sinc(2*f*d/c)
    arg = 2.0 * freqs[:, None, None] * distances[None, :, :] / SPEED_OF_SOUND
    coherence = np.sinc(arg)  # (n_bins, n_mics, n_mics)

    # Regularise and Cholesky decompose
    eps = 1e-8
    coherence += eps * np.eye(n_mics)[None, :, :]
    cholesky_factors = np.linalg.cholesky(coherence)  # (n_bins, n_mics, n_mics)

    # DC bin: all channels identical (skip Cholesky, just replicate)
    # We handle this by zeroing out the Cholesky factor and replacing
    # with a rank-1 matrix that replicates the input
    dc_factor = np.zeros((n_mics, n_mics))
    dc_factor[:, 0] = 1.0
    cholesky_factors[0] = dc_factor

    # Overlap-add synthesis
    window = np.hanning(fft_size)
    output = np.zeros((n_mics, n_samples), dtype=np.float64)

    # Pad input to allow full last frame
    padded = np.concatenate([mono_noise,
                             np.zeros(fft_size, dtype=np.float64)])

    n_frames = (len(padded) - fft_size) // hop_size + 1

    for frame_idx in range(n_frames):
        start = frame_idx * hop_size
        frame = padded[start:start + fft_size] * window

        # FFT of mono frame
        mono_spectrum = np.fft.rfft(frame)  # (n_bins,)

        # Generate independent complex Gaussian noise for each mic
        noise_real = rng.standard_normal((n_bins, n_mics))
        noise_imag = rng.standard_normal((n_bins, n_mics))
        independent = (noise_real + 1j * noise_imag) / np.sqrt(2)

        # Apply Cholesky factors to correlate channels
        # correlated[f, m] = sum_k cholesky[f, m, k] * independent[f, k]
        correlated = np.einsum('fmk,fk->fm', cholesky_factors, independent)

        # Scale by mono spectrum magnitude to preserve spectral shape
        magnitude = np.abs(mono_spectrum)  # (n_bins,)
        correlated *= magnitude[:, None]

        # IFFT each channel (no synthesis window -- single Hann window
        # at analysis satisfies COLA with 50% overlap)
        for mic_idx in range(n_mics):
            channel_frame = np.fft.irfft(correlated[:, mic_idx],
                                         n=fft_size)
            end = min(start + fft_size, n_samples)
            n_add = end - start
            output[mic_idx, start:end] += channel_frame[:n_add]

    return output


def extract_noise_rir(room, noise_src_idx, n_mics):
    """Extract the RIR for a noise source from a pyroomacoustics room.

    Args:
      room: pyroomacoustics room object (after compute_rir()).
      noise_src_idx: Index of the noise source in the room's source list.
      n_mics: Number of microphones.

    Returns:
      np.ndarray of shape (n_mics, rir_length).
    """
    max_len = 0
    for mic_idx in range(n_mics):
        rir = room.rir[mic_idx][noise_src_idx]
        max_len = max(max_len, len(rir))

    noise_rir = np.zeros((n_mics, max_len), dtype=np.float32)
    for mic_idx in range(n_mics):
        rir = room.rir[mic_idx][noise_src_idx]
        noise_rir[mic_idx, :len(rir)] = rir

    return noise_rir


def convolve_noise_with_rir(mono_noise, noise_rir):
    """Convolve mono noise with a multichannel RIR.

    Args:
      mono_noise: np.ndarray of mono noise (1D).
      noise_rir: np.ndarray of shape (n_mics, rir_length).

    Returns:
      np.ndarray of shape (n_mics, n_samples) where n_samples =
        len(mono_noise) + rir_length - 1.
    """
    n_mics = noise_rir.shape[0]
    out_len = len(mono_noise) + noise_rir.shape[1] - 1
    result = np.zeros((n_mics, out_len), dtype=np.float64)
    for mic_idx in range(n_mics):
        result[mic_idx] = fftconvolve(mono_noise, noise_rir[mic_idx],
                                      mode='full')
    return result


def add_noise_to_buffer(multichannel_buffer, noise_multichannel, snr_db,
                        rttm, sample_rate):
    """Scale noise to target SNR and add to the multichannel buffer.

    SNR is computed over speech-active regions only (from RTTM) to
    avoid dilution by silence. Modifies multichannel_buffer in-place.

    Args:
      multichannel_buffer: np.ndarray of shape (n_mics, n_samples),
        modified in-place.
      noise_multichannel: np.ndarray of shape (n_mics, n_samples).
      snr_db: Target signal-to-noise ratio in dB.
      rttm: List of (speaker_id, start_s, duration_s) tuples.
      sample_rate: Sample rate in Hz.
    """
    n_samples = multichannel_buffer.shape[1]

    # Build speech-active sample mask from RTTM
    mask = np.zeros(n_samples, dtype=bool)
    for _, start_s, dur_s in rttm:
        start_idx = int(start_s * sample_rate)
        end_idx = int((start_s + dur_s) * sample_rate)
        end_idx = min(end_idx, n_samples)
        if start_idx < n_samples:
            mask[start_idx:end_idx] = True

    n_active = mask.sum()
    if n_active == 0:
        logger.warning("No speech-active samples for SNR computation; "
                       "skipping noise addition.")
        return

    # Compute power over speech-active regions
    signal_power = np.mean(multichannel_buffer[:, mask] ** 2)
    noise_power = np.mean(noise_multichannel[:, mask] ** 2)

    if noise_power < 1e-10:
        logger.warning("Noise power is near zero; skipping noise addition.")
        return

    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)

    multichannel_buffer += scale * noise_multichannel
