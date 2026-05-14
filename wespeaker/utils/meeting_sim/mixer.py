"""Audio mixing for meeting simulation.

Assembles individual speaker turns into a multi-speaker meeting audio
by convolving with room impulse responses and summing contributions.
"""

import logging
import os

import numpy as np
import soundfile as sf
import torchaudio
from scipy.signal import fftconvolve

from wespeaker.utils.meeting_sim.noise import add_noise_to_buffer
from wespeaker.utils.meeting_sim.room import beamform, select_stereo_mics


logger = logging.getLogger('meeting_sim.mixer')


class MixConfig:
    """Configuration for audio mixing.

    Attributes:
      sample_rate: Output sample rate in Hz.
      output_channels: List of output formats to produce. Valid values:
        'mono' (beamformed), 'stereo' (2-mic ~17 cm baseline),
        'multichannel' (all array mics). A single string is also accepted
        and normalised to a list. Defaults to ['mono'].
      save_reverberant: Whether to save per-speaker reverberant signals.
      save_dry: Whether to save per-speaker dry (anechoic) signals.
    """

    def __init__(self, sample_rate=16000, output_channels=None,
                 save_reverberant=False, save_dry=False):
        if output_channels is None:
            output_channels = ['mono']
        elif isinstance(output_channels, str):
            output_channels = [output_channels]
        self.sample_rate = sample_rate
        self.output_channels = output_channels
        self.save_reverberant = save_reverberant
        self.save_dry = save_dry


def mix_meeting(turns, meeting_room, rirs, mix_cfg, speaker_index_map,
                noise_multichannel=None, noise_snr_db=None, rng=None,
                speaker_rms=None, target_rms=None):
    """Assemble all turns into meeting audio using room impulse responses.

    For each turn, loads the source audio, applies overlap energy
    scaling, convolves with the speaker's RIR, and places it at the
    correct position in the output buffer.

    Args:
      turns: List of Turn objects with meeting_onset_sample set.
      meeting_room: MeetingRoom instance.
      rirs: np.ndarray of shape (n_speakers, n_mics, rir_length) from
        compute_rirs().
      mix_cfg: MixConfig instance.
      speaker_index_map: Dict mapping speaker_id -> index into the rirs
        array (matching the order speakers were added to the room).
      noise_multichannel: Optional np.ndarray of shape (n_mics,
        n_samples) with unscaled multichannel noise. If provided,
        noise_snr_db must also be set.
      noise_snr_db: Target SNR in dB for noise mixing. Required if
        noise_multichannel is provided.
      rng: numpy random Generator.
      speaker_rms: Optional dict mapping speaker_id -> float RMS.
        When provided along with target_rms, each turn's audio is
        scaled to normalize speaker levels before RIR convolution.
      target_rms: Target RMS level for normalization. Required if
        speaker_rms is provided.

    Returns:
      Dict with keys:
        'mono': np.ndarray of beamformed mono signal, if requested.
        'stereo': np.ndarray of shape (n_samples, 2), if requested.
        'multichannel': np.ndarray (n_mics, n_samples) if requested.
        'reverberant': Dict of speaker_id -> np.ndarray if requested.
        'dry': Dict of speaker_id -> np.ndarray if requested.
        'rttm': List of (speaker_id, start_s, duration_s) tuples.
        'n_samples': int, length of the output in samples (always present).
    """
    if rng is None:
        rng = np.random.default_rng()

    sample_rate = mix_cfg.sample_rate
    n_mics = meeting_room.array_config.n_mics
    rir_length = rirs.shape[2]

    # Determine output buffer length
    max_end = 0
    for turn in turns:
        turn_end = turn.meeting_onset_sample + turn.segment.duration_samples
        max_end = max(max_end, turn_end)
    # Add RIR tail
    output_length = max_end + rir_length

    # Allocate output buffers
    multichannel_buffer = np.zeros(
        (n_mics, output_length), dtype=np.float64)

    # Per-speaker buffers (for optional reverberant/dry outputs)
    speaker_ids = set(t.segment.speaker_id for t in turns)
    reverberant_buffers = {}
    dry_buffers = {}
    if mix_cfg.save_reverberant:
        for sid in speaker_ids:
            reverberant_buffers[sid] = np.zeros(output_length,
                                                dtype=np.float64)
    if mix_cfg.save_dry:
        for sid in speaker_ids:
            dry_buffers[sid] = np.zeros(output_length, dtype=np.float64)

    # Process each turn
    for turn in turns:
        audio = _load_turn_audio(turn, sample_rate)
        if audio is None:
            continue

        # Normalize speaker level
        if speaker_rms is not None and target_rms is not None:
            spk_rms = speaker_rms.get(turn.segment.speaker_id)
            if spk_rms and spk_rms > 0:
                audio = audio * (target_rms / spk_rms)

        # Apply overlap energy scaling
        if turn.overlap is not None:
            audio = apply_overlap_energy(audio, turn.overlap)

        onset = turn.meeting_onset_sample
        speaker_id = turn.segment.speaker_id
        spk_idx = speaker_index_map[speaker_id]

        # Save dry signal
        if mix_cfg.save_dry:
            end_idx = min(onset + len(audio), output_length)
            dry_buffers[speaker_id][onset:end_idx] += audio[:end_idx - onset]

        # Convolve with each mic's RIR and add to multichannel buffer
        for mic_idx in range(n_mics):
            rir = rirs[spk_idx, mic_idx]
            reverberant = fftconvolve(audio, rir, mode='full')

            # Place in output buffer
            end_idx = min(onset + len(reverberant), output_length)
            n_to_add = end_idx - onset
            multichannel_buffer[mic_idx, onset:end_idx] += (
                reverberant[:n_to_add])

            # Reuse mic 0 result for per-speaker reverberant output
            if mic_idx == 0 and mix_cfg.save_reverberant:
                reverberant_buffers[speaker_id][onset:end_idx] += (
                    reverberant[:n_to_add])

    # Trim trailing silence
    output_length = _find_last_nonzero(multichannel_buffer) + 1
    multichannel_buffer = multichannel_buffer[:, :output_length]

    # Generate RTTM early (needed for noise SNR computation)
    rttm = generate_rttm(turns, sample_rate)

    # Add noise to multichannel buffer before output derivation
    if noise_multichannel is not None and noise_snr_db is not None:
        # Trim or pad noise to match output length
        noise_len = noise_multichannel.shape[1]
        if noise_len > output_length:
            noise_multichannel = noise_multichannel[:, :output_length]
        elif noise_len < output_length:
            pad = np.zeros((noise_multichannel.shape[0],
                            output_length - noise_len), dtype=np.float64)
            noise_multichannel = np.concatenate(
                [noise_multichannel, pad], axis=1)
        add_noise_to_buffer(multichannel_buffer, noise_multichannel,
                            noise_snr_db, rttm, sample_rate)

    # Build output dict
    result = {'n_samples': output_length}

    if 'mono' in mix_cfg.output_channels:
        mono = beamform(multichannel_buffer, meeting_room, sample_rate)
        result['mono'] = mono.astype(np.float32)

    if 'multichannel' in mix_cfg.output_channels:
        result['multichannel'] = multichannel_buffer.astype(np.float32)

    if 'stereo' in mix_cfg.output_channels:
        left, right = select_stereo_mics(meeting_room.array_config)
        result['stereo'] = np.stack(
            [multichannel_buffer[left], multichannel_buffer[right]], axis=1
        ).astype(np.float32)

    if mix_cfg.save_reverberant:
        result['reverberant'] = {
            sid: buf[:output_length].astype(np.float32)
            for sid, buf in reverberant_buffers.items()}

    if mix_cfg.save_dry:
        result['dry'] = {
            sid: buf[:output_length].astype(np.float32)
            for sid, buf in dry_buffers.items()}

    result['rttm'] = rttm

    return result


def _load_turn_audio(turn, target_sample_rate):
    """Load audio for a single turn segment.

    Supports both single-chunk and multi-chunk (concatenated)
    segments. Multi-chunk segments are joined with short silence
    pauses between them.

    Args:
      turn: Turn object with segment info.
      target_sample_rate: Desired sample rate.

    Returns:
      np.ndarray of mono audio samples, or None on failure.
    """
    chunks = turn.segment.chunks
    pause_samples = turn.segment.pause_samples

    audio_parts = []
    for chunk in chunks:
        part = _load_chunk_audio(chunk, target_sample_rate)
        if part is None:
            continue
        audio_parts.append(part)

    if not audio_parts:
        return None

    if len(audio_parts) == 1:
        return audio_parts[0]

    # Concatenate with silence pauses
    pause = np.zeros(pause_samples, dtype=np.float64)
    result = []
    for i, part in enumerate(audio_parts):
        if i > 0:
            result.append(pause)
        result.append(part)
    return np.concatenate(result)


def _load_chunk_audio(chunk, target_sample_rate):
    """Load audio for a single SegmentChunk.

    Args:
      chunk: SegmentChunk with wav_path, start_sample, end_sample.
      target_sample_rate: Desired sample rate.

    Returns:
      np.ndarray of mono audio samples, or None on failure.
    """
    try:
        waveform, sr = torchaudio.load(chunk.wav_path)
    except Exception as e:
        logger.warning("Failed to load %s: %s", chunk.wav_path, e)
        return None

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        waveform = resampler(waveform)

    audio = waveform.squeeze(0).numpy()

    # Extract the chunk boundaries
    start = chunk.start_sample
    end = chunk.end_sample
    if end > len(audio):
        end = len(audio)
    if start >= end:
        return None

    return audio[start:end]


def _find_last_nonzero(buffer, threshold=1e-8):
    """Find the last sample index with significant energy.

    Args:
      buffer: np.ndarray, can be 1D or 2D.
      threshold: Amplitude threshold for "non-zero".

    Returns:
      Index of last non-zero sample, or buffer length - 1.
    """
    if buffer.ndim == 2:
        energy = np.max(np.abs(buffer), axis=0)
    else:
        energy = np.abs(buffer)

    nonzero_indices = np.where(energy > threshold)[0]
    if len(nonzero_indices) == 0:
        return buffer.shape[-1] - 1
    return nonzero_indices[-1]


def apply_overlap_energy(audio, overlap):
    """Apply energy scaling for an overlap event.

    For most overlap types, multiplies by a constant scale factor with
    a short fade to avoid clicks. For 'collaborative' type, applies a
    linear ramp from 1.0 to 0.3.

    Args:
      audio: np.ndarray of audio samples.
      overlap: OverlapSpec instance.

    Returns:
      Scaled audio (new array, does not modify input).
    """
    audio = audio.copy()
    fade_samples = min(160, len(audio) // 4)  # ~10ms at 16kHz

    if overlap.energy_scale == 'ramp':
        # Linear ramp from 1.0 to 0.3
        ramp = np.linspace(1.0, 0.3, len(audio))
        audio *= ramp
    else:
        scale = float(overlap.energy_scale)
        audio *= scale

        # Apply fade-in to avoid click at onset
        if fade_samples > 0:
            fade_in = np.linspace(0.0, 1.0, fade_samples)
            audio[:fade_samples] *= fade_in

    return audio


def generate_rttm(turns, sample_rate):
    """Extract ground truth timing from the turn list.

    Args:
      turns: List of Turn objects.
      sample_rate: Sample rate for time conversion.

    Returns:
      List of (speaker_id, start_seconds, duration_seconds) tuples,
        sorted by start time.
    """
    entries = []
    for turn in turns:
        start_s = turn.meeting_onset_sample / sample_rate
        duration_s = turn.segment.duration_samples / sample_rate
        entries.append((turn.segment.speaker_id, start_s, duration_s))

    entries.sort(key=lambda x: x[1])
    return entries


def write_rttm(rttm_entries, filepath, meeting_id='meeting'):
    """Write RTTM format file.

    Format: SPEAKER <file> 1 <start> <dur> <NA> <NA> <spk> <NA> <NA>

    Args:
      rttm_entries: List of (speaker_id, start_s, duration_s) tuples.
      filepath: Output file path.
      meeting_id: Recording ID for the RTTM file field.
    """
    with open(filepath, 'w') as f:
        for speaker_id, start_s, duration_s in rttm_entries:
            f.write("SPEAKER %s 1 %.3f %.3f <NA> <NA> %s <NA> <NA>\n" % (
                meeting_id, start_s, duration_s, speaker_id))


def write_meeting_outputs(result, output_dir, meeting_id, mix_cfg):
    """Write all meeting outputs to disk.

    Creates the output directory and writes WAV files, RTTM, and
    optional per-speaker files.

    Args:
      result: Dict returned by mix_meeting().
      output_dir: Directory to write into.
      meeting_id: Meeting identifier for filenames.
      mix_cfg: MixConfig for sample rate info.
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_rate = mix_cfg.sample_rate

    if 'mono' in result:
        sf.write(os.path.join(output_dir, 'mixed.wav'),
                 result['mono'], sample_rate)

    if 'stereo' in result:
        sf.write(os.path.join(output_dir, 'stereo.wav'),
                 result['stereo'], sample_rate)

    if 'multichannel' in result:
        # soundfile expects (n_samples, n_channels)
        sf.write(os.path.join(output_dir, 'multichannel.wav'),
                 result['multichannel'].T, sample_rate)

    # Write RTTM
    rttm_path = os.path.join(output_dir, 'meeting.rttm')
    write_rttm(result['rttm'], rttm_path, meeting_id)

    # Write per-speaker files
    if 'reverberant' in result or 'dry' in result:
        spk_dir = os.path.join(output_dir, 'per_speaker')
        os.makedirs(spk_dir, exist_ok=True)

        if 'reverberant' in result:
            for spk_id, audio in result['reverberant'].items():
                path = os.path.join(spk_dir,
                                    '%s_reverberant.wav' % spk_id)
                sf.write(path, audio, sample_rate)

        if 'dry' in result:
            for spk_id, audio in result['dry'].items():
                path = os.path.join(spk_dir, '%s_dry.wav' % spk_id)
                sf.write(path, audio, sample_rate)
