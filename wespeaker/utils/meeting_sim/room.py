"""Room simulation for meeting audio generation.

Sets up a pyroomacoustics shoebox room with a linear microphone array
and configurable speaker positions, then computes room impulse
responses and performs beamforming.
"""

import logging

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pyroomacoustics as pra


logger = logging.getLogger('meeting_sim.room')


class RoomConfig:
    """Configuration for the shoebox room.

    Attributes:
      length: Room length in meters (x-axis).
      width: Room width in meters (y-axis).
      height: Room height in meters (z-axis).
      rt60: Reverberation time in seconds.
    """

    def __init__(self, length=4.0, width=3.0, height=2.4, rt60=0.4):
        self.length = length
        self.width = width
        self.height = height
        self.rt60 = rt60


class ArrayConfig:
    """Configuration for the linear microphone array.

    Attributes:
      n_mics: Number of microphone elements.
      spacing: Distance between adjacent elements in meters.
    """

    def __init__(self, n_mics=8, spacing=0.04):
        self.n_mics = n_mics
        self.spacing = spacing


class MeetingRoom:
    """A configured room with microphone array and speaker positions.

    Attributes:
      room: pyroomacoustics ShoeBox room object.
      mic_center: Center position of the microphone array (3D).
      speaker_positions: List of 3D position arrays, one per speaker.
      room_config: The RoomConfig used to create this room.
      array_config: The ArrayConfig used.
    """

    def __init__(self, room, mic_center, speaker_positions, room_config,
                 array_config):
        self.room = room
        self.mic_center = mic_center
        self.speaker_positions = speaker_positions
        self.room_config = room_config
        self.array_config = array_config


# Default speaker positions (before jitter):
# Speakers 0,1 sit close together on one side of the room,
# Speaker 2 sits on the opposite side.
DEFAULT_POSITIONS = [
    np.array([1.5, 1.0, 1.3]),  # Speaker 0 (close)
    np.array([2.5, 1.0, 1.3]),  # Speaker 1 (close)
    np.array([2.0, 2.0, 1.3]),  # Speaker 2 (far)
]


def create_room(room_cfg=None, array_cfg=None, n_speakers=3,
                speaker_position_jitter=0.0, configured_positions=None,
                rng=None, sample_rate=16000):
    """Create a shoebox room with a mic array and speaker positions.

    Places an 8-element linear microphone array at one end of the room,
    facing into the room. Speakers are positioned with two close
    together and one further away, with optional jitter.

    Args:
      room_cfg: RoomConfig instance. Uses defaults if None.
      array_cfg: ArrayConfig instance. Uses defaults if None.
      n_speakers: Number of speakers to place in the room.
      speaker_position_jitter: Half-width of uniform position jitter
        in meters, applied independently to each speaker's x/y/z.
      configured_positions: Optional list of [x, y, z] positions from
        config. If provided, overrides default positions. Each entry
        is a list or array of 3 floats.
      rng: numpy random Generator.
      sample_rate: Audio sample rate for RIR computation.

    Returns:
      MeetingRoom instance.
    """
    if room_cfg is None:
        room_cfg = RoomConfig()
    if array_cfg is None:
        array_cfg = ArrayConfig()
    if rng is None:
        rng = np.random.default_rng()

    # Compute room materials from RT60
    e_absorption, max_order = pra.inverse_sabine(
        room_cfg.rt60,
        [room_cfg.length, room_cfg.width, room_cfg.height])

    # Create shoebox room
    room = pra.ShoeBox(
        [room_cfg.length, room_cfg.width, room_cfg.height],
        fs=sample_rate,
        materials=pra.Material(e_absorption),
        max_order=max_order,
        air_absorption=True,
    )

    # Place linear microphone array at one end of the room
    # Array is along the y-axis, centered, at the x=0.1 wall
    mic_center = np.array([0.1, room_cfg.width / 2.0, 1.5])
    array_length = (array_cfg.n_mics - 1) * array_cfg.spacing
    mic_positions = np.zeros((3, array_cfg.n_mics))
    for i in range(array_cfg.n_mics):
        mic_positions[0, i] = mic_center[0]
        mic_positions[1, i] = (mic_center[1]
                               - array_length / 2.0
                               + i * array_cfg.spacing)
        mic_positions[2, i] = mic_center[2]

    room.add_microphone_array(mic_positions)

    # Place speakers
    speaker_positions = _compute_speaker_positions(
        n_speakers, room_cfg, speaker_position_jitter,
        configured_positions, rng)

    # Add sources (we'll use these positions for RIR computation)
    # Add a dummy signal -- actual audio will be convolved separately
    dummy_signal = np.zeros(1)
    for pos in speaker_positions:
        room.add_source(pos, signal=dummy_signal)

    logger.info("Created room: %.1f x %.1f x %.1f m, RT60=%.2f s, "
                "%d speakers, %d mics",
                room_cfg.length, room_cfg.width, room_cfg.height,
                room_cfg.rt60, n_speakers, array_cfg.n_mics)

    return MeetingRoom(room, mic_center, speaker_positions,
                       room_cfg, array_cfg)


def _auto_positions(n_speakers, room_cfg):
    """Generate automatic table-seating positions for n_speakers.

    Speakers are assigned to three zones:
      - Side A: y = width/4, running along the near side of the table.
      - Side B: y = 3*width/4, running along the far side.
      - Far end: x = length-0.5, running across the end opposite the mic.

    For n <= 6, speakers alternate A/B (even indices -> A, odd -> B).
    For 7 <= n <= 9, 6 speakers fill sides A and B, the rest go to the
    far end.
    For n > 9, speakers are split as evenly as possible across all three
    zones (A gets any +1 remainder first, B gets any +2 remainder).

    Within each zone, x (or y for the far end) positions are centred on
    the room midpoint and spaced 1.0 m apart.

    Args:
      n_speakers: Number of speakers to place.
      room_cfg: RoomConfig for room dimensions.

    Returns:
      List of np.ndarray positions (3D), length n_speakers.
    """
    cx = room_cfg.length / 2.0
    cy_a = room_cfg.width / 4.0
    cy_b = 3.0 * room_cfg.width / 4.0
    far_x = room_cfg.length - 0.5
    cy_mid = room_cfg.width / 2.0
    dx = 1.0
    z = 1.3

    def _row_x(count):
        """X positions for `count` seats, centred on cx."""
        offsets = np.arange(count) - (count - 1) / 2.0
        return cx + offsets * dx

    def _row_y(count):
        """Y positions for `count` seats at the far end, centred on cy_mid."""
        offsets = np.arange(count) - (count - 1) / 2.0
        return cy_mid + offsets * dx

    if n_speakers <= 6:
        n_a = (n_speakers + 1) // 2
        n_b = n_speakers // 2
        n_far = 0
    elif n_speakers <= 9:
        n_a, n_b = 3, 3
        n_far = n_speakers - 6
    else:
        n_far = n_speakers // 3
        n_a = n_speakers // 3 + (1 if n_speakers % 3 >= 1 else 0)
        n_b = n_speakers // 3 + (1 if n_speakers % 3 >= 2 else 0)

    xs_a = _row_x(n_a)
    xs_b = _row_x(n_b)
    ys_far = _row_y(n_far)

    # Interleave sides A and B (A first, then B) to assign position indices
    # 0,2,4,...  -> side A  and  1,3,5,... -> side B
    positions = [None] * n_speakers
    for i in range(n_a):
        positions[2 * i] = np.array([xs_a[i], cy_a, z])
    for i in range(n_b):
        positions[2 * i + 1] = np.array([xs_b[i], cy_b, z])
    for i in range(n_far):
        positions[n_a + n_b + i] = np.array([far_x, ys_far[i], z])

    return positions


def _compute_speaker_positions(n_speakers, room_cfg, jitter,
                               configured_positions, rng):
    """Compute speaker positions with optional jitter.

    If configured_positions are provided, uses those directly (must
    supply at least n_speakers entries).  For n_speakers == 3 with no
    configured positions, uses DEFAULT_POSITIONS.  For any other count,
    generates a table-seating layout via _auto_positions.

    Args:
      n_speakers: Number of speakers.
      room_cfg: RoomConfig for bounds checking.
      jitter: Position jitter half-width in meters.
      configured_positions: Optional list of [x, y, z] positions from
        config. If provided, must contain >= n_speakers entries.
      rng: numpy random Generator.

    Returns:
      List of np.ndarray positions (3D).

    Raises:
      ValueError: If configured_positions is provided but contains
        fewer than n_speakers entries.
    """
    if configured_positions is not None:
        if len(configured_positions) < n_speakers:
            raise ValueError(
                f"configured_positions has {len(configured_positions)} "
                f"entries but n_speakers={n_speakers}")
        base_positions = [np.array(p, dtype=float)
                          for p in configured_positions[:n_speakers]]
    elif n_speakers == 3:
        base_positions = [p.copy() for p in DEFAULT_POSITIONS]
    else:
        base_positions = _auto_positions(n_speakers, room_cfg)

    # Apply jitter and clamp to room bounds
    positions = []
    margin = 0.2  # minimum distance from walls
    for pos in base_positions:
        if jitter > 0:
            jittered = pos + rng.uniform(-jitter, jitter, size=3)
        else:
            jittered = pos.copy()

        # Clamp to room bounds
        jittered[0] = np.clip(
            jittered[0], margin, room_cfg.length - margin)
        jittered[1] = np.clip(
            jittered[1], margin, room_cfg.width - margin)
        jittered[2] = np.clip(jittered[2], 1.0, room_cfg.height - 0.3)
        positions.append(jittered)

    return positions


def plot_room_topdown(meeting_room):
    """Plot a bird's-eye (x-y) view of the room, mic array, and speakers.

    Args:
      meeting_room: MeetingRoom instance.

    Returns:
      matplotlib.figure.Figure with a single axes showing the top-down layout.
    """
    rc = meeting_room.room_config
    fig, ax = plt.subplots(figsize=(6, 5))

    # Room boundary
    ax.add_patch(patches.Rectangle(
        (0, 0), rc.length, rc.width,
        linewidth=2, edgecolor='black', facecolor='none'))

    # Microphone array positions (columns of the mic array matrix)
    mic_pos = meeting_room.room.mic_array.R  # shape (3, n_mics)
    ax.scatter(mic_pos[0], mic_pos[1], marker='s', c='blue', s=40, zorder=3)
    ax.annotate('Mic array', (mic_pos[0, 0], mic_pos[1, 0]),
                textcoords='offset points', xytext=(5, -10))

    # Speaker positions
    for i, pos in enumerate(meeting_room.speaker_positions):
        ax.scatter(pos[0], pos[1], marker='o', c='red', s=80, zorder=3)
        ax.annotate(f'Spk {i}', (pos[0], pos[1]),
                    textcoords='offset points', xytext=(5, 5))

    ax.set_xlim(-0.3, rc.length + 0.3)
    ax.set_ylim(-0.3, rc.width + 0.3)
    ax.set_aspect('equal')
    ax.set_xlabel('length (m)')
    ax.set_ylabel('width (m)')
    ax.set_title('Room layout (top-down)')
    return fig


def select_stereo_mics(array_cfg):
    """Select the mic pair closest to a 17 cm stereo baseline.

    Finds the spacing k (in number of elements) whose physical separation
    is nearest to 0.17 m, then picks the most-centered pair of that
    width. Ties in centrality are broken by choosing the lower first
    index (closer to mic 0).

    Args:
      array_cfg: ArrayConfig instance.

    Returns:
      Tuple (left_idx, right_idx) of mic indices, where left_idx < right_idx.
    """
    target = 0.17  # meters
    n = array_cfg.n_mics
    k = int(round(target / array_cfg.spacing))
    k = max(1, min(k, n - 1))
    best_i = min(
        range(n - k),
        key=lambda i: (abs(i + k / 2.0 - (n - 1) / 2.0), i)
    )
    return best_i, best_i + k


def compute_rirs(meeting_room, sample_rate=16000):
    """Compute room impulse responses for all source-mic pairs.

    Args:
      meeting_room: MeetingRoom instance with room and sources
        configured.
      sample_rate: Sample rate (should match the room's fs).

    Returns:
      np.ndarray of shape (n_speakers, n_mics, rir_length). Each
        RIR[i, j] is the impulse response from speaker i to mic j.
    """
    meeting_room.room.compute_rir()

    n_speakers = len(meeting_room.speaker_positions)
    n_mics = meeting_room.array_config.n_mics

    # Find the maximum RIR length across all pairs
    max_len = 0
    for src_idx in range(n_speakers):
        for mic_idx in range(n_mics):
            rir = meeting_room.room.rir[mic_idx][src_idx]
            max_len = max(max_len, len(rir))

    # Pack into array
    rirs = np.zeros((n_speakers, n_mics, max_len), dtype=np.float32)
    for src_idx in range(n_speakers):
        for mic_idx in range(n_mics):
            rir = meeting_room.room.rir[mic_idx][src_idx]
            rirs[src_idx, mic_idx, :len(rir)] = rir

    logger.info("Computed RIRs: shape %s, max length %d samples (%.3f s)",
                rirs.shape, max_len, max_len / sample_rate)
    return rirs


def beamform(multichannel, meeting_room, sample_rate=16000):
    """Apply delay-and-sum beamforming to produce mono output.

    Steers the beam toward the center of the room (average of speaker
    positions) using time-delay compensation.

    Args:
      multichannel: np.ndarray of shape (n_mics, n_samples).
      meeting_room: MeetingRoom instance with mic and speaker position
        info.
      sample_rate: Sample rate of the audio.

    Returns:
      np.ndarray of shape (n_samples,) -- mono beamformed signal.
    """
    mic_center = meeting_room.mic_center
    n_mics = meeting_room.array_config.n_mics
    spacing = meeting_room.array_config.spacing

    # Steer toward the center of speaker positions
    target = np.mean(meeting_room.speaker_positions, axis=0)

    # Compute direction vector from mic center to target
    direction = target - mic_center
    direction_norm = direction / np.linalg.norm(direction)

    # Compute delay for each mic element
    speed_of_sound = 343.0  # m/s
    array_length = (n_mics - 1) * spacing
    delays_samples = np.zeros(n_mics)

    for i in range(n_mics):
        # Mic position relative to center along the array axis (y-axis)
        mic_offset = -array_length / 2.0 + i * spacing
        # Project onto direction to get path length difference
        path_diff = mic_offset * direction_norm[1]
        delays_samples[i] = path_diff / speed_of_sound * sample_rate

    # Compensate delays: shift each channel and sum
    delays_samples -= delays_samples.min()
    max_delay = int(np.ceil(delays_samples.max()))

    n_samples = multichannel.shape[1]
    output_len = n_samples + max_delay
    output = np.zeros(output_len, dtype=np.float64)

    for i in range(n_mics):
        delay = int(round(delays_samples[i]))
        output[delay:delay + n_samples] += multichannel[i]

    # Normalize by number of mics
    output /= n_mics

    # Trim to original length
    output = output[:n_samples]
    return output.astype(np.float32)
