"""Wideband spectrogram computation.

The detector works on a power spectrogram whose scaling is calibrated: a
complex white-noise input of variance sigma^2 produces bins whose expected
value is exactly sigma^2. That property is what makes a CFAR threshold and an
absolute SNR readout possible later on, so every code path that builds a
:class:`Spectrogram` must preserve it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import signal as sps

__all__ = ["SpectrogramConfig", "Spectrogram", "compute_spectrogram"]

# Peak memory ceiling for the framing buffer, in complex samples.
_FRAME_CHUNK_SAMPLES = 1 << 22


@dataclass(frozen=True)
class SpectrogramConfig:
    """How the wideband capture is turned into a time/frequency image.

    ``nfft`` sets the frequency resolution and ``hop`` the time resolution.
    The defaults (50% overlapped Hann) are a reasonable compromise for scenes
    that mix narrowband continuous carriers with short wideband bursts.

    ``average`` coherently groups ``average`` consecutive frames by averaging
    their power. It trades time resolution for a lower detection threshold,
    which helps on weak continuous signals.
    """

    nfft: int = 1024
    hop: int | None = None  # defaults to nfft // 2
    window: str | tuple[str, float] = "hann"
    average: int = 1
    dtype: Any = np.float32

    def resolved_hop(self) -> int:
        return int(self.hop) if self.hop else max(1, self.nfft // 2)

    def validate(self) -> None:
        if self.nfft < 8:
            raise ValueError("nfft must be at least 8")
        if self.resolved_hop() < 1 or self.resolved_hop() > self.nfft:
            raise ValueError("hop must be in [1, nfft]")
        if self.average < 1:
            raise ValueError("average must be >= 1")


@dataclass
class Spectrogram:
    """A calibrated power spectrogram plus the axes needed to interpret it.

    ``power`` has shape ``(n_freq, n_time)`` and holds *linear* power per bin
    in the same units as ``|x|**2`` of the input samples.
    """

    power: np.ndarray
    freqs_hz: np.ndarray  # absolute, ascending, length n_freq
    times_s: np.ndarray  # centre of each frame, length n_time
    sample_rate_hz: float
    center_freq_hz: float
    nfft: int
    hop: int
    average: int
    enbw_bins: float  # equivalent noise bandwidth of the window, in bins
    window_name: str = "hann"

    # ---------------------------------------------------------------- axes --
    @property
    def n_freq(self) -> int:
        return int(self.power.shape[0])

    @property
    def n_time(self) -> int:
        return int(self.power.shape[1])

    @property
    def bin_width_hz(self) -> float:
        return float(self.sample_rate_hz / self.nfft)

    @property
    def enbw_hz(self) -> float:
        """Effective width of one bin — the smallest bandwidth we can resolve."""
        return float(self.enbw_bins * self.bin_width_hz)

    @property
    def frame_step_s(self) -> float:
        return float(self.hop * self.average / self.sample_rate_hz)

    @property
    def window_duration_s(self) -> float:
        return float(self.nfft / self.sample_rate_hz)

    @property
    def n_avg(self) -> int:
        """Number of averaged power samples behind each pixel."""
        return int(self.average)

    @property
    def power_db(self) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(self.power, 1e-30))

    # ------------------------------------------------------------ indexing --
    def freq_to_bin(self, freq_hz: float) -> int:
        idx = int(np.round((freq_hz - self.freqs_hz[0]) / self.bin_width_hz))
        return int(np.clip(idx, 0, self.n_freq - 1))

    def time_to_frame(self, t_s: float) -> int:
        if self.n_time <= 1:
            return 0
        idx = int(np.round((t_s - self.times_s[0]) / self.frame_step_s))
        return int(np.clip(idx, 0, self.n_time - 1))

    def crop(
        self,
        f_low_hz: float | None = None,
        f_high_hz: float | None = None,
        t_start_s: float | None = None,
        t_end_s: float | None = None,
    ) -> "Spectrogram":
        f0 = 0 if f_low_hz is None else self.freq_to_bin(f_low_hz)
        f1 = self.n_freq if f_high_hz is None else self.freq_to_bin(f_high_hz) + 1
        t0 = 0 if t_start_s is None else self.time_to_frame(t_start_s)
        t1 = self.n_time if t_end_s is None else self.time_to_frame(t_end_s) + 1
        return replace(
            self,
            power=self.power[f0:f1, t0:t1],
            freqs_hz=self.freqs_hz[f0:f1],
            times_s=self.times_s[t0:t1],
        )

    def summary(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.n_freq} bins x {self.n_time} frames | "
            f"df={self.bin_width_hz:.1f} Hz (enbw {self.enbw_hz:.1f} Hz) | "
            f"dt={self.frame_step_s * 1e6:.1f} us | "
            f"span={self.freqs_hz[0] / 1e6:.3f}..{self.freqs_hz[-1] / 1e6:.3f} MHz"
        )


def _window_taps(name: str | tuple[str, float], nfft: int) -> np.ndarray:
    return np.asarray(sps.get_window(name, nfft, fftbins=True), dtype=np.float64)


def compute_spectrogram(
    x: np.ndarray,
    sample_rate_hz: float,
    center_freq_hz: float = 0.0,
    config: SpectrogramConfig | None = None,
) -> Spectrogram:
    """Compute a calibrated power spectrogram of a complex wideband capture.

    ``x`` may be real, in which case it is treated as an analytic-free signal
    and the negative half of the spectrum is still returned (it is simply the
    mirror image), keeping the rest of the pipeline uniform.
    """
    config = config or SpectrogramConfig()
    config.validate()

    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array of samples")
    if not np.iscomplexobj(x):
        x = x.astype(np.complex64)
    elif x.dtype != np.complex64 and x.dtype != np.complex128:
        x = x.astype(np.complex64)

    nfft = int(config.nfft)
    hop = config.resolved_hop()
    if len(x) < nfft:
        raise ValueError(
            f"capture is shorter than one FFT frame ({len(x)} < {nfft} samples)"
        )

    win = _window_taps(config.window, nfft)
    win_energy = float(np.sum(win**2))
    enbw_bins = nfft * win_energy / float(np.sum(win) ** 2)

    n_frames = 1 + (len(x) - nfft) // hop
    n_groups = n_frames // config.average
    if n_groups < 1:
        raise ValueError(
            f"not enough frames ({n_frames}) for average={config.average}"
        )
    n_frames = n_groups * config.average

    power = np.empty((nfft, n_groups), dtype=config.dtype)
    frames_view = np.lib.stride_tricks.sliding_window_view(x, nfft)
    win_c = win.astype(x.dtype)

    chunk_frames = max(config.average, _FRAME_CHUNK_SAMPLES // nfft)
    chunk_frames -= chunk_frames % config.average
    for start in range(0, n_frames, chunk_frames):
        stop = min(n_frames, start + chunk_frames)
        block = frames_view[start * hop : (stop - 1) * hop + 1 : hop]
        spec = np.fft.fft(block * win_c, axis=-1)
        blk_power = (spec.real.astype(np.float64) ** 2) + (
            spec.imag.astype(np.float64) ** 2
        )
        blk_power /= win_energy
        if config.average > 1:
            blk_power = blk_power.reshape(-1, config.average, nfft).mean(axis=1)
        power[:, start // config.average : stop // config.average] = np.fft.fftshift(
            blk_power, axes=-1
        ).T.astype(config.dtype)

    freqs = center_freq_hz + np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / sample_rate_hz))
    # Timestamp each frame at the centre of its window; for averaged groups,
    # at the centre of the group.
    frame_centers = (np.arange(n_frames) * hop + (nfft - 1) / 2.0) / sample_rate_hz
    times = frame_centers.reshape(n_groups, config.average).mean(axis=1)

    return Spectrogram(
        power=power,
        freqs_hz=freqs,
        times_s=times,
        sample_rate_hz=float(sample_rate_hz),
        center_freq_hz=float(center_freq_hz),
        nfft=nfft,
        hop=hop,
        average=int(config.average),
        enbw_bins=float(enbw_bins),
        window_name=str(config.window),
    )
