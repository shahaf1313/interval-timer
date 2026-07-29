"""Synthetic wideband scenes, for testing and for the demo.

The generator builds a capture that looks like the problem the detector is
meant to solve: several emitters at once, mixing continuous carriers with
short bursts, over a wide range of bandwidths and modulations. Every source
also records ground truth (:class:`~wbmodem.types.TruthSignal`) so a detector
run can be scored automatically — see :mod:`wbmodem.evaluate`.

SNR is always defined as in-band SNR: the source's power divided by the noise
power inside the source's own bandwidth. That keeps "10 dB" meaningful
regardless of how wide the capture is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

from .types import TruthSignal

__all__ = ["SceneBuilder", "rrc_taps", "default_scene"]


def rrc_taps(sps_ratio: float, span_symbols: int = 10, beta: float = 0.35) -> np.ndarray:
    """Root-raised-cosine pulse, unit energy.

    ``sps_ratio`` may be fractional, which lets a scene use any symbol rate
    without resampling the whole capture.
    """
    n = int(round(span_symbols * sps_ratio))
    if n % 2 == 0:
        n += 1
    t = (np.arange(n) - (n - 1) / 2.0) / float(sps_ratio)
    beta = float(np.clip(beta, 1e-6, 1.0))

    h = np.empty_like(t)
    # Main branch, with the two removable singularities handled separately.
    denom = np.pi * t * (1.0 - (4.0 * beta * t) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = (
            np.sin(np.pi * t * (1.0 - beta))
            + 4.0 * beta * t * np.cos(np.pi * t * (1.0 + beta))
        ) / denom
    h[np.isclose(t, 0.0)] = 1.0 - beta + 4.0 * beta / np.pi
    sing = np.isclose(np.abs(t), 1.0 / (4.0 * beta), atol=1e-8)
    if np.any(sing):
        h[sing] = (beta / np.sqrt(2.0)) * (
            (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
            + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
        )
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    return h / np.sqrt(np.sum(h**2))


def _ramp(n: int, ramp_len: int) -> np.ndarray:
    """Raised-cosine on/off ramp, so burst edges do not splatter."""
    env = np.ones(n)
    ramp_len = int(min(ramp_len, n // 2))
    if ramp_len > 0:
        r = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp_len) / ramp_len))
        env[:ramp_len] = r
        env[-ramp_len:] = r[::-1]
    return env


@dataclass
class SceneBuilder:
    """Assembles a wideband capture from individual emitters."""

    sample_rate_hz: float
    duration_s: float
    center_freq_hz: float = 0.0
    noise_power: float = 1.0
    seed: int | None = 0

    _buffer: np.ndarray = field(init=False, repr=False)
    _truth: list[TruthSignal] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.n_samples = int(round(self.duration_s * self.sample_rate_hz))
        if self.n_samples < 16:
            raise ValueError("scene is too short")
        self._buffer = np.zeros(self.n_samples, dtype=np.complex128)
        self.rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------ helpers --
    @property
    def noise_psd(self) -> float:
        """Noise power per Hz."""
        return self.noise_power / self.sample_rate_hz

    def _amplitude_for(self, snr_db: float, bandwidth_hz: float) -> float:
        """Voltage scale giving the requested in-band SNR at unit signal power."""
        target_power = (10.0 ** (snr_db / 10.0)) * self.noise_psd * bandwidth_hz
        return float(np.sqrt(max(target_power, 0.0)))

    def _span(self, t_start_s: float | None, duration_s: float | None) -> tuple[int, int]:
        t0 = 0.0 if t_start_s is None else float(t_start_s)
        n0 = int(np.clip(round(t0 * self.sample_rate_hz), 0, self.n_samples - 1))
        if duration_s is None:
            n1 = self.n_samples
        else:
            n1 = int(np.clip(n0 + round(duration_s * self.sample_rate_hz), n0 + 1, self.n_samples))
        return n0, n1

    def _place(
        self,
        baseband: np.ndarray,
        n0: int,
        center_freq_hz: float,
        bandwidth_hz: float,
        snr_db: float,
        kind: str,
        name: str,
        params: dict | None = None,
    ) -> TruthSignal:
        """Normalise, scale, mix to RF and add a source into the capture."""
        n = len(baseband)
        power = float(np.mean(np.abs(baseband) ** 2))
        if power <= 0:
            raise ValueError("source has zero power")
        baseband = baseband / np.sqrt(power)
        baseband *= self._amplitude_for(snr_db, bandwidth_hz)

        idx = np.arange(n0, n0 + n, dtype=np.float64)
        offset = center_freq_hz - self.center_freq_hz
        phase = self.rng.uniform(0.0, 2.0 * np.pi)
        self._buffer[n0 : n0 + n] += baseband * np.exp(
            2j * np.pi * offset * idx / self.sample_rate_hz + 1j * phase
        )

        truth = TruthSignal(
            center_freq_hz=float(center_freq_hz),
            bandwidth_hz=float(bandwidth_hz),
            t_start_s=n0 / self.sample_rate_hz,
            t_end_s=(n0 + n) / self.sample_rate_hz,
            snr_db=float(snr_db),
            kind=kind,
            name=name or f"{kind}@{center_freq_hz / 1e6:.3f}MHz",
            params=params or {},
        )
        self._truth.append(truth)
        return truth

    def _noise(self, n: int) -> np.ndarray:
        return (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)) / np.sqrt(2.0)

    # ------------------------------------------------------------ sources --
    def add_tone(
        self,
        center_freq_hz: float,
        snr_db: float = 20.0,
        t_start_s: float | None = None,
        duration_s: float | None = None,
        nominal_bandwidth_hz: float | None = None,
        name: str = "",
    ) -> TruthSignal:
        """An unmodulated carrier — the narrowest thing the detector can see."""
        n0, n1 = self._span(t_start_s, duration_s)
        n = n1 - n0
        bw = nominal_bandwidth_hz or max(4.0 * self.sample_rate_hz / 1024.0, 1.0)
        base = np.ones(n, dtype=np.complex128)
        if duration_s is not None:
            base = base * _ramp(n, int(0.02 * n))
        return self._place(base, n0, center_freq_hz, bw, snr_db, "tone", name)

    def add_psk(
        self,
        center_freq_hz: float,
        symbol_rate_hz: float,
        order: int = 4,
        snr_db: float = 15.0,
        t_start_s: float | None = None,
        duration_s: float | None = None,
        rolloff: float = 0.35,
        name: str = "",
    ) -> TruthSignal:
        """RRC-shaped M-PSK; ``duration_s=None`` makes it continuous."""
        n0, n1 = self._span(t_start_s, duration_s)
        n = n1 - n0
        sps_ratio = self.sample_rate_hz / symbol_rate_hz
        if sps_ratio < 2.0:
            raise ValueError("symbol rate exceeds half the sample rate")

        n_symbols = int(np.ceil(n / sps_ratio)) + 16
        k = self.rng.integers(0, order, size=n_symbols)
        symbols = np.exp(2j * np.pi * k / order)

        taps = rrc_taps(sps_ratio, span_symbols=10, beta=rolloff)
        train = np.zeros(n + len(taps), dtype=np.complex128)
        pos = np.round(np.arange(n_symbols) * sps_ratio).astype(int)
        keep = pos < len(train)
        train[pos[keep]] = symbols[keep]
        base = np.convolve(train, taps)[: n + len(taps)]
        base = base[len(taps) // 2 : len(taps) // 2 + n]
        if duration_s is not None:
            base = base * _ramp(n, int(0.5 * sps_ratio))

        bw = symbol_rate_hz * (1.0 + rolloff)
        return self._place(
            base,
            n0,
            center_freq_hz,
            bw,
            snr_db,
            f"psk{order}",
            name,
            {"symbol_rate_hz": symbol_rate_hz, "rolloff": rolloff, "order": order},
        )

    def add_noise_band(
        self,
        center_freq_hz: float,
        bandwidth_hz: float,
        snr_db: float = 10.0,
        t_start_s: float | None = None,
        duration_s: float | None = None,
        name: str = "",
    ) -> TruthSignal:
        """Band-limited noise: a stand-in for an unknown wideband waveform."""
        n0, n1 = self._span(t_start_s, duration_s)
        n = n1 - n0
        taps = sps.firwin(
            255, 0.5 * bandwidth_hz, window=("kaiser", 8.0), fs=self.sample_rate_hz
        )
        raw = self._noise(n + 2 * len(taps))
        base = sps.lfilter(taps, [1.0], raw)[len(taps) : len(taps) + n]
        if duration_s is not None:
            base = base * _ramp(n, int(0.02 * n))
        return self._place(
            base, n0, center_freq_hz, bandwidth_hz, snr_db, "noise_band", name
        )

    def add_chirp(
        self,
        center_freq_hz: float,
        bandwidth_hz: float,
        t_start_s: float,
        duration_s: float,
        snr_db: float = 12.0,
        name: str = "",
    ) -> TruthSignal:
        """Linear FM sweep across ``bandwidth_hz``."""
        n0, n1 = self._span(t_start_s, duration_s)
        n = n1 - n0
        t = np.arange(n) / self.sample_rate_hz
        rate = bandwidth_hz / (n / self.sample_rate_hz)
        inst = -0.5 * bandwidth_hz * t + 0.5 * rate * t**2
        base = np.exp(2j * np.pi * inst) * _ramp(n, int(0.02 * n))
        return self._place(
            base,
            n0,
            center_freq_hz,
            bandwidth_hz,
            snr_db,
            "chirp",
            name,
            {"sweep_rate_hz_per_s": rate},
        )

    def add_fsk(
        self,
        center_freq_hz: float,
        symbol_rate_hz: float,
        deviation_hz: float,
        order: int = 2,
        snr_db: float = 15.0,
        t_start_s: float | None = None,
        duration_s: float | None = None,
        name: str = "",
    ) -> TruthSignal:
        """Continuous-phase M-FSK."""
        n0, n1 = self._span(t_start_s, duration_s)
        n = n1 - n0
        sps_ratio = self.sample_rate_hz / symbol_rate_hz
        n_symbols = int(np.ceil(n / sps_ratio)) + 1
        levels = (np.arange(order) - (order - 1) / 2.0) * 2.0 / max(order - 1, 1)
        k = self.rng.integers(0, order, size=n_symbols)
        freqs = levels[k] * deviation_hz
        per_sample = np.repeat(freqs, int(np.ceil(sps_ratio)))[:n]
        if len(per_sample) < n:
            per_sample = np.pad(per_sample, (0, n - len(per_sample)), mode="edge")
        phase = 2.0 * np.pi * np.cumsum(per_sample) / self.sample_rate_hz
        base = np.exp(1j * phase) * _ramp(n, int(0.5 * sps_ratio))
        bw = 2.0 * deviation_hz + symbol_rate_hz
        return self._place(
            base,
            n0,
            center_freq_hz,
            bw,
            snr_db,
            f"fsk{order}",
            name,
            {"symbol_rate_hz": symbol_rate_hz, "deviation_hz": deviation_hz},
        )

    def add_hopper(
        self,
        freqs_hz: list[float],
        symbol_rate_hz: float,
        dwell_s: float,
        t_start_s: float,
        n_hops: int | None = None,
        snr_db: float = 15.0,
        gap_s: float = 0.0,
        order: int = 4,
        name: str = "hop",
    ) -> list[TruthSignal]:
        """A frequency hopper: one PSK burst per dwell, cycling through freqs.

        Each dwell is recorded as its own ground-truth signal, which is also
        how an energy detector sees it before any tracking layer runs.
        """
        n_hops = n_hops or len(freqs_hz)
        out = []
        t = t_start_s
        for i in range(n_hops):
            f = freqs_hz[i % len(freqs_hz)]
            out.append(
                self.add_psk(
                    center_freq_hz=f,
                    symbol_rate_hz=symbol_rate_hz,
                    order=order,
                    snr_db=snr_db,
                    t_start_s=t,
                    duration_s=dwell_s,
                    name=f"{name}{i}",
                )
            )
            t += dwell_s + gap_s
        return out

    # -------------------------------------------------------------- build --
    def build(self, dtype=np.complex64) -> tuple[np.ndarray, list[TruthSignal]]:
        """Return ``(samples, ground_truth)`` with noise added."""
        noise = self._noise(self.n_samples) * np.sqrt(self.noise_power)
        x = (self._buffer + noise).astype(dtype)
        truth = sorted(self._truth, key=lambda s: (s.t_start_s, s.center_freq_hz))
        return x, truth


def default_scene(
    sample_rate_hz: float = 2_000_000.0,
    duration_s: float = 0.05,
    center_freq_hz: float = 100_000_000.0,
    seed: int | None = 7,
) -> tuple[np.ndarray, list[TruthSignal], SceneBuilder]:
    """A representative scene: continuous carriers, bursts and a hopper."""
    sb = SceneBuilder(
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
        center_freq_hz=center_freq_hz,
        seed=seed,
    )
    fc = center_freq_hz

    # Continuous, always on for the whole capture.
    sb.add_psk(fc - 700e3, symbol_rate_hz=50e3, order=4, snr_db=18.0, name="ctrl-qpsk")
    sb.add_tone(fc + 812e3, snr_db=25.0, name="beacon")
    sb.add_noise_band(fc + 300e3, bandwidth_hz=180e3, snr_db=12.0, name="wideband-cont")

    # Bursts of different widths and lengths.
    sb.add_psk(
        fc - 250e3,
        symbol_rate_hz=200e3,
        order=8,
        snr_db=16.0,
        t_start_s=0.006,
        duration_s=0.008,
        name="burst-8psk",
    )
    sb.add_fsk(
        fc + 60e3,
        symbol_rate_hz=20e3,
        deviation_hz=15e3,
        snr_db=17.0,
        t_start_s=0.024,
        duration_s=0.011,
        name="burst-fsk",
    )
    sb.add_chirp(
        fc - 480e3,
        bandwidth_hz=120e3,
        t_start_s=0.036,
        duration_s=0.006,
        snr_db=14.0,
        name="chirp",
    )
    sb.add_hopper(
        freqs_hz=[fc + 520e3, fc + 640e3, fc + 430e3],
        symbol_rate_hz=40e3,
        dwell_s=0.004,
        gap_s=0.003,
        t_start_s=0.010,
        n_hops=4,
        snr_db=15.0,
    )

    x, truth = sb.build()
    return x, truth, sb
