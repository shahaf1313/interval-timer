"""Channelizer: turn a detection into a stream of baseband samples.

Given the wideband capture and a :class:`~wbmodem.types.Detection`, this
module mixes the detection's centre frequency down to DC, low-pass filters to
its bandwidth and decimates to a rate that is just wide enough to carry it.
The result is a :class:`~wbmodem.types.SignalStream` — a self-describing chunk
of IQ that a demodulator can consume without ever seeing the wideband capture.

Two details matter for anything downstream:

* The mixer phase is referenced to the *capture's* sample index, not to the
  start of the extracted slice. Two streams cut from the same capture
  therefore share a phase reference, and re-extracting with different margins
  gives bit-identical samples.
* ``t_start_s`` on the returned stream is the true timestamp of its first
  sample, with the filter's group delay already removed, so
  ``stream.time_axis()`` lines up with the wideband timeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .types import Detection, SignalStream

__all__ = ["ExtractConfig", "extract_stream", "extract_streams"]


@dataclass(frozen=True)
class ExtractConfig:
    """Tuning for the per-signal channelizer."""

    oversample: float = 2.0
    """Output rate as a multiple of the detected bandwidth."""
    bandwidth_margin: float = 1.25
    """Filter passband as a multiple of the detected bandwidth.

    The detector reports an occupied bandwidth, which for a pulse-shaped
    carrier cuts into the roll-off. A margin keeps the signal's own skirts and
    any residual frequency offset inside the passband.
    """
    transition_frac: float = 0.25
    """Filter transition width as a fraction of the passband edge."""
    stopband_atten_db: float = 70.0
    guard_s: float = 0.0
    """Extra time kept on each side of the burst."""
    min_bandwidth_hz: float = 0.0
    """Floor on the passband, for detections narrower than one FFT bin."""
    max_taps: int = 8192
    max_decimation: int | None = None
    dtype: type = np.complex64

    def validate(self) -> None:
        if self.oversample < 1.0:
            raise ValueError("oversample must be >= 1")
        if self.bandwidth_margin < 1.0:
            raise ValueError("bandwidth_margin must be >= 1")
        if not 0.0 < self.transition_frac < 2.0:
            raise ValueError("transition_frac must be in (0, 2)")


def _design_decimator(
    sample_rate_hz: float, passband_hz: float, cfg: ExtractConfig
) -> tuple[np.ndarray, int, float]:
    """Pick a decimation factor and design the matching anti-alias filter.

    Returns ``(taps, decimation, output_rate_hz)``. The decimation factor is
    the largest one that keeps both the requested oversampling *and* the
    filter's transition band inside the output Nyquist zone, so nothing that
    survives the filter can alias.
    """
    cutoff = 0.5 * passband_hz
    transition = max(cutoff * cfg.transition_frac, 1e-9)
    # Everything below stop_edge must fit inside the output Nyquist zone.
    stop_edge = cutoff + transition
    max_dec_alias = int(np.floor(sample_rate_hz / (2.0 * stop_edge)))
    max_dec_rate = int(np.floor(sample_rate_hz / (cfg.oversample * passband_hz)))
    decim = max(1, min(max_dec_alias, max_dec_rate))
    if cfg.max_decimation:
        decim = min(decim, int(cfg.max_decimation))

    numtaps, beta = sps.kaiserord(cfg.stopband_atten_db, 2.0 * transition / sample_rate_hz)
    numtaps = int(min(max(numtaps, 15), cfg.max_taps))
    if numtaps % 2 == 0:
        numtaps += 1
    # A detection can be as wide as the capture itself, in which case there is
    # no room left for a transition band; keep the cutoff inside Nyquist and
    # let the filter be the widest one that is realisable.
    edge = min(cutoff + 0.5 * transition, 0.499 * sample_rate_hz)
    taps = sps.firwin(
        numtaps,
        edge,
        window=("kaiser", beta),
        pass_zero=True,
        fs=sample_rate_hz,
    )
    # Single-precision taps keep the whole filtering path in complex64, which
    # matters on a multi-million-sample capture.
    return taps.astype(np.float32), decim, sample_rate_hz / decim


def extract_stream(
    x: np.ndarray,
    sample_rate_hz: float,
    center_freq_hz: float,
    detection: Detection,
    config: ExtractConfig | None = None,
) -> SignalStream:
    """Filter one detection out of the wideband capture.

    ``center_freq_hz`` is the centre frequency of the capture ``x``; the
    detection's own centre frequency is absolute, as reported by the detector.
    """
    cfg = config or ExtractConfig()
    cfg.validate()

    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array of samples")
    n_total = len(x)

    passband = max(
        detection.bandwidth_hz * cfg.bandwidth_margin,
        cfg.min_bandwidth_hz,
        sample_rate_hz / n_total,
    )
    passband = min(passband, sample_rate_hz * 0.98)
    taps, decim, out_rate = _design_decimator(sample_rate_hz, passband, cfg)
    half_delay = (len(taps) - 1) // 2
    # An output sample only has full support once the filter has ingested
    # numtaps-1 inputs, so that is the runway the slice needs at each end —
    # half the filter length would leave a transient in the kept samples.
    runway = len(taps) - 1

    # Time window: the burst, plus the caller's guard, plus the runway.
    n_start = int(np.floor((detection.t_start_s - cfg.guard_s) * sample_rate_hz))
    n_stop = int(np.ceil((detection.t_end_s + cfg.guard_s) * sample_rate_hz))
    n_start = int(np.clip(n_start - runway, 0, max(0, n_total - 1)))
    n_stop = int(np.clip(n_stop + runway, n_start + 1, n_total))
    # Snap the slice to the decimation lattice. Together with the mixer's
    # absolute phase reference this pins the output grid to the capture, so
    # the same detection always yields the same samples no matter how much
    # guard time was asked for.
    n_start -= n_start % decim

    freq_offset = detection.center_freq_hz - center_freq_hz
    # Reduce the phase argument modulo one cycle before evaluating the
    # sinusoids: the raw product grows to millions of radians over a long
    # capture, and single precision would lose the phase entirely.
    cycles = np.arange(n_start, n_stop, dtype=np.float64) * (
        freq_offset / sample_rate_hz
    )
    cycles -= np.floor(cycles)
    phase = (-2.0 * np.pi) * cycles
    lo = np.empty(len(cycles), dtype=np.complex64)
    lo.real = np.cos(phase)
    lo.imag = np.sin(phase)
    mixed = x[n_start:n_stop].astype(np.complex64) * lo

    filtered = sps.upfirdn(taps, mixed, up=1, down=decim)

    # upfirdn output sample m draws on inputs [m*decim - runway, m*decim] and
    # sits at input index m*decim - half_delay once its group delay is removed.
    # Drop the outputs at each end whose support is not fully inside the slice.
    drop_head = int(np.ceil(runway / decim))
    drop_tail = int(np.ceil(runway / decim))
    if len(filtered) > drop_head + drop_tail:
        kept = filtered[drop_head : len(filtered) - drop_tail]
    else:  # pragma: no cover - degenerate, very short burst
        kept = filtered[drop_head : drop_head + 1]
        drop_tail = 0

    first_input_index = n_start + drop_head * decim - half_delay
    t_start = first_input_index / sample_rate_hz
    t_end = t_start + len(kept) / out_rate

    return SignalStream(
        samples=np.asarray(kept, dtype=cfg.dtype),
        sample_rate_hz=float(out_rate),
        center_freq_hz=float(detection.center_freq_hz),
        bandwidth_hz=float(detection.bandwidth_hz),
        t_start_s=float(t_start),
        t_end_s=float(t_end),
        snr_db=float(detection.snr_db),
        detection_id=int(detection.detection_id),
        source_center_freq_hz=float(center_freq_hz),
        source_sample_rate_hz=float(sample_rate_hz),
        decimation=int(decim),
        detection=detection,
        extra={
            "filter_taps": int(len(taps)),
            "filter_passband_hz": float(passband),
            "stopband_atten_db": float(cfg.stopband_atten_db),
            "oversample_actual": float(out_rate / max(detection.bandwidth_hz, 1e-9)),
        },
    )


def extract_streams(
    x: np.ndarray,
    sample_rate_hz: float,
    center_freq_hz: float,
    detections: list[Detection],
    config: ExtractConfig | None = None,
) -> list[SignalStream]:
    """Filter every detection out of the capture, in detection order."""
    cfg = config or ExtractConfig()
    return [
        extract_stream(x, sample_rate_hz, center_freq_hz, det, cfg)
        for det in detections
    ]
