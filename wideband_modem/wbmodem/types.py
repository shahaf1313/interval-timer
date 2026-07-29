"""Core data types shared by the detector, the extractor and the I/O layer.

Everything the rest of the package passes around is defined here so that a
detection produced by one stage can be serialised, stored and re-loaded by
another without importing the heavy DSP modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "Detection",
    "SignalStream",
    "TruthSignal",
]


def _round(value: float, digits: int = 6) -> float:
    return float(np.round(float(value), digits))


@dataclass
class Detection:
    """A single time/frequency region flagged by the energy detector.

    Frequencies are absolute (they include the capture centre frequency) and
    times are seconds relative to the first sample of the capture.
    """

    center_freq_hz: float
    bandwidth_hz: float
    t_start_s: float
    t_end_s: float

    # Band edges of the occupied bandwidth measurement.
    f_low_hz: float = 0.0
    f_high_hz: float = 0.0
    # Bandwidth before the window-smearing correction is removed.
    bandwidth_raw_hz: float = 0.0
    # Power-weighted mean frequency; differs from ``center_freq_hz`` for
    # asymmetric spectra (e.g. a carrier next to a modulated shoulder).
    centroid_freq_hz: float = 0.0

    snr_db: float = 0.0
    peak_snr_db: float = 0.0
    power_dbfs: float = -np.inf
    noise_dbfs_per_hz: float = -np.inf

    # Bookkeeping that helps a downstream classifier decide what it is looking
    # at before it has demodulated anything.
    n_pixels: int = 0
    fill_ratio: float = 0.0
    is_continuous: bool = False
    truncated_start: bool = False
    truncated_end: bool = False

    detection_id: int = 0
    label: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.t_end_s - self.t_start_s

    @property
    def f_low_edge_hz(self) -> float:
        return self.center_freq_hz - 0.5 * self.bandwidth_hz

    @property
    def f_high_edge_hz(self) -> float:
        return self.center_freq_hz + 0.5 * self.bandwidth_hz

    def overlaps(self, other: "Detection") -> bool:
        """True when the two detections share time *and* frequency support."""
        return (
            self.t_start_s < other.t_end_s
            and other.t_start_s < self.t_end_s
            and self.f_low_hz < other.f_high_hz
            and other.f_low_hz < self.f_high_hz
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration_s"] = self.duration_s
        return {k: (_round(v) if isinstance(v, float) else v) for k, v in data.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"#{self.detection_id:<3d} fc={self.center_freq_hz / 1e6:12.6f} MHz  "
            f"bw={self.bandwidth_hz / 1e3:9.2f} kHz  "
            f"t=[{self.t_start_s * 1e3:9.3f}, {self.t_end_s * 1e3:9.3f}] ms  "
            f"snr={self.snr_db:6.1f} dB"
            + ("  [continuous]" if self.is_continuous else "")
        )


@dataclass
class SignalStream:
    """A single detected signal, filtered out of the wideband capture.

    ``samples`` is complex baseband: the detection's centre frequency has been
    mixed to DC and the stream has been decimated to a rate that is just wide
    enough for its bandwidth. The metadata carried alongside is what a
    downstream demodulator needs in order to know what it received.
    """

    samples: np.ndarray
    sample_rate_hz: float
    center_freq_hz: float
    bandwidth_hz: float
    t_start_s: float
    t_end_s: float
    snr_db: float = 0.0
    detection_id: int = 0
    source_center_freq_hz: float = 0.0
    source_sample_rate_hz: float = 0.0
    decimation: int = 1
    detection: Detection | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate_hz

    @property
    def n_samples(self) -> int:
        return int(len(self.samples))

    @property
    def samples_per_symbol_for(self) -> Any:  # pragma: no cover - convenience
        def _f(symbol_rate_hz: float) -> float:
            return self.sample_rate_hz / symbol_rate_hz

        return _f

    def time_axis(self) -> np.ndarray:
        """Absolute capture-relative timestamps for every sample."""
        return self.t_start_s + np.arange(len(self.samples)) / self.sample_rate_hz

    def metadata(self) -> dict[str, Any]:
        meta = {
            "detection_id": self.detection_id,
            "center_freq_hz": _round(self.center_freq_hz),
            "bandwidth_hz": _round(self.bandwidth_hz),
            "t_start_s": _round(self.t_start_s, 9),
            "t_end_s": _round(self.t_end_s, 9),
            "duration_s": _round(self.duration_s, 9),
            "sample_rate_hz": _round(self.sample_rate_hz),
            "n_samples": self.n_samples,
            "snr_db": _round(self.snr_db, 3),
            "decimation": self.decimation,
            "source_center_freq_hz": _round(self.source_center_freq_hz),
            "source_sample_rate_hz": _round(self.source_sample_rate_hz),
            "freq_offset_from_source_hz": _round(
                self.center_freq_hz - self.source_center_freq_hz
            ),
            "dtype": str(self.samples.dtype),
        }
        meta.update(self.extra)
        if self.detection is not None:
            meta["detection"] = self.detection.to_dict()
        return meta

    def save(self, path: str | Path) -> Path:
        """Write ``<path>.cf32`` plus a ``<path>.json`` metadata sidecar."""
        path = Path(path)
        iq_path = path.with_suffix(".cf32")
        iq_path.parent.mkdir(parents=True, exist_ok=True)
        self.samples.astype(np.complex64).tofile(iq_path)
        meta_path = path.with_suffix(".json")
        meta = self.metadata()
        meta["data_file"] = iq_path.name
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return iq_path

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"stream #{self.detection_id} fc={self.center_freq_hz / 1e6:.6f} MHz "
            f"bw={self.bandwidth_hz / 1e3:.2f} kHz fs={self.sample_rate_hz / 1e3:.2f} kHz "
            f"n={self.n_samples}"
        )


@dataclass
class TruthSignal:
    """Ground truth emitted by the scene generator, used to score a detector."""

    center_freq_hz: float
    bandwidth_hz: float
    t_start_s: float
    t_end_s: float
    snr_db: float
    kind: str = "unknown"
    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.t_end_s - self.t_start_s

    @property
    def f_low_hz(self) -> float:
        return self.center_freq_hz - 0.5 * self.bandwidth_hz

    @property
    def f_high_hz(self) -> float:
        return self.center_freq_hz + 0.5 * self.bandwidth_hz

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration_s"] = self.duration_s
        return data
