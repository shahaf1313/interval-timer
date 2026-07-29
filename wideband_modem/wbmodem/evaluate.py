"""Scoring a detector run against ground truth.

Matching is deliberately not a strict IoU test. A tone has essentially zero
true bandwidth but is reported a couple of bins wide, and a burst's edges are
only knowable to within one analysis window, so a rectangle-IoU criterion
punishes a detector for the resolution limits of its own front end. Instead a
pair is *eligible* when the boxes overlap after both are widened by the
detector's resolution, and IoU is then used only to rank eligible pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import Detection, TruthSignal

__all__ = ["Match", "EvaluationReport", "match_detections"]


@dataclass
class Match:
    truth: TruthSignal
    detection: Detection
    iou: float

    @property
    def freq_error_hz(self) -> float:
        return self.detection.center_freq_hz - self.truth.center_freq_hz

    @property
    def bandwidth_error_hz(self) -> float:
        return self.detection.bandwidth_hz - self.truth.bandwidth_hz

    @property
    def bandwidth_ratio(self) -> float:
        return self.detection.bandwidth_hz / max(self.truth.bandwidth_hz, 1e-9)

    @property
    def t_start_error_s(self) -> float:
        return self.detection.t_start_s - self.truth.t_start_s

    @property
    def t_end_error_s(self) -> float:
        return self.detection.t_end_s - self.truth.t_end_s


@dataclass
class EvaluationReport:
    matches: list[Match] = field(default_factory=list)
    missed: list[TruthSignal] = field(default_factory=list)
    false_alarms: list[Detection] = field(default_factory=list)

    @property
    def recall(self) -> float:
        n = len(self.matches) + len(self.missed)
        return len(self.matches) / n if n else 1.0

    @property
    def precision(self) -> float:
        n = len(self.matches) + len(self.false_alarms)
        return len(self.matches) / n if n else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def error_summary(self) -> dict[str, float]:
        if not self.matches:
            return {}
        return {
            "median_abs_freq_error_hz": float(
                np.median([abs(m.freq_error_hz) for m in self.matches])
            ),
            "median_bandwidth_ratio": float(
                np.median([m.bandwidth_ratio for m in self.matches])
            ),
            "median_abs_t_start_error_s": float(
                np.median([abs(m.t_start_error_s) for m in self.matches])
            ),
            "median_abs_t_end_error_s": float(
                np.median([abs(m.t_end_error_s) for m in self.matches])
            ),
            "median_iou": float(np.median([m.iou for m in self.matches])),
        }

    def report(self) -> str:
        lines = [
            f"matched {len(self.matches)}  missed {len(self.missed)}  "
            f"false alarms {len(self.false_alarms)}",
            f"precision {self.precision:.3f}  recall {self.recall:.3f}  f1 {self.f1:.3f}",
        ]
        errs = self.error_summary()
        if errs:
            lines.append(
                f"|df| {errs['median_abs_freq_error_hz']:.0f} Hz  "
                f"bw ratio {errs['median_bandwidth_ratio']:.2f}  "
                f"|dt_start| {errs['median_abs_t_start_error_s'] * 1e6:.0f} us  "
                f"|dt_end| {errs['median_abs_t_end_error_s'] * 1e6:.0f} us"
            )
        for t in self.missed:
            lines.append(
                f"  MISSED  {t.name:<16} fc={t.center_freq_hz / 1e6:.3f} MHz "
                f"bw={t.bandwidth_hz / 1e3:.1f} kHz snr={t.snr_db:.0f} dB"
            )
        for d in self.false_alarms:
            lines.append(
                f"  FALSE   #{d.detection_id} fc={d.center_freq_hz / 1e6:.3f} MHz "
                f"bw={d.bandwidth_hz / 1e3:.1f} kHz snr={d.snr_db:.1f} dB"
            )
        return "\n".join(lines)


def _iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """IoU of two (f_low, f_high, t_start, t_end) rectangles."""
    fl = max(a[0], b[0])
    fh = min(a[1], b[1])
    ts = max(a[2], b[2])
    te = min(a[3], b[3])
    if fh <= fl or te <= ts:
        return 0.0
    inter = (fh - fl) * (te - ts)
    area_a = (a[1] - a[0]) * (a[3] - a[2])
    area_b = (b[1] - b[0]) * (b[3] - b[2])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_detections(
    truth: list[TruthSignal],
    detections: list[Detection],
    *,
    freq_tol_hz: float = 0.0,
    time_tol_s: float = 0.0,
    min_iou: float = 0.0,
) -> EvaluationReport:
    """Greedily pair ground-truth signals with detections.

    ``freq_tol_hz`` and ``time_tol_s`` widen both boxes before the overlap
    test; set them to roughly the spectrogram's ENBW and window duration.
    """
    candidates: list[tuple[float, int, int]] = []
    for i, t in enumerate(truth):
        t_box = (
            t.f_low_hz - freq_tol_hz,
            t.f_high_hz + freq_tol_hz,
            t.t_start_s - time_tol_s,
            t.t_end_s + time_tol_s,
        )
        for j, d in enumerate(detections):
            d_box = (
                d.f_low_edge_hz - freq_tol_hz,
                d.f_high_edge_hz + freq_tol_hz,
                d.t_start_s - time_tol_s,
                d.t_end_s + time_tol_s,
            )
            score = _iou(t_box, d_box)
            if score > 0:
                tight = _iou(
                    (t.f_low_hz, t.f_high_hz, t.t_start_s, t.t_end_s),
                    (d.f_low_edge_hz, d.f_high_edge_hz, d.t_start_s, d.t_end_s),
                )
                if tight >= min_iou:
                    candidates.append((score, i, j))

    candidates.sort(reverse=True)
    used_t: set[int] = set()
    used_d: set[int] = set()
    matches: list[Match] = []
    for score, i, j in candidates:
        if i in used_t or j in used_d:
            continue
        used_t.add(i)
        used_d.add(j)
        matches.append(
            Match(
                truth=truth[i],
                detection=detections[j],
                iou=_iou(
                    (
                        truth[i].f_low_hz,
                        truth[i].f_high_hz,
                        truth[i].t_start_s,
                        truth[i].t_end_s,
                    ),
                    (
                        detections[j].f_low_edge_hz,
                        detections[j].f_high_edge_hz,
                        detections[j].t_start_s,
                        detections[j].t_end_s,
                    ),
                ),
            )
        )

    matches.sort(key=lambda m: m.truth.t_start_s)
    return EvaluationReport(
        matches=matches,
        missed=[t for i, t in enumerate(truth) if i not in used_t],
        false_alarms=[d for j, d in enumerate(detections) if j not in used_d],
    )
