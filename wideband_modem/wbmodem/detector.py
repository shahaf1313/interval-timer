"""Wideband energy detector.

Stage one of the modem: given a wideband capture (or a spectrogram computed
elsewhere), find every emission in the time/frequency plane and describe each
one with the parameters a downstream stage needs — centre frequency,
bandwidth, start time and end time — without knowing anything about the
modulation.

Pipeline
--------
1. Calibrated power spectrogram.
2. Per-bin noise floor estimate (see :mod:`wbmodem.noise`).
3. CFAR threshold -> binary time/frequency mask.
4. Morphological closing to bridge sub-symbol gaps in time and spectral nulls
   in frequency, so one emission stays one blob.
5. Connected-component labelling; components too small in either axis are
   dropped.
6. Per-component parameter refinement on the underlying power, rather than on
   the binary mask: occupied bandwidth from the noise-subtracted spectrum, and
   start/end from an in-band power profile.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from .noise import estimate_noise_floor, threshold_factor_from_pfa
from .spectrogram import Spectrogram, SpectrogramConfig, compute_spectrogram
from .types import Detection

__all__ = ["DetectorConfig", "DetectionResult", "EnergyDetector"]


@dataclass(frozen=True)
class DetectorConfig:
    """Tuning for the energy detector.

    All physical parameters are in Hz and seconds, so a configuration survives
    a change of FFT size or sample rate.
    """

    # --- threshold ---
    pfa: float = 1e-6
    """Per-pixel probability of false alarm used to derive the CFAR threshold."""
    threshold_db: float | None = None
    """Explicit threshold above the noise floor, in dB. Overrides ``pfa``."""

    # --- multi-resolution integration ---
    freq_scales: tuple[int, ...] | None = None
    time_scales: tuple[int, ...] | None = None
    """Box-integration sizes, in bins and frames, tried in every combination.

    A single-pixel test has no processing gain at all: one bin of one frame
    holds one sample of a chi-squared variable, so the threshold has to sit
    ~12 dB above the noise to keep false alarms rare, and any signal below
    that in-band SNR is invisible however long it transmits. Averaging a
    ``(kf, kt)`` box first cuts the threshold by roughly the square root of
    the number of independent cells in it — a few dB above the noise instead
    of twelve — which is what makes a low-SNR emission detectable at all.
    Since the gain only materialises when the box sits *inside* the emission,
    the ladder runs a range of box shapes: wide-and-short, narrow-and-long,
    and everything between, with no prior knowledge of either.

    ``None`` selects powers of two up to a sixty-fourth of the band and a
    quarter of the capture, which adapts to whatever spectrogram it is handed.
    The frequency cap is deliberately tight: a box wide enough to span two
    emitters fires everywhere between them, and doubling a box only buys about
    1.5 dB of sensitivity, so the ladder stops well before it starts
    confusing neighbours. Set both to ``(1,)`` for a plain single-pixel
    detector.
    """
    clip_snr_db: float | None = 6.0
    """Cap on each pixel, in dB over the noise floor, before box integration.

    Without it, one strong carrier dominates every box that contains it, and
    the coarse scales report a detection as wide as their own box. Capping
    leaves a weak signal spread evenly across a box untouched — that is what
    the coarse scales exist to find — while a strong narrow one contributes
    only its cap and cannot carry the box on its own. ``None`` disables it.
    """
    scale_erosion: float = 1.0
    """Halo removal at integrated scales, as a fraction of the box size."""
    valley_tolerance: float = 0.9
    """Reject an integrated box dimmer than this fraction of both the boxes one
    width away — the signature of a bridge between two emitters rather than an
    emission of its own. Set to 0 to disable.
    """
    scale_margin_db: float = 0.2
    """Extra threshold on integrated scales, to absorb noise-estimate error.

    At large box sizes the CFAR threshold sits a fraction of a dB above the
    noise floor, so a small bias in the floor estimate would light up the whole
    plane. This margin buys robustness at the cost of a little sensitivity.
    """

    # --- what counts as a signal ---
    min_bandwidth_hz: float = 0.0
    min_duration_s: float = 0.0
    min_snr_db: float = 0.0
    min_pixels: int = 3

    # --- what counts as one signal ---
    merge_gap_hz: float = 0.0
    """Spectral nulls narrower than this do not split an emission in two."""
    merge_gap_s: float = 0.0
    """Time gaps shorter than this do not split a burst in two."""
    merge_gap_bins: int = 2
    merge_gap_frames: int = 2
    """Resolution-relative floor for the two gaps above, in pixels."""
    merge_overlap: float = 0.6
    """Fuse two detections when this fraction of the smaller one is inside the
    larger. A marginal emission comes out of the mask speckled, and its
    fragments would otherwise be reported as several overlapping signals.
    Set to 0 to report every connected component separately.
    """
    keep_nested_above_db: float = 6.0
    """Never fuse a fragment that is this much stronger than its container.

    A narrowband carrier sitting inside a wide weak emission is a real signal,
    not a fragment of the thing around it, and it announces itself by having a
    much better SNR than the band it sits in.
    """

    # --- measurement ---
    occupied_bw_fraction: float = 0.99
    """Fraction of in-band energy enclosed by the reported bandwidth."""
    deconvolve_window: bool = True
    """Subtract the analysis window's ENBW from the measured bandwidth."""
    refine_time: bool = True

    # --- exclusions ---
    edge_exclude_hz: float = 0.0
    """Ignore this much of the band at each edge (anti-alias filter roll-off)."""
    dc_notch_hz: float = 0.0
    """Ignore this much of the band around the capture centre (LO leakage)."""

    # --- noise estimation ---
    noise_percentile: float = 25.0
    noise_smooth_bins: int | None = None
    noise_coarse_bins: int | None = None
    noise_iterations: int = 4

    max_detections: int = 512

    def validate(self) -> None:
        if not 0.0 < self.pfa < 1.0:
            raise ValueError("pfa must be in (0, 1)")
        if not 0.0 < self.occupied_bw_fraction <= 1.0:
            raise ValueError("occupied_bw_fraction must be in (0, 1]")
        for scales in (self.freq_scales, self.time_scales):
            if scales is not None and (not scales or min(scales) < 1):
                raise ValueError("scales must be non-empty and >= 1")

    max_scale_rungs: int = 8
    """Cap on the octaves per axis, so cost stays bounded on a long capture."""

    @staticmethod
    def _octaves(limit: int, max_rungs: int = 8) -> tuple[int, ...]:
        out, k = [1], 2
        while k <= limit and len(out) < max_rungs:
            out.append(k)
            k *= 2
        return tuple(out)

    def scale_pairs(self, n_freq: int, n_time: int) -> list[tuple[int, int]]:
        """Box shapes to test, ordered fine to coarse by area."""
        rungs = max(1, int(self.max_scale_rungs))
        freq = self.freq_scales or self._octaves(max(1, n_freq // 64), rungs)
        time = self.time_scales or self._octaves(max(1, n_time // 4), rungs)
        pairs = {
            (kf, kt)
            for kf in freq
            for kt in time
            if kf <= n_freq and kt <= n_time
        }
        pairs.add((1, 1))
        return sorted(pairs, key=lambda p: (p[0] * p[1], p))

    def max_scales(self, n_freq: int, n_time: int) -> tuple[int, int]:
        pairs = self.scale_pairs(n_freq, n_time)
        return max(p[0] for p in pairs), max(p[1] for p in pairs)


@dataclass
class DetectionResult:
    """Everything the detector produced, including the intermediate planes."""

    detections: list[Detection]
    spectrogram: Spectrogram
    noise_floor: np.ndarray
    threshold: np.ndarray
    mask: np.ndarray
    threshold_db_over_noise: float
    labels: np.ndarray | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self):
        return iter(self.detections)

    @property
    def noise_floor_db(self) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(self.noise_floor, 1e-30))

    @property
    def occupancy(self) -> float:
        """Fraction of the time/frequency plane flagged as occupied."""
        return float(np.count_nonzero(self.mask) / self.mask.size)

    def to_dict(self) -> dict[str, Any]:
        spec = self.spectrogram
        return {
            "capture": {
                "sample_rate_hz": spec.sample_rate_hz,
                "center_freq_hz": spec.center_freq_hz,
                "duration_s": float(spec.times_s[-1] - spec.times_s[0])
                + spec.frame_step_s,
                "nfft": spec.nfft,
                "hop": spec.hop,
                "average": spec.average,
                "bin_width_hz": spec.bin_width_hz,
                "enbw_hz": spec.enbw_hz,
                "frame_step_s": spec.frame_step_s,
            },
            "detector": {
                "threshold_db_over_noise": self.threshold_db_over_noise,
                "median_noise_dbfs_per_bin": float(np.median(self.noise_floor_db)),
                "occupancy": self.occupancy,
            },
            "detections": [d.to_dict() for d in self.detections],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def table(self) -> str:
        """A fixed-width report of the detections, for logs and the CLI."""
        header = (
            f"{'id':>3}  {'fc [MHz]':>13}  {'bw [kHz]':>11}  {'t0 [ms]':>10}  "
            f"{'t1 [ms]':>10}  {'dur [ms]':>10}  {'SNR [dB]':>8}  {'type':<10}"
        )
        lines = [header, "-" * len(header)]
        for d in self.detections:
            kind = "continuous" if d.is_continuous else "burst"
            lines.append(
                f"{d.detection_id:>3}  {d.center_freq_hz / 1e6:>13.6f}  "
                f"{d.bandwidth_hz / 1e3:>11.3f}  {d.t_start_s * 1e3:>10.3f}  "
                f"{d.t_end_s * 1e3:>10.3f}  {d.duration_s * 1e3:>10.3f}  "
                f"{d.snr_db:>8.1f}  {kind:<10}"
            )
        if not self.detections:
            lines.append("(no detections)")
        return "\n".join(lines)


def _interp_index(cdf: np.ndarray, target: float) -> float:
    """Fractional bin index where a cumulative sum first reaches ``target``.

    ``cdf[k]`` is the energy accumulated through bin ``k`` inclusive, so the
    returned coordinate is measured in "bin edges": 0.0 is the low edge of bin
    0 and ``len(cdf)`` is the high edge of the last bin. The cumulative sum is
    not assumed to be monotonic — it is built from noise-subtracted power, so
    it wanders slightly downwards outside the signal — hence a first-crossing
    search rather than a binary search.
    """
    n = len(cdf)
    if n == 0:
        return 0.0
    above = cdf >= target
    if not above.any():
        return float(n)
    k = int(np.argmax(above))
    prev = float(cdf[k - 1]) if k > 0 else 0.0
    width = float(cdf[k]) - prev
    frac = 0.0 if width <= 0 else (target - prev) / width
    return float(k + np.clip(frac, 0.0, 1.0))


class EnergyDetector:
    """Detects emissions in a wideband capture and measures their parameters."""

    def __init__(
        self,
        spectrogram_config: SpectrogramConfig | None = None,
        config: DetectorConfig | None = None,
    ) -> None:
        self.spectrogram_config = spectrogram_config or SpectrogramConfig()
        self.config = config or DetectorConfig()
        self.config.validate()

    # ------------------------------------------------------------- entry --
    def detect(
        self,
        x: np.ndarray,
        sample_rate_hz: float,
        center_freq_hz: float = 0.0,
    ) -> DetectionResult:
        """Detect emissions in a complex wideband capture."""
        spec = compute_spectrogram(
            x, sample_rate_hz, center_freq_hz, self.spectrogram_config
        )
        return self.detect_spectrogram(spec)

    def detect_spectrogram(self, spec: Spectrogram) -> DetectionResult:
        """Detect emissions in an already-computed spectrogram.

        The spectrogram must be calibrated the way
        :func:`wbmodem.spectrogram.compute_spectrogram` produces it: linear
        power per bin, with white noise of variance sigma^2 reading sigma^2.
        """
        cfg = self.config
        power = spec.power

        noise = estimate_noise_floor(
            power,
            n_avg=spec.n_avg,
            percentile=cfg.noise_percentile,
            smooth_bins=cfg.noise_smooth_bins,
            coarse_bins=cfg.noise_coarse_bins,
            iterations=cfg.noise_iterations,
        )

        mask, base_factor, scale_info = self._build_mask(spec, noise)
        self._apply_exclusions(mask, spec)
        mask = self._close_gaps(mask, spec)

        labels, n_labels = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
        detections = self._measure_components(spec, labels, n_labels, noise)
        detections = self._filter_and_rank(detections)

        return DetectionResult(
            detections=detections,
            spectrogram=spec,
            noise_floor=noise,
            threshold=noise * base_factor,
            mask=mask,
            threshold_db_over_noise=float(10.0 * np.log10(base_factor)),
            labels=labels,
            stats={"n_components": int(n_labels), "scales": scale_info},
        )

    # -------------------------------------------------------- mask stages --
    def _build_mask(
        self, spec: Spectrogram, noise: np.ndarray
    ) -> tuple[np.ndarray, float, list[dict[str, float]]]:
        """Threshold the spectrogram at every integration scale, fine to coarse.

        Each scale gets its own threshold, derived from how many *independent*
        noise samples its box actually contains. Neighbouring bins overlap
        through the window's main lobe and neighbouring frames overlap through
        the hop, so the raw box size overstates the averaging gain; discounting
        by ENBW and by the overlap factor keeps the false-alarm rate honest.

        Integrated scales work on a clipped copy of the plane (see
        ``clip_snr_db``), which is what keeps a strong narrow carrier from
        painting a detection as wide as the coarsest box.
        """
        cfg = self.config
        frames_per_window = max(1.0, spec.nfft / (spec.hop * spec.average))
        pairs = cfg.scale_pairs(spec.n_freq, spec.n_time)
        n_tests = len(pairs)
        margin = 10.0 ** (cfg.scale_margin_db / 10.0)

        # Work in units of the noise floor, so every scale compares against a
        # single scalar and a sloping front-end response drops out.
        snr_plane = (spec.power / noise[:, None]).astype(np.float32)
        cap = (
            None
            if cfg.clip_snr_db is None
            else float(10.0 ** (cfg.clip_snr_db / 10.0))
        )
        clipped = snr_plane if cap is None else np.minimum(snr_plane, np.float32(cap))

        mask = np.zeros(snr_plane.shape, dtype=bool)
        base_factor = 1.0
        info: list[dict[str, float]] = []
        for kf, kt in pairs:
            n_eff = (
                max(1, int(round(kf / max(spec.enbw_bins, 1.0))))
                * max(1, int(round(kt / frames_per_window)))
                * spec.n_avg
            )
            single = (kf, kt) == (1, 1)
            if cfg.threshold_db is not None:
                factor = 10.0 ** (cfg.threshold_db / 10.0)
            else:
                factor = threshold_factor_from_pfa(
                    cfg.pfa / n_tests, n_eff, clip=None if single else cap
                )
                if not single:
                    factor *= margin

            if single:
                base_factor = factor
                hits = snr_plane > factor
            else:
                integrated = ndimage.uniform_filter(
                    clipped, size=(kf, kt), mode="nearest"
                )
                hits = integrated > factor
                if hits.any():
                    if kf > 1:
                        hits &= ~self._is_valley(integrated, kf, axis=0)
                    if kt > 1:
                        hits &= ~self._is_valley(integrated, kt, axis=1)
                    hits = self._erode_scale(hits, kf, kt)
            mask |= hits
            info.append(
                {
                    "freq_bins": kf,
                    "time_frames": kt,
                    "n_indep": n_eff,
                    "threshold_db": float(10.0 * np.log10(factor)),
                    "n_pixels": int(np.count_nonzero(hits)),
                }
            )
        return mask, base_factor, info

    def _is_valley(self, integrated: np.ndarray, k: int, axis: int) -> np.ndarray:
        """Find boxes that are only lit because both their neighbours are.

        Two emitters spaced about one box apart light up every box between
        them, and since that firing region is wider than the box, erosion
        cannot clear it — the two get welded into one component with the empty
        band between them swallowed. The signature of such a bridge is
        specific: the box sits in a valley, dimmer than the box one full width
        to either side. A real emission never does. A wide one looks like its
        own neighbours, a narrow one has noise on at least one side, and the
        edge of any emission has noise on the outward side.
        """
        tol = float(self.config.valley_tolerance)
        valley = np.zeros(integrated.shape, dtype=bool)
        n = integrated.shape[axis]
        if tol <= 0.0 or 2 * k >= n:
            return valley

        # Compare each pixel against the pixels one box-width to either side,
        # using slices rather than shifted copies of the whole plane. Pixels
        # within one box of an edge keep themselves as their outward
        # neighbour, so they can never register as a valley.
        def window(start, stop):
            sl = [slice(None), slice(None)]
            sl[axis] = slice(start, stop)
            return tuple(sl)

        below = integrated[window(0, n - 2 * k)]
        above = integrated[window(2 * k, n)]
        middle = window(k, n - k)
        valley[middle] = integrated[middle] < tol * np.minimum(below, above)
        return valley

    def _erode_scale(self, hits: np.ndarray, kf: int, kt: int) -> np.ndarray:
        """Keep only pixels whose every containing box cleared the threshold.

        A box centred half a box-width outside a strong emission still catches
        half of its energy, so the raw mask of a coarse scale is the emission
        *dilated* by half the box — wide enough, at the top of the ladder, to
        weld unrelated emitters into one component. Requiring every box that
        contains a pixel to be above threshold turns that around: for a signal
        of width ``n`` needing a fill fraction ``phi`` to trigger, the
        surviving core is ``n - 2*phi*k`` wide, always *inside* the true
        support. Coarse scales can then add sensitivity without ever inflating
        an extent or bridging two emitters.

        The lost extent is recovered downstream — the measurement stage re-reads
        the band edges and start/end times from the underlying power, searching
        half a box outside the mask.

        Implemented as a running mean over the boolean mask rather than a
        minimum filter, which makes it O(pixels) regardless of box size.
        """
        frac = float(self.config.scale_erosion)
        if frac <= 0.0:
            return hits
        ef = max(1, int(round(kf * frac)))
        et = max(1, int(round(kt * frac)))
        if ef <= 1 and et <= 1:
            return hits
        cover = hits.astype(np.float32)
        if ef > 1:
            cover = ndimage.uniform_filter1d(cover, ef, axis=0, mode="nearest")
        if et > 1:
            cover = ndimage.uniform_filter1d(cover, et, axis=1, mode="nearest")
        return cover >= 1.0 - 1e-4

    def _apply_exclusions(self, mask: np.ndarray, spec: Spectrogram) -> None:
        cfg = self.config
        if cfg.edge_exclude_hz > 0:
            n = int(np.ceil(cfg.edge_exclude_hz / spec.bin_width_hz))
            n = min(n, spec.n_freq // 2)
            if n > 0:
                mask[:n, :] = False
                mask[-n:, :] = False
        if cfg.dc_notch_hz > 0:
            half = int(np.ceil(0.5 * cfg.dc_notch_hz / spec.bin_width_hz))
            c = spec.freq_to_bin(spec.center_freq_hz)
            lo = max(0, c - half)
            hi = min(spec.n_freq, c + half + 1)
            mask[lo:hi, :] = False

    def _close_gaps(self, mask: np.ndarray, spec: Spectrogram) -> np.ndarray:
        cfg = self.config
        gap_f = max(
            int(round(cfg.merge_gap_hz / spec.bin_width_hz)), int(cfg.merge_gap_bins)
        )
        gap_t = max(
            int(round(cfg.merge_gap_s / spec.frame_step_s)), int(cfg.merge_gap_frames)
        )
        if gap_f > 0:
            mask = ndimage.binary_closing(
                mask, structure=np.ones((gap_f + 1, 1), dtype=bool)
            )
        if gap_t > 0:
            mask = ndimage.binary_closing(
                mask, structure=np.ones((1, gap_t + 1), dtype=bool)
            )
        return mask

    # ------------------------------------------------------- measurement --
    def _measure_components(
        self,
        spec: Spectrogram,
        labels: np.ndarray,
        n_labels: int,
        noise: np.ndarray,
    ) -> list[Detection]:
        cfg = self.config
        if n_labels == 0:
            return []

        bin_hz = spec.bin_width_hz
        step_s = spec.frame_step_s
        min_f_bins = max(1, int(np.floor(cfg.min_bandwidth_hz / bin_hz)))
        min_t_frames = max(1, int(np.floor(cfg.min_duration_s / step_s)))
        # Box integration marks a pixel from the average around it, and the
        # halo erosion trims that mask further, so a coarse-scale detection can
        # sit well inside the real emission. Search that far outside the
        # bounding box when re-measuring the true extent.
        max_kf, max_kt = cfg.max_scales(spec.n_freq, spec.n_time)
        pad_f = max(2, int(np.ceil(2 * spec.enbw_bins)), max_kf // 2 + 1)
        pad_t = max(4, max_kt // 2 + 1)

        objects = ndimage.find_objects(labels)
        detections: list[Detection] = []
        for lab_index, slc in enumerate(objects, start=1):
            if slc is None:
                continue
            f_slice, t_slice = slc
            n_f = f_slice.stop - f_slice.start
            n_t = t_slice.stop - t_slice.start
            if n_f < min_f_bins or n_t < min_t_frames:
                continue

            comp = labels[f_slice, t_slice] == lab_index
            n_pixels = int(np.count_nonzero(comp))
            if n_pixels < max(1, cfg.min_pixels):
                continue

            det = self._measure_one(
                spec, labels, noise, lab_index, f_slice, t_slice, pad_f, pad_t
            )
            if det is None:
                continue
            det.n_pixels = n_pixels
            det.fill_ratio = float(n_pixels / (n_f * n_t))
            detections.append(det)

        return detections

    def _measure_one(
        self,
        spec: Spectrogram,
        labels: np.ndarray,
        noise: np.ndarray,
        lab_index: int,
        f_slice: slice,
        t_slice: slice,
        pad_f: int,
        pad_t: int,
    ) -> Detection | None:
        cfg = self.config
        power = spec.power
        freqs = spec.freqs_hz
        bin_hz = spec.bin_width_hz
        step_s = spec.frame_step_s

        f0 = max(0, f_slice.start - pad_f)
        f1 = min(spec.n_freq, f_slice.stop + pad_f)
        t0, t1 = t_slice.start, t_slice.stop

        comp = labels[f0:f1, t0:t1] == lab_index
        active = comp.any(axis=0)
        block = power[f0:f1, t0:t1]
        psd = block[:, active].mean(axis=1).astype(np.float64)
        noise_slice = noise[f0:f1].astype(np.float64)
        excess = psd - noise_slice

        # Integrate the *signed* excess: noise-only bins in the padding
        # contribute about zero on average, so the energy quantiles land on the
        # real band edges. Clipping at zero first would bias the padding
        # upwards and inflate the bandwidth of weak signals.
        cdf = np.cumsum(excess)
        total = float(cdf[-1])
        if total <= 0.0:
            # Nothing survives noise subtraction; fall back to the mask box.
            u_lo, u_hi = float(f_slice.start - f0), float(f_slice.stop - f0)
        else:
            tail = 0.5 * (1.0 - cfg.occupied_bw_fraction)
            u_lo = _interp_index(cdf, total * tail)
            u_hi = _interp_index(cdf, total * (1.0 - tail))
            if u_hi <= u_lo:
                u_lo, u_hi = float(f_slice.start - f0), float(f_slice.stop - f0)
        sig = np.maximum(excess, 0.0)

        band_origin = freqs[f0] - 0.5 * bin_hz
        f_low = band_origin + u_lo * bin_hz
        f_high = band_origin + u_hi * bin_hz
        bw_raw = max(f_high - f_low, bin_hz)
        if cfg.deconvolve_window:
            bandwidth = max(bw_raw - spec.enbw_hz, bin_hz)
        else:
            bandwidth = bw_raw
        center = 0.5 * (f_low + f_high)

        # Integrate over the occupied band only, so a wide bounding box around
        # a narrow signal does not dilute the SNR.
        b_lo = int(np.clip(f0 + int(np.floor(u_lo)), 0, spec.n_freq - 1))
        b_hi = int(np.clip(f0 + int(np.ceil(u_hi)), b_lo + 1, spec.n_freq))
        w = sig[b_lo - f0 : b_hi - f0]
        w_total = float(w.sum())
        centroid = (
            float(np.sum(w * freqs[b_lo:b_hi]) / w_total)
            if w_total > 0
            else float(center)
        )
        i_lo, i_hi = b_lo - f0, b_hi - f0
        sig_sum = max(float(excess[i_lo:i_hi].sum()), 1e-30)
        noise_sum = float(noise_slice[i_lo:i_hi].sum())
        snr_db = 10.0 * np.log10(max(sig_sum, 1e-30) / max(noise_sum, 1e-30))
        peak_snr_db = float(
            10.0
            * np.log10(
                max(float(np.max(psd[i_lo:i_hi] / noise_slice[i_lo:i_hi])), 1e-30)
            )
        )
        # Summing bin powers over a band and dividing by nfft recovers the
        # signal power in that band (Parseval, with the window's energy
        # already normalised out by the spectrogram).
        power_dbfs = 10.0 * np.log10(max(sig_sum / spec.nfft, 1e-30))
        noise_dbfs_per_hz = 10.0 * np.log10(
            max(
                float(np.mean(noise_slice[i_lo:i_hi])) / spec.sample_rate_hz,
                1e-30,
            )
        )

        i_start, i_end = t0, t1 - 1
        if cfg.refine_time:
            i_start, i_end = self._refine_time(
                spec, noise, b_lo, b_hi, t0, t1, pad_t
            )

        t_start = float(spec.times_s[i_start] - 0.5 * step_s)
        t_end = float(spec.times_s[i_end] + 0.5 * step_s)

        truncated_start = i_start == 0
        truncated_end = i_end == spec.n_time - 1

        return Detection(
            center_freq_hz=float(center),
            bandwidth_hz=float(bandwidth),
            t_start_s=t_start,
            t_end_s=t_end,
            f_low_hz=float(f_low),
            f_high_hz=float(f_high),
            bandwidth_raw_hz=float(bw_raw),
            centroid_freq_hz=float(centroid),
            snr_db=float(snr_db),
            peak_snr_db=float(peak_snr_db),
            power_dbfs=float(power_dbfs),
            noise_dbfs_per_hz=float(noise_dbfs_per_hz),
            is_continuous=bool(truncated_start and truncated_end),
            truncated_start=bool(truncated_start),
            truncated_end=bool(truncated_end),
        )

    def _refine_time(
        self,
        spec: Spectrogram,
        noise: np.ndarray,
        b_lo: int,
        b_hi: int,
        t0: int,
        t1: int,
        pad_t: int,
    ) -> tuple[int, int]:
        """Walk outwards from the mask box along an in-band power profile.

        The mask box is set by a per-pixel threshold, which clips the leading
        and trailing edges of a burst. Averaging across the occupied band
        raises the processing gain, so the same false-alarm rate buys a much
        lower threshold and a tighter estimate of when the burst actually
        started.
        """
        e0 = max(0, t0 - max(pad_t, 4))
        e1 = min(spec.n_time, t1 + max(pad_t, 4))
        band = spec.power[b_lo:b_hi, e0:e1].astype(np.float64)
        profile = band.mean(axis=0)
        noise_in_band = float(np.mean(noise[b_lo:b_hi]))

        n_bins = b_hi - b_lo
        # Neighbouring FFT bins overlap through the window main lobe, so they
        # are not independent; discount them by the window's ENBW.
        n_indep = max(1, int(round(n_bins / max(spec.enbw_bins, 1.0))))
        factor = threshold_factor_from_pfa(self.config.pfa, n_indep * spec.n_avg)
        thr = noise_in_band * factor

        i_start = t0 - e0
        while i_start > 0 and profile[i_start - 1] > thr:
            i_start -= 1
        i_end = t1 - 1 - e0
        while i_end < len(profile) - 1 and profile[i_end + 1] > thr:
            i_end += 1
        return i_start + e0, i_end + e0

    # ------------------------------------------------------------ ranking --
    @staticmethod
    def _box(det: Detection) -> tuple[float, float, float, float]:
        return (det.f_low_edge_hz, det.f_high_edge_hz, det.t_start_s, det.t_end_s)

    def _merge_fragments(self, detections: list[Detection]) -> list[Detection]:
        """Fuse detections that are really one emission seen in pieces."""
        cfg = self.config
        if cfg.merge_overlap <= 0.0 or len(detections) < 2:
            return detections

        order = sorted(
            detections,
            key=lambda d: (d.f_high_edge_hz - d.f_low_edge_hz) * d.duration_s,
            reverse=True,
        )
        kept: list[Detection] = []
        for det in order:
            fl, fh, ts, te = self._box(det)
            area = max((fh - fl) * (te - ts), 1e-30)
            merged_into = None
            for host in kept:
                hfl, hfh, hts, hte = self._box(host)
                inter = max(0.0, min(fh, hfh) - max(fl, hfl)) * max(
                    0.0, min(te, hte) - max(ts, hts)
                )
                if inter / area < cfg.merge_overlap:
                    continue
                if det.snr_db > host.snr_db + cfg.keep_nested_above_db:
                    continue
                merged_into = host
                break
            if merged_into is None:
                kept.append(det)
                continue

            hfl, hfh = merged_into.f_low_edge_hz, merged_into.f_high_edge_hz
            f_low, f_high = min(fl, hfl), max(fh, hfh)
            merged_into.f_low_hz = min(merged_into.f_low_hz, det.f_low_hz)
            merged_into.f_high_hz = max(merged_into.f_high_hz, det.f_high_hz)
            merged_into.center_freq_hz = 0.5 * (f_low + f_high)
            merged_into.bandwidth_hz = f_high - f_low
            merged_into.t_start_s = min(merged_into.t_start_s, det.t_start_s)
            merged_into.t_end_s = max(merged_into.t_end_s, det.t_end_s)
            merged_into.snr_db = max(merged_into.snr_db, det.snr_db)
            merged_into.peak_snr_db = max(merged_into.peak_snr_db, det.peak_snr_db)
            merged_into.n_pixels += det.n_pixels
            merged_into.truncated_start |= det.truncated_start
            merged_into.truncated_end |= det.truncated_end
            merged_into.is_continuous = (
                merged_into.truncated_start and merged_into.truncated_end
            )
            merged_into.extra.setdefault("merged_fragments", 0)
            merged_into.extra["merged_fragments"] += 1
        return kept

    def _filter_and_rank(self, detections: list[Detection]) -> list[Detection]:
        cfg = self.config
        detections = self._merge_fragments(detections)
        kept = [
            d
            for d in detections
            if d.bandwidth_hz >= cfg.min_bandwidth_hz
            and d.duration_s >= cfg.min_duration_s
            and d.snr_db >= cfg.min_snr_db
        ]
        if len(kept) > cfg.max_detections:
            kept.sort(key=lambda d: d.snr_db, reverse=True)
            kept = kept[: cfg.max_detections]
        kept.sort(key=lambda d: (d.t_start_s, d.center_freq_hz))
        for i, det in enumerate(kept):
            det.detection_id = i
        return kept


def detect_signals(
    x: np.ndarray,
    sample_rate_hz: float,
    center_freq_hz: float = 0.0,
    *,
    spectrogram_config: SpectrogramConfig | None = None,
    config: DetectorConfig | None = None,
) -> DetectionResult:
    """Convenience wrapper around :class:`EnergyDetector`."""
    return EnergyDetector(spectrogram_config, config).detect(
        x, sample_rate_hz, center_freq_hz
    )
