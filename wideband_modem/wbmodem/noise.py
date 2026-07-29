"""Noise-floor estimation and CFAR thresholds.

Estimating the noise floor in a wideband scene is the hard part of energy
detection. Two effects fight each other:

* A *per-bin statistic over time* (median, low percentile) is blind to short
  bursts, which is exactly what we want — but a continuous carrier sits in the
  same bin for the whole capture, so the statistic locks onto the signal.
* A *statistic over frequency* rejects continuous carriers, but a wideband
  signal covering a large contiguous slice of the band drags it up too.

The estimator below combines both: a low percentile over time gives a first
guess per bin, a median filter over frequency rejects narrow spurs, and then a
few refinement passes mark bins that sit clearly above the current floor as
occupied and interpolate the floor across them. That handles wide continuous
emitters while still following a real, sloping analogue front-end response.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.special import gammaincinv, ndtri_exp

__all__ = [
    "clipped_exponential_moments",
    "estimate_noise_floor",
    "threshold_factor_from_pfa",
    "threshold_db_from_pfa",
]

_MIN_POWER = 1e-30


def clipped_exponential_moments(cap: float) -> tuple[float, float, float]:
    """Mean, variance and third central moment of ``min(X, cap)``, ``X~Exp(1)``.

    Clipping is what lets an integrated detector stay sensitive to a weak
    signal spread over a wide band while ignoring a strong narrow one nearby:
    the strong signal's pixels are capped, so they can no longer drag a large
    box over its threshold. The price is that the noise statistics change, and
    these moments are what the threshold is then computed from.
    """
    c = float(cap)
    e = float(np.exp(-c))
    m1 = 1.0 - e
    m2 = 2.0 - e * (2.0 * c + 2.0)
    m3 = 6.0 - e * (3.0 * c * c + 6.0 * c + 6.0)
    var = m2 - m1**2
    mu3 = m3 - 3.0 * m1 * m2 + 2.0 * m1**3
    return m1, var, mu3


def threshold_factor_from_pfa(
    pfa: float, n_avg: int = 1, clip: float | None = None
) -> float:
    """Linear threshold, as a ratio above the noise floor, for a target Pfa.

    Bin power of complex circular Gaussian noise is exponentially distributed;
    averaging ``n_avg`` independent frames makes it Gamma(n_avg, sigma^2/n_avg).
    The returned factor ``T`` satisfies ``P(power > T * sigma^2) = pfa``.

    With ``clip`` set, the statistic is the mean of ``n_avg`` samples each
    capped at ``clip`` times the noise floor. That distribution has no simple
    closed form, so the threshold comes from a normal approximation with a
    Cornish-Fisher correction for its skewness — accurate where it is used, on
    boxes averaging tens of samples or more.

    With overlapped windows neighbouring frames are correlated, so an averaged
    spectrogram delivers a slightly higher false-alarm rate than requested.
    That is a conservative direction for a detector front end: it errs towards
    reporting a marginal signal rather than dropping it.
    """
    if not 0.0 < pfa < 1.0:
        raise ValueError("pfa must be in (0, 1)")
    n_avg = max(1, int(n_avg))

    if clip is None or not np.isfinite(clip):
        return float(gammaincinv(n_avg, 1.0 - pfa) / n_avg)

    cap = float(clip)
    if cap <= 0.0:
        raise ValueError("clip must be positive")
    if n_avg == 1:
        return float(min(-np.log(pfa), cap))

    mean, var, mu3 = clipped_exponential_moments(cap)
    z = float(-ndtri_exp(np.log(pfa)))
    skew = (mu3 / var**1.5) / np.sqrt(n_avg)
    z_cf = z + (z * z - 1.0) * skew / 6.0
    return float(min(mean + z_cf * np.sqrt(var / n_avg), cap))


def threshold_db_from_pfa(
    pfa: float, n_avg: int = 1, clip: float | None = None
) -> float:
    """Same as :func:`threshold_factor_from_pfa` but expressed in dB."""
    return float(10.0 * np.log10(threshold_factor_from_pfa(pfa, n_avg, clip)))


def _percentile_bias(percentile: float, n_avg: int) -> float:
    """E[q_p] / sigma^2 for Gamma(n_avg, 1/n_avg) distributed bin power.

    Dividing the measured percentile by this factor removes the bias, so the
    estimate reads as a true noise power rather than "the 25th percentile of
    the noise power".
    """
    p = np.clip(percentile / 100.0, 1e-6, 1.0 - 1e-6)
    n_avg = max(1, int(n_avg))
    return float(gammaincinv(n_avg, p) / n_avg)


def _odd(n: int) -> int:
    n = int(max(1, n))
    return n if n % 2 == 1 else n + 1


def estimate_noise_floor(
    power: np.ndarray,
    *,
    n_avg: int = 1,
    percentile: float = 25.0,
    smooth_bins: int | None = None,
    coarse_bins: int | None = None,
    iterations: int = 6,
    guard_db: float | None = None,
    edge_guard_bins: int = 3,
    debias: bool = True,
) -> np.ndarray:
    """Estimate the per-frequency-bin noise power of a spectrogram.

    Parameters
    ----------
    power:
        Linear power spectrogram, shape ``(n_freq, n_time)``.
    n_avg:
        Frames averaged per pixel; used to de-bias the percentile.
    percentile:
        Percentile over time used as the initial per-bin estimate. Lower values
        tolerate a higher duty cycle before the estimate starts tracking the
        signal, at the price of a noisier estimate.
    smooth_bins:
        Width of the median filter that produces the final floor. Defaults to
        about 1/32 of the band, which follows a real front-end tilt while
        rejecting anything narrower than that.
    coarse_bins:
        Width of the *occupancy test* window. A median here would only tolerate
        an emitter filling half of it, so the test uses a low percentile over
        the window instead, which survives up to about three quarters. Wider
        emitters are still peeled off, from the edges inwards, because each
        iteration re-runs the test on the already cleaned data. Defaults to a
        quarter of the band.
    iterations:
        Number of peel-off passes.
    edge_guard_bins:
        Bins added either side of an occupied region before interpolating
        across it.
    guard_db:
        How far above the coarse floor a bin must sit to be called occupied.
        ``None`` derives it from the scatter of the estimate itself, which is
        what you want: a fixed 6 dB guard would quietly swallow every
        continuous signal weaker than 6 dB per bin into the noise floor, and
        those are exactly the signals the integrated scales are meant to find.

    Returns
    -------
    Array of shape ``(n_freq,)`` with the estimated noise power per bin.
    """
    power = np.asarray(power)
    if power.ndim != 2:
        raise ValueError("power must be a 2-D (freq, time) array")
    n_freq, n_time = power.shape

    if smooth_bins is None:
        smooth_bins = max(9, n_freq // 32)
    smooth_bins = _odd(min(smooth_bins, max(1, n_freq)))
    if coarse_bins is None:
        coarse_bins = max(65, n_freq // 4)
    coarse_bins = _odd(min(coarse_bins, max(1, n_freq)))

    if n_time == 1:
        base = power[:, 0].astype(np.float64)
    else:
        base = np.percentile(power.astype(np.float64), percentile, axis=1)
        if debias:
            base /= _percentile_bias(percentile, n_avg)
    base = np.maximum(base, _MIN_POWER)

    idx = np.arange(n_freq, dtype=np.float64)
    work = base.copy()
    occupied = np.zeros(n_freq, dtype=bool)

    for _ in range(max(1, int(iterations))):
        coarse = ndimage.percentile_filter(
            work, percentile=25.0, size=coarse_bins, mode="nearest"
        )
        ratio_db = 10.0 * np.log10(base / coarse)
        # Centre the test on the bulk of the band rather than on zero: a low
        # percentile of a noisy estimate is biased low by a few tenths of a dB,
        # and that bias would otherwise be read as occupancy everywhere.
        center = float(np.median(ratio_db))
        if guard_db is None:
            # Scale the guard to how much the per-bin estimate actually
            # scatters. A robust (MAD) spread ignores the occupied bins
            # themselves, so a few sigma above it separates "this bin carries
            # signal" from "this bin got a noisy estimate". A fixed guard of a
            # few dB would instead swallow every continuous emission weaker
            # than that into the floor, and it would never be detected.
            mad = float(np.median(np.abs(ratio_db - center)))
            guard_step = max(0.5, 4.0 * 1.4826 * mad)
        else:
            guard_step = float(guard_db)
        new_occupied = occupied | (ratio_db > center + guard_step)
        if new_occupied.all():
            # Every bin looks occupied. There is no in-band reference left, so
            # fall back to the most defensible global statistic we have.
            return np.full(n_freq, max(float(np.median(base)), _MIN_POWER))
        if np.array_equal(new_occupied, occupied):
            break
        occupied = new_occupied
        # Widen the exclusion before interpolating. The bins right at a
        # signal's edge are partly lit — not enough to trip the guard, but
        # enough to anchor the interpolation high and leave the floor raised
        # across the whole emission.
        if edge_guard_bins > 0:
            spread = ndimage.binary_dilation(
                occupied, structure=np.ones(2 * int(edge_guard_bins) + 1, dtype=bool)
            )
            if not spread.all():
                occupied = spread
        work = base.copy()
        work[occupied] = np.interp(idx[occupied], idx[~occupied], base[~occupied])

    floor = ndimage.median_filter(work, size=smooth_bins, mode="nearest")

    # A short uniform filter removes the staircase left by the median filter
    # without letting a single strong bin leak back in.
    floor = ndimage.uniform_filter1d(
        floor, size=_odd(max(3, smooth_bins // 4)), mode="nearest"
    )
    return np.maximum(floor, _MIN_POWER)
