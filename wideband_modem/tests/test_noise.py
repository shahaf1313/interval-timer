import numpy as np
import pytest

from wbmodem.generate import SceneBuilder
from wbmodem.noise import (
    clipped_exponential_moments,
    estimate_noise_floor,
    threshold_factor_from_pfa,
)
from wbmodem.spectrogram import compute_spectrogram


def noise_spectrogram(power=1.0, seed=0, n=400_000):
    rng = np.random.default_rng(seed)
    x = (
        (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        / np.sqrt(2.0)
        * np.sqrt(power)
    ).astype(np.complex64)
    return compute_spectrogram(x, 2e6, 100e6)


@pytest.mark.parametrize("power", [1.0, 0.05])
def test_floor_matches_the_true_noise_power(power):
    spec = noise_spectrogram(power)
    floor = estimate_noise_floor(spec.power)
    err_db = 10 * np.log10(floor / power)
    assert abs(np.median(err_db)) < 0.2
    assert np.max(np.abs(err_db)) < 1.0


def test_floor_ignores_a_wide_continuous_signal():
    """The hard case: an emitter that never goes away.

    A per-bin statistic over time cannot tell a continuous signal from a
    raised noise floor, so the estimator has to reject it across frequency
    instead. If this regresses, every wideband continuous signal silently
    disappears into the floor and is never detected.
    """
    sb = SceneBuilder(2e6, 0.15, 100e6, seed=11)
    sb.add_noise_band(100.3e6, bandwidth_hz=300e3, snr_db=20.0)
    x, _ = sb.build()
    spec = compute_spectrogram(x, 2e6, 100e6)
    floor = estimate_noise_floor(spec.power)

    inside = spec.freq_to_bin(100.3e6)
    outside = spec.freq_to_bin(99.4e6)
    assert 10 * np.log10(floor[inside]) < 1.0
    assert 10 * np.log10(floor[outside]) < 1.0


def test_floor_follows_a_sloping_front_end():
    spec = noise_spectrogram()
    tilt = np.linspace(0.1, 10.0, spec.n_freq)  # 20 dB of tilt across the band
    spec.power *= tilt[:, None].astype(spec.power.dtype)
    floor = estimate_noise_floor(spec.power)
    err_db = 10 * np.log10(floor / tilt)
    assert np.max(np.abs(err_db)) < 1.0


def test_fully_occupied_band_falls_back_instead_of_failing():
    spec = noise_spectrogram()
    spec.power *= 100.0
    floor = estimate_noise_floor(spec.power)
    assert np.all(np.isfinite(floor))
    assert np.all(floor > 0)


def test_threshold_hits_its_target_false_alarm_rate():
    rng = np.random.default_rng(5)
    samples = rng.exponential(1.0, size=2_000_000)
    for pfa in (1e-2, 1e-3, 1e-4):
        t = threshold_factor_from_pfa(pfa, 1)
        measured = float(np.mean(samples > t))
        assert measured == pytest.approx(pfa, rel=0.25)


def test_averaged_threshold_is_lower_but_still_correct():
    rng = np.random.default_rng(6)
    n_avg = 16
    means = rng.gamma(n_avg, 1.0 / n_avg, size=500_000)
    t = threshold_factor_from_pfa(1e-3, n_avg)
    assert t < threshold_factor_from_pfa(1e-3, 1)
    assert float(np.mean(means > t)) == pytest.approx(1e-3, rel=0.3)


def test_clipped_moments_match_monte_carlo():
    rng = np.random.default_rng(7)
    x = np.minimum(rng.exponential(1.0, size=4_000_000), 4.0)
    mean, var, mu3 = clipped_exponential_moments(4.0)
    assert mean == pytest.approx(x.mean(), rel=0.005)
    assert var == pytest.approx(x.var(), rel=0.01)
    assert mu3 == pytest.approx(float(np.mean((x - x.mean()) ** 3)), rel=0.05)


def test_clipped_threshold_controls_false_alarms():
    """The clipped statistic needs its own threshold, not the Gamma one."""
    rng = np.random.default_rng(8)
    n_avg, cap = 64, 4.0
    draws = np.minimum(rng.exponential(1.0, size=(200_000, n_avg)), cap).mean(axis=1)
    t = threshold_factor_from_pfa(1e-3, n_avg, clip=cap)
    measured = float(np.mean(draws > t))
    assert measured < 5e-3  # approximation must not be optimistic by much
    assert measured > 1e-5  # nor absurdly conservative
