import numpy as np
import pytest

from wbmodem.spectrogram import SpectrogramConfig, compute_spectrogram


def white_noise(n, power=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return (
        (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        / np.sqrt(2.0)
        * np.sqrt(power)
    ).astype(np.complex64)


@pytest.mark.parametrize("power", [1.0, 0.01, 25.0])
def test_noise_power_is_calibrated(power):
    """A bin must read the input noise power directly, in linear units.

    Every threshold in the package is expressed relative to the noise floor,
    so this calibration is what makes a CFAR threshold mean what it says.
    """
    spec = compute_spectrogram(white_noise(200_000, power), 1e6)
    assert spec.power.mean() == pytest.approx(power, rel=0.02)


def test_tone_lands_in_the_right_bin_with_the_right_power():
    fs, nfft = 1e6, 1024
    # Exactly on a bin centre, so there is no scalloping loss to account for.
    f_tone = 100e3
    n = np.arange(200_000)
    x = (2.0 * np.exp(2j * np.pi * f_tone * n / fs)).astype(np.complex64)
    spec = compute_spectrogram(x, fs, center_freq_hz=0.0, config=SpectrogramConfig(nfft=nfft))

    peak_bin = int(np.argmax(spec.power.mean(axis=1)))
    assert spec.freqs_hz[peak_bin] == pytest.approx(f_tone, abs=spec.bin_width_hz / 2)

    # Summing bin powers and dividing by nfft recovers the signal power
    # (Parseval), for a coherent tone exactly as for noise.
    total = spec.power[peak_bin - 4 : peak_bin + 5, :].mean(axis=1).sum()
    assert total / spec.nfft == pytest.approx(4.0, rel=0.05)


def test_hann_enbw_and_axes():
    fs = 2e6
    spec = compute_spectrogram(white_noise(100_000), fs, 1e9, SpectrogramConfig(nfft=512))
    assert spec.enbw_bins == pytest.approx(1.5, rel=0.01)  # Hann
    assert spec.bin_width_hz == pytest.approx(fs / 512)
    assert spec.freqs_hz[0] == pytest.approx(1e9 - fs / 2)
    assert spec.n_freq == 512
    # Frames are timestamped at the centre of their window.
    assert spec.times_s[0] == pytest.approx((512 - 1) / 2 / fs)
    assert spec.frame_step_s == pytest.approx(256 / fs)


def test_averaging_reduces_variance_without_moving_the_mean():
    x = white_noise(400_000, seed=3)
    plain = compute_spectrogram(x, 1e6, config=SpectrogramConfig(average=1))
    avg = compute_spectrogram(x, 1e6, config=SpectrogramConfig(average=8))
    assert avg.power.mean() == pytest.approx(plain.power.mean(), rel=0.02)
    assert avg.power.std() < plain.power.std() / 2
    assert avg.n_avg == 8
    assert avg.frame_step_s == pytest.approx(8 * plain.frame_step_s)


def test_crop_keeps_axes_consistent():
    spec = compute_spectrogram(white_noise(200_000), 1e6, 10e6)
    sub = spec.crop(f_low_hz=9.9e6, f_high_hz=10.1e6, t_start_s=0.02, t_end_s=0.08)
    assert sub.n_freq == len(sub.freqs_hz) == sub.power.shape[0]
    assert sub.n_time == len(sub.times_s) == sub.power.shape[1]
    assert sub.freqs_hz[0] >= 9.9e6 - spec.bin_width_hz
    assert sub.times_s[0] >= 0.02 - spec.frame_step_s


def test_rejects_capture_shorter_than_one_frame():
    with pytest.raises(ValueError):
        compute_spectrogram(white_noise(100), 1e6, config=SpectrogramConfig(nfft=1024))
