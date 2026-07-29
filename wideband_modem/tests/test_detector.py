import numpy as np
import pytest

from wbmodem import DetectorConfig, EnergyDetector, SceneBuilder, SpectrogramConfig
from wbmodem.spectrogram import compute_spectrogram

FS = 2e6
FC = 100e6


def detect(x, config=None, spec_config=None):
    return EnergyDetector(spec_config, config).detect(x, FS, FC)


def noise_only(duration=0.1, seed=0):
    sb = SceneBuilder(FS, duration, FC, seed=seed)
    return sb.build()[0]


def test_noise_alone_produces_no_detections():
    """The false-alarm floor. Everything else is worthless if this fails."""
    for seed in range(4):
        result = detect(noise_only(seed=seed))
        assert result.detections == [], result.table()
        assert result.occupancy < 1e-3


def test_finds_a_continuous_tone_and_calls_it_continuous():
    sb = SceneBuilder(FS, 0.1, FC, seed=1)
    sb.add_tone(FC + 321e3, snr_db=25.0)
    result = detect(sb.build()[0])

    assert len(result) == 1
    det = result.detections[0]
    assert det.center_freq_hz == pytest.approx(FC + 321e3, abs=result.spectrogram.enbw_hz)
    assert det.bandwidth_hz < 5 * result.spectrogram.bin_width_hz
    assert det.is_continuous
    assert det.snr_db > 20.0


def test_burst_start_and_end_land_within_one_analysis_window():
    sb = SceneBuilder(FS, 0.1, FC, seed=2)
    sb.add_psk(FC - 400e3, symbol_rate_hz=100e3, snr_db=20.0, t_start_s=0.030, duration_s=0.020)
    result = detect(sb.build()[0])

    assert len(result) == 1
    det = result.detections[0]
    tol = 2 * result.spectrogram.window_duration_s
    assert det.t_start_s == pytest.approx(0.030, abs=tol)
    assert det.t_end_s == pytest.approx(0.050, abs=tol)
    assert not det.is_continuous
    assert not det.truncated_start and not det.truncated_end


def test_bandwidth_tracks_the_symbol_rate():
    """Occupied bandwidth of an RRC-shaped carrier is (1+rolloff)*Rs."""
    for symbol_rate, rolloff in ((50e3, 0.35), (250e3, 0.2)):
        sb = SceneBuilder(FS, 0.08, FC, seed=3)
        sb.add_psk(FC, symbol_rate_hz=symbol_rate, snr_db=25.0, rolloff=rolloff)
        result = detect(sb.build()[0])
        assert len(result) == 1
        expected = symbol_rate * (1 + rolloff)
        assert result.detections[0].bandwidth_hz == pytest.approx(expected, rel=0.2)


def test_separates_two_signals_at_different_frequencies():
    sb = SceneBuilder(FS, 0.08, FC, seed=4)
    sb.add_psk(FC - 500e3, symbol_rate_hz=40e3, snr_db=20.0)
    sb.add_psk(FC + 500e3, symbol_rate_hz=40e3, snr_db=20.0, t_start_s=0.02, duration_s=0.03)
    result = detect(sb.build()[0])

    assert len(result) == 2
    centers = sorted(d.center_freq_hz for d in result.detections)
    assert centers[0] == pytest.approx(FC - 500e3, abs=10e3)
    assert centers[1] == pytest.approx(FC + 500e3, abs=10e3)


def test_integrated_scales_find_what_a_single_pixel_test_cannot():
    """A low-SNR wideband emission is the whole reason for the scale ladder.

    Per bin it sits below any sane single-pixel threshold; it only appears
    once the detector integrates over a box that fits inside it.
    """
    sb = SceneBuilder(FS, 0.15, FC, seed=5)
    sb.add_noise_band(FC + 200e3, bandwidth_hz=200e3, snr_db=6.0)
    x = sb.build()[0]

    single_pixel = detect(x, DetectorConfig(freq_scales=(1,), time_scales=(1,)))
    multi_scale = detect(x)

    # The single-pixel detector sees only the loudest speckles of the
    # emission, never the emission itself.
    assert not any(
        d.bandwidth_hz > 150e3 and d.duration_s > 0.05 for d in single_pixel
    )
    assert len(multi_scale) == 1
    det = multi_scale.detections[0]
    assert det.center_freq_hz == pytest.approx(FC + 200e3, abs=30e3)
    assert det.bandwidth_hz == pytest.approx(200e3, rel=0.35)
    assert det.is_continuous


def test_strong_narrow_carrier_does_not_smear_across_the_band():
    """Box integration must not report a tone as wide as its own box."""
    sb = SceneBuilder(FS, 0.1, FC, seed=6)
    sb.add_tone(FC - 100e3, snr_db=40.0)
    result = detect(sb.build()[0])

    assert len(result) == 1
    assert result.detections[0].bandwidth_hz < 20e3
    assert result.occupancy < 0.02


def test_neighbouring_emitters_stay_separate():
    sb = SceneBuilder(FS, 0.1, FC, seed=7)
    sb.add_psk(FC - 150e3, symbol_rate_hz=30e3, snr_db=25.0)
    sb.add_psk(FC + 150e3, symbol_rate_hz=30e3, snr_db=25.0)
    result = detect(sb.build()[0])
    assert len(result) == 2
    for det in result.detections:
        assert det.bandwidth_hz < 100e3


def test_explicit_threshold_overrides_the_cfar_setting():
    sb = SceneBuilder(FS, 0.08, FC, seed=8)
    sb.add_psk(FC + 200e3, symbol_rate_hz=50e3, snr_db=12.0)
    x = sb.build()[0]

    assert len(detect(x)) == 1
    # A threshold well above the signal must silence the detector entirely,
    # at every scale, proving the override reaches the whole ladder.
    assert len(detect(x, DetectorConfig(threshold_db=25.0))) == 0


def test_detect_spectrogram_accepts_a_precomputed_plane():
    sb = SceneBuilder(FS, 0.08, FC, seed=9)
    sb.add_tone(FC + 250e3, snr_db=25.0)
    x = sb.build()[0]

    spec = compute_spectrogram(x, FS, FC, SpectrogramConfig(nfft=2048))
    result = EnergyDetector().detect_spectrogram(spec)
    assert len(result) == 1
    assert result.detections[0].center_freq_hz == pytest.approx(FC + 250e3, abs=5e3)
    assert result.spectrogram is spec


def test_min_bandwidth_and_duration_filters_apply():
    sb = SceneBuilder(FS, 0.1, FC, seed=10)
    sb.add_tone(FC - 300e3, snr_db=30.0)  # narrow, continuous
    sb.add_psk(FC + 300e3, symbol_rate_hz=200e3, snr_db=25.0, t_start_s=0.04, duration_s=0.004)

    assert len(detect(sb.build()[0])) == 2
    wide_only = detect(sb.build()[0], DetectorConfig(min_bandwidth_hz=100e3))
    assert len(wide_only) == 1
    assert wide_only.detections[0].center_freq_hz > FC
    long_only = detect(sb.build()[0], DetectorConfig(min_duration_s=0.02))
    assert len(long_only) == 1
    assert long_only.detections[0].center_freq_hz < FC


def test_dc_notch_and_edge_exclusion():
    sb = SceneBuilder(FS, 0.08, FC, seed=11)
    sb.add_tone(FC, snr_db=30.0)  # LO leakage sitting at the capture centre
    assert len(detect(sb.build()[0])) == 1
    assert len(detect(sb.build()[0], DetectorConfig(dc_notch_hz=40e3))) == 0


def test_result_serialises_to_json():
    sb = SceneBuilder(FS, 0.05, FC, seed=12)
    sb.add_tone(FC + 100e3, snr_db=25.0)
    result = detect(sb.build()[0])
    import json

    payload = json.loads(result.to_json())
    assert payload["capture"]["sample_rate_hz"] == FS
    assert len(payload["detections"]) == len(result)
    assert "center_freq_hz" in payload["detections"][0]
    assert "bandwidth_hz" in payload["detections"][0]
    assert "t_start_s" in payload["detections"][0]
    assert "t_end_s" in payload["detections"][0]


def test_reported_power_matches_the_signal_that_was_generated():
    sb = SceneBuilder(FS, 0.1, FC, noise_power=1.0, seed=13)
    truth = sb.add_noise_band(FC + 250e3, bandwidth_hz=100e3, snr_db=20.0)
    result = detect(sb.build()[0])

    assert len(result) == 1
    # in-band SNR is the generator's own definition, so it must come back out
    assert result.detections[0].snr_db == pytest.approx(truth.snr_db, abs=2.0)
    # and absolute power: P = snr * N0 * B
    expected_dbfs = 10 * np.log10(10 ** (20 / 10) * (1.0 / FS) * 100e3)
    assert result.detections[0].power_dbfs == pytest.approx(expected_dbfs, abs=1.5)
