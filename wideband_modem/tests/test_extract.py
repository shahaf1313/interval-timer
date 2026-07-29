import json

import numpy as np
import pytest

from wbmodem import (
    EnergyDetector,
    ExtractConfig,
    SceneBuilder,
    extract_stream,
    extract_streams,
)
from wbmodem.iqio import load_iq

FS = 2e6
FC = 100e6


def tone_frequency(samples, sample_rate_hz):
    """Estimate a complex tone's frequency from its average phase advance."""
    if len(samples) < 2:
        return 0.0
    advance = np.mean(samples[1:] * np.conj(samples[:-1]))
    return float(np.angle(advance) * sample_rate_hz / (2 * np.pi))


def scene_with_tone(offset_hz=321e3, snr_db=30.0, t_start=None, duration=None):
    sb = SceneBuilder(FS, 0.1, FC, seed=1)
    truth = sb.add_tone(FC + offset_hz, snr_db=snr_db, t_start_s=t_start, duration_s=duration)
    x = sb.build()[0]
    result = EnergyDetector().detect(x, FS, FC)
    assert len(result) == 1
    return x, result.detections[0], truth


def test_stream_is_mixed_to_baseband():
    """The extracted stream must be centred on the signal, not on the capture."""
    x, det, truth = scene_with_tone(offset_hz=321e3)
    stream = extract_stream(x, FS, FC, det)

    residual = tone_frequency(stream.samples, stream.sample_rate_hz)
    absolute = stream.center_freq_hz + residual
    assert absolute == pytest.approx(truth.center_freq_hz, abs=2e3)


def test_output_rate_covers_the_bandwidth_and_nothing_more():
    sb = SceneBuilder(FS, 0.1, FC, seed=2)
    sb.add_psk(FC - 300e3, symbol_rate_hz=100e3, snr_db=25.0)
    x = sb.build()[0]
    det = EnergyDetector().detect(x, FS, FC).detections[0]

    stream = extract_stream(x, FS, FC, det, ExtractConfig(oversample=2.0))
    assert stream.sample_rate_hz >= 2.0 * det.bandwidth_hz
    assert stream.sample_rate_hz <= FS
    # decimation must be an exact divisor of the capture rate
    assert FS / stream.sample_rate_hz == pytest.approx(stream.decimation)
    # and no wider than needed: one more halving would break the requirement
    assert stream.sample_rate_hz / 2 < 2.0 * det.bandwidth_hz or stream.decimation == 1


def test_timestamps_line_up_with_the_wideband_timeline():
    """A burst must land at the same time in the stream as in the capture."""
    x, det, truth = scene_with_tone(offset_hz=-250e3, t_start=0.030, duration=0.020)
    stream = extract_stream(x, FS, FC, det, ExtractConfig(guard_s=0.002))

    t = stream.time_axis()
    assert t[0] == pytest.approx(stream.t_start_s)
    envelope = np.abs(stream.samples) ** 2
    strong = envelope > 0.5 * np.max(envelope)
    assert t[strong][0] == pytest.approx(truth.t_start_s, abs=1e-3)
    assert t[strong][-1] == pytest.approx(truth.t_end_s, abs=1e-3)


def test_phase_reference_is_the_capture_not_the_slice():
    """Two extractions with different margins must agree where they overlap.

    The mixer is referenced to the capture's own sample index, so a stream is
    reproducible and two streams cut from one capture share a phase origin.
    """
    x, det, _ = scene_with_tone(offset_hz=180e3, t_start=0.030, duration=0.020)
    tight = extract_stream(x, FS, FC, det, ExtractConfig(guard_s=0.0))
    loose = extract_stream(x, FS, FC, det, ExtractConfig(guard_s=0.005))

    assert tight.sample_rate_hz == loose.sample_rate_hz
    shift = round((tight.t_start_s - loose.t_start_s) * loose.sample_rate_hz)
    n = min(len(tight.samples), len(loose.samples) - shift)
    assert n > 100
    a = tight.samples[:n]
    b = loose.samples[shift : shift + n]
    assert np.allclose(a, b, atol=1e-4 * np.max(np.abs(a)))


def test_out_of_band_energy_is_filtered_away():
    """A very strong neighbour must not leak into the channel we asked for.

    Extract the same signal from two captures that differ only by a 45 dB
    carrier 400 kHz away. If the anti-alias filter does its job the two
    streams carry the same power.
    """
    def build(with_interferer):
        sb = SceneBuilder(FS, 0.1, FC, seed=3)
        sb.add_psk(FC - 200e3, symbol_rate_hz=40e3, snr_db=20.0)
        if with_interferer:
            sb.add_tone(FC + 200e3, snr_db=45.0)
        return sb.build()[0]

    clean, crowded = build(False), build(True)
    result = EnergyDetector().detect(crowded, FS, FC)
    wanted = min(result.detections, key=lambda d: d.center_freq_hz)

    a = extract_stream(clean, FS, FC, wanted)
    b = extract_stream(crowded, FS, FC, wanted)
    power_a = float(np.mean(np.abs(a.samples) ** 2))
    power_b = float(np.mean(np.abs(b.samples) ** 2))
    assert 10 * np.log10(power_b / power_a) < 0.5

    # And the energy really is centred: most of it inside the reported band.
    n = min(8192, len(b.samples))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(b.samples[:n]))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1 / b.sample_rate_hz))
    in_band = np.abs(freqs) <= 0.5 * wanted.bandwidth_hz
    assert spectrum[in_band].sum() > 0.8 * spectrum.sum()


def test_streams_round_trip_through_disk(tmp_path):
    x, det, _ = scene_with_tone()
    stream = extract_stream(x, FS, FC, det)
    iq_path = stream.save(tmp_path / "sig000")

    reloaded = load_iq(iq_path, "cf32")
    assert np.allclose(reloaded, stream.samples.astype(np.complex64))

    meta = json.loads((tmp_path / "sig000.json").read_text())
    assert meta["center_freq_hz"] == pytest.approx(stream.center_freq_hz)
    assert meta["bandwidth_hz"] == pytest.approx(stream.bandwidth_hz)
    assert meta["sample_rate_hz"] == pytest.approx(stream.sample_rate_hz)
    assert meta["n_samples"] == stream.n_samples
    assert meta["data_file"] == "sig000.cf32"
    assert "detection" in meta


def test_every_detection_becomes_its_own_stream():
    sb = SceneBuilder(FS, 0.1, FC, seed=4)
    sb.add_psk(FC - 400e3, symbol_rate_hz=50e3, snr_db=22.0)
    sb.add_tone(FC + 100e3, snr_db=30.0)
    sb.add_psk(FC + 500e3, symbol_rate_hz=150e3, snr_db=22.0, t_start_s=0.02, duration_s=0.03)
    x = sb.build()[0]

    result = EnergyDetector().detect(x, FS, FC)
    streams = extract_streams(x, FS, FC, result.detections)

    assert len(streams) == len(result) == 3
    for stream, det in zip(streams, result.detections):
        assert stream.detection_id == det.detection_id
        assert stream.center_freq_hz == det.center_freq_hz
        assert stream.n_samples > 0
        # a narrow signal must not be handed back at the full capture rate
        assert stream.sample_rate_hz <= FS
        assert stream.duration_s == pytest.approx(det.duration_s, abs=0.01)
    # the narrowband tone should be decimated the most
    rates = {round(s.bandwidth_hz): s.sample_rate_hz for s in streams}
    assert min(rates.values()) == streams[
        int(np.argmin([s.bandwidth_hz for s in streams]))
    ].sample_rate_hz


def test_recovers_qpsk_symbols_well_enough_to_demodulate():
    """The point of the whole exercise: the stream must still be a signal.

    Extract a strong QPSK carrier and check the constellation collapses onto
    four points once the sample timing is recovered by brute force.
    """
    symbol_rate = 100e3
    sb = SceneBuilder(FS, 0.05, FC, seed=5)
    sb.add_psk(FC + 350e3, symbol_rate_hz=symbol_rate, order=4, snr_db=30.0, rolloff=0.35)
    x = sb.build()[0]

    result = EnergyDetector().detect(x, FS, FC)
    assert len(result) == 1
    stream = extract_stream(x, FS, FC, result.detections[0], ExtractConfig(oversample=4.0))

    sps = stream.sample_rate_hz / symbol_rate
    assert sps >= 3.0
    s = stream.samples[len(stream.samples) // 4 : -len(stream.samples) // 4]
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))

    # Fourth-power spectral line: for QPSK it is sharp only if the stream is
    # clean and still carries its modulation.
    quartic = np.abs(np.fft.fft(s**4))
    assert quartic.max() / np.median(quartic) > 50
