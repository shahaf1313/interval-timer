"""End-to-end behaviour: scene in, labelled streams out."""

import json

import numpy as np
import pytest

from wbmodem import (
    Detection,
    EnergyDetector,
    TruthSignal,
    default_scene,
    extract_streams,
    match_detections,
)
from wbmodem.cli import main, parse_hz, parse_seconds
from wbmodem.iqio import load_iq, save_iq
from wbmodem.plotting import ascii_waterfall


def run_default_scene():
    x, truth, _ = default_scene()
    result = EnergyDetector().detect(x, 2e6, 100e6)
    spec = result.spectrogram
    report = match_detections(
        truth,
        result.detections,
        freq_tol_hz=2 * spec.enbw_hz,
        time_tol_s=2 * spec.window_duration_s,
    )
    return x, truth, result, report


def test_mixed_scene_is_mostly_recovered():
    """The headline claim: continuous and bursty signals, varying widths."""
    _, truth, result, report = run_default_scene()
    assert report.recall >= 0.8, report.report()
    assert report.precision >= 0.8, report.report()

    errors = report.error_summary()
    assert errors["median_abs_freq_error_hz"] < 5e3
    assert 0.6 < errors["median_bandwidth_ratio"] < 1.6
    assert errors["median_abs_t_start_error_s"] < 2e-3
    assert errors["median_abs_t_end_error_s"] < 2e-3


def test_continuous_signals_are_flagged_as_continuous():
    _, truth, result, report = run_default_scene()
    for match in report.matches:
        spans_capture = match.truth.duration_s > 0.9 * 0.05
        if spans_capture:
            assert match.detection.is_continuous, match.truth.name


def test_every_detection_extracts_to_a_usable_stream():
    x, _, result, _ = run_default_scene()
    streams = extract_streams(x, 2e6, 100e6, result.detections)

    assert len(streams) == len(result)
    for stream, det in zip(streams, result.detections):
        assert stream.n_samples > 16
        assert stream.sample_rate_hz >= 2 * det.bandwidth_hz
        assert np.all(np.isfinite(stream.samples.view(np.float32)))
        meta = stream.metadata()
        for key in ("center_freq_hz", "bandwidth_hz", "t_start_s", "t_end_s"):
            assert np.isfinite(meta[key])


def test_scene_is_reproducible_from_its_seed():
    a, truth_a, _ = default_scene(seed=42)
    b, truth_b, _ = default_scene(seed=42)
    assert np.array_equal(a, b)
    assert [t.name for t in truth_a] == [t.name for t in truth_b]


def test_matching_pairs_by_overlap_and_reports_misses():
    truth = [
        TruthSignal(100e6, 50e3, 0.01, 0.02, 20.0, name="a"),
        TruthSignal(101e6, 50e3, 0.01, 0.02, 20.0, name="b"),
    ]
    dets = [
        Detection(center_freq_hz=100e6, bandwidth_hz=52e3, t_start_s=0.0101, t_end_s=0.0199),
        Detection(center_freq_hz=105e6, bandwidth_hz=50e3, t_start_s=0.01, t_end_s=0.02),
    ]
    report = match_detections(truth, dets)
    assert len(report.matches) == 1
    assert report.matches[0].truth.name == "a"
    assert [t.name for t in report.missed] == ["b"]
    assert len(report.false_alarms) == 1
    assert report.recall == 0.5
    assert report.precision == 0.5
    assert report.matches[0].freq_error_hz == pytest.approx(0.0, abs=1.0)


@pytest.mark.parametrize("fmt", ["cf32", "ci16", "ci8", "cu8"])
def test_iq_formats_round_trip(tmp_path, fmt):
    rng = np.random.default_rng(0)
    x = ((rng.standard_normal(4096) + 1j * rng.standard_normal(4096)) * 0.2).astype(
        np.complex64
    )
    path = save_iq(tmp_path / f"cap.{fmt}", x, fmt)
    back = load_iq(path, fmt)
    assert len(back) == len(x)
    # Integer formats quantise; the tolerance is one LSB of each.
    tol = {"cf32": 1e-6, "ci16": 1e-4, "ci8": 1e-2, "cu8": 1e-2}[fmt]
    assert np.max(np.abs(back - x)) < tol


def test_parse_helpers():
    assert parse_hz("20M") == 20e6
    assert parse_hz("2.412G") == 2.412e9
    assert parse_hz("48k") == 48e3
    assert parse_hz("2e7") == 2e7
    assert parse_seconds("5ms") == pytest.approx(5e-3)
    assert parse_seconds("200us") == pytest.approx(200e-6)
    assert parse_seconds("1.5") == pytest.approx(1.5)


def test_ascii_waterfall_renders_without_a_plotting_library():
    _, _, result, _ = run_default_scene()
    art = ascii_waterfall(result, width=60, height=12)
    lines = art.splitlines()
    assert len(lines) > 12
    assert all(len(line) < 200 for line in lines)
    assert "MHz" in lines[0]


def test_cli_detect_and_extract(tmp_path, capsys):
    x, _, _ = default_scene()
    capture = tmp_path / "capture.cf32"
    save_iq(capture, x, "cf32")

    out_json = tmp_path / "dets.json"
    rc = main(
        [
            "detect",
            str(capture),
            "-r", "2M",
            "-f", "100M",
            "--json", str(out_json),
        ]
    )
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["capture"]["sample_rate_hz"] == 2e6
    assert len(payload["detections"]) >= 5
    printed = capsys.readouterr().out
    assert "fc [MHz]" in printed

    out_dir = tmp_path / "streams"
    rc = main(
        ["extract", str(capture), "-r", "2M", "-f", "100M", "-o", str(out_dir), "--quiet"]
    )
    assert rc == 0
    index = json.loads((out_dir / "index.json").read_text())
    assert len(index) == len(payload["detections"])
    for entry in index:
        assert (out_dir / entry["file"]).exists()
        assert entry["n_samples"] > 0


def test_cli_demo_runs(capsys):
    assert main(["demo", "-d", "20ms", "--quiet"]) == 0
    printed = capsys.readouterr().out
    assert "precision" in printed
