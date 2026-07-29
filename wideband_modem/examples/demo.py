"""End-to-end walkthrough: build a scene, detect it, extract every signal.

    python examples/demo.py [output_dir]

Prints the detections, scores them against the generator's ground truth, and
writes one IQ file plus metadata sidecar per detected signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wbmodem import (
    EnergyDetector,
    ascii_waterfall,
    default_scene,
    extract_streams,
    match_detections,
)

SAMPLE_RATE = 2e6
CENTER_FREQ = 100e6


def main(out_dir: Path | None) -> int:
    x, truth, _ = default_scene(
        sample_rate_hz=SAMPLE_RATE, duration_s=0.05, center_freq_hz=CENTER_FREQ
    )
    print(f"capture: {len(x)} samples, {len(x) / SAMPLE_RATE * 1e3:.1f} ms "
          f"at {SAMPLE_RATE / 1e6:.1f} MS/s centred on {CENTER_FREQ / 1e6:.3f} MHz")
    print(f"scene contains {len(truth)} emitters\n")

    result = EnergyDetector().detect(x, SAMPLE_RATE, CENTER_FREQ)
    print(result.spectrogram.summary())
    print(f"threshold +{result.threshold_db_over_noise:.1f} dB over the noise floor, "
          f"occupancy {result.occupancy * 100:.1f}%\n")
    print(result.table())

    print()
    print(ascii_waterfall(result, width=96, height=28))

    report = match_detections(
        truth,
        result.detections,
        freq_tol_hz=2 * result.spectrogram.enbw_hz,
        time_tol_s=2 * result.spectrogram.window_duration_s,
    )
    print("\nscored against ground truth")
    print(report.report())

    print("\nper-signal streams")
    streams = extract_streams(x, SAMPLE_RATE, CENTER_FREQ, result.detections)
    for stream in streams:
        print(
            f"  #{stream.detection_id:<3} {stream.center_freq_hz / 1e6:11.6f} MHz  "
            f"bw {stream.bandwidth_hz / 1e3:8.2f} kHz  "
            f"rate {stream.sample_rate_hz / 1e3:8.2f} kHz  "
            f"{stream.n_samples:>7} samples  "
            f"(1/{stream.decimation} of the capture rate)"
        )

    total = sum(s.n_samples for s in streams)
    print(
        f"\n{total} samples across {len(streams)} streams, "
        f"{total / len(x) * 100:.1f}% of the wideband capture"
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for stream in streams:
            stream.save(out_dir / f"sig{stream.detection_id:03d}")
        print(f"wrote {len(streams)} streams to {out_dir}/")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    raise SystemExit(main(target))
