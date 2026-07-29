"""Command-line front end.

    python -m wbmodem demo --ascii
    python -m wbmodem detect capture.cf32 -r 20M -f 2.412G --json dets.json
    python -m wbmodem extract capture.cf32 -r 20M -f 2.412G -o streams/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .detector import DetectorConfig, EnergyDetector
from .evaluate import match_detections
from .extract import ExtractConfig, extract_streams
from .generate import default_scene
from .iqio import FORMATS, load_iq, write_detections_json
from .plotting import ascii_waterfall
from .spectrogram import SpectrogramConfig

_SUFFIXES = {"k": 1e3, "K": 1e3, "m": 1e6, "M": 1e6, "g": 1e9, "G": 1e9}


def parse_hz(text: str) -> float:
    """Parse ``"20M"``, ``"2.412G"``, ``"48k"`` or plain ``"2e7"`` as Hz."""
    text = str(text).strip()
    if text and text[-1] in _SUFFIXES:
        return float(text[:-1]) * _SUFFIXES[text[-1]]
    return float(text)


def parse_seconds(text: str) -> float:
    """Parse ``"5ms"``, ``"200us"``, ``"1.5"`` (seconds) or ``"2e-3"``."""
    text = str(text).strip().lower()
    for suffix, scale in (("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9), ("s", 1.0)):
        if text.endswith(suffix):
            return float(text[: -len(suffix)]) * scale
    return float(text)


def _add_common_args(p: argparse.ArgumentParser, needs_input: bool = True) -> None:
    if needs_input:
        p.add_argument("input", help="raw IQ recording")
        p.add_argument(
            "-r",
            "--sample-rate",
            required=True,
            type=parse_hz,
            help="capture sample rate, e.g. 20M",
        )
        p.add_argument(
            "-f",
            "--center-freq",
            default="0",
            type=parse_hz,
            help="capture centre frequency, e.g. 2.412G",
        )
        p.add_argument(
            "--format", default="cf32", choices=sorted(FORMATS), help="sample format"
        )
        p.add_argument(
            "--max-samples", type=int, default=None, help="read at most N samples"
        )

    g = p.add_argument_group("spectrogram")
    g.add_argument("--nfft", type=int, default=1024)
    g.add_argument("--hop", type=int, default=None, help="default nfft/2")
    g.add_argument("--window", default="hann")
    g.add_argument(
        "--average", type=int, default=1, help="average N frames per pixel"
    )

    g = p.add_argument_group("detector")
    g.add_argument("--pfa", type=float, default=1e-6)
    g.add_argument(
        "--threshold-db",
        type=float,
        default=None,
        help="fixed threshold over the noise floor; overrides --pfa",
    )
    g.add_argument("--min-bandwidth", type=parse_hz, default=0.0)
    g.add_argument("--min-duration", type=parse_seconds, default=0.0)
    g.add_argument("--min-snr", type=float, default=0.0)
    g.add_argument(
        "--merge-gap-freq",
        type=parse_hz,
        default=0.0,
        help="bridge spectral nulls up to this wide",
    )
    g.add_argument(
        "--merge-gap-time",
        type=parse_seconds,
        default=0.0,
        help="bridge time gaps up to this long",
    )
    g.add_argument("--occupied-bw", type=float, default=0.99)
    g.add_argument("--edge-exclude", type=parse_hz, default=0.0)
    g.add_argument("--dc-notch", type=parse_hz, default=0.0)

    g = p.add_argument_group("output")
    g.add_argument("--json", default=None, help="write detections to this JSON file")
    g.add_argument("--ascii", action="store_true", help="print an ASCII waterfall")
    g.add_argument("--ascii-width", type=int, default=100)
    g.add_argument("--ascii-height", type=int, default=40)
    g.add_argument("--plot", default=None, help="write a PNG (needs matplotlib)")
    g.add_argument("--quiet", action="store_true")


def _configs(args: argparse.Namespace) -> tuple[SpectrogramConfig, DetectorConfig]:
    spec_cfg = SpectrogramConfig(
        nfft=args.nfft, hop=args.hop, window=args.window, average=args.average
    )
    det_cfg = DetectorConfig(
        pfa=args.pfa,
        threshold_db=args.threshold_db,
        min_bandwidth_hz=args.min_bandwidth,
        min_duration_s=args.min_duration,
        min_snr_db=args.min_snr,
        merge_gap_hz=args.merge_gap_freq,
        merge_gap_s=args.merge_gap_time,
        occupied_bw_fraction=args.occupied_bw,
        edge_exclude_hz=args.edge_exclude,
        dc_notch_hz=args.dc_notch,
    )
    return spec_cfg, det_cfg


def _emit(result, args: argparse.Namespace) -> None:
    if not args.quiet:
        print(result.spectrogram.summary())
        print(
            f"noise floor {np.median(result.noise_floor_db):.1f} dB/bin | "
            f"threshold +{result.threshold_db_over_noise:.1f} dB | "
            f"occupancy {result.occupancy * 100:.2f}%"
        )
        print(result.table())
    if args.ascii:
        print()
        print(
            ascii_waterfall(result, width=args.ascii_width, height=args.ascii_height)
        )
    if args.json:
        write_detections_json(args.json, result.to_dict())
        if not args.quiet:
            print(f"\nwrote {args.json}")
    if args.plot:
        from .plotting import plot_result

        plot_result(result, args.plot)
        if not args.quiet:
            print(f"wrote {args.plot}")


def _load(args: argparse.Namespace) -> np.ndarray:
    x = load_iq(args.input, args.format, count=args.max_samples)
    if len(x) == 0:
        raise SystemExit(f"{args.input}: no samples read")
    return x


def cmd_detect(args: argparse.Namespace) -> int:
    x = _load(args)
    spec_cfg, det_cfg = _configs(args)
    result = EnergyDetector(spec_cfg, det_cfg).detect(
        x, args.sample_rate, args.center_freq
    )
    _emit(result, args)
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    x = _load(args)
    spec_cfg, det_cfg = _configs(args)
    result = EnergyDetector(spec_cfg, det_cfg).detect(
        x, args.sample_rate, args.center_freq
    )
    _emit(result, args)

    ex_cfg = ExtractConfig(
        oversample=args.oversample,
        bandwidth_margin=args.bandwidth_margin,
        guard_s=args.guard,
    )
    streams = extract_streams(
        x, args.sample_rate, args.center_freq, result.detections, ex_cfg
    )
    out_dir = Path(args.out_dir)
    index = []
    for stream in streams:
        name = (
            f"sig{stream.detection_id:03d}"
            f"_{stream.center_freq_hz / 1e6:.4f}MHz"
            f"_{stream.bandwidth_hz / 1e3:.1f}kHz"
        )
        iq_path = stream.save(out_dir / name)
        index.append({"file": iq_path.name, **stream.metadata()})
        if not args.quiet:
            print(
                f"  #{stream.detection_id:>3}  {iq_path.name}  "
                f"{stream.n_samples} samples @ {stream.sample_rate_hz / 1e3:.2f} kHz"
            )
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    if not args.quiet:
        print(f"\nwrote {len(streams)} streams to {out_dir}/")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    x, truth, scene = default_scene(
        sample_rate_hz=args.sample_rate,
        duration_s=args.duration,
        center_freq_hz=args.center_freq,
        seed=args.seed,
    )
    spec_cfg, det_cfg = _configs(args)
    result = EnergyDetector(spec_cfg, det_cfg).detect(
        x, args.sample_rate, args.center_freq
    )
    _emit(result, args)

    spec = result.spectrogram
    report = match_detections(
        truth,
        result.detections,
        freq_tol_hz=2 * spec.enbw_hz,
        time_tol_s=2 * spec.window_duration_s,
    )
    print("\nground truth vs detections")
    print(report.report())

    if args.out_dir:
        streams = extract_streams(
            x, args.sample_rate, args.center_freq, result.detections
        )
        out_dir = Path(args.out_dir)
        for stream in streams:
            stream.save(out_dir / f"sig{stream.detection_id:03d}")
        print(f"\nwrote {len(streams)} streams to {out_dir}/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wbmodem",
        description="Wideband energy detector and per-signal channelizer.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("detect", help="detect signals in a recording")
    _add_common_args(p)
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("extract", help="detect, then write one IQ file per signal")
    _add_common_args(p)
    g = p.add_argument_group("extraction")
    g.add_argument("-o", "--out-dir", default="streams")
    g.add_argument("--oversample", type=float, default=2.0)
    g.add_argument("--bandwidth-margin", type=float, default=1.25)
    g.add_argument("--guard", type=parse_seconds, default=0.0)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("demo", help="generate a synthetic scene and score the detector")
    _add_common_args(p, needs_input=False)
    p.add_argument("-r", "--sample-rate", type=parse_hz, default=2e6)
    p.add_argument("-f", "--center-freq", type=parse_hz, default=100e6)
    p.add_argument("-d", "--duration", type=parse_seconds, default=0.05)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("-o", "--out-dir", default=None, help="also write the IQ streams")
    p.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
