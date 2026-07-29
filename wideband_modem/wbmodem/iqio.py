"""Reading and writing raw IQ recordings.

Recorders disagree about sample formats, so the loader takes an explicit one.
Integer formats are scaled to roughly +/-1.0 full scale, which keeps the
``dBFS`` figures the detector reports comparable across formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

__all__ = ["FORMATS", "load_iq", "save_iq", "iter_iq_blocks", "write_detections_json"]

FORMATS: dict[str, tuple[Any, float, bool]] = {
    # name: (numpy dtype of one component, full-scale divisor, already complex)
    "cf32": (np.complex64, 1.0, True),
    "cf64": (np.complex128, 1.0, True),
    "ci16": (np.int16, 32768.0, False),
    "ci8": (np.int8, 128.0, False),
    "cu8": (np.uint8, 128.0, False),  # offset binary, as used by RTL-SDR
}


def load_iq(
    path: str | Path,
    fmt: str = "cf32",
    offset_samples: int = 0,
    count: int | None = None,
) -> np.ndarray:
    """Load a raw interleaved IQ file as ``complex64``."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {sorted(FORMATS)}")
    dtype, scale, is_complex = FORMATS[fmt]
    itemsize = np.dtype(dtype).itemsize
    per_sample = 1 if is_complex else 2

    n_items = -1 if count is None else count * per_sample
    raw = np.fromfile(
        path, dtype=dtype, count=n_items, offset=offset_samples * per_sample * itemsize
    )
    if is_complex:
        return raw.astype(np.complex64)
    if len(raw) % 2:
        raw = raw[:-1]
    vals = raw.astype(np.float32)
    if fmt == "cu8":
        vals -= 128.0
    vals /= scale
    return (vals[0::2] + 1j * vals[1::2]).astype(np.complex64)


def iter_iq_blocks(
    path: str | Path,
    fmt: str = "cf32",
    block_samples: int = 1 << 20,
    overlap_samples: int = 0,
) -> Iterator[tuple[int, np.ndarray]]:
    """Stream a long recording in overlapping blocks.

    Yields ``(start_sample_index, samples)``. The overlap lets a caller run a
    spectrogram per block without losing the frames that straddle a boundary;
    detections found in the overlap region will appear in both blocks and need
    de-duplicating by the caller.
    """
    if block_samples <= overlap_samples:
        raise ValueError("block_samples must exceed overlap_samples")
    start = 0
    step = block_samples - overlap_samples
    while True:
        block = load_iq(path, fmt, offset_samples=start, count=block_samples)
        if len(block) == 0:
            return
        yield start, block
        if len(block) < block_samples:
            return
        start += step


def save_iq(path: str | Path, x: np.ndarray, fmt: str = "cf32") -> Path:
    """Write complex samples to a raw interleaved file."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}")
    dtype, scale, is_complex = FORMATS[fmt]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_complex:
        np.asarray(x).astype(dtype).tofile(path)
        return path
    inter = np.empty(2 * len(x), dtype=np.float64)
    inter[0::2] = np.real(x)
    inter[1::2] = np.imag(x)
    inter *= scale
    if fmt == "cu8":
        inter += 128.0
    # Round rather than truncate: casting straight to an integer type biases
    # every sample towards zero by up to a full LSB.
    np.rint(inter, out=inter)
    lo, hi = np.iinfo(dtype).min, np.iinfo(dtype).max
    np.clip(inter, lo, hi, out=inter)
    inter.astype(dtype).tofile(path)
    return path


def write_detections_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
