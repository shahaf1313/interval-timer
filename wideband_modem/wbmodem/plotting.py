"""Rendering a detection result.

Two renderers: a dependency-free ASCII waterfall that works over SSH, and a
matplotlib PNG for when the extra dependency is available. Both draw the
detections on top of the spectrogram so a bad threshold is obvious at a
glance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .detector import DetectionResult

__all__ = ["ascii_waterfall", "plot_result"]

_SHADES = " .:-=+*#%@"


def ascii_waterfall(
    result: DetectionResult,
    width: int = 100,
    height: int = 40,
    db_range: float = 30.0,
    mark_detections: bool = True,
) -> str:
    """Render the spectrogram as text, frequency across, time down.

    Intensity is SNR over the estimated noise floor, clipped to ``db_range``.
    Detections are outlined with box-drawing characters.
    """
    spec = result.spectrogram
    snr_db = 10.0 * np.log10(
        np.maximum(spec.power, 1e-30) / np.maximum(result.noise_floor[:, None], 1e-30)
    )

    width = max(16, min(width, spec.n_freq))
    height = max(4, min(height, spec.n_time))
    f_edges = np.linspace(0, spec.n_freq, width + 1).astype(int)
    t_edges = np.linspace(0, spec.n_time, height + 1).astype(int)

    grid = np.empty((height, width))
    for r in range(height):
        rows = snr_db[:, t_edges[r] : max(t_edges[r + 1], t_edges[r] + 1)]
        for c in range(width):
            block = rows[f_edges[c] : max(f_edges[c + 1], f_edges[c] + 1), :]
            grid[r, c] = block.max() if block.size else -np.inf

    levels = np.clip(grid / db_range, 0.0, 1.0)
    idx = np.clip((levels * (len(_SHADES) - 1)).round().astype(int), 0, len(_SHADES) - 1)
    canvas = [[_SHADES[i] for i in row] for row in idx]

    if mark_detections:
        for det in result.detections:
            c0 = int(np.clip(np.searchsorted(f_edges, spec.freq_to_bin(det.f_low_edge_hz)) - 1, 0, width - 1))
            c1 = int(np.clip(np.searchsorted(f_edges, spec.freq_to_bin(det.f_high_edge_hz)) - 1, 0, width - 1))
            r0 = int(np.clip(np.searchsorted(t_edges, spec.time_to_frame(det.t_start_s)) - 1, 0, height - 1))
            r1 = int(np.clip(np.searchsorted(t_edges, spec.time_to_frame(det.t_end_s)) - 1, 0, height - 1))
            for c in range(c0, c1 + 1):
                canvas[r0][c] = "-" if canvas[r0][c] == " " else canvas[r0][c]
                canvas[r1][c] = "-" if canvas[r1][c] == " " else canvas[r1][c]
            for r in range(r0, r1 + 1):
                canvas[r][c0] = "|"
                canvas[r][c1] = "|"
            tag = str(det.detection_id)
            if r0 > 0 and c0 + len(tag) <= width:
                for k, ch in enumerate(tag):
                    canvas[r0 - 1][c0 + k] = ch

    lines = []
    f_lo = spec.freqs_hz[0] / 1e6
    f_hi = spec.freqs_hz[-1] / 1e6
    lines.append(f"{'':>10}{f_lo:<.4f} MHz{'':^{max(1, width - 26)}}{f_hi:>.4f} MHz")
    lines.append(f"{'':>10}+{'-' * (width - 2)}+")
    for r, row in enumerate(canvas):
        t_ms = spec.times_s[min(t_edges[r], spec.n_time - 1)] * 1e3
        lines.append(f"{t_ms:>8.2f}ms|{''.join(row)}|")
    lines.append(f"{'':>10}+{'-' * (width - 2)}+")
    lines.append(
        f"{'':>10}shading = SNR over noise floor, 0..{db_range:.0f} dB  "
        f"('{_SHADES[1]}' low, '{_SHADES[-1]}' high)"
    )
    return "\n".join(lines)


def plot_result(
    result: DetectionResult,
    path: str | Path,
    db_range: float = 40.0,
    dpi: int = 130,
) -> Path:
    """Save a PNG of the spectrogram with detection boxes. Needs matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "plot_result needs matplotlib; install it or use ascii_waterfall()"
        ) from exc

    spec = result.spectrogram
    snr_db = 10.0 * np.log10(
        np.maximum(spec.power, 1e-30) / np.maximum(result.noise_floor[:, None], 1e-30)
    )

    extent = [
        spec.times_s[0] * 1e3,
        spec.times_s[-1] * 1e3,
        (spec.freqs_hz[0] - spec.center_freq_hz) / 1e6,
        (spec.freqs_hz[-1] - spec.center_freq_hz) / 1e6,
    ]
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(
        snr_db,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=db_range,
        cmap="viridis",
        interpolation="nearest",
    )
    for det in result.detections:
        ax.add_patch(
            Rectangle(
                (det.t_start_s * 1e3, (det.f_low_edge_hz - spec.center_freq_hz) / 1e6),
                det.duration_s * 1e3,
                det.bandwidth_hz / 1e6,
                fill=False,
                edgecolor="white",
                linewidth=1.0,
            )
        )
        ax.text(
            det.t_start_s * 1e3,
            (det.f_high_edge_hz - spec.center_freq_hz) / 1e6,
            f"{det.detection_id}",
            color="white",
            fontsize=8,
            va="bottom",
        )
    ax.set_xlabel("time [ms]")
    ax.set_ylabel(f"frequency offset from {spec.center_freq_hz / 1e6:.3f} MHz [MHz]")
    ax.set_title(
        f"{len(result.detections)} detections | "
        f"threshold {result.threshold_db_over_noise:.1f} dB over noise"
    )
    fig.colorbar(im, ax=ax, label="SNR over noise floor [dB]")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
