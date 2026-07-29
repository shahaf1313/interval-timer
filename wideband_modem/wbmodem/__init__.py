"""wbmodem — a generic wideband receiver front end.

Stage one of a modulation-agnostic modem: take a wideband capture containing
an unknown mix of continuous and bursty emissions at unknown bandwidths, find
them all with an energy detector, and hand each one downstream as its own
stream of baseband samples labelled with centre frequency, bandwidth, start
time and end time.

    from wbmodem import EnergyDetector, extract_streams

    result = EnergyDetector().detect(iq, sample_rate_hz=20e6, center_freq_hz=2.412e9)
    print(result.table())
    streams = extract_streams(iq, 20e6, 2.412e9, result.detections)

Nothing in the detection path assumes a modulation, a symbol rate or a frame
structure — only that a signal has more energy than the noise around it.
"""

from __future__ import annotations

from .detector import DetectionResult, DetectorConfig, EnergyDetector, detect_signals
from .evaluate import EvaluationReport, Match, match_detections
from .extract import ExtractConfig, extract_stream, extract_streams
from .generate import SceneBuilder, default_scene
from .iqio import load_iq, save_iq
from .noise import estimate_noise_floor, threshold_db_from_pfa
from .plotting import ascii_waterfall, plot_result
from .spectrogram import Spectrogram, SpectrogramConfig, compute_spectrogram
from .types import Detection, SignalStream, TruthSignal

__version__ = "0.1.0"

__all__ = [
    "Detection",
    "DetectionResult",
    "DetectorConfig",
    "EnergyDetector",
    "EvaluationReport",
    "ExtractConfig",
    "Match",
    "SceneBuilder",
    "SignalStream",
    "Spectrogram",
    "SpectrogramConfig",
    "TruthSignal",
    "ascii_waterfall",
    "compute_spectrogram",
    "default_scene",
    "detect_signals",
    "estimate_noise_floor",
    "extract_stream",
    "extract_streams",
    "load_iq",
    "match_detections",
    "plot_result",
    "save_iq",
    "threshold_db_from_pfa",
    "__version__",
]
