# wbmodem — wideband energy detector and channelizer

Stage one of a modulation-agnostic modem. It takes a wideband capture holding
an unknown mix of continuous and bursty emissions, at unknown bandwidths and
unknown modulations, and hands each one downstream as its own stream of
baseband samples labelled with **centre frequency, bandwidth, start time and
end time**.

Nothing in the detection path assumes a modulation, a symbol rate or a frame
structure. The only assumption is that a signal has more energy than the noise
around it.

```
wideband IQ ──► spectrogram ──► noise floor ──► multi-scale CFAR ──► components
                                                                        │
                        ┌───────────────────────────────────────────────┘
                        ▼
                  measure fc / bw / t_start / t_end
                        │
                        ▼
              mix ──► low-pass ──► decimate ──► SignalStream (+ metadata)
```

## Quick start

```bash
pip install -e .            # numpy + scipy; matplotlib optional, for plots

python -m wbmodem demo --ascii                       # synthetic scene, scored
python -m wbmodem detect capture.cf32 -r 20M -f 2.412G --json dets.json
python -m wbmodem extract capture.cf32 -r 20M -f 2.412G -o streams/
```

```python
from wbmodem import EnergyDetector, extract_streams, load_iq

iq = load_iq("capture.cf32", "cf32")
result = EnergyDetector().detect(iq, sample_rate_hz=20e6, center_freq_hz=2.412e9)
print(result.table())

for stream in extract_streams(iq, 20e6, 2.412e9, result.detections):
    print(stream.center_freq_hz, stream.bandwidth_hz, stream.sample_rate_hz)
    stream.save(f"sig{stream.detection_id:03d}")   # .cf32 + .json sidecar
```

Already have a spectrogram? Feed it directly, as long as it is calibrated the
way `compute_spectrogram` produces it:

```python
result = EnergyDetector().detect_spectrogram(spec)
```

On the bundled synthetic scene — three continuous emitters (QPSK, an
unmodulated beacon, a wideband noise-like carrier), an 8-PSK burst, an FSK
burst, a chirp and a four-dwell frequency hopper — the detector currently
finds all ten with no false alarms, a median centre-frequency error of 340 Hz
and a median start-time error of 128 µs (half a frame).

## How it works, and why

### 1. A calibrated spectrogram

`compute_spectrogram` normalises by the window's energy, so complex white
noise of variance σ² reads exactly σ² per bin. Every threshold downstream is
expressed as a ratio to the noise floor, which is what makes a CFAR threshold
mean what it says and an SNR readout an actual SNR. Summing bin powers over a
band and dividing by `nfft` recovers the signal power in that band.

### 2. Noise floor

Estimating the floor in a wideband scene is the hard part, because two
effects fight each other:

* A **per-bin statistic over time** (a low percentile) ignores bursts — which
  is what we want — but a continuous carrier occupies its bin for the whole
  capture, so the statistic locks onto the signal instead of the noise.
* A **statistic over frequency** rejects continuous carriers, but a wide
  emitter drags it up too.

The estimator combines both. A de-biased 25th percentile over time gives a
first guess per bin; bins sitting significantly above a low percentile taken
*across* frequency are marked occupied and the floor is interpolated across
them; the pass repeats, so emitters wider than the test window are peeled off
from their edges inwards.

Two details that are easy to get wrong and expensive when you do:

* The occupancy guard is derived from the measured scatter of the estimate
  (a MAD, a few sigma), not fixed at some round number of dB. A fixed 6 dB
  guard silently swallows every continuous emission weaker than 6 dB per bin
  into the noise floor, after which it can never be detected.
* The exclusion is widened by a few bins before interpolating. The bins right
  at an emission's edge are partly lit — not enough to trip the guard, but
  enough to anchor the interpolation high and leave the floor raised across
  the whole emission.

### 3. Multi-scale detection

A single-pixel threshold has no processing gain: one bin of one frame is one
sample of a chi-squared variable, so the threshold sits ~12 dB above the noise
to keep false alarms rare, and any emission below that in-band SNR is
invisible however long it transmits.

So the detector also thresholds **box averages** over a ladder of shapes
(`kf` bins × `kt` frames, powers of two in both axes). Averaging cuts the
threshold by roughly the square root of the number of independent cells in the
box — a fraction of a dB instead of twelve. Each box gets its own threshold,
computed from how many *independent* samples it really holds: neighbouring
bins overlap through the window main lobe and neighbouring frames overlap
through the hop, so the raw box size overstates the gain and is discounted by
the window ENBW and the overlap factor.

Three mechanisms keep that sensitivity from destroying the localisation, which
is the part that a naive OR-over-scales gets badly wrong:

* **Clipping.** Pixels are capped at 6 dB over the floor before integration.
  A weak signal spread evenly across a box is untouched; a strong narrow one
  contributes only its cap and can no longer carry a large box on its own.
  The clipped statistic has different noise moments, so its threshold comes
  from those moments (with a Cornish-Fisher skewness correction) rather than
  from the Gamma law.
* **Erosion.** A pixel is claimed at a scale only if *every* box containing it
  cleared the threshold. For an emission of width `n` needing a fill fraction
  `phi` to trigger, the surviving core is `n - 2·phi·k` wide — always inside
  the true support, so a coarse scale can add sensitivity without ever
  inflating an extent.
* **Valley rejection.** Erosion alone still fails for two emitters spaced
  about one box apart: every box between them is lit, that region is wider
  than the box, and the two get welded together with the empty band between
  them swallowed. Such a bridge has a specific signature — the box is dimmer
  than the boxes one full width to either side. A real emission never is.

The frequency ladder is capped at 1/64 of the band for the same reason:
doubling a box buys about 1.5 dB, and a box wide enough to span two emitters
costs far more than that.

### 4. Measurement

Parameters are measured from the underlying power, not from the binary mask,
which is only ever an approximation of the emission's support:

* **Bandwidth** is the occupied bandwidth — the band holding 99% of the
  noise-subtracted energy — with the window's ENBW deconvolved out. The
  cumulative sum runs over the *signed* excess so that noise-only bins in the
  search margin contribute about zero; clipping at zero first would bias the
  margin upwards and inflate the bandwidth of weak signals.
* **Start and end** come from an in-band power profile rather than the mask.
  Averaging across the occupied band buys processing gain, so the same
  false-alarm rate buys a much lower threshold and a tighter estimate of when
  the burst actually started.
* Anything touching the first or last frame is flagged `truncated_start` /
  `truncated_end`, and something touching both is reported as `continuous`.

Fragments of one marginal emission are then fused when most of the smaller
sits inside the larger — unless the fragment is much stronger than its
container, which is how a narrowband carrier inside a wide weak emission
announces that it is a signal in its own right rather than a piece of the
band around it.

### 5. Extraction

Each detection is mixed to DC, low-pass filtered to its own bandwidth and
decimated to the lowest rate that still carries it. Two properties matter
downstream:

* The mixer phase is referenced to the **capture's** sample index and the
  output grid is snapped to the decimation lattice, so a detection always
  yields the same samples regardless of how much guard time was requested, and
  two streams cut from one capture share a phase origin.
* `stream.t_start_s` is the true timestamp of the first sample, with the
  filter group delay already removed, so `stream.time_axis()` lines up with
  the wideband timeline.

The decimation factor is the largest one that keeps both the requested
oversampling and the filter's transition band inside the output Nyquist zone,
so nothing that survives the filter can alias.

## Tuning

Everything physical is in Hz and seconds, so a configuration survives a change
of FFT size or sample rate.

| Knob | Meaning |
| --- | --- |
| `pfa` | per-pixel false-alarm rate behind every threshold (default 1e-6) |
| `threshold_db` | fixed threshold over the floor; overrides `pfa` at every scale |
| `min_bandwidth_hz`, `min_duration_s`, `min_snr_db` | discard anything smaller |
| `merge_gap_hz`, `merge_gap_s` | spectral nulls / time gaps that must not split an emission |
| `merge_overlap`, `keep_nested_above_db` | fragment fusing, and the exception for a strong signal nested in a weak one |
| `freq_scales`, `time_scales` | the integration ladder; `(1,)` for a plain single-pixel detector |
| `clip_snr_db`, `scale_erosion`, `valley_tolerance` | the three localisation mechanisms above |
| `occupied_bw_fraction` | 0.99 by default, the ITU occupied-bandwidth definition |
| `edge_exclude_hz`, `dc_notch_hz` | ignore filter roll-off at the band edges and LO leakage at the centre |

`SpectrogramConfig` sets the resolution trade: `nfft` for frequency, `hop` for
time, `average` to trade time resolution for a lower threshold.

## Cost

Detection is `O(pixels × scales)`; the spectrogram holds `2 × len(x)` pixels at
50% overlap, and the ladder is capped at 8 octaves per axis. A 0.2 s capture at
20 MS/s (4 M samples, 8 M pixels, 40 scales) takes about 10 s to detect and 9 s
to extract 11 streams. To go faster, restrict `freq_scales`/`time_scales`,
raise `hop`, or process in blocks with `iter_iq_blocks`.

## Limitations

* One emission per time/frequency region. Two signals overlapping in *both*
  time and frequency come out as one detection; separating those needs
  spatial or cyclostationary processing, not energy.
* A scale ladder cannot localise better than its box. A detection that only
  the coarsest scale found carries an extent uncertain by roughly that box,
  which is why the measurement stage re-reads the parameters from the power.
* A frequency hopper is reported as one detection per dwell. Tying the dwells
  together into one emitter is a tracking problem, and belongs in stage two.
* The noise floor is estimated per bin over the whole capture. A gain change
  part-way through a capture is not tracked; split the capture instead.
* Bandwidth is measured as 99% occupied bandwidth, which for a pulse-shaped
  carrier reads a little under the `(1 + rolloff) × symbol_rate` figure a
  datasheet would quote.

## Layout

| Module | Role |
| --- | --- |
| `spectrogram.py` | calibrated STFT |
| `noise.py` | noise floor, CFAR thresholds, clipped-statistic moments |
| `detector.py` | multi-scale detection, component measurement |
| `extract.py` | per-signal mixer / filter / decimator |
| `types.py` | `Detection`, `SignalStream`, `TruthSignal` |
| `generate.py` | synthetic scenes with ground truth |
| `evaluate.py` | scoring a run against ground truth |
| `iqio.py` | raw IQ formats (cf32, ci16, ci8, cu8) |
| `plotting.py` | ASCII waterfall, and a matplotlib PNG if available |
| `cli.py` | `detect` / `extract` / `demo` |

## Next stages

The detector deliberately stops at "here is a stream of samples and what we
know about it". A stage two would take each `SignalStream` and estimate symbol
rate (cyclostationary or from the envelope spectrum), classify the modulation,
recover carrier and timing, and demodulate. The metadata carried on the stream
— centre frequency, bandwidth, duration, SNR — is what those stages need to
choose their own parameters.
