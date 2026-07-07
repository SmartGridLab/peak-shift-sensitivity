# Peak-Shift Sensitivity: BESS Peak Shaving under Forecast Peak-Time Error

Reproduction code for the simulation experiment (Experiment 2) of the paper:

> Quantifying the impact of demand-forecast peak-time shifts on battery energy
> storage system (BESS) peak shaving. Submitted to IEEE Access.
> (Citation and DOI will be added upon publication.)

The pipeline combines an LSTM demand forecaster, a mixed-integer linear
programming (MILP) charge/discharge optimizer, and a circular-shift
sensitivity analysis that quantifies how peak-time mismatch between forecast
and actual demand affects the peak-reduction rate.

## Scope

This repository contains only the simulation experiment (Experiment 2 in the
paper: LSTM forecast + MILP peak shaving + peak-time shift sensitivity).

The DERMS-integrated verification experiment (Experiment 1, Fig. 4) relies on
a commercial DERMS verification environment and its telemetry; the related
code and data are environment-specific and are not published here.

This code was extracted from a larger internal simulator; the git history
starts at this public release.

## Setup

Python 3.10+ is required.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The MILP solver is CBC, which ships with PuLP; no separate solver
installation should be needed.

## Data

### Bundled fixtures (`data/fixtures/`)

- `actual_demand_2024_08_05.csv` — actual demand of the target day (2024-08-05).
- `lstm_august_2024_08_05.csv`, `lstm_forecast_2024_08_05.csv` — the LSTM
  forecast used in the paper.
- `lstm_august_2024_08_05__shift_{-8..+8}slot.csv` — the forecast circularly
  shifted by −8 to +8 slots (1 slot = 30 min), generated with
  `scripts/shift_forecast_peak.py`.

Provenance: the demand values are derived from the "Area Supply and Demand
Results" (30-minute values) published by TEPCO Power Grid, Inc.
(https://www.tepco.co.jp/forecast/html/area_jukyu-j.html). As described in
the paper, the numerical value of the TEPCO-area demand in MW is used as a
portfolio demand in kW (an intentional 1/1000 scaling). Note that, for
historical reasons, some columns are named `demand_mw` / `demand_forecast_mw`
although the values are interpreted as portfolio kW.

### Raw TEPCO data (not bundled)

The raw TEPCO CSVs are not redistributed here. To run the full pipeline
(Tier 2 below), download them with:

```bash
make download-tepco   # or download manually into data/tepco/
```

## Reproducing the paper

| Paper artifact | Command | Output |
|---|---|---|
| Table 8 (shift sensitivity, Δ = −8..+8) | `make shift-analysis` | `data/output/peak_shift/shift_analysis_20240805.{csv,png}` |
| Fig. 5 (sensitivity bar chart, paper style) | `make plot-sensitivity` | `data/output/peak_shift/shift_sensitivity_20240805_eff092.png` |
| Fig. 6 (overlay, Δ = 0) | `make peak-shaving SHIFT=0` | `data/output/peak_shift/peak_shaving_overlay_20240805*.png` |
| Fig. 7 (overlay, Δ = −7, best case) | `make peak-shaving SHIFT=-7` | 〃 |
| Fig. 8 (overlay, Δ = +8, worst case) | `make peak-shaving SHIFT=+8` | 〃 |
| Table 5 (LSTM forecast metrics) | `make evaluate-forecast` | printed metrics table |
| Numerical check against the paper | `make verify` | pytest vs `expected/` |

Representative expected values (Table 8 of the paper):

| Shift Δ [slot] | Reduction rate R [%] |
|---|---|
| −7 (best) | 8.27 |
| −6 (peak-time match) | 7.95 |
| 0 (no shift) | 2.56 |
| +3 | −5.64 |
| +8 (worst) | −15.99 |

### Two-tier reproduction

- **Tier 1 (default, deterministic).** All commands above start from the
  bundled LSTM forecast fixtures, so the MILP optimization and the
  sensitivity analysis reproduce the paper values deterministically
  (CBC is deterministic for this problem).
- **Tier 2 (full pipeline, approximate).** `make download-tepco`,
  `make prepare-input`, then `make lstm-forecast` retrains the LSTM
  (Jan 1 – Jul 31, 2024; 10 epochs) and forecasts August 2024. The random
  seed is not fixed, so Table 5 values are reproduced only approximately;
  the bundled fixtures preserve the exact forecast used in the paper.
  Shifted forecasts can then be regenerated with
  `scripts/shift_forecast_peak.py --slots N`.

## Model and optimization parameters

Parameters not fully specified in the paper are documented here:

- Solver: PuLP + CBC (`PULP_CBC_CMD`), formulated as MILP with a binary
  variable prohibiting simultaneous charge/discharge.
- Objective: minimize peak + λ·Σ(c_t + d_t), with action penalty λ = 1e-4.
- BESS: capacity 40,000 kWh, max power 10,000 kW, charge/discharge
  efficiency 0.92, Δt = 0.5 h, initial SoC 50 % (20,000 kWh),
  terminal SoC within ±5 % of the initial SoC.
- LSTM: 2 layers, 128 hidden units, input length 96 slots, output 48 slots,
  MSE loss, Adam (lr = 1e-3), 10 epochs. Features: time-of-day / day-of-week /
  month with sin–cos encodings, lag-48, MA-48, MA-336, SD-336.

## License and citation

MIT License. If you use this code, please cite the paper above (BibTeX will
be added upon publication). The underlying demand data are provided by
TEPCO Power Grid, Inc.; please credit the original source when reusing them.
