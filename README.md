# BESS peak-shaving sensitivity to circular shifts of demand forecasts

This repository reproduces the numerical experiment (Experiment 2) of the
revised IEEE Access manuscript. It uses the original-scale Tokyo-area demand
published by TEPCO Power Grid and a hypothetical aggregate BESS.

## Reproduce the revised results

Use Python 3.12. The requirements pin the numerical libraries, including
PuLP 3.3.0 and its bundled CBC solver, and the manuscript's Matplotlib 3.9.4.
The verified environment is macOS
arm64, Python 3.12.13 and CBC 2.10.3. The numerical verification below
will flag differences at manuscript precision on another platform.

```sh
python3.12 -m venv .venv-revision
source .venv-revision/bin/activate
python -m pip install -r revision/requirements.txt
python revision/scripts/reproduce.py \
    --raw-dir data/tepco/revision --out-dir data/output/revised --download
python revision/scripts/verify_outputs.py data/output/revised
```

The run directory must be new. Keep completed runs and choose a different
`--out-dir` when repeating a run. To run without network access, omit
`--download` and point `--raw-dir` at the official TEPCO input files
described below. With `--main-only`, the pipeline skips the BESS and lambda
sweeps. Pass the same flag to `verify_outputs.py` for this run.

| Output | Path within the run |
| --- | --- |
| Fig. 5, reduction distributions over 17 shifts | `figures/fig5.pdf` and `fig5.svg` |
| Fig. 6, three unshifted planning inputs | `figures/fig6.pdf` and `fig6.svg` |
| Fig. 7, paired daily changes from three ex post corrections | `figures/fig7.pdf` and `fig7.svg` |
| Fig. 8, demand, BESS power and SoC for 3 March 2026 | `figures/fig8.pdf` and `fig8.svg` |
| Tables 3 and 5, forecast errors and peak-reduction statistics | `tables/table3.csv`, `tables/table5.csv` |
| Tables S1–S3, lambda, BESS and seasonal statistics | `tables/tableS1.csv` through `tableS3.csv` |

The run also saves selected dates, interval-level schedules, daily metrics,
input hashes, and CBC logs. Verification checks the tabulated values and the
figure statistics at the precision printed in the revised manuscript. The
observed CBC wall-clock times depend on the host.

## Inputs and evaluation

- The official source URLs and archive checksums are listed in
  [the input manifest](revision/data/inputs.json). For an offline run place
  `AREA_YYYYMM.zip` files under `<raw-dir>/primary/` and
  `eria_jukyu_202506_03.csv` under `<raw-dir>/supplement/`. A flat directory
  containing these same files also works. TEPCO archives are downloaded from
  the provider or supplied locally.
- The 30-minute interval-energy records (`AREA_JISEKI`, April 2022 to March
  2026) are converted from kWh to average kW. The 38 intervals on 14 June 2025,
  05:00–24:00 JST, use the separate official mean-power series. The assembled
  series is checked against the manuscript snapshot before case selection.
  If the provider has revised the source values and the snapshot check fails,
  supply the original archives at `--raw-dir`.
- Each season contributes the five evaluation-year dates with the largest
  uncontrolled daily peak. The earlier date breaks a tie. All dates are
  selected from actual demand, before reading the forecasts. The supplied
  LSTM forecast fixture contains timestamps and the frozen forecasts only.
  Seasonal naïve input is computed from the same slots one week earlier.
- Use `--retrain` to train the LSTM from the official data and run the full
  pipeline with the resulting predictions. Training uses April 2022–March
  2024, validation uses April 2024–March 2025, and evaluation uses April
  2025–March 2026, all in JST. It uses seed 42, two CPU threads, and early
  stopping. The manuscript run selected epoch 44. Frozen forecasts are the
  reference for reproducing the exact manuscript numbers across platforms.
- The baseline BESS is 11.4012 GW and 45.6048 GWh, with equal initial and
  terminal SoC of 50% and 85% round-trip efficiency. The values retain the
  fixed settings of the preceding regional-scale experiment. They were not
  resized using the expanded demand history. The daily runs use the
  same two-stage MILP, 30-minute slots, `lambda = 1e-4`, and the fixed
  schedule applied to actual demand. The additional BESS settings are
  tested at zero shift. The lambda sweep uses the baseline BESS.
- Circular shifting displaces the whole forecast, including wraparound at
  the day boundary. Perfect foresight and the ex post corrections use
  evaluation-day actuals as diagnostic inputs. The separate DERMS staging
  experiment is outside the scope of this reproduction.

The earlier single-day experiment is retained in
[the initial-submission instructions](docs/initial-submission.md).

The revised pipeline reuses the simulation and forecast routines from the
experiment-development snapshot `d8b8b79964bdef234fa81ac915b344797844b029`.
