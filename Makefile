# ==============================================================================
# Peak-Shift Sensitivity – Makefile
# ==============================================================================

PYTHON         ?= python
DATE           ?= 2024-08-05
OUT_DIR        ?= data/output/peak_shift
FIXTURES       := data/fixtures
LP_EFF         ?= 0.92
LP_CAPACITY    ?= 40000
LP_POWER       ?= 10000
LP_INITIAL_SOC ?= 50
SHIFT          ?= 0
VERSION        ?=

ACTUAL_CSV     := $(FIXTURES)/actual_demand_2024_08_05.csv
FORECAST_CSV   := $(FIXTURES)/lstm_forecast_2024_08_05.csv
FIXTURE_PREFIX := lstm_august_2024_08_05__shift

.DEFAULT_GOAL := help

.PHONY: help shift-analysis plot-sensitivity peak-shaving peak-shaving-all-shifts evaluate-forecast verify \
        download-tepco prepare-input lstm-forecast clean

# ------------------------------------------------------------------------------
help:
	@echo ""
	@echo "Usage: make <target> [DATE=YYYY-MM-DD] [SHIFT=+N] [VERSION=label]"
	@echo ""
	@echo "  -- Tier 1: reproduce the paper from bundled forecast fixtures --"
	@echo "  shift-analysis           Sensitivity analysis over -8..+8 slot shifts (Table 8)"
	@echo "  plot-sensitivity         Paper-style sensitivity bar chart (Fig. 5)"
	@echo "  peak-shaving             Single-shift LP peak-shaving overlay (Fig. 6: SHIFT=0, Fig. 7: SHIFT=-7, Fig. 8: SHIFT=+8)"
	@echo "  peak-shaving-all-shifts  Overlay figures for all 17 shifts"
	@echo "  evaluate-forecast        LSTM forecast error metrics on the target day (Table 5)"
	@echo "  verify                   Run shift-analysis and check results against expected/"
	@echo ""
	@echo "  -- Tier 2: full pipeline from TEPCO data (results are approximate; no RNG seed) --"
	@echo "  download-tepco           Download TEPCO area demand CSVs into data/tepco/"
	@echo "  prepare-input            Build the LSTM input CSV from TEPCO data"
	@echo "  lstm-forecast            Train the LSTM (Jan-Jul 2024) and forecast August 2024"
	@echo ""
	@echo "Defaults: DATE=$(DATE)  LP_EFF=$(LP_EFF)  LP_CAPACITY=$(LP_CAPACITY)  LP_POWER=$(LP_POWER)"
	@echo ""

# ------------------------------------------------------------------------------
# シフト感度分析: 予測を -8〜+8 コマシフトして LP を一括実行 (Fig. 5, Table 8)
# ------------------------------------------------------------------------------
shift-analysis:
	$(PYTHON) scripts/run_shift_analysis.py \
		--actual   $(ACTUAL_CSV) \
		--actual-col demand_mw \
		--forecast $(FORECAST_CSV) \
		--forecast-col demand_forecast_mw \
		--fixture-dir $(FIXTURES) \
		--fixture-prefix $(FIXTURE_PREFIX) \
		--date $(DATE) \
		--out-dir $(OUT_DIR) \
		--capacity-kwh $(LP_CAPACITY) \
		--power-kw $(LP_POWER) \
		--eff $(LP_EFF) \
		--initial-soc-pct $(LP_INITIAL_SOC) \
		$(if $(VERSION),--version $(VERSION),)

# ------------------------------------------------------------------------------
# 論文体裁の感度棒グラフ (Fig. 5)
# ------------------------------------------------------------------------------
plot-sensitivity:
	$(PYTHON) scripts/plot_shift_sensitivity.py \
		--csv $(FIXTURES)/lstm_august_2024_08_05.csv \
		--date $(DATE) \
		--actual-col demand_kw \
		--forecast-col demand_forecast_kw \
		--slots-min -8 --slots-max 8 \
		--eff $(LP_EFF) \
		--capacity-kwh $(LP_CAPACITY) \
		--power-kw $(LP_POWER) \
		--out-dir $(OUT_DIR) \
		--version eff092

# ------------------------------------------------------------------------------
# 単一シフトの LP ピークシェービング可視化 (SHIFT=0 がデフォルト)
# SHIFT=0  → lstm_august_2024_08_05__shift_0slot.csv
# SHIFT=+3 → lstm_august_2024_08_05__shift_+3slot.csv
# ------------------------------------------------------------------------------
_FORECAST_FOR_SHIFT = $(FIXTURES)/$(FIXTURE_PREFIX)_$(SHIFT)slot.csv

peak-shaving:
	$(PYTHON) scripts/lp_peak_shaving_analysis.py \
		--csv-actual   $(ACTUAL_CSV) \
		--csv-forecast $(_FORECAST_FOR_SHIFT) \
		--actual-col demand_mw \
		--forecast-col demand_forecast_mw \
		--date $(DATE) \
		--out-dir $(OUT_DIR) \
		--lp \
		--lp-capacity-kwh $(LP_CAPACITY) \
		--lp-power-kw $(LP_POWER) \
		--lp-eff $(LP_EFF) \
		--lp-initial-soc-pct $(LP_INITIAL_SOC) \
		$(if $(VERSION),--version $(VERSION),)

# ------------------------------------------------------------------------------
# 全シフト (-8〜+8) の LP ピークシェービング可視化を一括生成
# ------------------------------------------------------------------------------
SHIFT_VALUES := -8 -7 -6 -5 -4 -3 -2 -1 0 +1 +2 +3 +4 +5 +6 +7 +8

peak-shaving-all-shifts:
	@for s in $(SHIFT_VALUES); do \
		label="shift$${s}"; \
		if [ -n "$(VERSION)" ]; then label="$(VERSION)_$${label}"; fi; \
		echo "==> peak-shaving SHIFT=$${s} VERSION=$${label}"; \
		$(MAKE) peak-shaving SHIFT=$${s} VERSION=$${label} || exit $$?; \
	done

# ------------------------------------------------------------------------------
# LSTM 予測誤差メトリクス (Table 5)
# ------------------------------------------------------------------------------
evaluate-forecast:
	$(PYTHON) scripts/evaluate_shift_forecast_metrics.py \
		--csv $(FIXTURES)/lstm_august_2024_08_05.csv \
		--date $(DATE)

# ------------------------------------------------------------------------------
# 再現検証: shift-analysis を実行し、論文掲載値 (expected/) と照合
# ------------------------------------------------------------------------------
verify: shift-analysis
	$(PYTHON) -m pytest tests/ -v

# ------------------------------------------------------------------------------
# Tier 2: TEPCO 公開データからのフル再現（近似）
# ------------------------------------------------------------------------------
download-tepco:
	bash scripts/download_tepco_data.sh data/tepco

prepare-input:
	$(PYTHON) scripts/prepare_lstm_input.py \
		--tepco-dir data/tepco \
		--year 2024 \
		--output data/output/lstm_input_2024.csv

lstm-forecast:
	$(PYTHON) scripts/run_lstm_demand_prediction.py \
		--input data/output/lstm_input_2024.csv \
		--output data/output/lstm_predictions_2024_august.csv \
		--start-day 2024-08-01 --end-day 2024-09-01 --epochs 10

# ------------------------------------------------------------------------------
clean:
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete
	find . -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
