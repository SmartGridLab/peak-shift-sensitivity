"""shift-analysis の出力が論文掲載値（Table 8 相当）と一致することを確認する。

事前に `make shift-analysis`（または `make verify`）で
data/output/peak_shift/shift_analysis_20240805.csv を生成しておくこと。
"""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / 'data' / 'output' / 'peak_shift' / 'shift_analysis_20240805.csv'
EXPECTED_CSV = ROOT / 'expected' / 'shift_analysis_2024-08-05_expected.csv'

TOLERANCE_PCT_POINTS = 0.01


@pytest.fixture(scope='module')
def merged() -> pd.DataFrame:
    if not OUTPUT_CSV.exists():
        pytest.skip(f'{OUTPUT_CSV} not found. Run `make shift-analysis` first.')
    actual = pd.read_csv(OUTPUT_CSV)
    expected = pd.read_csv(EXPECTED_CSV)
    return expected.merge(actual, on='shift_steps', suffixes=('_expected', '_actual'))


def test_all_shifts_present(merged):
    assert sorted(merged['shift_steps']) == list(range(-8, 9))


def test_reduction_rates_match_paper(merged):
    diff = (merged['reduction_rate_pct_actual'] - merged['reduction_rate_pct_expected']).abs()
    report = merged.assign(abs_diff=diff)[
        ['shift_steps', 'reduction_rate_pct_expected', 'reduction_rate_pct_actual', 'abs_diff']
    ]
    assert (diff <= TOLERANCE_PCT_POINTS).all(), f'mismatch:\n{report.to_string(index=False)}'
