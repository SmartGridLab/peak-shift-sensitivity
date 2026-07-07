# -*- coding: utf-8 -*-
"""
Overlay visualizer: LSTM予測 × LPピークシェービング
- Gray:  Actual Demand (Before)
- Orange dashed: LSTM Forecast (Input)
- Red:   Actual Demand (After, LP) …… LPPeakShavingOptimizer（予測に基づく制御を実績に適用）

【概要】
このスクリプトは、電力需要データ（実績・予測）に対して、線形計画法（LP）を用いた蓄電池による
ピークカット（ピークシェービング）制御をシミュレーションし、その効果を可視化・分析するためのツールです。
「もしこの予測データに基づいて蓄電池を制御していたら、どれくらいピークを削減できたか？」を検証するのに使われます。

【主な機能】
1. ピークカットのシミュレーション
   - 指定した日の需要データ（予測または実績）を入力として、最適な蓄電池の充放電スケジュールを計算します。
   - 蓄電池のスペック（容量、出力、効率、初期SoCなど）を細かく設定可能です。

2. 結果の可視化（グラフ作成）
   - 以下の3つのグラフを1枚の画像（PNG）として出力します。
     - 上段: 需要電力の推移（灰色：制御前の実績、オレンジ点線：予測、赤：制御後の実績）。ピークがどれだけ下がったかを確認できます。
     - 中段: 充放電電力（赤バー）。いつ充電・放電したかがわかります。
     - 下段: SoC（蓄電残量）の推移。バッテリー切れや満充電の状況を確認できます。

3. 詳細データのCSV出力
   - シミュレーション結果（時刻ごとの充放電量、制御後の需要、SoCなど）をCSVファイルとして保存します。

【使い方】
python scripts/lp_peak_shaving_analysis.py \
    --csv data/input/demand_forecast_sample.csv \
    --date 2024-08-05 \
    --out-dir data/output/battery_control/deterministic \
    --lp \
    --lp-capacity-kwh 50 --lp-power-mw 25 --lp-eff 0.92 \
    --lp-initial-soc-pct 50 --lp-terminal-soc-min-pct 0 --lp-timestep-h 0.5

必要列：timestamp, demand_mw, demand_predict
    ※ 列名が異なる場合は --ts-col/--actual-col/--forecast-col で指定

注：
- LP（LPPeakShavingOptimizer）が見つからなければ自動スキップ
- タイムゾーンは Asia/Tokyo に正規化（NaiveはJSTとして取り込み）

修正ポイント
- FDS に関する描画は全て除去（青線、充放電バー、SoC、ピーク線、凡例）
- LP に timestep_h を明示（30分刻み → 0.5h）
- 初期 SoC と終端 SoC を指定可能（既定: 初期 50%, 終端は initSoC ±5% 許容。--lp-final-soc-equality 指定時は等式）
- 互換性のため、最小引数で初期化 → 追加引数は try/except で柔軟に設定
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== optional imports =====
_HAS_LP = False
try:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from peakshift.lp_peak_shaving_optimizer import LPPeakShavingOptimizer
    _HAS_LP = True
except Exception:
    pass

JST = 'Asia/Tokyo'


def _resolve_col_name(df, primary, defaults):
    if primary in df.columns:
        return primary
    for d in defaults:
        if d in df.columns:
            return d
    return None


def _read_csv(csv_path, ts_col, actual_col, forecast_col, tz=JST):
    df = pd.read_csv(csv_path)
    if ts_col not in df.columns:
        raise ValueError(f"timestamp列 '{ts_col}' が見つかりません")
    
    act = _resolve_col_name(df, actual_col, ['demand_mw', 'demand_kw'])
    if act is None:
        raise ValueError(f"列 '{actual_col}' (または demand_mw/kw) が見つかりません")
    
    # forecast is optional in some contexts but required by this function signature generally
    fst = _resolve_col_name(df, forecast_col, ['demand_predict', 'demand_forecast_kw', 'demand_forecast_mw', 'forecast'])
    if fst is None:
         # Warning or raise? existing code raised.
         raise ValueError(f"列 '{forecast_col}' (または demand_predict/forecast_kw 等) が見つかりません")

    ts = pd.to_datetime(df[ts_col], errors='coerce')
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(tz)
    else:
        ts = ts.dt.tz_convert(tz)

    df = df.assign(timestamp=ts).dropna(subset=['timestamp'])
    df = df.sort_values('timestamp')
    
    # 元データはTEPCO東京エリア需要（MW単位）だが、数値をそのままkWとして扱う。
    # これは意図的な 1/1000 スケーリング相当であり、ピーク約50,000 kW（50 MW）の
    # 小売事業者顧客ポートフォリオを模したシナリオとして位置づける。
    result_df = pd.DataFrame()
    result_df['timestamp'] = df['timestamp']
    result_df['actual_kw'] = df[act]
    result_df['forecast_kw'] = df[fst]
    return result_df


def _read_csv_separate(actual_csv_path, forecast_csv_path, ts_col, actual_col, forecast_col, tz=JST):
    """実測CSVと予測CSVを別々に読み、timestampで内部結合する。
    forecast_csv_path が None の場合は実測のみ返す（予測列なし）。
    返却: DataFrame: columns ⊆ {timestamp, actual_kw, forecast_kw}
    """
    adf = pd.read_csv(actual_csv_path)
    if ts_col not in adf.columns:
        raise ValueError(f"actual CSV に timestamp列 '{ts_col}' が必要です")
        
    act = _resolve_col_name(adf, actual_col, ['demand_mw', 'demand_kw'])
    if act is None:
        raise ValueError(f"actual CSV に実測列 '{actual_col}' (または demand_mw/kw) がありません")

    ts = pd.to_datetime(adf[ts_col], errors='coerce')
    ts = ts.dt.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz)
    
    # MW→kW読み替え：元データはMW単位だが数値そのままkWとして扱う（意図的な1/1000スケーリング）
    adf = adf.assign(timestamp=ts, actual_kw=adf[act]).dropna(subset=['timestamp'])
    adf = adf[['timestamp', 'actual_kw']].sort_values('timestamp')

    if forecast_csv_path:
        fdf = pd.read_csv(forecast_csv_path)
        if ts_col not in fdf.columns:
            raise ValueError(f"forecast CSV に timestamp列 '{ts_col}' が必要です")
            
        fst = _resolve_col_name(fdf, forecast_col, ['demand_predict', 'demand_forecast_kw', 'demand_forecast_mw'])
        if fst is None:
             raise ValueError(f"forecast CSV に予測列 '{forecast_col}' (または demand_predict/forecast_kw 等) がありません")

        tsf = pd.to_datetime(fdf[ts_col], errors='coerce')
        tsf = tsf.dt.tz_localize(tz) if tsf.dt.tz is None else tsf.dt.tz_convert(tz)
        
        fdf = fdf.assign(timestamp=tsf, forecast_kw=fdf[fst]).dropna(subset=['timestamp'])
        fdf = fdf[['timestamp', 'forecast_kw']].sort_values('timestamp')

        df = pd.merge(adf, fdf, on='timestamp', how='inner')
        return df
    else:
        return adf


def _align_on_intersection(a: pd.Series, b: pd.Series):
    """インデックス共通部分でアライン（欠損は落とす）"""
    common = a.index.intersection(b.index)
    return a.loc[common], b.loc[common]


def visualize_overlay(
    csv_path: str,
    target_date_str: str,
    ts_col: str = 'timestamp',
    actual_col: str = 'demand_mw',
    forecast_col: str = 'demand_predict',
    out_dir: str = 'data/output/battery_control/deterministic',
    csv_actual_path: str | None = None,
    csv_forecast_path: str | None = None,
    # LP params (kWh/kW 系) - Optimizerクラスのデフォルトと統一
    lp_capacity_kwh: float = 40000.0,
    lp_power_kw: float = 10000.0,
    lp_eff: float = 0.92,  # 論文用の既定（効率 92%）
    lp_timestep_h: float = 0.5,
    lp_initial_soc_pct: float = 50.0,
    lp_terminal_soc_min_pct: float = 0.0,  # 等式で戻さない
    lp_final_soc_equality: bool = False,   # 既定: 等式なし（initSoC ±5%を許容）
    lp_action_penalty: float = 1e-4,  # Optimizerクラスのデフォルトと統一
    use_lp: bool = True,
    lp_input_source: str = 'forecast',  # 'forecast' or 'actual' を選択
    version: str = '',
    dpi: int = 600,
):
    # === 読み込み & 日抽出 ===
    target_day = pd.Timestamp(target_date_str).tz_localize(JST)
    if csv_path:
        df = _read_csv(csv_path, ts_col, actual_col, forecast_col, tz=JST)
    else:
        df = _read_csv_separate(csv_actual_path, csv_forecast_path, ts_col, actual_col, forecast_col, tz=JST)

    day_mask = (df['timestamp'] >= target_day) & (df['timestamp'] < target_day + pd.Timedelta(days=1))
    day_df = df.loc[day_mask].copy()
    if day_df.empty:
        print(f"No data found for {target_date_str} in {csv_path}")
        return

    print(f"Debug: day_df columns before set_index: {list(day_df.columns)}")
    print(f"Debug: actual_col='{actual_col}', forecast_col='{forecast_col}'")
    
    day_df = day_df.set_index('timestamp')
    
    # 読み込み関数で列名を正規化済みとして処理 (fallback to arguments if needed)
    if 'actual_kw' in day_df.columns:
        actual_kw = day_df['actual_kw'].astype(float)
    else:
        actual_kw = day_df[actual_col].astype(float)

    forecast_kw = None
    if 'forecast_kw' in day_df.columns:
        forecast_kw = day_df['forecast_kw'].astype(float)
    elif actual_col == forecast_col and 'demand_forecast' in day_df.columns: # Legacy handling just in case
        forecast_kw = day_df['demand_forecast'].astype(float)
    elif forecast_col in day_df.columns:
         forecast_kw = day_df[forecast_col].astype(float)

    # === プロット準備 ===
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # 論文体裁：白背景・グリッドなし・フォント16pt
    plt.style.use('seaborn-v0_8-white')
    plt.rcParams.update({'font.size': 16})
    # 3つのサブプロットを作成：上部に需要、中部に充放電、下部にSoC
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[3, 1, 1])

    fig.subplots_adjust(hspace=0.35, left=0.08, right=0.97, top=0.95, bottom=0.07)

    # X軸用の数値配列
    time_numeric = np.arange(len(actual_kw.index))

    # 上部：需要データのプロット（タイトルは付けない）
    print(f"Debug: actual_kw range: {actual_kw.min():.1f} - {actual_kw.max():.1f}")
    if forecast_kw is not None:
        print(f"Debug: forecast_kw range: {forecast_kw.min():.1f} - {forecast_kw.max():.1f}")

    plot_scale = 1000.0  # kW -> 10^3 kW units for display only
    actual_plot = actual_kw / plot_scale
    forecast_plot = forecast_kw / plot_scale if forecast_kw is not None else None

    ax1.plot(time_numeric, actual_plot.values, label='Actual Demand (Before)', color='gray', alpha=0.7, marker='.', linestyle='-')
    if forecast_kw is not None:
        ax1.plot(time_numeric, forecast_plot.values, label='LSTM Forecast (Input)', color='#ff7f0e', linestyle='--', alpha=0.9, marker='x')

    peak_before = float(actual_kw.max(skipna=True))
    actual_peak_pos = int(np.nanargmax(actual_kw.to_numpy(dtype=float)))
    actual_peak_time = actual_kw.index[actual_peak_pos].strftime('%H:%M')
    forecast_peak_pos = None
    forecast_peak_time = None
    if forecast_kw is not None:
        forecast_peak_pos = int(np.nanargmax(forecast_kw.to_numpy(dtype=float)))
        forecast_peak_time = forecast_kw.index[forecast_peak_pos].strftime('%H:%M')

    # 充放電データとSoCデータを保存するための変数（LPのみ）
    lp_charge_data = None
    lp_discharge_data = None
    lp_soc_data = None

    # FDS関連の描画・計算は論文用のため除去（計算・描画ともに実施しない）

    # === LP: 入力系列（予測 or 実績）で最適化し、そのスケジュールを実績に適用（赤） ===
    used_lp = False
    residual_actual_lp = None
    if use_lp and _HAS_LP:
        try:
            # 入力系列の選択（kW）
            if lp_input_source == 'actual' or forecast_kw is None:
                if lp_input_source != 'actual' and forecast_kw is None:
                    print('[WARN] forecast が未提供のため LP 入力を actual にフォールバックします')
                lp_input_series = actual_kw  # 実測（kW）
            else:
                lp_input_series = forecast_kw  # 予測（kW, 既定）

            # 初期/終端SoCを計算（終端は等式モードのみサポート）
            initial_kwh = lp_capacity_kwh * (lp_initial_soc_pct / 100.0)
            # terminal_min_kwh は現行 Optimizer では未対応のため利用しない

            # Optimizer を明示引数で初期化（型安全）
            optimizer = LPPeakShavingOptimizer(
                battery_capacity_kwh=lp_capacity_kwh,
                max_power_kw=lp_power_kw,
                charge_efficiency=lp_eff,
                discharge_efficiency=lp_eff,
                initial_soc_kwh=initial_kwh,
                action_penalty_lambda=lp_action_penalty,
                timestep_h=lp_timestep_h,
                final_soc_equality=lp_final_soc_equality,
            )

            # 最適化呼び出し（指定の入力系列で実行）
            # 最適化実行（numpy 配列 + DatetimeIndex を明示）
            demand_array = lp_input_series.to_numpy(dtype=float)
            time_index = pd.DatetimeIndex(lp_input_series.index)
            results = optimizer.optimize_peak_shaving(demand_array, time_index)

            # 充放電 [kW]
            c_t = pd.Series(results['c_t'], index=lp_input_series.index)
            d_t = pd.Series(results['d_t'], index=lp_input_series.index)

            # スケジュールを実績に適用：残余 = 実績 + 充電 - 放電（符号定義）
            residual_actual_lp = (actual_kw + c_t - d_t).astype(float)
            residual_actual_lp, _ = _align_on_intersection(residual_actual_lp, actual_kw)
            residual_actual_lp_plot = residual_actual_lp / plot_scale
            lp_label = 'Actual Demand (After, LP)' if lp_input_source == 'forecast' else 'Actual Demand (After, LP input=actual)'
            ax1.plot(time_numeric, residual_actual_lp_plot.values,
                     label=lp_label, color='red', linewidth=2.5)

            # LPの充放電データとSoCデータを保存（kW単位のまま）
            lp_charge_data = c_t  # kW
            lp_discharge_data = d_t  # kW
            
            # SoCデータを保存（kWh単位のまま）
            if 'soc_t' in results:
                # SoCデータのインデックスを作成（N+1個のデータ用）
                soc_index = list(lp_input_series.index) + [lp_input_series.index[-1] + pd.Timedelta(minutes=30)]
                lp_soc_data = pd.Series(results['soc_t'], index=soc_index)

            used_lp = True
        except Exception as e:
            print(f"[WARN] LP overlay skipped: {e}")

    # === ピークライン ===
    ax1.axhline(peak_before / plot_scale, linestyle=':', color='gray', linewidth=2, label='_nolegend_')

    # FDSのピークラインは非表示

    if used_lp and residual_actual_lp is not None:
        peak_after_lp = float(residual_actual_lp.max(skipna=True))
        ax1.axhline(peak_after_lp / plot_scale, linestyle=':', color='red', linewidth=2, label='_nolegend_')

    ax1.axvline(actual_peak_pos, linestyle='--', color='gray', linewidth=1.8,
                label=f'Actual Peak Time ({actual_peak_time})')
    if forecast_peak_pos is not None and forecast_peak_time is not None:
        ax1.axvline(forecast_peak_pos, linestyle='--', color='#ff7f0e', linewidth=1.8,
                    label=f'Forecast Peak Time ({forecast_peak_time})')

    # === 上部グラフの体裁 ===
    ax1.set_ylabel('Electricity Demand [kW]', fontsize=16)
    ax1.text(0.0, 1.02, '×10³', transform=ax1.transAxes, fontsize=16, va='bottom', ha='left')
    if version in ('shift_0slot', 'shift+8', 'shift-7'):
        ax1.legend(fontsize=15, loc='upper left', bbox_to_anchor=(0.0, 0.72))
    else:
        ax1.legend(fontsize=16, loc='lower left', bbox_to_anchor=(0.6, 0.05))
    ax1.grid(False)
    ax1.tick_params(axis='both', labelsize=16)

    # Y軸の範囲を適切に設定
    all_values = []
    all_values.extend(actual_plot.values)
    if forecast_plot is not None:
        all_values.extend(forecast_plot.values)
    # FDS系列は非表示
    if used_lp and residual_actual_lp is not None:
        all_values.extend((residual_actual_lp / plot_scale).values)

    y_min = min(all_values) * 0.95
    y_max = max(all_values) * 1.05
    ax1.set_ylim(y_min, y_max)
    print(f"Debug: Y-axis range set to {y_min:.1f} - {y_max:.1f}")

    # 上部グラフのX軸も設定
    ax1.set_xlim(-0.5, len(time_numeric) - 0.5)
    tick_positions = range(0, len(time_numeric), 4)  # 2時間間隔
    ax1.set_xticks(list(tick_positions))
    ax1.set_xticklabels([f"{i//2:02d}:00" for i in tick_positions])

    # === 下部グラフ：充放電量 ===
    bar_width = 0.4
    # time_numeric は上で定義済み

    # データのデバッグ出力（LPのみ）
    print(f"Debug: LP charge max: {max(lp_charge_data) if lp_charge_data is not None else 'None'}")
    print(f"Debug: LP discharge max: {max(lp_discharge_data) if lp_discharge_data is not None else 'None'}")
    # LPの充放電バー
    if used_lp and lp_charge_data is not None and lp_discharge_data is not None:
        ax2.bar(time_numeric + bar_width/2, lp_charge_data / plot_scale, bar_width,
                label='LP Charge', color="#6cb9e6", alpha=0.7)
        ax2.bar(time_numeric + bar_width/2, -lp_discharge_data / plot_scale, bar_width,
                label='LP Discharge', color='red', alpha=0.5)

    # 下部グラフの体裁（Xラベルは下段でまとめて表示するため中段では非表示）
    ax2.set_xlabel('')
    ax2.set_ylabel('Charging and Discharging [kW]', fontsize=16)
    ax2.text(0.0, 1.02, '×10³', transform=ax2.transAxes, fontsize=16, va='bottom', ha='left')
    ax2_legend_loc = 'lower left' if version in ('shift_-7slot', 'shift-7') else 'upper left'
    ax2.legend(fontsize=16, loc=ax2_legend_loc)
    ax2.grid(False)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.axhline(0, color='black', linewidth=0.5)

    # X軸の設定を簡素化
    ax2.set_xlim(-0.5, len(time_numeric) - 0.5)
    tick_positions = range(0, len(time_numeric), 4)  # 2時間間隔（30分刻み→4ステップ）
    ax2.set_xticks(list(tick_positions))
    ax2.set_xticklabels([f"{i//2:02d}:00" for i in tick_positions])  # 30分刻みなので2で割る

    # 充放電のY軸範囲を固定（放電側を -lp_power_kw まで表示）
    try:
        ymax = max(lp_power_kw, float(np.nanmax(lp_charge_data)) if lp_charge_data is not None else lp_power_kw) / plot_scale
    except Exception:
        ymax = lp_power_kw / plot_scale
    ax2.set_ylim(-lp_power_kw / plot_scale * 1.15, ymax)

    # === 3つ目のサブプロット: SoC追跡グラフ ===
    ax3.set_ylabel('SoC [kWh]', fontsize=16)
    ax3.set_xlabel('Hour', fontsize=16)

    # FDSのSoC折れ線は非表示

    # LPのSoC折れ線  
    if used_lp and lp_soc_data is not None:
        # LPのSoCデータ（N+1個）の時系列を作成
        soc_time_numeric_lp = np.arange(len(lp_soc_data))
        ax3.plot(soc_time_numeric_lp, lp_soc_data,
                label='LP SoC', color='red', linewidth=2.5, marker='s', markersize=4)

    # SoCグラフの体裁
    ax3.legend(fontsize=16)
    ax3.grid(False)
    ax3.tick_params(axis='both', labelsize=16)
    
    # X軸設定を他のサブプロットと合わせる（48半時間分）
    ax3.set_xlim(-0.5, len(time_numeric) + 0.5)  # SoCは+1個のデータポイントがある
    ax3.set_xticks(list(tick_positions))
    ax3.set_xticklabels([f"{i//2:02d}:00" for i in tick_positions])

    # SoCの範囲設定（0%〜100%）
    battery_capacity = lp_capacity_kwh if used_lp else 40000.0
    ax3.set_ylim(-battery_capacity * 0.05, battery_capacity * 1.1)
    
    # SoC参考線（50%、100%）
    ax3.axhline(battery_capacity * 0.5, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax3.axhline(battery_capacity, color='gray', linestyle=':', alpha=0.5, label='100%')

    version_suffix = f'_{version}' if version else ''
    out_path = Path(out_dir) / f'peak_shaving_overlay_{target_day.strftime("%Y%m%d")}{version_suffix}.png'
    fig.align_ylabels([ax1, ax2, ax3])
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {out_path}")

    # === 充放電スケジュールをCSVに保存 ===
    if used_lp and lp_charge_data is not None and lp_discharge_data is not None:
        # 充放電スケジュールデータをDataFrameに整形
        schedule_df = pd.DataFrame({
            'timestamp': actual_kw.index,
            'demand_before_kw': actual_kw.values,
            'charge_kw': lp_charge_data.values,
            'discharge_kw': lp_discharge_data.values,
        })
        
        # 予測データがある場合は追加
        if forecast_kw is not None:
            schedule_df['forecast_kw'] = forecast_kw.values
        
        # 制御後の需要を追加
        if residual_actual_lp is not None:
            schedule_df['demand_after_kw'] = residual_actual_lp.values
        
        # SoCデータを追加（時刻インデックスを1つずらして対応）
        if lp_soc_data is not None:
            # SoCデータは各時刻の終了時の値なので、次の時刻の開始時SoCとして記録
            # 最初のタイムステップには初期SoCを設定
            soc_values = lp_soc_data.values[:-1]  # 最後の1つを除く（N個）
            schedule_df['soc_kwh'] = soc_values
        
        # CSVファイルとして保存
        csv_out_path = Path(out_dir) / f'battery_schedule_{target_day.strftime("%Y%m%d")}{version_suffix}.csv'
        schedule_df.to_csv(csv_out_path, index=False)
        print(f"Battery schedule saved to: {csv_out_path}")

    # サマリー出力
    print('-' * 50)
    print('Summary:')
    print(f"  Peak (Before): {peak_before:.1f} kW")
    # FDS に関するサマリー出力は非表示
    if used_lp and residual_actual_lp is not None:
        peak_after_lp = float(residual_actual_lp.max(skipna=True))
        print(f"  Peak (After, LP):  {peak_after_lp:.1f} kW  | Reduction: {(peak_before-peak_after_lp)/peak_before*100:.2f}%")
    else:
        print("  LP: not used")
    print('-' * 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=False, help='入力CSVのパス（実測と予測が同一ファイルの場合）')
    ap.add_argument('--csv-actual', required=False, help='実測データCSVのパス（timestamp, actual列）')
    ap.add_argument('--csv-forecast', required=False, help='予測データCSVのパス（timestamp, forecast列）')
    ap.add_argument('--date', required=True, help='対象日 (YYYY-MM-DD) JST')
    ap.add_argument('--ts-col', default='timestamp')
    ap.add_argument('--actual-col', default='demand_mw')
    ap.add_argument('--forecast-col', default='demand_predict')
    ap.add_argument('--out-dir', default='data/output/battery_control/deterministic')

    # LP
    ap.add_argument('--lp', action='store_true', help='LPを重ね描きする')
    ap.add_argument('--lp-capacity-kwh', type=float, default=40000.0)   # FDSと同じ: 40000 kWh
    ap.add_argument('--lp-power-kw', type=float, default=10000.0)       # FDSと同じ: 10000 kW
    ap.add_argument('--lp-eff', type=float, default=0.92, help='LP効率（デフォルト: 0.92）')
    ap.add_argument('--lp-timestep-h', type=float, default=0.5)
    ap.add_argument('--lp-initial-soc-pct', type=float, default=50.0)
    ap.add_argument('--lp-terminal-soc-min-pct', type=float, default=0.0)
    ap.add_argument('--lp-final-soc-equality', action='store_true', help='終端SoCを等式制約 E(T)=E(0) にする（指定時のみ有効。未指定時は初期SoC±5%%を許容）')
    ap.add_argument('--lp-action-penalty', type=float, default=1e-4, help='LP動作ペナルティ（デフォルト: 1e-4, Optimizerクラスと統一）')
    ap.add_argument('--lp-input', choices=['forecast', 'actual'], default='forecast', help='LP の入力系列を選択（既定: forecast）')
    ap.add_argument('--version', type=str, default='', help='出力ファイル名に追加するバージョン文字列')

    args = ap.parse_args()

    use_lp = args.lp and _HAS_LP
    if args.lp and not _HAS_LP:
        print('[WARN] LPPeakShavingOptimizer が見つからないため LP はスキップされます')

    visualize_overlay(
        csv_path=args.csv,
        csv_actual_path=args.csv_actual,
        csv_forecast_path=args.csv_forecast,
        target_date_str=args.date,
        ts_col=args.ts_col,
        actual_col=args.actual_col,
        forecast_col=args.forecast_col,
        out_dir=args.out_dir,
        # LP
        lp_capacity_kwh=args.lp_capacity_kwh,
        lp_power_kw=args.lp_power_kw,
        lp_eff=args.lp_eff,
        lp_timestep_h=args.lp_timestep_h,
        lp_initial_soc_pct=args.lp_initial_soc_pct,
        lp_terminal_soc_min_pct=args.lp_terminal_soc_min_pct,
        lp_final_soc_equality=args.lp_final_soc_equality,
        lp_action_penalty=args.lp_action_penalty,
        lp_input_source=args.lp_input,
        use_lp=use_lp,
        version=args.version,
    )


if __name__ == '__main__':
    main()
