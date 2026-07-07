"""
LP Peak Shaving Optimizer
線形計画法を使用したピークシェービング最適化

要求仕様:
- 30分コマの需要予測系列 demand_forecast_t（オレンジ線）を入力に、
  蓄電池の充放電計画を線形計画（LP）で最適化し、
  当日の制御後需要の最大値（ピーク）demand_peak_forecast をできるだけ小さくする

記号・前提:
- 時間刻み: Δt = 0.5 h（30分）
- 時刻インデックス: t = 0, …, N-1（通常 N = 48）
- 予測需要（LP前）: demand_forecast_t [kW]
- 充電・放電（グリッド側）: c_t ≥ 0, d_t ≥ 0 [kW]
- 蓄電量（State of Charge）: SoC_t [kWh]
- 効率: η_c, η_d ∈ (0, 1]（初期は η_c = η_d = 1.0）
- 蓄電池仕様: 容量 E_cap [kWh]、出力上限 P_cap [kW]、初期SoC SoC_init [kWh]

制御後需要（予測基準）:
demand_t = demand_forecast_t + c_t - d_t

LP定式化:
minimize   demand_peak_forecast + λ * Σ_t (c_t + d_t)

制約:
- demand_t = demand_forecast_t + c_t - d_t
- SoC_{t+1} = SoC_t + (η_c * c_t - d_t / η_d) * Δt
- 0 ≤ SoC_t ≤ E_cap
- 0 ≤ c_t ≤ P_cap, 0 ≤ d_t ≤ P_cap
- SoC_0 = SoC_init, SoC_N = SoC_init
- demand_peak_forecast ≥ demand_t for all t
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import pulp
from pathlib import Path
import json


class LPPeakShavingOptimizer:
    """LP（線形計画法）によるピークシェービング最適化器"""
    
    def __init__(
        self,
        battery_capacity_kwh: float = 40000.0,
        max_power_kw: float = 10000.0,
        charge_efficiency: float = 1.0,
        discharge_efficiency: float = 1.0,
        initial_soc_kwh: Optional[float] = None,
        action_penalty_lambda: float = 1e-4,
        timestep_h: float = 0.5,
        solver_name: str = "PULP_CBC_CMD",
        final_soc_equality: bool = True
    ):
        """
        LP ピークシェービング最適化器を初期化
        
        Args:
            battery_capacity_kwh: 蓄電池容量 E_cap [kWh]
            max_power_kw: 最大出力 P_cap [kW]  
        
            charge_efficiency: 充電効率 η_c
            discharge_efficiency: 放電効率 η_d
            initial_soc_kwh: 初期SoC [kWh]（Noneの場合は容量の50%）
            action_penalty_lambda: 動作ペナルティ係数 λ
            timestep_h: 時間刻み Δt [時間]
            solver_name: ソルバー名
            final_soc_equality: 終端SoC等式制約 SoC_N = SoC_init を適用するか
        """
        self.E_cap = battery_capacity_kwh
        self.P_cap = max_power_kw
        self.eta_c = charge_efficiency
        self.eta_d = discharge_efficiency
        self.SoC_init = initial_soc_kwh if initial_soc_kwh is not None else battery_capacity_kwh * 0.5
        self.lambda_penalty = action_penalty_lambda
        self.dt = timestep_h
        self.solver_name = solver_name
        self.final_soc_equality = final_soc_equality
        
        # 妥当性チェック用の閾値
        self.eps_tolerance = 10.0  # kW（同時充放電チェック用）
        
    def optimize_peak_shaving(
        self,
        demand_forecast: np.ndarray,
        time_index: Optional[pd.DatetimeIndex] = None
    ) -> Dict:
        """
        ピークシェービング最適化を実行
        
        Args:
            demand_forecast: 需要予測 demand_forecast_t [kW] (長さN)
            time_index: 時刻インデックス（可視化用、オプション）
            
        Returns:
            最適化結果辞書:
                - c_t: 充電電力 [kW] (長さN)
                - d_t: 放電電力 [kW] (長さN)
                - soc_t: SoC [kWh] (長さN+1, SoC_0からSoC_Nまで)
                - demand_t: 制御後需要 [kW] (長さN)
                - demand_peak_forecast: 制御後ピーク [kW]
                - metrics: 評価指標
                - validation: 妥当性チェック結果
        """
        N = len(demand_forecast)
        
        # LP問題の作成
        prob = pulp.LpProblem("Peak_Shaving_LP", pulp.LpMinimize)
        
        # 変数定義
        c_vars = [pulp.LpVariable(f"c_{t}", 0, self.P_cap) for t in range(N)]  # 充電電力
        d_vars = [pulp.LpVariable(f"d_{t}", 0, self.P_cap) for t in range(N)]  # 放電電力
        soc_vars = [pulp.LpVariable(f"soc_{t}", 0, self.E_cap) for t in range(N+1)]  # SoC
        demand_vars = [pulp.LpVariable(f"demand_{t}") for t in range(N)]  # 制御後需要
        peak_var = pulp.LpVariable("demand_peak_forecast")  # ピーク需要
        
        # バイナリ変数：充電フラグ（1=充電中, 0=放電中または待機中）
        is_charging = [pulp.LpVariable(f"is_charging_{t}", cat='Binary') for t in range(N)]
        
        # 制約の追加
        
        # 1. 制御後需要の定義: demand_t = demand_forecast_t + c_t - d_t
        for t in range(N):
            prob += demand_vars[t] == demand_forecast[t] + c_vars[t] - d_vars[t]
        
        # 2. SoC更新制約: SoC_{t+1} = SoC_t + (η_c * c_t - d_t / η_d) * Δt
        prob += soc_vars[0] == self.SoC_init  # 初期SoC
        for t in range(N):
            prob += (soc_vars[t+1] == soc_vars[t] 
                    + (self.eta_c * c_vars[t] - d_vars[t] / self.eta_d) * self.dt)
        
        # 3. 同時充放電禁止制約
        for t in range(N):
            # 充電時: c_t <= P_cap * is_charging_t
            prob += c_vars[t] <= self.P_cap * is_charging[t]
            # 放電時: d_t <= P_cap * (1 - is_charging_t)
            prob += d_vars[t] <= self.P_cap * (1 - is_charging[t])
        
        # 4. 終端SoC制約: SoC_N = SoC_init（オプション）
        if self.final_soc_equality:
            prob += soc_vars[N] == self.SoC_init
        else:
            # FDSと同じ±5%範囲制約（柔軟な制約）
            prob += soc_vars[N] >= self.SoC_init * 0.95
            prob += soc_vars[N] <= self.SoC_init * 1.05
        
        # 5. ピーク制約: demand_peak_forecast ≥ demand_t for all t
        for t in range(N):
            prob += peak_var >= demand_vars[t]
        
        # 目的関数: minimize demand_peak_forecast + λ * Σ_t (c_t + d_t)
        action_penalty = pulp.lpSum([c_vars[t] + d_vars[t] for t in range(N)])
        prob += peak_var + self.lambda_penalty * action_penalty
        
        # ソルバーの設定と求解
        try:
            solver = pulp.getSolver(self.solver_name, msg=False)
        except:
            # ソルバーが利用できない場合はデフォルトを使用
            print(f"⚠️  {self.solver_name} が利用できません。デフォルトソルバーを使用します。")
            solver = None  # PuLPがデフォルトソルバーを自動選択
        
        prob.solve(solver)
        
        # 結果の抽出
        if prob.status != pulp.LpStatusOptimal:
            raise RuntimeError(f"最適化が失敗しました。ステータス: {pulp.LpStatus[prob.status]}")
        
        # 最適解の取得
        c_optimal = np.array([c_vars[t].varValue or 0.0 for t in range(N)])
        d_optimal = np.array([d_vars[t].varValue or 0.0 for t in range(N)])
        soc_optimal = np.array([soc_vars[t].varValue or 0.0 for t in range(N+1)])
        demand_optimal = np.array([demand_vars[t].varValue or 0.0 for t in range(N)])
        peak_optimal = peak_var.varValue or 0.0
        
        # 評価指標の計算
        metrics = self._calculate_metrics(
            demand_forecast, demand_optimal, c_optimal, d_optimal, peak_optimal
        )
        
        # 妥当性チェック
        validation = self._validate_solution(
            c_optimal, d_optimal, soc_optimal, demand_forecast, demand_optimal
        )
        
        return {
            "c_t": c_optimal,
            "d_t": d_optimal,
            "soc_t": soc_optimal,
            "demand_t": demand_optimal,
            "demand_peak_forecast": peak_optimal,
            "metrics": metrics,
            "validation": validation,
            "time_index": time_index,
            "problem_status": pulp.LpStatus[prob.status]
        }
    
    def _calculate_metrics(
        self,
        demand_forecast: np.ndarray,
        demand_controlled: np.ndarray,
        c_t: np.ndarray,
        d_t: np.ndarray,
        peak_controlled: float
    ) -> Dict:
        """評価指標を計算"""
        peak_before = np.max(demand_forecast)
        peak_after = peak_controlled
        peak_reduction_mw = peak_before - peak_after
        peak_reduction_pct = (peak_reduction_mw / peak_before * 100) if peak_before > 0 else 0.0
        
        total_charge = np.sum(c_t) * self.dt  # kWh
        total_discharge = np.sum(d_t) * self.dt  # kWh
        total_action = np.sum(c_t + d_t) * self.dt  # kWh
        
        return {
            "peak_before_kw": peak_before,
            "peak_after_kw": peak_after,
            "h1_reduction_kw": peak_reduction_mw,  # 要求仕様の評価指標
            "h1_reduction_pct": peak_reduction_pct,  # 要求仕様の評価指標
            "total_charge_kwh": total_charge,
            "total_discharge_kwh": total_discharge,
            "total_action_kwh": total_action,
            "energy_efficiency_pct": (total_discharge / total_charge * 100) if total_charge > 0 else 0.0
        }
    
    def _validate_solution(
        self,
        c_t: np.ndarray,
        d_t: np.ndarray,
        soc_t: np.ndarray,
        demand_forecast: np.ndarray,
        demand_controlled: np.ndarray
    ) -> Dict:
        """妥当性チェック（A1-A4の達成基準）"""
        N = len(c_t)
        
        # A1: 出力・容量・SoC範囲チェック
        power_bounds_ok = (
            np.all(c_t >= 0) and np.all(c_t <= self.P_cap + 1e-6) and
            np.all(d_t >= 0) and np.all(d_t <= self.P_cap + 1e-6)
        )
        soc_bounds_ok = (
            np.all(soc_t >= -1e-6) and np.all(soc_t <= self.E_cap + 1e-6)
        )
        
        # A2: SoC更新チェック
        soc_update_ok = True
        soc_errors = []
        for t in range(N):
            expected_soc = (soc_t[t] + 
                          (self.eta_c * c_t[t] - d_t[t] / self.eta_d) * self.dt)
            error = abs(soc_t[t+1] - expected_soc)
            soc_errors.append(error)
            if error > 1.0:  # 1.0 kWh tolerance
                soc_update_ok = False
        
        # A3: SoC戻しチェック
        soc_return_error = abs(soc_t[N] - soc_t[0])
        same_soc_ok = soc_return_error < 1.0  # 1 kWh tolerance
        
        # A4: 同時充放電チェック
        simultaneous_actions = np.minimum(c_t, d_t)
        max_simultaneous = np.max(simultaneous_actions)
        no_simul_charge_discharge_ok = max_simultaneous <= self.eps_tolerance
        
        # 制御後需要チェック
        demand_error = np.max(np.abs(
            demand_controlled - (demand_forecast + c_t - d_t)
        ))
        demand_definition_ok = demand_error < 1e-6
        
        return {
            "constraints_ok": power_bounds_ok and soc_bounds_ok,
            "soc_update_ok": soc_update_ok,
            "same_soc_ok": same_soc_ok,
            "no_simul_charge_discharge_ok": no_simul_charge_discharge_ok,
            "demand_definition_ok": demand_definition_ok,
            "all_valid": (power_bounds_ok and soc_bounds_ok and soc_update_ok and 
                         same_soc_ok and no_simul_charge_discharge_ok and demand_definition_ok),
            "details": {
                "power_bounds_ok": power_bounds_ok,
                "soc_bounds_ok": soc_bounds_ok,
                "max_soc_update_error": max(soc_errors) if soc_errors else 0.0,
                "soc_return_error": soc_return_error,
                "max_simultaneous_action": max_simultaneous,
                "demand_definition_error": demand_error
            }
        }
    
    def save_results(
        self, 
        results: Dict, 
        schedule_path: str, 
        metrics_path: str
    ) -> None:
        """結果をCSVファイルに保存"""
        # スケジュールの保存
        N = len(results["c_t"])
        schedule_df = pd.DataFrame({
            "timestep": range(N),
            "demand_forecast_kw": results["demand_t"] - results["c_t"] + results["d_t"],  # 元の予測
            "charge_kw": results["c_t"],
            "discharge_kw": results["d_t"],
            "demand_controlled_kw": results["demand_t"],
            "soc_kwh": results["soc_t"][:-1]  # 最後のSoCは除く
        })
        
        if results["time_index"] is not None:
            schedule_df["timestamp"] = results["time_index"]
        
        schedule_df.to_csv(schedule_path, index=False)
        
        # メトリクスの保存
        metrics_data = {
            "metric": list(results["metrics"].keys()) + list(results["validation"].keys()),
            "value": list(results["metrics"].values()) + list(results["validation"].values())
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(metrics_path, index=False)
        
        print(f"✅ LP ピークシェービング結果を保存しました:")
        print(f"   スケジュール: {schedule_path}")
        print(f"   メトリクス: {metrics_path}")


if __name__ == "__main__":
    # サンプル実行
    import matplotlib.pyplot as plt
    
    # サンプルデータの生成（24時間、48ステップ）
    N = 48
    t = np.arange(N) * 0.5  # 時刻 [時間]
    
    # 典型的な需要パターン（朝夕にピーク）
    base_demand = 30000.0  # kW
    morning_peak = 15000.0 * np.exp(-((t - 8)**2) / 8)  # 8時頃の朝ピーク
    evening_peak = 20000.0 * np.exp(-((t - 18)**2) / 12)  # 18時頃の夕ピーク
    noise = np.random.normal(0, 2000, N)
    
    demand_forecast = base_demand + morning_peak + evening_peak + noise
    demand_forecast = np.maximum(demand_forecast, 15000.0)  # 最小15MW = 15000kW
    
    # 最適化器の設定
    optimizer = LPPeakShavingOptimizer(
        battery_capacity_kwh=40000.0,
        max_power_kw=10000.0,
        charge_efficiency=0.95,
        discharge_efficiency=0.95
    )
    
    # 最適化実行
    results = optimizer.optimize_peak_shaving(demand_forecast)
    
    # 結果の表示
    print("\n=== LP ピークシェービング最適化結果 ===")
    print(f"最適化ステータス: {results['problem_status']}")
    print(f"制御前ピーク: {results['metrics']['peak_before_kw']:.2f} kW")
    print(f"制御後ピーク: {results['metrics']['peak_after_kw']:.2f} kW")
    print(f"削減量: {results['metrics']['h1_reduction_kw']:.2f} kW ({results['metrics']['h1_reduction_pct']:.1f}%)")
    print(f"妥当性チェック: {'✅ 全て合格' if results['validation']['all_valid'] else '❌ 不合格'}")
    
    # 簡単な可視化
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # 需要の比較
    ax1.plot(t, demand_forecast, 'o-', color='orange', label='制御前需要（予測）', alpha=0.8)
    ax1.plot(t, results["demand_t"], 's-', color='red', label='制御後需要', alpha=0.8)
    ax1.axhline(results['metrics']['peak_before_kw'], color='orange', linestyle='--', alpha=0.6)
    ax1.axhline(results['metrics']['peak_after_kw'], color='red', linestyle='--', alpha=0.6)
    ax1.set_ylabel('需要 [kW]')
    ax1.set_title('LP ピークシェービング結果')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 充放電電力
    ax2.bar(t, results["c_t"], width=0.4, alpha=0.7, color='blue', label='充電')
    ax2.bar(t, -results["d_t"], width=0.4, alpha=0.7, color='green', label='放電')
    ax2.set_ylabel('電力 [kW]')
    ax2.set_title('バッテリー充放電スケジュール')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # SoC
    ax3.plot(np.arange(N+1) * 0.5, results["soc_t"], 'o-', color='purple', label='SoC')
    ax3.axhline(optimizer.SoC_init, color='purple', linestyle='--', alpha=0.6)
    ax3.set_ylabel('SoC [kWh]')
    ax3.set_xlabel('時刻 [時間]')
    ax3.set_title('バッテリーSoC推移')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()