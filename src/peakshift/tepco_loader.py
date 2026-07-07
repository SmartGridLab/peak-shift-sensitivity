"""TEPCO エリア需給実績 CSV の読み込み・正規化。

東京電力パワーグリッドが公開する「エリア需給実績データ」CSV を読み込み、
[timestamp, demand_mw] 形式の 30 分値 DataFrame に正規化する。
（元プロジェクトの master_dataframe.py から TEPCODataLoader のみを抽出したもの）
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


class TEPCODataLoader:
    """TEPCO特化CSV読み込みクラス"""

    # 元の 'area-YYYY.csv' ファイルの固定ヘッダー
    NEW_HEADER_AREA = [
        "DATE", "TIME", "Tokyo Area Demand", "Nuclear", "Thermal", "Hydro",
        "Geothermal", "Biomass", "Solar Power Generation (Actual)",
        "Solar Power Curtailment", "Wind Power Generation (Actual)",
        "Wind Power Curtailment", "Pumped Storage", "Interconnection", "Total"
    ]

    # 新しい 'eria_jukyu_YYYYMM_XX.csv' ファイルの固定ヘッダー
    NEW_HEADER_ERIA = [
        "DATE", "TIME", "Tokyo Area Demand", "Nuclear", "Thermal (LNG)", "Thermal (Coal)",
        "Thermal (Oil)", "Thermal (Other)", "Hydro", "Geothermal", "Biomass",
        "Solar Power Generation (Actual)", "Solar Power Curtailment",
        "Wind Power Generation (Actual)", "Wind Power Curtailment", "Pumped Storage",
        "Battery Storage", "Interconnection", "Others", "Total"
    ]

    # 両形式のすべての列を含む統一ヘッダー
    UNIFIED_HEADER = [
        "DATE", "TIME", "Tokyo Area Demand",
        # 電源
        "Nuclear",
        "Thermal",  # area-*.csvから
        "Thermal (LNG)", "Thermal (Coal)", "Thermal (Oil)", "Thermal (Other)",  # eria_jukyu-*.csvから
        "Hydro", "Geothermal", "Biomass",
        "Solar Power Generation (Actual)", "Solar Power Curtailment",
        "Wind Power Generation (Actual)", "Wind Power Curtailment",
        # 蓄電・系統
        "Pumped Storage", "Battery Storage", "Interconnection",
        # 集計
        "Others", "Total"
    ]

    @staticmethod
    def _normalize_demand_to_mw(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
        """ファイル名から単位を判定し demand_mw を MW に揃える。

        - area-YYYY.csv (旧形式): 値は万kW → ×10 して MW へ変換
        - eria_jukyu_*.csv (新形式): 値は既に MW → 変換なし
        - それ以外: 単位不明として警告のみ出力し変換しない
        """
        if df.empty:
            return df
        name = source_path.name
        if name.startswith("area-"):
            df["demand_mw"] = df["demand_mw"] * 10.0
            print(f"[TEPCODataLoader] {name}: 単位変換 万kW -> MW (×10)")
        elif name.startswith("eria_jukyu_"):
            print(f"[TEPCODataLoader] {name}: MW のため単位変換なし")
        else:
            print(f"[TEPCODataLoader][WARN] {name}: 未知のファイル名形式、単位変換をスキップ")
        return df

    @classmethod
    def load_annual_demand_csv(cls, paths: Iterable[Path], year: int) -> pd.DataFrame:
        """指定された年の需要CSVを読み込み・正規化（通年）。

        この関数は、すべてのソースCSVに可変数のヘッダー行があり、その後に
        データセクションが続くことを前提としています。初期ヘッダーを
        プログラム的にスキップし、新しい標準化された英語ヘッダーを適用します。

        Parameters
        ----------
        paths: iterable of Path
            候補となるCSVファイルパス。名前に対象年（4桁の部分文字列）を
            含むもののみが対象となります。
        year: int
            抽出対象年（例: 2024）。

        Returns
        -------
        列 [timestamp(datetime64[ns]), demand_mw(float)] を持つDataFrame
        通年分のデータをタイムスタンプでソート済み。
        """
        year_str = str(year)
        dfs: List[pd.DataFrame] = []

        for p in paths:
            if year_str not in p.name:
                continue

            # ファイル名の命名規則に基づいて使用するヘッダーを決定
            original_header = None
            if p.name.startswith("area-"):
                original_header = cls.NEW_HEADER_AREA
            elif p.name.startswith("eria_jukyu"):
                original_header = cls.NEW_HEADER_ERIA
            else:
                logging.debug(f"未知の形式のファイルをスキップ: {p.name}")
                continue

            # "DATE"と"TIME"を含む行を探して実際のデータの開始位置を特定。
            # データは"DATE"と"TIME"を含む行の*次の行*から始まります。
            header_row_index = -1
            encodings_to_try = ["utf-8", "shift_jis", "cp932"]
            lines = []
            file_encoding = None

            for enc in encodings_to_try:
                try:
                    with p.open("r", encoding=enc) as f:
                        lines = f.readlines()
                    file_encoding = enc
                    break
                except (UnicodeDecodeError, IOError):
                    continue

            if not file_encoding:
                logging.warning(f"ファイル {p} のエンコーディング検出に失敗しました")
                continue

            for i, line in enumerate(lines[:15]):  # 最初の15行をチェック
                if "DATE" in line and "TIME" in line:
                    header_row_index = i
                    break

            if header_row_index == -1:
                logging.warning(f"{p} でデータヘッダー行が見つかりません。ファイルをスキップします。")
                continue

            try:
                df = pd.read_csv(
                    p,
                    encoding=file_encoding,
                    skiprows=header_row_index + 1,
                    header=None,  # pandasに見えるデータにはヘッダーなし
                    names=original_header,
                    engine="python",
                )
            except Exception as e:
                logging.warning(f"ヘッダースキップ後、pandasが {p} の読み込みに失敗: {e}")
                continue

            # 統一ヘッダーでDataFrameを再インデックス。
            # 元ファイルに存在しない列は追加され、NaNで埋められます。
            df = df.reindex(columns=cls.UNIFIED_HEADER)

            # --- データクリーニングと処理 ---
            # DATEとTIMEを結合してタイムスタンプを作成
            df["timestamp"] = pd.to_datetime(df["DATE"] + " " + df["TIME"], errors="coerce")
            df = df.dropna(subset=["timestamp"])

            # 需要列を設定してクリーニング
            df = df.rename(columns={"Tokyo Area Demand": "demand_mw"})
            df["demand_mw"] = pd.to_numeric(
                df["demand_mw"].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )
            df = df.dropna(subset=["demand_mw"])

            # 対象年でフィルタ（月フィルタなし - 通年使用）
            df = df[df["timestamp"].dt.year == year]

            if df.empty:
                continue

            # 必要な列のみを選択
            final_df = df[["timestamp", "demand_mw"]].copy()

            # 旧形式(area-*: 万kW) → ×10 で MW / 新形式(eria_jukyu_*: MW) → そのまま
            final_df = cls._normalize_demand_to_mw(final_df, p)

            dfs.append(final_df)

        if not dfs:
            raise FileNotFoundError(f"指定されたパスで年 {year} の有効な年間CSVデータが見つかりません")

        out = pd.concat(dfs, ignore_index=True).sort_values("timestamp")
        out = out.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        return out

    @classmethod
    def combine_all_years_data(cls, paths: Iterable[Path], years: Sequence[int]) -> pd.DataFrame:
        """指定された年の全CSVを単一の正規化された年間DataFrameに結合。

        Returns
        -------
        DataFrame列: [timestamp, demand_mw, year]
        """
        combined: List[pd.DataFrame] = []
        skipped_years = []
        for y in years:
            try:
                df_y = cls.load_annual_demand_csv(paths, y)
                df_y["year"] = y
                combined.append(df_y)
            except FileNotFoundError:
                logging.warning(f"年間データが見つからないか読み込みエラーのため年 {y} をスキップします。")
                skipped_years.append(y)
                continue

        if not combined:
            raise FileNotFoundError(
                f"指定された年のいずれについても年間データを読み込めませんでした: {years}"
            )

        logging.info(
            f"データ読み込み成功年: {[y for y in years if y not in skipped_years]}"
        )
        if skipped_years:
            logging.warning(f"スキップされた年（データなし）: {skipped_years}")

        df_all = pd.concat(combined, ignore_index=True)
        return df_all
