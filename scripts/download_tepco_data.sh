#!/usr/bin/env bash
# 東京電力パワーグリッド「エリア需給実績データ」の 2024 年分 CSV を取得する。
#
# 出典: 東京電力パワーグリッド でんき予報 エリア需給実績データ
#   https://www.tepco.co.jp/forecast/html/area_jukyu-j.html
#
# 再配布条件が明示されていないため、本リポジトリには生データを同梱せず、
# このスクリプトで各自取得する方式とする。
# URL 構成が変更された場合は、上記ページから手動でダウンロードし
# data/tepco/ に eria_jukyu_YYYYMM_03.csv の名前のまま置くこと。

set -u

OUT_DIR="${1:-data/tepco}"
BASE_URL="https://www.tepco.co.jp/forecast/html/images"

mkdir -p "$OUT_DIR"

failed=0
for month in 01 02 03 04 05 06 07 08; do
    fname="eria_jukyu_2024${month}_03.csv"
    dest="$OUT_DIR/$fname"
    if [ -s "$dest" ]; then
        echo "[SKIP] $dest already exists"
        continue
    fi
    url="$BASE_URL/$fname"
    echo "[GET ] $url"
    if ! curl -fsSL -o "$dest" "$url"; then
        echo "[FAIL] $fname could not be downloaded automatically."
        rm -f "$dest"
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    echo ""
    echo "一部のファイルを自動取得できませんでした。"
    echo "https://www.tepco.co.jp/forecast/html/area_jukyu-j.html から手動でダウンロードし、"
    echo "$OUT_DIR/ に配置してください。"
    exit 1
fi

echo "done: $OUT_DIR"
