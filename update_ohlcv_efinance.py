#!/usr/bin/env python3
"""更新 CSI300 成分股 OHLCV CSV — 使用 efinance（东方财富后端，直连）"""
import os, time, random
from pathlib import Path
from datetime import datetime, timedelta

for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        del os.environ[k]

import efinance as ef
import pandas as pd

SAVE_DIR = Path.home() / ".qlib/a_share_csv_enhanced"


def main():
    csv_files = sorted(SAVE_DIR.glob("sh*.csv")) + sorted(SAVE_DIR.glob("sz*.csv"))
    total = len(csv_files)
    updated = skipped = failed = 0
    new_rows_total = 0
    today = datetime.now()

    print(f"CSI300 OHLCV 增量更新 — efinance 数据源")
    print(f"共 {total} 个 CSV 文件")
    print(f"{'='*60}")

    for i, fp in enumerate(csv_files):
        symbol = fp.stem  # e.g., sh600000
        code = symbol[2:]  # e.g., 600000

        try:
            existing = pd.read_csv(fp)
            existing["date"] = pd.to_datetime(existing["date"])
            last_date = existing["date"].max()
        except Exception:
            failed += 1
            continue

        # 如果最新数据换手率>0且日期已到昨天则跳过；否则强制刷新最近4天
        needs_update = True
        if last_date >= today - timedelta(days=1):
            last_row = existing.iloc[-1]
            if float(last_row.get("turn", 0) or 0) > 0:
                skipped += 1
                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{total}] {updated}更新 {skipped}跳过 {failed}失败")
                continue
            # turn=0 说明之前下载缺字段，强制刷新
            start_dl = (last_date - timedelta(days=4)).strftime("%Y%m%d")
        else:
            start_dl = (last_date + timedelta(days=1)).strftime("%Y%m%d")

        end_dl = today.strftime("%Y%m%d")

        try:
            df_new = ef.stock.get_quote_history(code, beg=start_dl, end=end_dl)
            time.sleep(0.15 + random.random() * 0.2)
        except Exception:
            time.sleep(2 + random.random() * 2)
            try:
                df_new = ef.stock.get_quote_history(code, beg=start_dl, end=end_dl)
            except Exception as e:
                failed += 1
                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{total}] {symbol} 下载失败: {e}")
                continue

        if df_new is None or df_new.empty:
            skipped += 1
            continue

        # efinance 列: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 涨跌幅, 换手率
        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df_new["日期"])
        out["open"] = pd.to_numeric(df_new["开盘"], errors="coerce")
        out["high"] = pd.to_numeric(df_new["最高"], errors="coerce")
        out["low"] = pd.to_numeric(df_new["最低"], errors="coerce")
        out["close"] = pd.to_numeric(df_new["收盘"], errors="coerce")
        # 成交量是手（lot），×100 转为 股
        out["volume"] = (pd.to_numeric(df_new["成交量"], errors="coerce") * 100).fillna(0)
        out["amount"] = pd.to_numeric(df_new["成交额"], errors="coerce").fillna(0)
        out["turn"] = pd.to_numeric(df_new["换手率"], errors="coerce").fillna(0)
        out["pctChg"] = pd.to_numeric(df_new["涨跌幅"], errors="coerce").fillna(0)
        out["symbol"] = symbol.upper()

        out = out[out["date"] > last_date]
        if out.empty:
            skipped += 1
            continue

        # 对齐列
        for col in existing.columns:
            if col not in out.columns:
                out[col] = 0.0
        out = out[existing.columns]

        combined = pd.concat([existing, out], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last")
        combined = combined.sort_values("date")
        combined.to_csv(fp, index=False)

        updated += 1
        new_rows_total += len(out)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] {updated}更新({new_rows_total}行) {skipped}跳过 {failed}失败")

    print(f"\n{'='*60}")
    print(f"完成! {updated} 文件 ({new_rows_total} 新行) | {skipped} 跳过 | {failed} 失败")


if __name__ == "__main__":
    main()
