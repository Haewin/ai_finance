#!/usr/bin/env python3
"""从腾讯数据源下载 CSI300 最新K线，计算涨跌幅后增量更新CSV"""
import os, time, random
from pathlib import Path
from datetime import datetime, timedelta

for k in list(os.environ.keys()):
    if 'proxy' in k.lower():
        del os.environ[k]

import akshare as ak
import pandas as pd

SAVE_DIR = Path.home() / ".qlib/a_share_csv_enhanced"


def main():
    csv_files = sorted(SAVE_DIR.glob("sh*.csv")) + sorted(SAVE_DIR.glob("sz*.csv"))
    total = len(csv_files)
    updated = skipped = failed = 0
    new_rows_total = 0
    today = datetime.now()

    print(f"CSI300 OHLCV 增量更新 — 腾讯数据源")
    print(f"共 {total} 个 CSV 文件")
    print(f"{'='*60}")

    for i, fp in enumerate(csv_files):
        symbol = fp.stem
        code = symbol[2:]

        try:
            existing = pd.read_csv(fp)
            existing["date"] = pd.to_datetime(existing["date"])
            last_date = existing["date"].max()
            prev_close = float(existing["close"].iloc[-1])  # 用于算涨跌幅
        except Exception:
            failed += 1
            continue

        # 已有今天的数据且涨跌幅非0则跳过
        if last_date >= today - timedelta(days=1):
            last_row = existing.iloc[-1]
            if float(last_row.get("pctChg", 0) or 0) != 0:
                skipped += 1
                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{total}] {updated}更新 {skipped}跳过 {failed}失败")
                continue

        start_dl = (last_date - timedelta(days=1)).strftime("%Y%m%d")  # 多拉一天确保能算涨跌幅
        end_dl = today.strftime("%Y%m%d")

        tx_symbol = f"sh{code}" if symbol.startswith("sh") else f"sz{code}"

        df_new = None
        for attempt in range(3):
            try:
                df_new = ak.stock_zh_a_hist_tx(
                    symbol=tx_symbol,
                    start_date=start_dl,
                    end_date=end_dl,
                    adjust="qfq",
                    timeout=15,
                )
                if df_new is not None and not df_new.empty:
                    break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 + random.random() * 3)
                else:
                    failed += 1
                    if (i + 1) % 20 == 0:
                        print(f"  [{i+1}/{total}] {symbol} 3次重试均失败: {e}")
                    break

        if df_new is None or df_new.empty:
            skipped += 1
            continue

        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df_new["date"])
        out["open"] = pd.to_numeric(df_new["open"], errors="coerce")
        out["high"] = pd.to_numeric(df_new["high"], errors="coerce")
        out["low"] = pd.to_numeric(df_new["low"], errors="coerce")
        out["close"] = pd.to_numeric(df_new["close"], errors="coerce")
        # amount 字段在腾讯API是成交量(手), ×100 转为股
        out["volume"] = (pd.to_numeric(df_new["amount"], errors="coerce") * 100).fillna(0)
        out["amount"] = 0.0
        out["turn"] = 0.0
        out["symbol"] = symbol.upper()

        # 计算涨跌幅: (close - prev_close) / prev_close * 100
        # 用 existing 最后一天的收盘价作为基准
        all_close = [prev_close] + out["close"].tolist()
        pct_chg = []
        for j in range(len(out)):
            cur = all_close[j + 1]
            prv = all_close[j]
            pct_chg.append(round((cur - prv) / prv * 100, 2) if prv and prv > 0 else 0.0)
        out["pctChg"] = pct_chg

        # 只用新日期
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

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] {updated}更新({new_rows_total}行) {skipped}跳过 {failed}失败")

        time.sleep(0.15 + random.random() * 0.3)

    print(f"\n{'='*60}")
    print(f"完成! {updated} 更新 ({new_rows_total} 新行) | {skipped} 跳过 | {failed} 失败")


if __name__ == "__main__":
    main()
