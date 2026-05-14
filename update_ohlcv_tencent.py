#!/usr/bin/env python3
"""更新 CSI300 成分股 OHLCV CSV — 使用 akshare 腾讯数据源（直连，不走代理）"""
import os, time, random
from pathlib import Path
from datetime import datetime, timedelta

# 清空代理环境变量，走直连
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
        symbol = fp.stem  # e.g., sh600000

        try:
            existing = pd.read_csv(fp)
            existing["date"] = pd.to_datetime(existing["date"])
            last_date = existing["date"].max()
        except Exception:
            failed += 1
            continue

        # 如果已有昨天或今天的数据就跳过
        if last_date >= today - timedelta(days=1):
            skipped += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{total}] 进度 — {updated} 更新, {skipped} 跳过, {failed} 失败")
            continue

        # 用腾讯数据源下载
        start_dl = (last_date + timedelta(days=1)).strftime("%Y%m%d")
        end_dl = today.strftime("%Y%m%d")

        try:
            df_new = ak.stock_zh_a_hist_tx(
                symbol=f"sh{symbol[2:]}" if symbol.startswith("sh") else f"sz{symbol[2:]}",
                start_date=start_dl,
                end_date=end_dl,
                adjust="qfq",
                timeout=15,
            )
        except Exception:
            # 等一会儿重试一次
            time.sleep(2 + random.random() * 2)
            try:
                df_new = ak.stock_zh_a_hist_tx(
                    symbol=f"sh{symbol[2:]}" if symbol.startswith("sh") else f"sz{symbol[2:]}",
                    start_date=start_dl,
                    end_date=end_dl,
                    adjust="qfq",
                    timeout=15,
                )
            except Exception as e:
                failed += 1
                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{total}] {symbol} 下载失败: {e}")
                continue

        if df_new is None or df_new.empty:
            skipped += 1
            continue

        # 转换为 CSV 格式
        # akshare Tencent 返回: date, open, close, high, low, amount
        # amount 实际是成交量（手），需 ×100 转为 股
        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df_new["date"])
        out["open"] = pd.to_numeric(df_new["open"], errors="coerce")
        out["high"] = pd.to_numeric(df_new["high"], errors="coerce")
        out["low"] = pd.to_numeric(df_new["low"], errors="coerce")
        out["close"] = pd.to_numeric(df_new["close"], errors="coerce")
        out["volume"] = (pd.to_numeric(df_new["amount"], errors="coerce") * 100).fillna(0)
        out["amount"] = 0.0          # 腾讯不返回成交额
        out["turn"] = 0.0            # 腾讯不返回换手率
        out["pctChg"] = 0.0          # 腾讯不返回涨跌幅
        out["symbol"] = symbol.upper()

        # 只保留新日期
        out = out[out["date"] > last_date]
        if out.empty:
            skipped += 1
            continue

        # 与现有数据对齐列
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
            print(f"  [{i+1}/{total}] 进度 — {updated} 更新 ({new_rows_total}行), {skipped} 跳过, {failed} 失败")

        time.sleep(0.15 + random.random() * 0.3)

    print(f"\n{'='*60}")
    print(f"完成! {updated} 文件更新 ({new_rows_total} 新行) | {skipped} 跳过 | {failed} 失败")


if __name__ == "__main__":
    main()
