#!/usr/bin/env python3
"""从 akshare 下载 CSI300 成分股最新交易日数据，增量追加到 CSV"""
import os, time, random
from pathlib import Path
from datetime import datetime, timedelta

# 走本地代理，避免直连被墙
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

import akshare as ak
import pandas as pd

SAVE_DIR = Path.home() / ".qlib/a_share_csv_enhanced"
START_DATE = "20260511"  # akshare 用 YYYYMMDD 格式


def get_csv_list() -> list[Path]:
    """获取已有CSV文件列表"""
    return sorted(SAVE_DIR.glob("sh*.csv")) + sorted(SAVE_DIR.glob("sz*.csv"))


def download_stock_hist(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """
    从 akshare 下载个股日K数据。
    symbol: sh600000 或 sz000001
    start/end: YYYYMMDD
    """
    for attempt in range(3):
        try:
            # 确定市场和代码
            if symbol.startswith("sh"):
                mkt, code = "sh", symbol[2:]
                ak_symbol = code  # akshare 用纯数字代码
            else:
                mkt, code = "sz", symbol[2:]
                ak_symbol = code

            df = ak.stock_zh_a_hist(
                symbol=ak_symbol,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="qfq",  # 前复权
            )
            if df is None or df.empty:
                return None
            return df
        except Exception as e:
            if attempt < 2:
                time.sleep(2 + random.random() * 3)
            else:
                print(f"  [{symbol}] 下载失败: {e}")
                return None


def convert_to_csv_format(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """将 akshare 输出转为与现有 CSV 一致的格式"""
    # akshare columns: 日期, 开盘, 最高, 最低, 收盘, 成交量, 成交额, 涨跌幅, 涨跌额, 换手率, ...
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["日期"])
    out["open"] = pd.to_numeric(df["开盘"], errors="coerce")
    out["high"] = pd.to_numeric(df["最高"], errors="coerce")
    out["low"] = pd.to_numeric(df["最低"], errors="coerce")
    out["close"] = pd.to_numeric(df["收盘"], errors="coerce")
    out["volume"] = pd.to_numeric(df["成交量"], errors="coerce")
    out["amount"] = pd.to_numeric(df["成交额"], errors="coerce")
    out["turn"] = pd.to_numeric(df.get("换手率", pd.Series([0] * len(df))), errors="coerce").fillna(0)
    out["pctChg"] = pd.to_numeric(df.get("涨跌幅", pd.Series([0] * len(df))), errors="coerce").fillna(0)
    out["symbol"] = symbol.upper()
    return out.dropna(subset=["open", "close"])


def main():
    csv_files = get_csv_list()
    print(f"找到 {len(csv_files)} 个 CSV 文件")
    print(f"下载起始日期: {START_DATE}\n")

    updated = 0
    skipped = 0
    failed = 0
    total_new_rows = 0

    for i, fp in enumerate(csv_files):
        symbol = fp.stem  # e.g., sh600000
        # 读取已有数据，获取最后日期
        try:
            existing = pd.read_csv(fp)
            existing["date"] = pd.to_datetime(existing["date"])
            last_date = existing["date"].max()
        except Exception:
            print(f"  [{symbol}] 读取CSV失败，跳过")
            failed += 1
            continue

        # 如果已经是最新，跳过
        target_end = datetime.now().strftime("%Y%m%d")
        if last_date.strftime("%Y%m%d") >= target_end:
            skipped += 1
            if i % 100 == 0:
                print(f"  [{i+1}/{len(csv_files)}] 进度: {updated} 更新, {skipped} 跳过, {failed} 失败")
            continue

        # 下载数据
        start_dl = (last_date + timedelta(days=1)).strftime("%Y%m%d")
        df_new = download_stock_hist(symbol, start_dl, target_end)

        if df_new is None or df_new.empty:
            skipped += 1
            if i % 50 == 0:
                print(f"  [{i+1}/{len(csv_files)}] 进度: {updated} 更新, {skipped} 跳过, {failed} 失败")
            continue

        # 转换格式
        new_rows = convert_to_csv_format(df_new, symbol)
        if new_rows.empty:
            skipped += 1
            continue

        # 只保留新日期
        new_rows = new_rows[new_rows["date"] > last_date]
        if new_rows.empty:
            skipped += 1
            continue

        # 追加到 CSV
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last")
        combined = combined.sort_values("date")
        combined.to_csv(fp, index=False)

        updated += 1
        total_new_rows += len(new_rows)

        if i % 50 == 0:
            print(f"  [{i+1}/{len(csv_files)}] 进度: {updated} 更新, {skipped} 跳过, {failed} 失败")

        # 反爬延时
        time.sleep(0.3 + random.random() * 0.5)

    print(f"\n{'='*60}")
    print(f"完成! {updated} 文件更新 ({total_new_rows} 新行) | {skipped} 跳过 | {failed} 失败")


if __name__ == "__main__":
    main()
