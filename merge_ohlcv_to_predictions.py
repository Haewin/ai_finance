"""
将 ~/.qlib/a_share_csv_enhanced/ 中的最新 OHLCV 数据合入预测 CSV
修复涨跌幅/换手率等字段为 0 的问题
"""
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

OHLCV_DIR = Path.home() / ".qlib/a_share_csv_enhanced"
CSV_PATH = Path(__file__).parent / "data" / "Qlib_沪深300_全量预测&回测.csv"
BACKUP_PATH = CSV_PATH.with_suffix(".csv.bak2")

print("读取预测 CSV ...")
df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])
print(f"  总行数: {len(df)}, 日期: {df['date'].min().date()} → {df['date'].max().date()}")

# 加载 OHLCV 缓存
print(f"\n读取 OHLCV CSVs from {OHLCV_DIR} ...")
ohlcv = {}
csv_files = list(OHLCV_DIR.glob("*.csv"))
for fp in csv_files:
    try:
        cdf = pd.read_csv(fp)
        cdf["date"] = pd.to_datetime(cdf["date"])
        sym = fp.stem.upper()
        for _, r in cdf.iterrows():
            ohlcv[(r["date"], sym)] = r
    except Exception:
        continue
print(f"  缓存条目: {len(ohlcv)} from {len(csv_files)} files")

# 找出涨跌幅为零的行并尝试修复
zero_mask = df["涨跌幅"] == 0
print(f"\n涨跌幅==0 的行: {zero_mask.sum()}/{len(df)}")
print(f"其中换手率也为0: {(df['换手率'] == 0).sum()}")

# 只修复最近的日期（加速）
recent_dates = sorted(df["date"].unique())[-10:]
print(f"修复范围: {recent_dates[0].date()} → {recent_dates[-1].date()}")

fixed = 0
for idx in df.index:
    row = df.loc[idx]
    if row["涨跌幅"] != 0:
        continue
    if row["date"] not in recent_dates:
        continue
    key = (row["date"], str(row["symbol"]).upper())
    if key in ohlcv:
        r = ohlcv[key]
        df.at[idx, "open"] = float(r.get("open", row["open"]) or row["open"])
        df.at[idx, "close"] = float(r.get("close", row["close"]) or row["close"])
        df.at[idx, "high"] = float(r.get("high", row["high"]) or row["high"])
        df.at[idx, "low"] = float(r.get("low", row["low"]) or row["low"])
        df.at[idx, "volume"] = float(r.get("volume", row["volume"]) or row["volume"])
        df.at[idx, "amount"] = float(r.get("amount", row["amount"]) or row["amount"])
        df.at[idx, "振幅"] = float(r.get("pctChg", row["振幅"]) or row["振幅"])
        df.at[idx, "涨跌幅"] = float(r.get("pctChg", df.at[idx, "涨跌幅"]) or 0)
        df.at[idx, "换手率"] = float(r.get("turn", row["换手率"]) or row["换手率"])
        fixed += 1

print(f"修复了 {fixed} 行")

if fixed > 0:
    # 验证
    recent = df[df["date"] >= recent_dates[-3]]
    for d in sorted(recent["date"].unique()):
        day = recent[recent["date"] == d]
        nz = (day["涨跌幅"] != 0).sum()
        print(f"  {d.date()}: {nz}/{len(day)} 涨跌幅非零")

    # 备份并保存
    shutil.copy2(CSV_PATH, BACKUP_PATH)
    print(f"已备份到: {BACKUP_PATH}")
    df.to_csv(CSV_PATH, index=False)
    print(f"已保存: {CSV_PATH}")
else:
    print("无行被修复 — OHLCV 缓存可能已是最新或缓存缺失")
