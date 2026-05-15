"""
一键补齐 2020-2024 年缺失的 上涨概率参考 / 信号强度
基于全量数据的 IsotonicRegression 校准
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

PROJECT_DIR = Path(__file__).parent
CSV_PATH = PROJECT_DIR / "data" / "Qlib_沪深300_全量预测&回测.csv"
BACKUP_PATH = PROJECT_DIR / "data" / "Qlib_沪深300_全量预测&回测.csv.bak"

print(f"读取 {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])
total = len(df)
print(f"  总行数: {total}")

# ── 1. 补齐原始预测分数 ──
raw_col = "原始预测分数"
df[raw_col] = pd.to_numeric(df[raw_col], errors="coerce")
raw_before = df[raw_col].notna().sum()
raw_null = df[raw_col].isna()
if raw_null.any():
    df.loc[raw_null, raw_col] = pd.to_numeric(df.loc[raw_null, "预测分数"], errors="coerce")
print(f"  原始预测分数: {raw_before} → {df[raw_col].notna().sum()} (补齐 {raw_null.sum()} 行)")

# ── 2. 计算次日涨跌标签 ──
df = df.sort_values(["symbol", "date"])
df["next_ret"] = df.groupby("symbol")["涨跌幅"].shift(-1)

# ── 3. Isotonic 校准 ──
train_mask = df[raw_col].notna() & df["next_ret"].notna()
print(f"  训练样本: {train_mask.sum()}")

label_up = (df.loc[train_mask, "next_ret"] > 0).astype(int)
calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
calibrator.fit(df.loc[train_mask, raw_col], label_up)
print(f"  保序回归拟合完成，X 范围 [{calibrator.X_min_:.4f}, {calibrator.X_max_:.4f}]")

# ── 4. 预测概率 ──
prob_col = "上涨概率参考"
prob_before = df[prob_col].notna().sum()
pred_mask = df[raw_col].notna()
df.loc[pred_mask, prob_col] = calibrator.predict(df.loc[pred_mask, raw_col])
df[prob_col] = df[prob_col].clip(0.01, 0.99)
print(f"  上涨概率参考: {prob_before} → {df[prob_col].notna().sum()} (补齐 {pred_mask.sum() - prob_before} 行)")

# ── 5. 重算预测分数 & 信号强度 ──
df["预测分数"] = (df[prob_col] - 0.5) * 2
df["信号强度"] = df["预测分数"].abs()
print(f"  预测分数 & 信号强度已基于校准概率重算")

# ── 6. 清理临时列，备份 + 覆写 ──
df = df.drop(columns=["next_ret"])

import shutil
shutil.copy2(CSV_PATH, BACKUP_PATH)
print(f"  已备份: {BACKUP_PATH}")

df.to_csv(CSV_PATH, index=False)
print(f"  已写入: {CSV_PATH}")

# ── 7. 快速验证 ──
latest = df["date"].max()
print(f"\n验证: 最新日期 {latest}, 总行数 {len(df)}")
print(f"  各列完整度:")
for c in ["原始预测分数", "上涨概率参考", "预测分数", "信号强度"]:
    n = df[c].notna().sum()
    print(f"    {c}: {n}/{total} ({n/total*100:.1f}%)")
