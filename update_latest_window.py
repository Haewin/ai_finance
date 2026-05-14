#!/usr/bin/env python3
"""增量更新：仅重跑最后一个 Walk-Forward 窗口（2025→最新），合并回完整CSV"""
import json, multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha158
from qlib.utils import init_instance_by_config
from qlib.tests.config import get_record_xgboost_config

XGB_PARAMS = {
    "eval_metric": "rmse",
    "colsample_bytree": 0.85,
    "eta": 0.04,
    "max_depth": 6,
    "subsample": 0.85,
    "nthread": 16,
}

QLIB_PROVIDER = Path.home() / ".qlib/qlib_data/csi300_full"
CSV_DIR = Path.home() / ".qlib/a_share_csv_enhanced"
DASHBOARD_DIR = Path(__file__).resolve().parent
CSV_PATH = DASHBOARD_DIR / "data" / "Qlib_沪深300_全量预测&回测.csv"

# ── 初始化 ──
mp.set_start_method("fork", force=True)
qlib.init(provider_uri=QLIB_PROVIDER.as_posix(), region=REG_CN)

instruments = "all"
cal = [pd.Timestamp(x) for x in D.calendar()]
latest_cal_day = cal[-1]  # 最新交易日
print(f"交易日历: {cal[0].date()} → {latest_cal_day.date()} ({len(cal)}天)")

# ── 预加载 OHLCV 缓存 ──
ohlcv_cache = {}
csv_files = list(CSV_DIR.glob("*.csv"))
for csv_f in csv_files:
    try:
        cdf = pd.read_csv(csv_f)
        cdf["date"] = pd.to_datetime(cdf["date"])
        sym = csv_f.stem.upper()
        for _, r in cdf.iterrows():
            ohlcv_cache[(r["date"], sym)] = r
    except Exception:
        continue
print(f"OHLCV cache: {len(ohlcv_cache)} entries from {len(csv_files)} files")

# ── Window 6: Train 2020-2024 → Test 2025 → 最新 ──
tr_start, tr_end = "2020-01-02", "2024-12-31"
te_start = "2025-01-02"
te_end = latest_cal_day.strftime("%Y-%m-%d")

print(f"\nWindow 6: Train={tr_start}→{tr_end}, Test={te_start}→{te_end}")

# Valid: train 最后 ~90 天
tr_end_ts = pd.Timestamp(tr_end)
va_start = max(pd.Timestamp(tr_start), tr_end_ts - pd.Timedelta(days=120)).strftime("%Y-%m-%d")

handler = Alpha158(instruments=instruments, start_time=tr_start, end_time=te_end, freq="day")
dataset = DatasetH(
    handler=handler,
    segments={
        "train": (pd.Timestamp(tr_start), pd.Timestamp(tr_end)),
        "valid": (pd.Timestamp(va_start), pd.Timestamp(tr_end)),
        "test": (pd.Timestamp(te_start), pd.Timestamp(te_end)),
    },
)

# 构建模型
conf = get_record_xgboost_config()["model"].copy()
conf["kwargs"] = XGB_PARAMS.copy()
model = init_instance_by_config(conf)
model.fit(dataset, num_boost_round=500, early_stopping_rounds=50, verbose_eval=False)

# 校准器：把原始 score 映射成上涨概率参考
valid_pred = model.predict(dataset, segment="valid")
valid_label = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_I)
if isinstance(valid_label, pd.DataFrame):
    valid_label = valid_label.iloc[:, 0]
valid_joined = pd.concat([valid_pred.rename("score"), valid_label.rename("label")], axis=1).dropna()
calibrator = None
if not valid_joined.empty and valid_joined["score"].nunique() >= 10:
    valid_joined["label_up"] = (valid_joined["label"] > 0).astype(int)
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(valid_joined["score"].values, valid_joined["label_up"].values)

# 预测
pred = model.predict(dataset, segment="test")
pred_reset = pred.reset_index()
pred_reset.columns = ["date", "instrument", "score"]
pred_reset["原始预测分数"] = pred_reset["score"]
if calibrator is not None:
    pred_reset["上涨概率参考"] = calibrator.predict(pred_reset["score"].values)
else:
    pred_reset["上涨概率参考"] = pred_reset.groupby("date")["score"].rank(method="average", pct=True).clip(0.01, 0.99)
pred_reset["预测分数"] = (pred_reset["上涨概率参考"] - 0.5) * 2
pred_reset["信号强度"] = pred_reset["预测分数"].abs()

# 组装 CSV 行
rows = []
for _, row in pred_reset.iterrows():
    d, instr, score = row["date"], row["instrument"], row["score"]
    key = (d, instr.upper())
    if key in ohlcv_cache:
        r = ohlcv_cache[key]
        entry = {
            "date": d,
            "股票代码": instr.lower(),
            "open": float(r.get("open", 0) or 0),
            "close": float(r.get("close", 0) or 0),
            "high": float(r.get("high", 0) or 0),
            "low": float(r.get("low", 0) or 0),
            "volume": float(r.get("volume", 0) or 0),
            "amount": float(r.get("amount", 0) or 0),
            "振幅": float(r.get("pctChg", 0) or 0),
            "涨跌幅": float(r.get("pctChg", 0) or 0),
            "涨跌额": 0.0,
            "换手率": float(r.get("turn", 0) or 0),
            "symbol": instr,
            "原始预测分数": float(row["原始预测分数"]),
            "上涨概率参考": float(row["上涨概率参考"]),
            "预测分数": float(row["预测分数"]),
            "信号强度": float(row["信号强度"]),
        }
    else:
        entry = {
            "date": d, "股票代码": instr.lower(),
            "open": 0, "close": 0, "high": 0, "low": 0,
            "volume": 0, "amount": 0, "振幅": 0, "涨跌幅": 0,
            "涨跌额": 0, "换手率": 0, "symbol": instr,
            "原始预测分数": float(row["原始预测分数"]),
            "上涨概率参考": float(row["上涨概率参考"]),
            "预测分数": float(row["预测分数"]),
            "信号强度": float(row["信号强度"]),
        }
    rows.append(entry)

new_window_df = pd.DataFrame(rows)
new_window_df["date"] = pd.to_datetime(new_window_df["date"])

# ── 合并：保留旧CSV中2025年之前的数据 + 新的Window6数据 ──
old_df = pd.read_csv(CSV_PATH)
old_df["date"] = pd.to_datetime(old_df["date"])

# 删除旧的2025年之后数据
cutoff = pd.Timestamp("2025-01-01")
old_before = old_df[old_df["date"] < cutoff]
merged = pd.concat([old_before, new_window_df], ignore_index=True)
merged = merged.sort_values(["date", "股票代码"])
merged.to_csv(CSV_PATH, index=False)

# ── 统计 ──
print(f"\n旧数据 (before 2025): {len(old_before)}行, {old_before['date'].nunique()}天")
print(f"新窗口6: {len(new_window_df)}行, {new_window_df['date'].nunique()}天")
print(f"合并后: {len(merged)}行, {merged['date'].nunique()}天, {merged['股票代码'].nunique()}只")
print(f"日期范围: {merged['date'].min().date()} → {merged['date'].max().date()}")
pos_rate = (merged["预测分数"] > 0).mean() * 100
print(f"预测均值: {merged['预测分数'].mean():.4f}, 看涨占比: {pos_rate:.1f}%")

# 验证质量
test_merged = merged[merged["date"] >= cutoff]
test_metrics = test_merged.copy()
test_metrics["pred_sign"] = test_metrics["预测分数"] > 0
# IC
ic_vals = []
for d in sorted(test_merged["date"].unique()):
    day = test_merged[test_merged["date"] == d]
    # Need returns for IC — use 涨跌幅 as proxy for next-day
    if len(day) >= 30:
        ic = day["预测分数"].corr(day["涨跌幅"])
        ic_vals.append(ic)
avg_ic = float(np.mean(ic_vals)) if ic_vals else None
print(f"新窗口 IC (同日): {avg_ic:.4f}" if avg_ic else "IC: N/A")
print(f"\n保存到: {CSV_PATH}")
print("完成.")
