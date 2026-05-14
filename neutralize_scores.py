#!/usr/bin/env python3
"""行业+市值中性化：对原始预测分数做横截面回归取残差，消除行业/市值偏差"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "Qlib_沪深300_全量预测&回测.csv"
INDUSTRY_CACHE = ROOT / "industry_cache.json"


def build_industry_cache(codes: list[str]) -> dict:
    """批量获取行业和市值数据"""
    cache = {}
    for i, code in enumerate(codes):
        try:
            ticker = yf.Ticker(code)
            info = ticker.info
            cache[code] = {
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "mcap": info.get("marketCap", 0) or 0,
            }
        except Exception:
            cache[code] = {"sector": "Unknown", "industry": "Unknown", "mcap": 0}
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(codes)}")
        time.sleep(0.15)
    return cache


def neutralize(df: pd.DataFrame, industry_map: dict) -> pd.DataFrame:
    """按日期做横截面回归：score ~ C(industry) + log(mcap)，返残差

    df 需包含: date, symbol, 预测分数
    industry_map: {symbol: {"sector": ..., "mcap": ...}}
    """
    df = df.copy()
    df["sector"] = df["symbol"].map(lambda s: industry_map.get(s, {}).get("sector", "Unknown"))
    df["mcap"] = df["symbol"].map(lambda s: industry_map.get(s, {}).get("mcap", 0))
    # 用 close 做日频市值代理（跟 mcap 高度相关）
    df["log_price"] = np.log(df["close"].clip(lower=0.01))

    # 将 sector 做 one-hot
    sector_dummies = pd.get_dummies(df["sector"], prefix="sec")

    neutralized = []
    for date, grp in df.groupby("date"):
        if len(grp) < 50:  # 样本太少跳过
            grp["中性化预测分"] = grp["预测分数"]
            neutralized.append(grp)
            continue

        X = sector_dummies.loc[grp.index].copy().astype(float)
        X["log_price"] = grp["log_price"].values.astype(float)
        # 去掉常数项少的 dummy（跨截面回归需要满秩）
        X = X.loc[:, X.std() > 0.01]
        if X.shape[1] == 0:
            grp["中性化预测分"] = grp["预测分数"]
            neutralized.append(grp)
            continue

        y = grp["预测分数"].values
        try:
            # OLS: score = X * beta + epsilon
            beta = np.linalg.lstsq(X.values, y, rcond=None)[0]
            y_hat = X.values @ beta
            residuals = y - y_hat
            # 标准化残差，保持与原分数同量纲
            residuals = residuals * (y.std() / residuals.std()) if residuals.std() > 0 else residuals
            grp["中性化预测分"] = residuals
        except np.linalg.LinAlgError:
            grp["中性化预测分"] = grp["预测分数"]

        neutralized.append(grp)

    result = pd.concat(neutralized, ignore_index=True)
    result.drop(columns=["sector", "mcap", "log_price"], inplace=True, errors="ignore")
    return result


def evaluate(df: pd.DataFrame, score_col: str) -> dict:
    """评估某一预测分列的质量"""
    daily = df.copy().sort_values(["symbol", "date"])
    daily["next_ret"] = daily.groupby("symbol")["涨跌幅"].shift(-1)
    valid = daily.dropna(subset=["next_ret"])

    ics = []
    dir_accs = []
    for date, grp in valid.groupby("date"):
        if len(grp) < 30:
            continue
        ic = grp[score_col].corr(grp["next_ret"])
        ics.append(ic)
        pred_up = grp[score_col] > 0
        actual_up = grp["next_ret"] > 0
        dir_accs.append((pred_up == actual_up).mean())

    # Top20 策略
    def topk_daily(grp, k=20):
        top = grp.nlargest(k, score_col)
        return top["next_ret"].mean() / 100

    top20_ret = valid.groupby("date").apply(topk_daily, include_groups=False)
    cumret = float((1 + top20_ret).prod() - 1) if len(top20_ret) > 0 else 0
    n_days = len(top20_ret)
    ann_ret = float((1 + cumret) ** (252 / n_days) - 1) if n_days > 0 else 0

    return {
        "avg_ic": float(np.mean(ics)) if ics else 0,
        "pos_ic_ratio": float(np.mean([i > 0 for i in ics])) if ics else 0,
        "dir_acc": float(np.mean(dir_accs)) if dir_accs else 0,
        "top20_cumret": cumret,
        "top20_ann": ann_ret,
    }


def main():
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    print(f"数据: {len(df)}行, {df['date'].nunique()}天, {df['symbol'].nunique()}只")

    # ── 获取行业数据 ──
    symbols = sorted(df["symbol"].unique())
    # yfinance 需要纯代码.SS / .SZ 后缀（注意：symbol 是 sh600000 格式）
    yf_codes = []
    for s in symbols:
        code = s[2:]  # 去掉 sh/sz 前缀
        suffix = ".SS" if s[:2] in ("sh", "SH") else ".SZ"
        yf_codes.append(code + suffix)
    code_map = dict(zip(symbols, yf_codes))

    if INDUSTRY_CACHE.exists():
        print("加载行业缓存...")
        industry_map = json.loads(INDUSTRY_CACHE.read_text())
    else:
        print(f"获取 {len(yf_codes)} 只股票的行业数据（约需 {len(yf_codes)*0.15/60:.0f} 分钟）...")
        raw = build_industry_cache(yf_codes)
        # 映射回原始 symbol
        industry_map = {}
        for sym, yf_code in code_map.items():
            if yf_code in raw:
                industry_map[sym] = raw[yf_code]
        INDUSTRY_CACHE.write_text(json.dumps(industry_map, ensure_ascii=False, indent=2))
        print(f"行业缓存已保存: {len(industry_map)} 只")

    # ── 中性化 ──
    print("\n执行横截面中性化...")
    df_neut = neutralize(df, industry_map)
    print(f"中性化完成: {len(df_neut)}行")

    # ── 对比评估 ──
    print("\n========== 原始 vs 中性化 ==========")
    raw_metrics = evaluate(df, "预测分数")
    neut_metrics = evaluate(df_neut, "中性化预测分")

    print(f"{'指标':<20} {'原始预测分':>12} {'中性化预测分':>12} {'变化':>10}")
    print("-" * 55)
    for key, label in [("avg_ic", "日均IC"), ("pos_ic_ratio", "IC>0占比"),
                        ("dir_acc", "方向准确率"), ("top20_ann", "Top20年化")]:
        raw_v = raw_metrics[key]
        neut_v = neut_metrics[key]
        delta = neut_v - raw_v
        if "ann" in key:
            print(f"{label:<20} {raw_v:>11.1%} {neut_v:>11.1%} {delta:>+9.1%}")
        elif "ratio" in key:
            print(f"{label:<20} {raw_v:>11.1%} {neut_v:>11.1%} {delta:>+9.1%}")
        else:
            print(f"{label:<20} {raw_v:>11.4f} {neut_v:>11.4f} {delta:>+9.4f}")

    # ── 保存中性化版本到CSV ──
    # 保留中性化预测分作为新列
    out_cols = ["date", "股票代码", "open", "close", "high", "low",
                "volume", "amount", "振幅", "涨跌幅", "涨跌额", "换手率",
                "symbol", "预测分数", "中性化预测分"]
    df_neut[out_cols].to_csv(CSV_PATH, index=False)
    print(f"\n已保存到 {CSV_PATH} (新增列: 中性化预测分)")
    print("完成.")


if __name__ == "__main__":
    main()
