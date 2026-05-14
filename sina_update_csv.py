#!/usr/bin/env python3
"""使用新浪API增量更新CSI300成分股CSV文件

仅下载 2026-05-08 之后的新交易日数据，追加到已有CSV
"""
import json
import time
import random
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd

SAVE_DIR = Path.home() / ".qlib/a_share_csv_enhanced"
HIST_API = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"

HEADERS = {
    "Referer": "https://finance.sina.com.cn",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
}
PROXY = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
CUTOFF = datetime(2026, 5, 11)


def get_stock_list() -> list[tuple[str, str]]:
    """从已有CSV文件获取股票列表"""
    stocks = []
    for fp in sorted(SAVE_DIR.glob("*.csv")):
        name = fp.stem.lower()  # e.g., sh600000
        if len(name) >= 8 and name[:2] in ("sh", "sz"):
            market = name[:2]
            code = name[2:].zfill(6) if name[2:].isdigit() else name[2:]
            stocks.append((market, code))
    print(f"Found {len(stocks)} CSV files in {SAVE_DIR}")
    return stocks


def get_hist_kline(market: str, code: str, datalen: int = 10) -> list[dict]:
    """从新浪历史K线API获取数据"""
    symbol = f"{market}{code}"
    params = {"symbol": symbol, "scale": 240, "ma": "no", "datalen": datalen}
    for attempt in range(3):
        try:
            r = requests.get(HIST_API, params=params, headers=HEADERS,
                             proxies=PROXY, timeout=20)
            if r.status_code == 200:
                return json.loads(r.text)
        except Exception:
            pass
        time.sleep(1 + attempt)
    return []


def main():
    stocks = get_stock_list()
    total = len(stocks)
    updated = skipped = failed = 0
    new_rows_total = 0
    start_time = time.time()

    print(f"Updating CSV files — cutoff: {CUTOFF.date()}")
    print("=" * 60)

    for i, (market, code) in enumerate(stocks):
        symbol = f"{market}{code}"
        filepath = SAVE_DIR / f"{symbol}.csv"

        if not filepath.exists():
            skipped += 1
            continue

        try:
            existing = pd.read_csv(filepath)
            existing["date"] = pd.to_datetime(existing["date"])
            last_date = existing["date"].max()
        except Exception:
            failed += 1
            continue

        # 检查是否已经有最新数据（今天或昨天）
        if last_date >= datetime.now() - timedelta(days=1):
            skipped += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{total}] already up-to-date through {last_date.date()}")
            continue

        hist_data = get_hist_kline(market, code, datalen=8)
        if not hist_data:
            failed += 1
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{total}] {symbol}: API failed")
            continue

        new_rows = []
        for d in hist_data:
            d_date = datetime.strptime(d["day"], "%Y-%m-%d")
            if d_date > CUTOFF:
                close_val = float(d["close"])
                new_rows.append({
                    "date": d["day"],
                    "open": float(d["open"]),
                    "high": float(d["high"]),
                    "low": float(d["low"]),
                    "close": close_val,
                    "volume": float(d["volume"]),
                    "amount": 0.0,
                    "turn": 0.0,
                    "pctChg": 0.0,  # placeholder, computed below
                    "symbol": symbol.upper(),
                })

        if not new_rows:
            skipped += 1
            continue

        # Compute pctChg: use last known close as base
        all_closes = [float(d["close"]) for d in hist_data]
        new_rows_sorted = sorted(new_rows, key=lambda r: r["date"])
        prev_close = float(existing["close"].iloc[-1])
        for nr in new_rows_sorted:
            if prev_close > 0:
                nr["pctChg"] = round((nr["close"] - prev_close) / prev_close * 100, 4)
            prev_close = nr["close"]

        new_df = pd.DataFrame(new_rows_sorted)
        new_df["date"] = pd.to_datetime(new_df["date"])
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
        merged.to_csv(filepath, index=False, encoding="utf-8-sig")

        new_rows_total += len(new_rows_sorted)
        updated += 1
        latest = new_df["date"].max().strftime("%Y-%m-%d")
        print(f"  [{i+1}/{total}] {symbol}: +{len(new_rows_sorted)} rows → {latest}")

        delay = 0.2 + random.uniform(0, 0.2)
        time.sleep(delay)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Done! {elapsed:.0f}s | {updated} files updated ({new_rows_total} new rows) | "
          f"{skipped} skipped | {failed} failed")


if __name__ == "__main__":
    main()
