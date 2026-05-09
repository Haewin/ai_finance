"""
实验数据自动解析器
遍历 Codex 结果目录，自动提取所有实验指标
"""
import json
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

CODEX_BASE = Path(__file__).parent / "results"

# 实验结果目录名 → 显示名映射
EXPERIMENT_DIRS = {
    "results_xgb_top300": "XGBoost Top20",
    "results_xgb_top300_v2": "XGBoost Top50",
    "results_lgb_top300": "LightGBM Top50",
    "results_ensemble_top300": "LGB+XGB 融合",
    "results_xgb_top300_enhanced": "XGB+增强因子",
    "results_xgb_enhanced_direct": "XGB+增强因子(直投)",
    "native_qlib_a_share_results": "原生XGB Top50",
    "results_improved_v1": "改进版V1 (旧)",
    "results_improved_v2": "改进版V2 (当前)",
}

TOP_SECONDARY_DIRS = {
    "topk_compare_results": "TopK策略对比",
    "test_window_compare_results": "窗口稳定性",
}


def parse_md_summary(path: Path) -> dict:
    """解析 summary.md 格式"""
    text = path.read_text(encoding="utf-8")
    result = {}

    # 标题
    m = re.search(r"^# (.+)$", text, re.MULTILINE)
    if m:
        result["title"] = m.group(1).strip()

    # 字段
    patterns = {
        "model": r"- Model:\s*(.+)",
        "strategy": r"- Strategy:\s*(.+)",
        "directional_accuracy": r"Directional accuracy:\s*([\d.]+)",
        "annualized_return": r"Annualized return:\s*([\d.]+)",
        "information_ratio": r"Information ratio:\s*([\d.]+)",
        "max_drawdown": r"Max drawdown:\s*(-?[\d.]+)",
        "train_period": r"- Train:\s*(.+)",
        "valid_period": r"- Valid:\s*(.+)",
        "test_period": r"- Test:\s*(.+)",
        "recorder_id": r"Recorder id:\s*(\S+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            val = m.group(1).strip()
            if key in ("directional_accuracy", "annualized_return", "information_ratio", "max_drawdown"):
                result[key] = float(val)
            else:
                result[key] = val

    return result


def parse_json_summary(path: Path) -> dict:
    """解析 summary.json 格式"""
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {
        "title": f"Native Qlib A-share Experiment ({data.get('model', 'unknown')})",
        "model": data.get("model", ""),
        "strategy": f"TopkDropout(topk={data.get('topk')}, n_drop={data.get('n_drop')})",
        "directional_accuracy": data.get("directional_accuracy"),
        "annualized_return": data.get("annualized_return"),
        "information_ratio": data.get("information_ratio"),
        "max_drawdown": data.get("max_drawdown"),
        "train_period": f"{data.get('train_start', '')} to {data.get('train_end', '')}",
        "valid_period": f"{data.get('valid_start', '')} to {data.get('valid_end', '')}",
        "test_period": f"{data.get('test_start', '')} to {data.get('test_end', '')}",
    }
    # 处理 NaN
    import math
    for k in ("directional_accuracy", "annualized_return", "information_ratio", "max_drawdown"):
        v = result.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            result[k] = None
    return result


def load_all_experiments() -> pd.DataFrame:
    """加载所有实验结果，返回统一 DataFrame"""
    rows = []

    for dirname, display_name in EXPERIMENT_DIRS.items():
        exp_dir = CODEX_BASE / dirname
        if not exp_dir.exists():
            continue

        # 尝试两种格式
        md_path = exp_dir / "summary.md"
        json_path = exp_dir / "summary.json"

        data = None
        if md_path.exists():
            data = parse_md_summary(md_path)
        elif json_path.exists():
            data = parse_json_summary(json_path)

        if data is None:
            continue

        rows.append({
            "实验名称": display_name,
            "目录名": dirname,
            "模型": data.get("model", ""),
            "方向准确率": data.get("directional_accuracy"),
            "年化收益": data.get("annualized_return"),
            "信息比率": data.get("information_ratio"),
            "最大回撤": data.get("max_drawdown"),
            "训练区间": data.get("train_period", ""),
            "验证区间": data.get("valid_period", ""),
            "测试区间": data.get("test_period", ""),
            "策略": data.get("strategy", ""),
        })

    return pd.DataFrame(rows)


def load_topk_comparison() -> pd.DataFrame:
    """加载 TopK 对比数据"""
    md_path = CODEX_BASE / "topk_compare_results" / "compare_topk_summary.md"
    if not md_path.exists():
        return pd.DataFrame()

    text = md_path.read_text(encoding="utf-8")

    # 解析 markdown 表格
    lines = text.split("\n")
    table_start = None
    for i, line in enumerate(lines):
        if line.startswith("| model"):
            table_start = i
            break

    if table_start is None:
        return pd.DataFrame()

    header_line = lines[table_start]
    sep_line = lines[table_start + 1]
    headers = [h.strip() for h in header_line.split("|")[1:-1]]

    rows = []
    for line in lines[table_start + 2:]:
        if not line.startswith("|"):
            break
        vals = [v.strip() for v in line.split("|")[1:-1]]
        if len(vals) == len(headers):
            rows.append(vals)

    df = pd.DataFrame(rows, columns=headers)

    # 转换数值列
    num_cols = ["topk", "n_drop", "directional_accuracy", "ic", "rank_ic",
                "annualized_return", "information_ratio", "max_drawdown"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 重命名
    col_map = {
        "model": "模型", "topk": "TopK", "n_drop": "Drop",
        "directional_accuracy": "方向准确率", "ic": "IC", "rank_ic": "Rank IC",
        "annualized_return": "年化收益", "information_ratio": "信息比率",
        "max_drawdown": "最大回撤",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "TopK" in df.columns:
        df["策略"] = "Top " + df["TopK"].astype(int).astype(str)

    return df


def load_window_comparison() -> pd.DataFrame:
    """加载测试窗口稳定性对比数据"""
    md_path = CODEX_BASE / "test_window_compare_results" / "compare_test_windows_summary.md"
    if not md_path.exists():
        return pd.DataFrame()

    text = md_path.read_text(encoding="utf-8")

    lines = text.split("\n")
    table_start = None
    for i, line in enumerate(lines):
        if line.startswith("| model"):
            table_start = i
            break

    if table_start is None:
        return pd.DataFrame()

    header_line = lines[table_start]
    headers = [h.strip() for h in header_line.split("|")[1:-1]]

    rows = []
    for line in lines[table_start + 2:]:
        if not line.startswith("|"):
            break
        vals = [v.strip() for v in line.split("|")[1:-1]]
        if len(vals) == len(headers):
            rows.append(vals)

    df = pd.DataFrame(rows, columns=headers)

    num_cols = ["topk", "n_drop", "valid_days", "test_days", "directional_accuracy",
                "ic", "rank_ic", "annualized_return", "information_ratio", "max_drawdown"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 添加窗口标签
    if "test_days" in df.columns:
        day_labels = {21: "1个月 (21天)", 42: "2个月 (42天)", 63: "3个月 (63天)"}
        df["窗口"] = df["test_days"].map(day_labels)

    return df


def get_latest_kpi(experiments_df: pd.DataFrame) -> dict:
    """从实验数据中提取最优 KPI"""
    best = experiments_df[experiments_df["方向准确率"].notna()].copy()
    if best.empty:
        return {}

    best_row = best.sort_values("方向准确率", ascending=False).iloc[0]
    return {
        "best_model": best_row.get("实验名称", ""),
        "accuracy": best_row.get("方向准确率"),
        "annual_return": best_row.get("年化收益"),
        "information_ratio": best_row.get("信息比率"),
        "max_drawdown": best_row.get("最大回撤"),
        "data_updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def get_data_timestamp() -> str:
    """获取数据最后更新时间"""
    pred_path = Path(__file__).parent / "data" / "Qlib_沪深300_全量预测&回测.csv"
    if pred_path.exists():
        return datetime.fromtimestamp(pred_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
