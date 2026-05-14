"""
AI 量化选股看板 · Qlib XGBoost · A股 CSI300
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
try:
    from sklearn.isotonic import IsotonicRegression
except Exception:
    IsotonicRegression = None

from data_loader import (
    load_all_experiments,
    load_topk_comparison,
    load_window_comparison,
    get_latest_kpi,
    get_data_timestamp,
)

# ── 页面配置 ──
st.set_page_config(
    page_title="AI量化智投",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"

# ── 全局 CSS（浅色主题） ──
st.markdown("""
<style>
    .stApp { background: #f0f2f5; }

    /* 卡片 */
    .card {
        background: #ffffff; border: 1px solid #e8e8ec; border-radius: 12px;
        padding: 1.2rem 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .card-header {
        font-size: 0.78rem; color: #8e8e93; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 0.35rem;
    }
    .card-value {
        font-size: 1.8rem; font-weight: 700; color: #1a1a2e; line-height: 1.2;
    }
    .card-sub {
        font-size: 0.8rem; margin-top: 0.25rem;
    }
    .card-sub.up { color: #e8453c; }
    .card-sub.down { color: #10b981; }
    .card-sub.neutral { color: #8e8e93; }

    /* section 标题 */
    .section-title {
        font-size: 1.05rem; font-weight: 700; color: #1a1a2e;
        margin: 0.6rem 0 0.8rem 0; padding-left: 0.3rem;
        border-left: 4px solid #3b82f6;
    }
    .section-subtitle {
        font-size: 0.8rem; color: #8e8e93; margin-bottom: 0.6rem;
    }

    /* 副标题 */
    .hero-title { font-size: 1.6rem; font-weight: 800; color: #1a1a2e; }
    .hero-sub { font-size: 0.82rem; color: #8e8e93; margin-top: 0.15rem; }

    /* 徽章 */
    .badge {
        display: inline-block; padding: 0.15rem 0.55rem; border-radius: 6px;
        font-size: 0.72rem; font-weight: 600;
    }
    .badge-blue { background: #dbeafe; color: #1e40af; }
    .badge-green { background: #d1fae5; color: #065f46; }
    .badge-red { background: #fef2f2; color: #991b1b; }

    /* 页脚 */
    .footer {
        text-align: center; color: #a1a1aa; font-size: 0.72rem;
        padding: 1.2rem 0 0.4rem 0; margin-top: 1rem;
        border-top: 1px solid #e8e8ec;
    }

    /* sidebar 品牌 */
    .brand { font-size: 1.3rem; font-weight: 800; color: #1a1a2e; }
    .brand-sub { font-size: 0.75rem; color: #8e8e93; margin-bottom: 1rem; }

    /* sidebar radio 间距修复 */
    div[data-testid="stRadio"] label {
        padding: 0.35rem 0; margin-left: 0.15rem;
    }

    /* 移动端适配 */
    @media (max-width: 768px) {
        .card-value { font-size: 1.3rem !important; }
        .card-header { font-size: 0.68rem !important; }
        .hero-title { font-size: 1.2rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ── Plotly 全局配置（禁止拖拽/缩放） ──
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
    "doubleClick": False,
    "showTips": False,
    "displaylogo": False,
}

# ── 数据加载（缓存） ──
import json

@st.cache_data(ttl=3600)
def load_stock_names():
    path = PROJECT_DIR / "stock_names.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=120)
def load_full_predictions():
    path = DATA_DIR / "Qlib_沪深300_全量预测&回测.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = enrich_prediction_columns(df)
        return df
    return None

@st.cache_data(ttl=300)
def load_exp_data():
    exps = load_all_experiments()
    return {
        "experiments": exps,
        "topk": load_topk_comparison(),
        "windows": load_window_comparison(),
        "kpi": get_latest_kpi(exps),
        "timestamp": get_data_timestamp(),
    }

# ── 辅助函数 ──
def make_card(col, label, value, sub_text="", sub_class="neutral"):
    with col:
        st.markdown(f"""
        <div class="card">
            <div class="card-header">{label}</div>
            <div class="card-value">{value}</div>
            <div class="card-sub {sub_class}">{sub_text}</div>
        </div>
        """, unsafe_allow_html=True)


def fallback_prob_from_rank(series: pd.Series) -> pd.Series:
    """无校准器时，用横截面分位数生成 0~1 概率参考。"""
    clean = pd.to_numeric(series, errors="coerce")
    rank = clean.rank(method="average", pct=True)
    return rank.clip(0.01, 0.99)


def enrich_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一预测分口径：保留原始分，并补出概率参考与信号强度。"""
    if df is None or df.empty:
        return df

    data = df.copy()
    if "原始预测分数" not in data.columns:
        data["原始预测分数"] = pd.to_numeric(data["预测分数"], errors="coerce")
    else:
        data["原始预测分数"] = pd.to_numeric(data["原始预测分数"], errors="coerce")

    if "上涨概率参考" not in data.columns:
        next_ret = data.groupby("symbol")["涨跌幅"].shift(-1)
        train_mask = data["原始预测分数"].notna() & next_ret.notna()
        prob_series = pd.Series(index=data.index, dtype=float)

        if train_mask.sum() >= 200 and IsotonicRegression is not None:
            label_up = (next_ret[train_mask] > 0).astype(int)
            calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            calibrator.fit(data.loc[train_mask, "原始预测分数"], label_up)
            prob_series.loc[data["原始预测分数"].notna()] = calibrator.predict(
                data.loc[data["原始预测分数"].notna(), "原始预测分数"]
            )
        else:
            prob_series = data.groupby("date")["原始预测分数"].transform(fallback_prob_from_rank)

        data["上涨概率参考"] = prob_series.clip(0.01, 0.99)
    else:
        data["上涨概率参考"] = pd.to_numeric(data["上涨概率参考"], errors="coerce").clip(0.01, 0.99)

    data["预测分数"] = (data["上涨概率参考"] - 0.5) * 2
    data["信号强度"] = data["预测分数"].abs()
    return data


def calc_compound_return(pct_series: pd.Series) -> float:
    """把日涨跌幅序列转换为复利累计收益。"""
    clean = pd.to_numeric(pct_series, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    return float((1 + clean / 100).prod() - 1)


def score_rank_stats(series: pd.Series, value: float) -> tuple[int, int, float]:
    """返回当前分数在横截面中的排名、总数和头部占比。"""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    total = len(clean)
    if total == 0:
        return 0, 0, 100.0
    better_count = int((clean > value).sum())
    rank = better_count + 1
    top_share = rank / total * 100
    return rank, total, top_share


def prob_to_label(prob: float, confidence: float) -> tuple[str, str]:
    """把概率参考与信号强弱映射成更直观的观点标签。"""
    if prob >= 0.62 and confidence >= 0.22:
        return "强烈关注", "up"
    if prob >= 0.55:
        return "偏看好", "up"
    if prob <= 0.38 and confidence >= 0.22:
        return "明显谨慎", "down"
    if prob <= 0.45:
        return "偏谨慎", "down"
    return "中性观察", "neutral"


def summarize_market_regime(latest_data: pd.DataFrame) -> tuple[str, str, str]:
    """基于全市场分数分布生成简短的市场状态解读。"""
    if latest_data is None or latest_data.empty:
        return "暂无数据", "neutral", "请先更新预测数据。"

    mean_score = float(latest_data["预测分数"].mean())
    pos_ratio = float((latest_data["预测分数"] > 0).mean() * 100)

    if pos_ratio >= 58 and mean_score > 0.02:
        return "市场偏强", "up", "多数股票信号为正，短期情绪偏多。"
    if pos_ratio <= 42 and mean_score < -0.02:
        return "市场偏谨慎", "down", "负分股票更多，说明模型整体更保守。"
    if mean_score > 0:
        return "温和偏多", "up", "整体不是普涨，但强于中性。"
    if mean_score < 0:
        return "多空分化", "neutral", "仍有结构性机会，但更适合精选个股。"
    return "中性震荡", "neutral", "模型没有给出明显的单边倾向。"


def describe_strategy(strategy: str) -> tuple[str, str]:
    meta = {
        "score_weighted": (
            "分数加权",
            "每天都持有全市场股票，但会把更多仓位分给模型更看好的标的，适合讲“全市场增强”。",
        ),
        "topk_equal": (
            "Top20 等权",
            "每天只买入模型最看好的 20 只股票，每只仓位相同，最贴近推荐页的产品表达。",
        ),
        "long_short": (
            "Top20/Bottom20 多空",
            "同时做多最强和做空最弱，强调模型区分能力，但不适合作为普通用户直观买入建议。",
        ),
    }
    return meta[strategy]


def compute_drawdown_details(curves: pd.DataFrame) -> dict:
    if curves is None or curves.empty:
        return {"max_dd": 0.0, "peak_date": None, "trough_date": None, "recovery_date": None}

    wealth = 1 + curves["策略累计"]
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1
    trough_idx = drawdown.idxmin()
    if pd.isna(trough_idx):
        return {"max_dd": 0.0, "peak_date": None, "trough_date": None, "recovery_date": None}

    peak_idx = wealth.loc[:trough_idx].idxmax()
    peak_date = curves.loc[peak_idx, "date"]
    trough_date = curves.loc[trough_idx, "date"]
    recovery_candidates = curves.loc[trough_idx:]
    recovery_mask = (1 + recovery_candidates["策略累计"]) >= wealth.loc[peak_idx]
    recovery_date = recovery_candidates.loc[recovery_mask, "date"].iloc[0] if recovery_mask.any() else None

    return {
        "max_dd": float(drawdown.loc[trough_idx]),
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
    }


full_pred = load_full_predictions()
try:
    exp_data = load_exp_data()
except Exception:
    exp_data = {"experiments": None, "topk": None, "windows": None, "kpi": {}, "timestamp": "N/A"}
df_exp = exp_data.get("experiments")
kpi = exp_data.get("kpi", {})
stock_names = load_stock_names()


@st.cache_data(ttl=3600)
def prepare_eval_frame(df: pd.DataFrame) -> pd.DataFrame:
    """对齐到预测日 -> 下一交易日收益"""
    daily = df.copy().sort_values(["symbol", "date"])
    daily["next_ret"] = daily.groupby("symbol")["涨跌幅"].shift(-1)
    daily["pred_up"] = daily["预测分数"] > 0
    daily["actual_up_next"] = daily["next_ret"] > 0
    daily["correct_next"] = daily["pred_up"] == daily["actual_up_next"]
    return daily


@st.cache_data(ttl=3600)
def build_strategy_curves(df: pd.DataFrame, strategy: str = "score_weighted", topk: int = 20) -> pd.DataFrame:
    """构建同口径的基准与AI策略累计收益曲线

    strategy:
      - 'score_weighted': 每日全部股票按预测分归一化加权
      - 'topk_equal':    每日取预测分最高的 topk 只，等权
      - 'long_short':    每日做多 topk 只 + 做空 bottom topk 只，各等权
    """
    valid = df.dropna(subset=["next_ret"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["date", "等权", "AI策略", "等权累计", "策略累计"])

    if strategy == "topk_equal":
        # 每日取 topk 只预测分最高的，等权
        def _topk_ret(grp):
            top = grp.nlargest(topk, "预测分数")
            return pd.Series({
                "等权": grp["next_ret"].mean() / 100,
                "AI策略": top["next_ret"].mean() / 100,
            })
        curves = valid.groupby("date").apply(_topk_ret, include_groups=False).reset_index()

    elif strategy == "long_short":
        def _ls_ret(grp):
            top = grp.nlargest(topk, "预测分数")
            bot = grp.nsmallest(topk, "预测分数")
            return pd.Series({
                "等权": grp["next_ret"].mean() / 100,
                "AI策略": (top["next_ret"].mean() - bot["next_ret"].mean()) / 100,
            })
        curves = valid.groupby("date").apply(_ls_ret, include_groups=False).reset_index()

    else:  # score_weighted
        valid["norm_weight"] = valid.groupby("date")["预测分数"].transform(
            lambda s: ((s - s.min()) / (s - s.min()).sum()) if (s - s.min()).sum() else (1.0 / len(s))
        )
        curves = valid.groupby("date").apply(
            lambda x: pd.Series({
                "等权": x["next_ret"].mean() / 100,
                "AI策略": (x["norm_weight"] * x["next_ret"] / 100).sum(),
            }),
            include_groups=False,
        ).reset_index()

    curves["等权累计"] = (1 + curves["等权"]).cumprod() - 1
    curves["策略累计"] = (1 + curves["AI策略"]).cumprod() - 1
    return curves

# ═══════════════════════════════════
# 侧边栏导航（方块按钮）
# ═══════════════════════════════════
with st.sidebar:
    st.markdown('<p class="brand">📈 AI量化智投</p>', unsafe_allow_html=True)
    st.markdown('<p class="brand-sub">CSI300 · Walk-Forward · XGBoost</p>', unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "导航",
        ["🏠 市场概览", "⭐ AI推荐", "📊 回测追踪", "🔍 个股追踪"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<span class="badge badge-blue">XGBoost · Alpha158 · Walk-Forward</span>',
                unsafe_allow_html=True)
    st.caption(f"数据刷新: {exp_data['timestamp']}")

    with st.expander("📋 关于"):
        st.caption("""
        CSI300 量化研究平台。Alpha158 因子 + XGBoost + Walk-Forward 滚动回测。
        仅供研究参考，不构成投资建议。
        """)


# ── 默认策略（市场概览用分数加权）──
_strategy = "score_weighted"

# ═══════════════════════════════════════════════════
# 🏠 市场概览
# ═══════════════════════════════════════════════════
if page == "🏠 市场概览":
    st.markdown('<p class="hero-title">市场概览</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">预测 T+1 涨跌方向 · Walk-Forward 6窗口 · 交易成本已计入</p>',
                unsafe_allow_html=True)

    # 从预测CSV计算真实回测指标
    if full_pred is not None:
        daily = prepare_eval_frame(full_pred)
        daily_acc = daily.groupby("date").agg(
            acc=("correct_next", "mean"), cnt=("correct_next", "count")
        ).reset_index()
        daily_acc = daily_acc[daily_acc["cnt"] >= 30]

        avg_acc = daily_acc["acc"].mean()
        win_days = (daily_acc["acc"] > 0.5).sum()
        total_days = len(daily_acc)

        # 逐日IC
        ics = []
        for d in sorted(daily["date"].unique()):
            day = daily[daily["date"] == d]
            day = day.dropna(subset=["next_ret"])
            if len(day) >= 30:
                ic = day["预测分数"].corr(day["next_ret"])
                ics.append(ic)
        avg_ic = float(np.mean(ics)) if ics else None
        pos_ic = sum(i > 0 for i in ics) if ics else 0

        # 累计收益
        cumret = build_strategy_curves(daily, _strategy)
        final_ret = float(cumret["策略累计"].iloc[-1]) if len(cumret) > 0 else 0

        st.markdown('<p class="section-title">Walk-Forward 核心指标</p>', unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)
        make_card(k1, "方向准确率", f"{avg_acc*100:.1f}%",
                  f"{win_days}/{total_days}天超50%",
                  "up" if avg_acc > 0.5 else "neutral")
        make_card(k2, "IC", f"{avg_ic:.4f}" if avg_ic else "N/A",
                  f"{pos_ic}/{len(ics)}天为正" if ics else "",
                  "up" if avg_ic and avg_ic > 0 else "neutral")
        make_card(k3, "预测覆盖", f"{daily_acc['cnt'].sum():,}次",
                  f"{daily['股票代码'].nunique()}只×{len(daily_acc)}天")
        make_card(k4, "累计收益", f"{final_ret*100:.1f}%",
                  "T+1策略收益",
                  "up" if final_ret > 0 else "down")
        make_card(k5, "数据范围", f"{daily['date'].min().date()} → {daily['date'].max().date()}",
                  f"{len(daily_acc)}个交易日")
    else:
        st.warning("未找到预测数据")

    # 预测分布
    if full_pred is not None:
        st.markdown('<p class="section-title">预测分数分布</p>', unsafe_allow_html=True)
        latest_date = full_pred["date"].max()
        latest_data = full_pred[full_pred["date"] == latest_date]

        c1, c2 = st.columns([2, 1])
        with c1:
            fig_dist = px.histogram(
                latest_data, x="原始预测分数", nbins=40,
                color_discrete_sequence=["#3b82f6"],
            )
            fig_dist.update_layout(
                title=dict(text=f"全市场原始预测分布 ({latest_date.strftime('%Y-%m-%d')})",
                           font=dict(color="#1a1a2e", size=14)),
                xaxis=dict(title="原始预测分数 (模型原始排序分)", color="#52525b"),
                yaxis=dict(title="股票数量", color="#52525b"),
                bargap=0.05, showlegend=False, height=360,
                dragmode=False,
                paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
                margin=dict(l=0, r=0, t=44, b=0),
            )
            st.plotly_chart(fig_dist, width="stretch", config=PLOTLY_CONFIG)

        with c2:
            s1, s2, s3, s4 = st.columns(1), st.columns(1), st.columns(1), st.columns(1)
            make_card(s1[0], "原始分均值", f"{latest_data['原始预测分数'].mean():.4f}",
                      "模型原始排序输出的均值")
            make_card(s2[0], "原始分中位数", f"{latest_data['原始预测分数'].median():.4f}")
            make_card(s3[0], "原始分标准差", f"{latest_data['原始预测分数'].std():.4f}",
                      "越大表示个股分化越明显")
            regime_label, regime_class, regime_note = summarize_market_regime(latest_data)
            make_card(s4[0], "信号正分占比",
                      f"{(latest_data['预测分数'] > 0).mean()*100:.1f}%",
                      regime_label, regime_class)
            st.caption(regime_note)

    # 当前模型说明
    st.markdown('<p class="section-title">当前模型</p>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    latest_pool = full_pred[full_pred["date"] == full_pred["date"].max()]["股票代码"].nunique() if full_pred is not None else 300
    history_pool = full_pred["股票代码"].nunique() if full_pred is not None else 300
    make_card(m1, "模型", "XGBoost", "500轮, early stopping")
    make_card(m2, "因子", "Alpha158", "Qlib 原生因子集")
    make_card(m3, "训练方式", "Walk-Forward", "5年×6窗口滚动")
    make_card(m4, "股票池", "CSI300", f"当日{latest_pool}只 · 历史覆盖{history_pool}只")

# ═══════════════════════════════════════════════════
# ⭐ AI推荐
# ═══════════════════════════════════════════════════
elif page == "⭐ AI推荐":
    st.markdown('<p class="hero-title">AI 相对看好股票</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">从全市场中选出当日预测分最高的 Top 20，表示相对更看好，不等于市场整体看多</p>',
                unsafe_allow_html=True)

    if full_pred is not None:
        latest_date = full_pred["date"].max()
        latest_data = full_pred[full_pred["date"] == latest_date].copy()
        top_day = latest_data.nlargest(20, "预测分数").copy()
        display_df = top_day[["股票代码", "symbol", "close", "涨跌幅", "换手率", "预测分数", "上涨概率参考", "信号强度"]].copy()
        # 取纯代码（去 sh/sz 前缀）
        display_df["代码"] = display_df["symbol"].str.replace("sh", "").str.replace("sz", "")
        # 公司名称（symbol 转小写匹配）
        display_df["名称"] = display_df["symbol"].str.lower().map(stock_names).fillna(display_df["代码"])
        display_df["预测分"] = display_df["预测分数"].round(4)
        display_df["AI观点"] = display_df.apply(
            lambda row: prob_to_label(float(row["上涨概率参考"]), float(row["信号强度"]))[0],
            axis=1,
        )
        display_df["收盘价"] = display_df["close"]
        display_df["涨跌%"] = display_df["涨跌幅"]
        display_df["换手%"] = display_df["换手率"]
        display_df["概率参考"] = (display_df["上涨概率参考"] * 100).round(1).astype(str) + "%"
        display_df = display_df[["代码", "名称", "收盘价", "涨跌%", "换手%", "预测分", "概率参考", "AI观点"]]
        display_df = display_df.reset_index(drop=True)
        display_df.insert(0, "#", range(1, len(display_df) + 1))

        regime_label, regime_class, regime_note = summarize_market_regime(latest_data)
        pct_up = (latest_data["预测分数"] > 0).mean() * 100
        top1_score = float(top_day["预测分数"].iloc[0])
        top20_threshold = float(top_day["预测分数"].min())
        top1_code = str(top_day["symbol"].iloc[0]).replace("sh", "").replace("sz", "")
        top1_name = stock_names.get(str(top_day["symbol"].iloc[0]).lower(), top1_code)

        k1, k2, k3, k4 = st.columns(4)
        make_card(k1, "市场状态", regime_label, regime_note, regime_class)
        make_card(k2, "全市场正分占比", f"{pct_up:.1f}%",
                  "衡量模型对市场整体偏多还是偏谨慎")
        make_card(k3, "Top1 标的", top1_name,
                  f"{top1_code} · 分数 {top1_score:.4f}", "up")
        make_card(k4, "Top20 门槛分", f"{top20_threshold:.4f}",
                  "进入今日推荐列表的最低分")

        tab_list, tab_dist, tab_how = st.tabs(["📋 今日 Top20", "📊 全市场分布", "🧭 如何解读"])

        with tab_list:
            st.markdown('<p class="section-title">Top 20 推荐</p>', unsafe_allow_html=True)
            left, right = st.columns([1.35, 0.95])

            with left:
                st.dataframe(display_df, width="stretch", hide_index=True, height=720)
                st.caption(f"数据日期: {latest_date.strftime('%Y-%m-%d')} · 按模型预测分降序，分数越高表示相对更值得优先关注")

            with right:
                st.markdown('<p class="section-title" style="border-color: transparent;">推荐解读</p>',
                            unsafe_allow_html=True)
                st.info("Top 20 是从全市场里挑出当日分数最高的股票；即使市场不是全面看多，这里仍会展示相对更强的标的。")
                s1, s2 = st.columns(2)
                make_card(s1, "均值", f"{latest_data['预测分数'].mean():.4f}")
                make_card(s2, "中位数", f"{latest_data['预测分数'].median():.4f}")
                s3, s4 = st.columns(2)
                make_card(s3, "标准差", f"{latest_data['预测分数'].std():.4f}",
                          "越大表示分化越明显")
                make_card(s4, "市场状态", regime_label, regime_note, regime_class)

        with tab_dist:
            fig_dist = px.histogram(
                latest_data, x="原始预测分数", nbins=40,
                color_discrete_sequence=["#3b82f6"],
            )
            fig_dist.update_layout(
                title=dict(text="原始预测分数分布", font=dict(color="#1a1a2e", size=13)),
                xaxis=dict(title="原始预测分数", color="#52525b"),
                yaxis=dict(title="股票数", color="#52525b"),
                bargap=0.05, showlegend=False, height=320,
                dragmode=False,
                paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_dist, width="stretch", config=PLOTLY_CONFIG)
            st.caption("如果分布整体右移，说明模型对大部分股票更乐观；如果只有右侧长尾，通常代表更适合做精选。")

        with tab_how:
            st.markdown("""
            **怎么看这个页面**

            1. `Top 20 推荐` 看的是横向相对强弱，适合先筛出值得研究的候选股票。
            2. `全市场正分占比` 看的是整体市场温度，用来判断现在更像普涨环境还是结构性行情。
            3. `Top20 门槛分` 越高，说明当天真正能挤进推荐名单的股票更少、竞争更强。
            4. 单只股票别只看分数绝对值，更要结合它在当天 CSI300 里的相对排名一起看。
            """)
    else:
        st.warning("请先运行预测脚本生成数据文件")

# ═══════════════════════════════════════════════════
# 📊 回测追踪
# ═══════════════════════════════════════════════════
elif page == "📊 回测追踪":
    st.markdown('<p class="hero-title">回测追踪</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">预测 T+1 涨跌方向 · 累计收益曲线 · 交易成本已计入</p>',
                unsafe_allow_html=True)

    strategy = st.selectbox(
        "回测策略",
        ["score_weighted", "topk_equal", "long_short"],
        format_func=lambda s: {
            "score_weighted": "分数加权（全量股票按预测分分配仓位）",
            "topk_equal": "Top20 等权（每日取预测最高的20只，等权重买入）",
            "long_short": "Top20/Bottom20 多空（做多Top20 + 做空Bottom20）",
        }[s],
    )
    strategy_name, strategy_desc = describe_strategy(strategy)

    if full_pred is not None:
        daily = prepare_eval_frame(full_pred)
        daily_acc = daily.groupby("date").agg(
            准确率=("correct_next", "mean"), 股票数=("correct_next", "count")
        ).reset_index()
        daily_acc = daily_acc[daily_acc["股票数"] >= 30]
        cumret = build_strategy_curves(daily, strategy)
        drawdown_meta = compute_drawdown_details(cumret)

        st.markdown(f"""
        <div style="background:#ffffff; border:1px solid #e8e8ec; border-radius:12px; padding:1rem 1.2rem; margin:0.6rem 0 1rem 0;">
            <div style="font-size:0.9rem; font-weight:700; color:#1a1a2e; margin-bottom:0.35rem;">当前策略：{strategy_name}</div>
            <div style="font-size:0.82rem; color:#52525b; line-height:1.7;">{strategy_desc}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-title">T+1 方向准确率 & 累计收益</p>', unsafe_allow_html=True)
        left, right = st.columns([1, 1])

        with left:
            # 按月聚合折线图（1534天柱状图会挤成一条线）
            daily_acc["月"] = daily_acc["date"].dt.to_period("M").dt.to_timestamp()
            monthly = daily_acc.groupby("月").agg(
                准确率=("准确率", "mean"), 天数=("准确率", "count")
            ).reset_index()

            fig_day = go.Figure()
            fig_day.add_trace(go.Scatter(
                x=monthly["月"], y=monthly["准确率"],
                mode="lines+markers",
                line=dict(width=2, color="#3b82f6"),
                marker=dict(size=5, color="#3b82f6"),
                name="月均准确率",
                fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
            ))
            fig_day.add_hline(
                y=0.5, line_dash="dash", line_color="#d4d4d8",
                annotation_text="随机 50%", annotation_font=dict(color="#a1a1aa"),
            )
            fig_day.update_layout(
                title=dict(text="方向预测准确率（月均）", font=dict(color="#1a1a2e", size=14)),
                yaxis=dict(tickformat=".0%", color="#52525b"),
                xaxis=dict(color="#52525b"),
                height=380, showlegend=False, dragmode=False,
                paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
            )
            st.plotly_chart(fig_day, width="stretch", config=PLOTLY_CONFIG)

        with right:
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=cumret["date"], y=cumret["等权累计"],
                mode="lines", name="等权持有",
                line=dict(width=2, color="#d4d4d8"),
            ))
            fig_cum.add_trace(go.Scatter(
                x=cumret["date"], y=cumret["策略累计"],
                mode="lines", name="AI策略",
                line=dict(width=3, color="#3b82f6"),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.06)",
            ))
            fig_cum.update_layout(
                title=dict(text="累计收益 · 等权 vs AI策略",
                           font=dict(color="#1a1a2e", size=14)),
                yaxis=dict(tickformat=".1%", color="#52525b"),
                xaxis=dict(color="#52525b"),
                height=380, dragmode=False,
                legend=dict(x=0.01, y=0.99, font=dict(color="#52525b")),
                paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
            )
            if drawdown_meta["peak_date"] is not None and drawdown_meta["trough_date"] is not None:
                fig_cum.add_vrect(
                    x0=drawdown_meta["peak_date"], x1=drawdown_meta["trough_date"],
                    fillcolor="rgba(239,68,68,0.10)", line_width=0
                )
            st.plotly_chart(fig_cum, width="stretch", config=PLOTLY_CONFIG)

        # 汇总卡片
        st.markdown('<p class="section-title">回测汇总</p>', unsafe_allow_html=True)
        total_days = len(daily_acc)
        avg_acc = daily_acc["准确率"].mean()
        win_days = (daily_acc["准确率"] > 0.5).sum()
        final_eq = cumret["等权累计"].iloc[-1] if len(cumret) > 0 else 0
        final_w = cumret["策略累计"].iloc[-1] if len(cumret) > 0 else 0
        alpha = final_w - final_eq

        # 年化 / 夏普 / 最大回撤
        ann_eq = (1 + final_eq) ** (252 / total_days) - 1 if total_days > 0 else 0
        ann_w = (1 + final_w) ** (252 / total_days) - 1 if total_days > 0 else 0
        ai_daily = cumret["AI策略"].dropna() if len(cumret) > 0 else pd.Series(dtype=float)
        sharpe = float(ai_daily.mean() / ai_daily.std() * np.sqrt(252)) if len(ai_daily) > 1 and ai_daily.std() > 0 else 0
        max_dd = drawdown_meta["max_dd"]

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        make_card(m1, "回测天数", f"{total_days}d",
                  f"{daily['date'].min().date()} → {daily['date'].max().date()}")
        make_card(m2, "方向准确率", f"{avg_acc:.1%}",
                  f"{win_days}/{total_days}天>50%", "up" if avg_acc > 0.5 else "neutral")
        make_card(m3, "AI年化收益", f"{ann_w:.1%}",
                  {"score_weighted": "分数加权", "topk_equal": "Top20等权", "long_short": "多空组合"}[strategy],
                  "up" if ann_w > 0 else "down")
        make_card(m4, "等权年化", f"{ann_eq:.1%}",
                  f"超额 α={alpha:.1%}", "up" if alpha > 0 else "down")
        make_card(m5, "夏普比率", f"{sharpe:.2f}",
                  "优秀" if sharpe > 1.5 else ("良好" if sharpe > 0.8 else "一般"),
                  "up" if sharpe > 0 else "down")
        make_card(m6, "最大回撤", f"{max_dd:.1%}",
                  "策略峰值→谷底", "down")

        alpha_curve = cumret[["date"]].copy()
        alpha_curve["超额累计"] = cumret["策略累计"] - cumret["等权累计"]
        alpha_curve["年份"] = alpha_curve["date"].dt.year
        year_summary = cumret.copy()
        year_summary["年份"] = year_summary["date"].dt.year
        yearly = year_summary.groupby("年份").agg(
            AI策略=("AI策略", lambda s: (1 + s).prod() - 1),
            等权=("等权", lambda s: (1 + s).prod() - 1),
        ).reset_index()
        yearly["超额"] = yearly["AI策略"] - yearly["等权"]
        yearly["AI策略"] = yearly["AI策略"].map(lambda x: f"{x:.1%}")
        yearly["等权"] = yearly["等权"].map(lambda x: f"{x:.1%}")
        yearly["超额"] = yearly["超额"].map(lambda x: f"{x:.1%}")

        dd_text = "未恢复" if drawdown_meta["recovery_date"] is None else pd.Timestamp(drawdown_meta["recovery_date"]).strftime("%Y-%m-%d")
        insight_left, insight_right = st.columns([1.15, 0.85])
        with insight_left:
            fig_alpha = go.Figure()
            fig_alpha.add_trace(go.Scatter(
                x=alpha_curve["date"], y=alpha_curve["超额累计"],
                mode="lines", name="累计超额",
                line=dict(width=2.5, color="#0f766e"),
                fill="tozeroy", fillcolor="rgba(15,118,110,0.08)",
            ))
            fig_alpha.add_hline(y=0, line_dash="dash", line_color="#d4d4d8")
            fig_alpha.update_layout(
                title=dict(text="累计超额收益拆解", font=dict(color="#1a1a2e", size=14)),
                yaxis=dict(tickformat=".1%", color="#52525b"),
                xaxis=dict(color="#52525b"),
                height=340, showlegend=False, dragmode=False,
                paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
            )
            st.plotly_chart(fig_alpha, width="stretch", config=PLOTLY_CONFIG)

        with insight_right:
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e8e8ec; border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.8rem;">
                <div style="font-size:0.92rem; font-weight:700; color:#1a1a2e; margin-bottom:0.35rem;">最大回撤时间段</div>
                <div style="font-size:0.82rem; color:#52525b; line-height:1.8;">
                    峰值日：<b>{pd.Timestamp(drawdown_meta['peak_date']).strftime('%Y-%m-%d') if drawdown_meta['peak_date'] is not None else 'N/A'}</b><br>
                    谷底日：<b>{pd.Timestamp(drawdown_meta['trough_date']).strftime('%Y-%m-%d') if drawdown_meta['trough_date'] is not None else 'N/A'}</b><br>
                    恢复日：<b>{dd_text}</b><br>
                    最大回撤：<b>{max_dd:.1%}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(yearly, width="stretch", hide_index=True)
    else:
        st.warning("请先运行预测脚本生成数据文件")

# ═══════════════════════════════════════════════════
# ═══════════════════════════════════════════════════
# 🔍 个股追踪
# ═══════════════════════════════════════════════════
elif page == "🔍 个股追踪":
    st.markdown('<p class="hero-title">个股追踪</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">看真实行情，也看 AI 对下一交易日方向的判断强弱 · 沪深300成分股</p>',
                unsafe_allow_html=True)

    if full_pred is not None:
        all_stocks = sorted(full_pred["symbol"].unique())
        # 构建搜索选项：纯代码 + 名称
        stock_options = {}
        for s in all_stocks:
            code = s.replace("sh", "").replace("sz", "")
            name = stock_names.get(s.lower(), "")
            label = f"{code}  {name}" if name else code
            stock_options[label] = s

        # 搜索框 — 用卡片样式突出
        st.markdown("""
        <div style="background:#ffffff; border:2px solid #3b82f6; border-radius:12px;
                    padding:1.2rem 1.5rem; margin:0.8rem 0 1.2rem 0;">
            <div style="font-size:0.9rem; font-weight:700; color:#1a1a2e; margin-bottom:0.6rem;">
                🔍 选择股票开始分析
            </div>
        """, unsafe_allow_html=True)
        # 若从快捷卡片点进来，自动选中
        if "stock_pick" not in st.session_state:
            st.session_state.stock_pick = None
        pick_label = st.session_state.stock_pick
        st.session_state.stock_pick = None  # 用完清零

        selected_label = st.selectbox(
            "股票代码或名称",
            options=sorted(stock_options.keys()),
            index=sorted(stock_options.keys()).index(pick_label) if pick_label and pick_label in stock_options else None,
            placeholder="输入 600000 或 浦发银行 搜索...",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if not selected_label:
            st.markdown('<p class="section-title">今日 AI 最看好</p>', unsafe_allow_html=True)
            st.caption("点击任意股票直接查看K线分析")
            latest_date = full_pred["date"].max()
            top10 = full_pred[full_pred["date"] == latest_date].nlargest(10, "预测分数")

            # 10只股票排成两行5列，可点击
            for row_idx in range(0, 10, 5):
                cols = st.columns(5)
                for col_idx in range(5):
                    i = row_idx + col_idx
                    if i >= len(top10):
                        break
                    row = top10.iloc[i]
                    code = str(row["symbol"]).replace("sh", "").replace("sz", "")
                    name = stock_names.get(str(row["symbol"]).lower(), code)
                    score = row["预测分数"]
                    pct = row["涨跌幅"]
                    color = "#e8453c" if pct > 0 else "#10b981"
                    with cols[col_idx]:
                        st.markdown(f"""
                        <div style="background:#ffffff; border:1px solid #e8e8ec; border-radius:10px;
                                    padding:0.6rem 0.5rem; text-align:center; cursor:pointer;
                                    transition:box-shadow 0.15s;"
                             onmouseover="this.style.boxShadow='0 2px 8px rgba(0,0,0,0.1)'"
                             onmouseout="this.style.boxShadow='none'">
                            <div style="font-size:0.82rem; font-weight:700; color:#1a1a2e;">{code}</div>
                            <div style="font-size:0.68rem; color:#8e8e93; margin:0.15rem 0;">{name}</div>
                            <div style="font-size:0.78rem; font-weight:600; color:{color};">{pct:+.2f}%</div>
                            <div style="font-size:0.65rem; color:#8e8e93;">预测 {score:.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        label = f"{code}  {name}"
                        if label in stock_options:
                            if st.button("查看", key=f"pick_{row['symbol']}", width="stretch"):
                                st.session_state.stock_pick = label
                                st.rerun()

        if selected_label:
            selected_symbol = stock_options[selected_label]

            sdf = full_pred[full_pred["symbol"] == selected_symbol].sort_values("date")
            stock_code = selected_symbol.replace("sh", "").replace("sz", "")
            stock_name = stock_names.get(selected_symbol.lower(), "")

            sd = sdf.copy()
            sd["next_ret"] = sd["涨跌幅"].shift(-1)
            sd["pred_dir"] = sd["预测分数"] > 0
            sd["actual_dir"] = sd["next_ret"] > 0
            match = (sd.dropna(subset=["next_ret"])["pred_dir"] == sd.dropna(subset=["next_ret"])["actual_dir"]).mean() * 100
            latest_score = float(sdf["预测分数"].iloc[-1])
            latest_prob = float(sdf["上涨概率参考"].iloc[-1])
            latest_conf = float(sdf["信号强度"].iloc[-1])
            latest_market = full_pred[full_pred["date"] == sdf["date"].iloc[-1]]
            latest_rank, latest_total, latest_top_share = score_rank_stats(latest_market["预测分数"], latest_score)
            latest_label, latest_class = prob_to_label(latest_prob, latest_conf)
            latest_rank_text = f"第 {latest_rank}/{latest_total} 名"
            period_return = calc_compound_return(sdf["涨跌幅"])

            # 指标卡片
            st.markdown(f'<p class="section-title">{stock_code}  {stock_name}</p>',
                        unsafe_allow_html=True)
            s1, s2, s3, s4, s5 = st.columns(5)
            make_card(s1, "最新收盘", f"{sdf['close'].iloc[-1]:.2f}",
                      f"{sdf['涨跌幅'].iloc[-1]:.2f}%",
                      "up" if sdf["涨跌幅"].iloc[-1] > 0 else "down")
            make_card(s2, "AI观点", latest_label,
                      f"上涨概率参考 {latest_prob * 100:.1f}% · 信号强度 {latest_conf:.2f}",
                      latest_class)
            make_card(s3, "同日排名", latest_rank_text,
                      "排名越靠前，说明当天相对更受模型关注")
            make_card(s4, "方向一致率", f"{match:.1f}%",
                      "历史判断较稳定" if match > 55 else "仅作辅助参考",
                      "up" if match > 50 else "down")
            make_card(s5, "区间累计收益", f"{period_return * 100:.2f}%",
                      "按复利计算，而不是简单相加")

            # 时间范围切换
            range_option = st.radio(
                "走势时间范围",
                ["近一周", "近1月", "近3月"],
                horizontal=True,
                index=2,
                key=f"range_{selected_symbol}",
                label_visibility="collapsed",
            )
            max_date = sdf["date"].max()
            if range_option == "近一周":
                cutoff = max_date - pd.DateOffset(days=7)
            elif range_option == "近1月":
                cutoff = max_date - pd.DateOffset(months=1)
            else:
                cutoff = max_date - pd.DateOffset(months=3)
            chart_df = sdf[sdf["date"] >= cutoff]
            recent_mean_score = chart_df["预测分数"].mean()
            chart_period_return = calc_compound_return(chart_df["涨跌幅"])
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e8e8ec; border-radius:12px; padding:1rem 1.2rem; margin:0.4rem 0 1rem 0;">
                <div style="font-size:0.92rem; font-weight:700; color:#1a1a2e; margin-bottom:0.4rem;">AI 结论解读</div>
                <div style="font-size:0.82rem; color:#52525b; line-height:1.8;">
                    当前模型对 <b>{stock_code} {stock_name}</b> 的最新判断为 <b>{latest_label}</b>。<br>
                    当前上涨概率参考为 <b>{latest_prob * 100:.1f}%</b>，信号强度为 <b>{latest_conf:.2f}</b>，在最近一个交易日的 CSI300 成分股中排在 <b>{latest_rank_text}</b>。<br>
                    当前区间平均预测分为 <b>{recent_mean_score:.4f}</b>，区间累计收益为 <b>{chart_period_return * 100:.2f}%</b>，方向一致率为 <b>{match:.1f}%</b>。<br>
                    这里的“上涨概率参考”来自历史分数校准，适合帮助理解强弱，但仍应与趋势、成交和基本面一起看，不建议单独当成交易指令。
                </div>
            </div>
            """, unsafe_allow_html=True)

            # K线 + 预测分 → 彻底分开，各占一栏
            left_chart, right_chart = st.columns(2)

            with left_chart:
                fig_price = go.Figure()
                fig_price.add_trace(go.Candlestick(
                    x=chart_df["date"], open=chart_df["open"], high=chart_df["high"],
                    low=chart_df["low"], close=chart_df["close"],
                    increasing_line_color="#e8453c",
                    decreasing_line_color="#10b981",
                    name="K线",
                ))
                fig_price.update_layout(
                    title=dict(text="股价走势", font=dict(color="#1a1a2e", size=14)),
                    yaxis=dict(title="价格", color="#52525b"),
                    height=450, dragmode=False,
                    xaxis_rangeslider_visible=False,
                    paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
                    margin=dict(l=0, r=0, t=44, b=0),
                )
                st.plotly_chart(fig_price, width="stretch", config=PLOTLY_CONFIG)

            with right_chart:
                fig_score = go.Figure()
                fig_score.add_trace(go.Scatter(
                    x=chart_df["date"], y=chart_df["预测分数"],
                    mode="lines", name="AI预测分",
                    line=dict(color="#3b82f6", width=2),
                    fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
                ))
                fig_score.add_hline(y=0, line_dash="solid", line_color="#d4d4d8")
                fig_score.update_layout(
                    title=dict(text="AI 信号强弱", font=dict(color="#1a1a2e", size=14)),
                    yaxis=dict(title="预测分（越高越看好）", color="#52525b"),
                    height=450, dragmode=False, showlegend=False,
                    paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
                    margin=dict(l=0, r=0, t=44, b=0),
                )
                st.plotly_chart(fig_score, width="stretch", config=PLOTLY_CONFIG)

            st.divider()

            # 涨跌幅走势（全宽）
            fig_pct = go.Figure()
            fig_pct.add_trace(go.Scatter(
                x=chart_df["date"], y=chart_df["涨跌幅"],
                mode="lines", name="涨跌幅",
                line=dict(color="#3b82f6", width=1.5),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
            ))
            fig_pct.update_layout(
                title=dict(text="每日涨跌幅 (%)", font=dict(color="#1a1a2e", size=14)),
                height=380, showlegend=False, dragmode=False,
                paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
                xaxis=dict(color="#52525b"), yaxis=dict(color="#52525b"),
                margin=dict(l=0, r=0, t=44, b=0),
            )
            st.plotly_chart(fig_pct, width="stretch", config=PLOTLY_CONFIG)

            with st.expander("📄 历史数据（最近60天）"):
                show = sdf[["date", "open", "close", "涨跌幅", "换手率", "预测分数"]] \
                    .sort_values("date", ascending=False).head(60)
                show.columns = ["日期", "开盘", "收盘", "涨跌%", "换手%", "预测分"]
                st.dataframe(show, width="stretch", hide_index=True)
    else:
        st.warning("请先运行预测脚本生成数据文件")

# ═══════════════════════════════════════════════════
# 页脚
# ═══════════════════════════════════════════════════
st.markdown('<p class="footer">AI 量化选股看板 · Qlib XGBoost · 仅供研究参考，不构成投资建议</p>',
            unsafe_allow_html=True)
