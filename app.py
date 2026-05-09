"""
AI 量化选股看板 · Qlib XGBoost · A股 CSI300
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

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
</style>
""", unsafe_allow_html=True)

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

full_pred = load_full_predictions()
exp_data = load_exp_data()
df_exp = exp_data["experiments"]
kpi = exp_data["kpi"]
stock_names = load_stock_names()

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


def prepare_eval_frame(df: pd.DataFrame) -> pd.DataFrame:
    """对齐到预测日 -> 下一交易日收益"""
    daily = df.copy().sort_values(["symbol", "date"])
    daily["next_ret"] = daily.groupby("symbol")["涨跌幅"].shift(-1)
    daily["pred_up"] = daily["预测分数"] > 0
    daily["actual_up_next"] = daily["next_ret"] > 0
    daily["correct_next"] = daily["pred_up"] == daily["actual_up_next"]
    return daily


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
                latest_data, x="预测分数", nbins=40,
                color_discrete_sequence=["#3b82f6"],
            )
            fig_dist.update_layout(
                title=dict(text=f"全市场预测分布 ({latest_date.strftime('%Y-%m-%d')})",
                           font=dict(color="#1a1a2e", size=14)),
                xaxis=dict(title="预测分数 (越高越看好)", color="#52525b"),
                yaxis=dict(title="股票数量", color="#52525b"),
                bargap=0.05, showlegend=False, height=360,
                paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
                margin=dict(l=0, r=0, t=44, b=0),
            )
            st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

        with c2:
            s1, s2, s3, s4 = st.columns(1), st.columns(1), st.columns(1), st.columns(1)
            make_card(s1[0], "均值", f"{latest_data['预测分数'].mean():.4f}")
            make_card(s2[0], "中位数", f"{latest_data['预测分数'].median():.4f}")
            make_card(s3[0], "标准差", f"{latest_data['预测分数'].std():.4f}")
            make_card(s4[0], "看涨占比",
                      f"{(latest_data['预测分数'] > 0).mean()*100:.1f}%")

    # 当前模型说明
    st.markdown('<p class="section-title">当前模型</p>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    make_card(m1, "模型", "XGBoost", "500轮, early stopping")
    make_card(m2, "因子", "Alpha158", "Qlib 原生因子集")
    make_card(m3, "训练方式", "Walk-Forward", "5年×6窗口滚动")
    make_card(m4, "股票池", "CSI300", f"{full_pred['股票代码'].nunique() if full_pred is not None else 300}只")

# ═══════════════════════════════════════════════════
# ⭐ AI推荐
# ═══════════════════════════════════════════════════
elif page == "⭐ AI推荐":
    st.markdown('<p class="hero-title">AI 推荐股票</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">基于模型预测分数排序的 Top 推荐 · 每日更新</p>',
                unsafe_allow_html=True)

    if full_pred is not None:
        latest_date = full_pred["date"].max()
        top_day = full_pred[full_pred["date"] == latest_date].nlargest(20, "预测分数")
        display_df = top_day[["股票代码", "symbol", "close", "涨跌幅", "换手率", "预测分数"]].copy()
        # 取纯代码（去 sh/sz 前缀）
        display_df["代码"] = display_df["symbol"].str.replace("sh", "").str.replace("sz", "")
        # 公司名称（symbol 转小写匹配）
        display_df["名称"] = display_df["symbol"].str.lower().map(stock_names).fillna(display_df["代码"])
        display_df["预测分"] = display_df["预测分数"].round(4)
        display_df["收盘价"] = display_df["close"]
        display_df["涨跌%"] = display_df["涨跌幅"]
        display_df["换手%"] = display_df["换手率"]
        display_df = display_df[["代码", "名称", "收盘价", "涨跌%", "换手%", "预测分"]]
        display_df = display_df.reset_index(drop=True)
        display_df.insert(0, "#", range(1, len(display_df) + 1))

        st.markdown('<p class="section-title">Top 20 推荐</p>', unsafe_allow_html=True)
        left, right = st.columns([1.3, 1])

        with left:
            # 构建颜色编码的表格
            fig_tbl = go.Figure(data=[go.Table(
                header=dict(
                    values=list(display_df.columns),
                    fill_color="#f0f2f5",
                    font=dict(color="#1a1a2e", size=13, family="Inter, sans-serif"),
                    align="center", height=36,
                    line=dict(color="#e8e8ec", width=1),
                ),
                cells=dict(
                    values=[display_df[c] for c in display_df.columns],
                    fill_color=[["#ffffff" if i % 2 == 0 else "#fafbfc"
                                 for i in range(len(display_df))]],
                    font=dict(color="#3f3f46", size=12, family="Inter, sans-serif"),
                    align="center", height=34,
                    line=dict(color="#f0f0f3", width=1),
                ),
            )])
            fig_tbl.update_layout(
                margin=dict(l=0, r=0, t=0, b=0), height=720,
                paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            )
            st.plotly_chart(fig_tbl, use_container_width=True, config={"displayModeBar": False})
            st.caption(f"数据日期: {latest_date.strftime('%Y-%m-%d')} · 按模型预测分降序")

        with right:
            latest_data = full_pred[full_pred["date"] == latest_date]
            st.markdown('<p class="section-title" style="border-color: transparent;">预测统计</p>',
                        unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            make_card(s1, "均值", f"{latest_data['预测分数'].mean():.4f}")
            make_card(s2, "中位数", f"{latest_data['预测分数'].median():.4f}")
            s3, s4 = st.columns(2)
            make_card(s3, "标准差", f"{latest_data['预测分数'].std():.4f}")
            pct_up = (latest_data["预测分数"] > 0).mean() * 100
            make_card(s4, "看涨占比", f"{pct_up:.1f}%",
                      "偏多" if pct_up > 50 else "偏空",
                      "up" if pct_up > 50 else "down")

            st.markdown("<br>", unsafe_allow_html=True)
            fig_dist = px.histogram(
                latest_data, x="预测分数", nbins=40,
                color_discrete_sequence=["#3b82f6"],
            )
            fig_dist.update_layout(
                title=dict(text="预测分数分布", font=dict(color="#1a1a2e", size=13)),
                xaxis=dict(title="预测分数", color="#52525b"),
                yaxis=dict(title="股票数", color="#52525b"),
                bargap=0.05, showlegend=False, height=320,
                paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})
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

    if full_pred is not None:
        daily = prepare_eval_frame(full_pred)
        daily_acc = daily.groupby("date").agg(
            准确率=("correct_next", "mean"), 股票数=("correct_next", "count")
        ).reset_index()
        daily_acc = daily_acc[daily_acc["股票数"] >= 30]

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
                height=380, showlegend=False,
                paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
            )
            st.plotly_chart(fig_day, use_container_width=True, config={"displayModeBar": False})

        with right:
            cumret = build_strategy_curves(daily, strategy)

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
                height=380,
                legend=dict(x=0.01, y=0.99, font=dict(color="#52525b")),
                paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
            )
            st.plotly_chart(fig_cum, use_container_width=True, config={"displayModeBar": False})

        # 汇总卡片
        st.markdown('<p class="section-title">回测汇总</p>', unsafe_allow_html=True)
        total_days = len(daily_acc)
        avg_acc = daily_acc["准确率"].mean()
        win_days = (daily_acc["准确率"] > 0.5).sum()
        final_eq = cumret["等权累计"].iloc[-1] if len(cumret) > 0 else 0
        final_w = cumret["策略累计"].iloc[-1] if len(cumret) > 0 else 0
        m1, m2, m3, m4, m5 = st.columns(5)
        make_card(m1, "回测天数", f"{total_days}d")
        make_card(m2, "平均准确率", f"{avg_acc:.1%}",
                  f"{win_days}/{total_days}天超50%", "up" if avg_acc > 0.5 else "neutral")
        make_card(m3, "等权累计收益", f"{final_eq:.2%}",
                  "买入持有基准 (T+1)", "up" if final_eq >= 0 else "down")
        make_card(m4, "AI策略累计收益", f"{final_w:.2%}",
                  {"score_weighted": "分数归一化加权", "topk_equal": "Top20等权做多", "long_short": "Top20/Btm20多空"}[strategy], "up" if final_w >= 0 else "down")
        alpha = final_w - final_eq
        make_card(m5, "AI超额收益 α", f"{alpha:.2%}",
                  "优于等权" if alpha > 0 else "不及等权",
                  "up" if alpha > 0 else "down")
    else:
        st.warning("请先运行预测脚本生成数据文件")

# ═══════════════════════════════════════════════════
# ═══════════════════════════════════════════════════
# 🔍 个股追踪
# ═══════════════════════════════════════════════════
elif page == "🔍 个股追踪":
    st.markdown('<p class="hero-title">个股追踪</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">K线走势 + AI预测分数叠加 · 沪深300成分股</p>',
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
        selected_label = st.selectbox(
            "股票代码或名称",
            options=sorted(stock_options.keys()),
            index=None,
            placeholder="输入 600000 或 浦发银行 搜索...",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

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

            # 指标卡片
            st.markdown(f'<p class="section-title">{stock_code}  {stock_name}</p>',
                        unsafe_allow_html=True)
            s1, s2, s3, s4, s5 = st.columns(5)
            make_card(s1, "最新收盘", f"{sdf['close'].iloc[-1]:.2f}",
                      f"{sdf['涨跌幅'].iloc[-1]:.2f}%",
                      "up" if sdf["涨跌幅"].iloc[-1] > 0 else "down")
            make_card(s2, "AI预测分", f"{sdf['预测分数'].iloc[-1]:.4f}",
                      "看好" if sdf["预测分数"].iloc[-1] > 0 else "看跌",
                      "up" if sdf["预测分数"].iloc[-1] > 0 else "down")
            make_card(s3, "平均预测分", f"{sdf['预测分数'].mean():.4f}")
            make_card(s4, "方向一致率", f"{match:.1f}%",
                      "优秀" if match > 55 else "一般",
                      "up" if match > 50 else "down")
            make_card(s5, "累计涨跌", f"{sdf['涨跌幅'].sum():.2f}%")

            # K线 + 预测分叠加
            fig_stock = go.Figure()
            fig_stock.add_trace(go.Candlestick(
                x=sdf["date"], open=sdf["open"], high=sdf["high"],
                low=sdf["low"], close=sdf["close"],
                increasing_line_color="#e8453c",
                decreasing_line_color="#10b981",
                name="K线",
            ))
            fig_stock.add_trace(go.Scatter(
                x=sdf["date"], y=sdf["预测分数"],
                mode="lines+markers", yaxis="y2",
                line=dict(color="#f59e0b", width=2, dash="dot"),
                marker=dict(size=5), name="AI预测分",
            ))
            fig_stock.update_layout(
                title=dict(text=f"{stock_code} {stock_name} · 走势 & AI预测",
                           font=dict(color="#1a1a2e", size=15)),
                yaxis=dict(title="价格", color="#52525b"),
                yaxis2=dict(title="预测分数", overlaying="y", side="right",
                            color="#f59e0b"),
                height=500, xaxis_rangeslider_visible=False,
                paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
                legend=dict(font=dict(color="#52525b")),
            )
            st.plotly_chart(fig_stock, use_container_width=True, config={"displayModeBar": False})

            # 预测分走势 + 涨跌幅对比
            left, right = st.columns([1, 1])
            with left:
                fig_pred = go.Figure()
                colors = ["#e8453c" if v > 0 else "#10b981" for v in sdf["预测分数"]]
                fig_pred.add_trace(go.Bar(
                    x=sdf["date"], y=sdf["预测分数"], marker_color=colors,
                    name="预测分",
                ))
                fig_pred.add_hline(y=0, line_dash="solid", line_color="#d4d4d8")
                fig_pred.update_layout(
                    title=dict(text="每日AI预测分数", font=dict(color="#1a1a2e", size=14)),
                    height=320, showlegend=False,
                    paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
                    xaxis=dict(color="#52525b"), yaxis=dict(color="#52525b"),
                )
                st.plotly_chart(fig_pred, use_container_width=True, config={"displayModeBar": False})

            with right:
                pct_colors = ["#e8453c" if v > 0 else "#10b981" for v in sdf["涨跌幅"]]
                fig_pct = go.Figure()
                fig_pct.add_trace(go.Bar(
                    x=sdf["date"], y=sdf["涨跌幅"], marker_color=pct_colors,
                    name="涨跌幅",
                ))
                fig_pct.update_layout(
                    title=dict(text="每日涨跌幅(%)", font=dict(color="#1a1a2e", size=14)),
                    height=320, showlegend=False,
                    paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
                    xaxis=dict(color="#52525b"), yaxis=dict(color="#52525b"),
                )
                st.plotly_chart(fig_pct, use_container_width=True, config={"displayModeBar": False})

            with st.expander("📄 历史数据（最近60天）"):
                show = sdf[["date", "open", "close", "涨跌幅", "换手率", "预测分数"]] \
                    .sort_values("date", ascending=False).head(60)
                show.columns = ["日期", "开盘", "收盘", "涨跌%", "换手%", "预测分"]
                st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.warning("请先运行预测脚本生成数据文件")

# ═══════════════════════════════════════════════════
# 页脚
# ═══════════════════════════════════════════════════
st.markdown('<p class="footer">AI 量化选股看板 · Qlib XGBoost · 仅供研究参考，不构成投资建议</p>',
            unsafe_allow_html=True)
