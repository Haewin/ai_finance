#!/bin/bash
# ═══════════════════════════════════════════════════════════
# AI 量化看板 · 一键数据更新脚本
# 用法: bash dashboard/update_data.sh
# ═══════════════════════════════════════════════════════════
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
QLIB_DIR="$PROJ_DIR/qlib"

echo "========================================"
echo "  AI 量化看板 · 数据更新"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# ── 步骤1: 运行改进版模型训练 ──────────────────────
echo "[1/2] 运行沪深300预测 (60因子 + XGBoost)..."
cd "$QLIB_DIR"

if [ -f "improved_pipeline.py" ]; then
    echo "  → 运行 improved_pipeline.py --csi300 csi300_stocks.json"
    python3 improved_pipeline.py --csi300 csi300_stocks.json
else
    echo "  ⚠ 未找到 improved_pipeline.py"
fi

# ── 步骤2: 检查数据文件 ──────────────────────────────
echo ""
echo "[2/2] 检查数据文件..."

check_file() {
    if [ -f "$QLIB_DIR/$1" ]; then
        lines=$(wc -l < "$QLIB_DIR/$1")
        echo "  ✅ $1 (${lines} 行)"
    else
        echo "  ❌ $1 不存在"
    fi
}

check_file "Qlib_沪深300_全量预测&回测.csv"

# ── 步骤3: 检查实验数据 ──────────────────────────────
echo ""
echo "[3/3] 检查实验对照数据..."
CODEX_DIR="/Users/haewin/Documents/Codex/2026-04-23-files-mentioned-by-the-user-ai"
if [ -d "$CODEX_DIR" ]; then
    exp_count=$(find "$CODEX_DIR" -maxdepth 2 -name "summary.md" -o -name "summary.json" | wc -l)
    echo "  ✅ 找到 ${exp_count} 个实验记录"
    echo "  → 看板会自动从 summary.md / summary.json 解析"
else
    echo "  ⚠ Codex 结果目录不存在，模型对比数据将无法加载"
fi

echo ""
echo "========================================"
echo "  更新完成!"
echo "  启动看板: streamlit run dashboard/app.py"
echo "========================================"
