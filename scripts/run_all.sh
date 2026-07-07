#!/usr/bin/env bash
# ============================================================
# run_all.sh — 集群一键训练流程
# 数据准备 → (可选)合成冷启动 → GRPO 训练 → 训前训后评测
# ============================================================
set -e

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"
echo "项目根目录: $PROJECT_DIR"

# ---- 0. 环境检查 ----
echo ""
echo "============================================================"
echo "  [0/5] 环境检查"
echo "============================================================"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"

# ---- 1. 安装依赖（首次运行）----
if [ ! -d ".venv" ] && [ "$1" != "--no-install" ]; then
    echo ""
    echo "============================================================"
    echo "  [1/5] 安装依赖"
    echo "============================================================"
    pip install -r requirements.txt
fi

# ---- 2. 数据准备 ----
echo ""
echo "============================================================"
echo "  [2/5] 数据准备"
echo "============================================================"
# 集群上联网下载 iSarcasm；离线用 sample
if python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=5)" 2>/dev/null; then
    echo "网络可用，下载 iSarcasm..."
    python data/prepare_data.py --source isarcasm --max-train 4000 --max-eval 500
    SOURCE="isarcasm"
else
    echo "[warn] 无法联网，降级为内置小样本（仅走通流程，不可用于真实训练）"
    python data/prepare_data.py --source sample --max-train 4000 --max-eval 500
    SOURCE="sample"
fi

# ---- 3. 可选：合成冷启动数据 ----
echo ""
echo "============================================================"
echo "  [3/5] 合成冷启动数据（可选，需要 API key）"
echo "============================================================"
if [ -n "$OPENAI_API_KEY$DEEPSEEK_API_KEY" ]; then
    echo "检测到 API key，合成冷启动数据..."
    python data/synthesize.py --input data/processed/train.jsonl \
        --output data/processed/sft_synth.jsonl --num 200 || echo "[warn] 合成失败，跳过"
else
    echo "未设置 API key，跳过合成（GRPO 仍可直接跑，只是冷启动效果略差）"
fi

# ---- 4. Baseline 评测（训前）----
echo ""
echo "============================================================"
echo "  [4/5] Baseline 评测（训前）"
echo "============================================================"
BASELINE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
python eval/evaluate.py \
    --model "$BASELINE_MODEL" \
    --eval-file data/processed/eval.jsonl \
    --output results/baseline_metrics.json || echo "[warn] baseline 评测失败"

# ---- 5. GRPO 训练 ----
echo ""
echo "============================================================"
echo "  [5/5] GRPO 训练"
echo "============================================================"
python train/grpo_train.py --config configs/default.yaml

# ---- 6. 训后评测 + 对比 ----
echo ""
echo "============================================================"
echo "  训后评测 + 训前训后对比"
echo "============================================================"
AFTER_MODEL=$(python -c "import yaml; print(yaml.safe_load(open('configs/default.yaml'))['grpo']['output_dir'])")
python eval/evaluate.py \
    --model "$AFTER_MODEL" \
    --eval-file data/processed/eval.jsonl \
    --baseline-model "$BASELINE_MODEL" \
    --output results/after_metrics.json || echo "[warn] 训后评测失败"

echo ""
echo "============================================================"
echo "  ✅ 全流程完成"
echo "============================================================"
echo "训练产出: $AFTER_MODEL"
echo "评测结果: results/baseline_metrics.json + results/after_metrics.json"
echo ""
echo "下一步：把训练曲线、评测对比截图填入 results/experiment_record.md"
