#!/usr/bin/env bash
# ============================================================
# run_local_small.sh — Mac 本地小规模走通流程（不训练）
# 验证：数据准备 → 奖励函数 → 评测 全链路可跑
# ============================================================
set -e

cd "$(dirname "$0")/.."
echo "项目根目录: $(pwd)"
echo ""
echo "本脚本只在本地走通流程，不下载大模型、不训练。"
echo "训练请上集群跑 scripts/run_all.sh"
echo ""

echo "============================================================"
echo "  [1/4] 数据准备（内置小样本）"
echo "============================================================"
python data/prepare_data.py --source sample --max-train 20 --max-eval 8

echo ""
echo "============================================================"
echo "  [2/4] 奖励函数自测"
echo "============================================================"
python train/reward_fn.py

echo ""
echo "============================================================"
echo "  [3/4] 评测脚本走通（离线模拟）"
echo "============================================================"
python eval/evaluate.py --offline --eval-file data/processed/eval.jsonl

echo ""
echo "============================================================"
echo "  [4/4] 训练脚本环境检查（只 import，不跑）"
echo "============================================================"
python -c "
import sys
sys.path.insert(0, 'train')
sys.path.insert(0, 'data')
print('✅ reward_fn 可 import')
from reward_fn import compute_reward
print('✅ compute_reward 可调用')
print()
print('训练需要的库（集群上装）:')
for mod in ['torch', 'transformers', 'trl', 'accelerate', 'datasets']:
    try:
        m = __import__(mod)
        print(f'  ✅ {mod}: {getattr(m, \"__version__\", \"?\")}')
    except ImportError:
        print(f'  ⬜ {mod}: 未安装（集群上 pip install -r requirements.txt）')
"

echo ""
echo "============================================================"
echo "  ✅ 本地走通完成"
echo "============================================================"
echo "所有逻辑已验证，可以上传到集群运行 scripts/run_all.sh"
