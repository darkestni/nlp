#!/usr/bin/env bash
# ============================================================
# update_results.sh — 训练完成后,一键提交并推送结果
#
# 用法(在 sarcasm-r1-lite 目录下):
#   1. 先把训练产物拷到 results/ 目录:
#      - results/baseline_metrics.json  (训前评测)
#      - results/after_metrics.json     (训后评测)
#      - results/training_log.png       (训练曲线图)
#      - results/experiment_record.md   (填好的实验记录,把 template 复制改名)
#   2. 运行本脚本:
#      bash scripts/update_results.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  提交训练结果到 GitHub"
echo "============================================================"

# 检查 results 目录下有没有产物
echo "results/ 目录内容:"
ls -la results/ 2>/dev/null

# 取消 results/*.png 的 gitignore(让训练曲线图能进 git)
if grep -q "results/\*.png" .gitignore; then
  sed -i '' '/results\/\*\.png/d' .gitignore
  echo "✅ 已取消 results/*.png 的忽略"
fi

# 添加所有变更
git add -A
echo ""
echo "待提交内容:"
git status --short

# 提交
echo ""
git commit -m "results: 加入训前训后评测数据与训练曲线

- baseline_metrics.json: 训前 baseline 评测
- after_metrics.json: GRPO 训后评测
- training_log.png: 训练 reward 曲线
- experiment_record.md: 完整实验记录" 2>&1 | tail -3

# 推送
echo ""
echo "推送到 GitHub..."
git push origin main

echo ""
echo "============================================================"
echo "  ✅ 完成! 训练结果已公开"
echo "============================================================"
echo "仓库: https://github.com/darkestni/nlp"
echo ""
echo "现在 GitHub 上能看到完整的:代码 + 训练结果 + 实验记录"
