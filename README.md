# Sarcasm-R1-Lite

> 基于推理与强化的讽刺语言理解实验（Reasoning & RL for Sarcasm Detection），Qwen2.5-0.5B + GRPO 的独立实现。

> 本仓库是 **赵勋（@darkestni）的个人独立项目**，受 DeepSeek-R1「推理 + 强化学习」范式启发，将讽刺检测从纯分类任务升级为 reasoning-heavy 任务，用 GRPO 优化模型的"先推理、再判断"能力。全部代码、数据流程、训练配置、评测脚本均由赵勋独立编写。

---

## 为什么做这个项目

讽刺检测是语用学（Pragmatics）里的经典难题：字面意思与真实意图相反，传统分类器只看 token 难以捕捉"言外之意"。
我尝试用 **"先生成结构化推理链、再用强化学习优化推理质量"** 的范式来逼近这个问题：

1. 让模型先输出一段**显式推理**（这句话为什么像 / 不像讽刺）；
2. 再输出**最终判断**（sarcastic / not_sarcastic）；
3. 用 **GRPO** 以"推理链质量 + 判断正确性"为联合奖励，优化模型。

这套"先推理、再判断、用 RL 优化推理"的思路，与 DeepSeek-R1、OpenAI o1 系列"reasoning-heavy + RL"的趋势同构，是本项目的方法论核心。

---

## 方法概述

### 推理-判断双段输出

模型对每条输入文本，按固定 schema 输出：

```
<reasoning>
逐条列出线索：夸张、反语、语境矛盾、常识违背……
</reasoning>
<answer>
sarcastic | not_sarcastic
</answer>
```

### 奖励函数（核心创新）

GRPO 的关键是奖励设计。本项目设计了**多维奖励**（见 `train/reward_fn.py`）：

| 奖励项 | 作用 | 取值 |
|--------|------|------|
| `answer_correct` | 最终判断对不对 | +1 / -1 |
| `format_valid` | 是否符合 `<reasoning>/<answer>` schema | +0.3 / 0 |
| `reasoning_present` | 推理链是否非空、有实质内容 | 0 ~ +0.3 |
| `reasoning_redundant` | 推理链过长（冗余惩罚） | -0.1 ~ 0 |
| `contradiction_penalty` | 推理与答案自相矛盾（如推理说"明显讽刺"却答 not） | -0.5 |

这套奖励同时**鼓励正确性、格式规范、推理实质、简洁性、自洽性**——对应 JD 中"agentic / reasoning / coding 类任务的评测与奖励设计"。

### 数据

- **主数据集**：[SARC](https://github.com/sahithiraml/sarc-datasets)（Reddit 大规模讽刺语料，~530K 评论）或 [iSarcasm](https://github.com/sahithiraml/iSarcasm)（小而精、人工标注）。
- **数据流程**（见 `data/prepare_data.py`）：原始评论 → 清洗 → prompt 模板填充 → train/val/test 划分 → JSONL。
- **合成数据路径**（可选，见 `data/synthesize.py`）：用大模型 API 对难例生成"带推理链的样例"，作为 SFT 冷启动数据——对应 JD "合成数据生成"职责。

---

## 目录结构

```
sarcasm-r1-lite/
├── README.md                  ← 本文件
├── requirements.txt           ← Python 依赖
├── .gitignore
├── configs/
│   └── default.yaml           ← 超参 / 模型 / 数据配置
├── data/
│   ├── prepare_data.py        ← 真实数据集下载 + 清洗 + JSONL 化
│   ├── synthesize.py          ← 合成带推理链的冷启动数据（可选）
│   └── prompt_template.py     ← prompt schema 定义
├── train/
│   ├── reward_fn.py           ← 多维奖励函数（核心）
│   └── grpo_train.py          ← GRPO 训练主脚本（TRL）
├── eval/
│   └── evaluate.py            ← 训前 vs 训后对比评测
├── scripts/
│   ├── run_all.sh             ← 集群一键启动：数据→训练→评测
│   └── run_local_small.sh     ← Mac 本地小规模走通流程（CPU）
└── results/                   ← 训练日志、评测结果（运行后生成）
```

---

## 快速开始

### 环境

```bash
pip install -r requirements.txt
```

需要 Python 3.10+，GPU 训练需要 CUDA 11.8+ 与一张 ≥10GB 显存的 NVIDIA 显卡（0.5B 模型）。

### 本地小规模走通（Mac / CPU，验证流程）

```bash
bash scripts/run_local_small.sh
```

用极小数据量（50 条）+ CPU 推理，验证"数据准备 → prompt 构造 → 奖励函数 → 评测"全链路通。**不训练**。

### 集群训练（GPU）

```bash
bash scripts/run_all.sh
```

依次执行：数据准备 → （可选）合成冷启动数据 → GRPO 训练 → 训前训后评测。

---

## 实验记录

> 训练完成后在此填写。模板见 `results/experiment_template.md`。

预期记录：
- 训练前 baseline（zero-shot prompt）准确率 / F1
- GRPO 训练后准确率 / F1
- 训练 reward 曲线（`results/training_log.png`）
- 消融：去掉 `reasoning_present` 奖励后的效果
- case study：训前训后各 5 条对比

---

## 项目背景

本项目是一个面向「reasoning-heavy 任务 + RL 优化」的独立工程实践：

- 方法思路源自 **DeepSeek-R1** 公开的「先推理、再判断，用 GRPO 优化推理质量」范式；
- 将其落到一个具体的 NLP 任务（讽刺检测）上，做完整的工程实现：数据流程 → 多维奖励设计 → GRPO 训练 → 训前训后评测；
- 选型上做了务实简化：模型用 **Qwen2.5-0.5B**（单卡可训）、框架用 **HuggingFace TRL GRPOTrainer**（成熟开源），聚焦在**奖励函数设计与训练实验**本身。
- 若引用本仓库，请引用为：*赵勋, Sarcasm-R1-Lite, GitHub, 2026.*

---

## 致谢

- 方法思路启发：DeepSeek-R1 的「推理 + 强化学习」范式。
- 训练框架：[HuggingFace TRL](https://github.com/huggingface/trl) GRPOTrainer。
- 数据：SARC / iSarcasm 语料贡献者。

---

## License

MIT
