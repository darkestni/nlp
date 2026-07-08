"""
grpo_train.py
=============
用 TRL 的 GRPOTrainer 对 Qwen2.5-0.5B 做 GRPO 训练，
优化"先推理、再判断"的讽刺检测能力。

⚠️ 本脚本需要 GPU + torch + transformers + trl。
   在 Mac（无 GPU）上 import 即可，不要实际运行训练。
   集群上：python train/grpo_train.py --config configs/default.yaml

核心组件：
    1. 从 JSONL 读数据 → 构造 prompt + gold label
    2. 加载 Qwen2.5-0.5B-Instruct + tokenizer
    3. 注册多维奖励函数（来自 reward_fn.py）
    4. GRPOTrainer 训练，保存 checkpoint + 训练日志
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "data"))

from reward_fn import make_reward_funcs, DEFAULT_WEIGHTS  # noqa: E402


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str):
    """读 JSONL，返回 list[dict]。"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_grpo_dataset(records):
    """
    把记录转成 GRPOTrainer 期望的 prompt-only 数据集。
    GRPO 不需要 "completion"——它自己采样生成 completions，
    但需要传 gold label 进奖励函数（通过 dataset 列）。
    """
    prompts = []
    labels = []
    for r in records:
        # GRPOTrainer 期望 prompt 是字符串（apply_chat_template 会处理）
        # 或已 tokenized 的 input_ids。这里用字符串形式。
        prompts.append({"prompt": r["prompt"]})
        labels.append(r["label"])
    return prompts, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--train-file", default=str(ROOT / "data" / "processed" / "train.jsonl"))
    ap.add_argument("--output-dir", default=None, help="覆盖 config 里的 output_dir")
    ap.add_argument("--sft-checkpoint", default=None,
                    help="SFT 冷启动 checkpoint 路径（推荐用，作为 GRPO 起点）")
    ap.add_argument("--smoke-test", action="store_true",
                    help="只 import 库 + 载模型，不训练（验证环境用）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    grpo_cfg = cfg["grpo"]
    reward_weights = cfg.get("reward_weights", DEFAULT_WEIGHTS)
    seed = cfg["run"]["seed"]

    print("=" * 60)
    print("Sarcasm-R1-Lite · GRPO 训练")
    print("=" * 60)
    model_name = args.sft_checkpoint or model_cfg["name"]
    print(f"模型: {model_name}" + (" (从 SFT 接续)" if args.sft_checkpoint else " (基座)"))
    print(f"输出: {grpo_cfg['output_dir']}")

    # ---- 1. 读数据 ----
    print("\n[1/4] 加载训练数据...")
    records = load_jsonl(args.train_file)
    print(f"      训练样本: {len(records)}")
    from collections import Counter
    print(f"      标签分布: {dict(Counter(r['label'] for r in records))}")

    prompts, gold_labels = build_grpo_dataset(records)

    # ---- 2. 加载模型与 tokenizer ----
    print("\n[2/4] 加载模型与 tokenizer...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # model_name 已在上方根据 sft_checkpoint 确定
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"      模型加载完成: {model_name}")

    if args.smoke_test:
        print("\n[smoke-test] 环境验证通过，跳过训练。")
        # 试一条推理
        test_prompt = prompts[0]["prompt"]
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        print("      样例推理输出:")
        print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        return

    # ---- 3. 配置奖励函数 ----
    print("\n[3/4] 注册多维奖励函数...")
    reward_funcs_named = make_reward_funcs(weights=reward_weights)
    for name, _ in reward_funcs_named:
        print(f"      - {name}")

    # ---- 4. GRPO 训练 ----
    print("\n[4/4] 初始化 GRPOTrainer 并启动训练...")
    from trl import GRPOConfig, GRPOTrainer

    output_dir = args.output_dir or grpo_cfg["output_dir"]

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=grpo_cfg["learning_rate"],
        per_device_train_batch_size=grpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=grpo_cfg["gradient_accumulation_steps"],
        num_train_epochs=grpo_cfg["num_train_epochs"],
        max_prompt_length=grpo_cfg["max_prompt_length"],
        max_completion_length=grpo_cfg["max_completion_length"],
        num_generations=grpo_cfg["num_generations"],
        temperature=grpo_cfg["temperature"],
        beta=grpo_cfg["beta"],
        save_steps=grpo_cfg["save_steps"],
        logging_steps=grpo_cfg["logging_steps"],
        seed=seed,
        report_to="none",          # 集群上可改 "tensorboard" / "wandb"
        log_completions=True,      # 记录采样结果，便于分析
    )

    # GRPOTrainer 需要一个返回 prompt 的数据集；奖励函数通过 column 名拿 gold
    # 我们把 gold_labels 作为 dataset 的一列 "label"
    import datasets
    ds_dict = {"prompt": [p["prompt"] for p in prompts], "label": gold_labels}
    train_dataset = datasets.Dataset.from_dict(ds_dict)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[fn for _, fn in reward_funcs_named],
        processing_class=tokenizer,
    )

    print("\n[start] 开始训练...")
    trainer.train()

    print(f"\n[done] 训练完成，保存到 {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("\n下一步：python eval/evaluate.py --model {} --eval-file {}".format(
        output_dir,
        str(ROOT / "data" / "processed" / "eval.jsonl"),
    ))


if __name__ == "__main__":
    main()
