"""
sft_cold_start.py
=================
SFT 冷启动：在 GRPO 之前，先用监督学习教模型基本的输出格式。

为什么需要：
    Qwen2.5-0.5B 随机生成时几乎不会输出 <reasoning>/<answer> 格式，
    导致奖励函数全给 0，GRPO 无梯度可学。
    先用 SFT 让模型"学会格式"，GRPO 再"优化内容"。

做法：
    用训练集的 gold label 构造标准格式的 SFT 样本：
        prompt = 原始讽刺检测 prompt
        completion = <reasoning>...线索...</reasoning><answer>gold_label</answer>

    推理链用模板生成（基于文本特征），保证格式正确。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "train"))

from prompt_template import build_user_prompt, SYSTEM_PROMPT, normalize_label  # noqa: E402


# ============================================================
# 构造 SFT 样本（标准格式的推理链）
# ============================================================
def make_reasoning(text: str, label: str) -> str:
    """
    根据文本和标签，生成一条"看起来合理"的推理链。
    模板化，保证格式正确 + 推理方向正确（不自相矛盾）。
    """
    # 启发式线索：检测文本里的讽刺/非讽刺特征
    sarcasm_cues = []
    if any(w in text for w in ["真是", "简直是", "太", "完美", "幸福"]):
        sarcasm_cues.append("文本含夸张表达")
    if any(w in text for w in ["停电", "挂了", "迟到", "故障", "丢了", "报错"]):
        sarcasm_cues.append("语境涉及负面事件")
    if any(w in text for w in ["当然", "梦想", "真是体贴"]):
        sarcasm_cues.append("语气与字面意思可能冲突")
    if any(w in text for w in ["真准", "预报", "精彩", "厉害"]):
        sarcasm_cues.append("词义可能为反语")

    literal_cues = []
    if any(w in text for w in ["研究", "表明", "测试", "方法"]):
        literal_cues.append("文本为客观陈述")
    if any(w in text for w in ["公园", "散步", "跑步", "起床"]):
        literal_cues.append("描述日常行为")
    if any(w in text for w in ["喜欢", "适合", "合理", "美丽"]):
        literal_cues.append("表达真实感受")

    if label == "sarcastic":
        cues = sarcasm_cues if sarcasm_cues else ["字面意思与真实意图可能相反"]
        cue_text = "；".join(cues[:3])
        return f"<reasoning>\n1. {cue_text}\n2. 综合判断为讽刺\n</reasoning>"
    else:
        cues = literal_cues if literal_cues else ["文本字面意思连贯合理"]
        cue_text = "；".join(cues[:3])
        return f"<reasoning>\n1. {cue_text}\n2. 综合判断为非讽刺\n</reasoning>"


def make_completion(text: str, label: str) -> str:
    """构造完整的标准格式输出。"""
    reasoning = make_reasoning(text, label)
    norm = normalize_label(label)
    return f"{reasoning}\n<answer>\n{norm}\n</answer>"


def generate_sft_data(train_file: str, output_file: str, num_samples: int = 2000):
    """从训练集生成 SFT 数据。"""
    with open(train_file, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    records = records[:num_samples]
    print(f"[SFT] 从 {len(records)} 条训练样本生成 SFT 数据")

    out_dir = Path(output_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            prompt = build_user_prompt(r["text"])
            completion = make_completion(r["text"], r["label"])
            sample = {
                "prompt": prompt,
                "completion": completion,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                "text": r["text"],
                "label": r["label"],
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n += 1

    print(f"[SFT] 生成 {n} 条 SFT 样本 → {output_file}")
    return output_file


# ============================================================
# SFT 训练
# ============================================================
def run_sft_train(sft_file: str, model_path: str, output_dir: str,
                  num_epochs: int = 3, lr: float = 2e-5, batch_size: int = 4):
    """用 transformers Trainer 做 SFT。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from datasets import Dataset
    import yaml

    cfg_path = ROOT / "configs" / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print(f"\n[SFT] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )

    # 加载 SFT 数据
    with open(sft_file, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    print(f"[SFT] {len(samples)} 条样本")

    # tokenize：prompt + completion 拼接，对 prompt 部分掩码
    def tokenize_fn(sample):
        messages = sample["messages"]
        # 用 chat template 格式化
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        # 只取 user 部分（计算 prompt 长度用于掩码）
        prompt_messages = messages[:2]  # system + user
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True)

        full_enc = tokenizer(full_text, truncation=True,
                             max_length=cfg["model"]["max_length"],
                             padding="max_length", return_tensors=None)
        prompt_enc = tokenizer(prompt_text, truncation=True,
                               max_length=cfg["model"]["max_length"])

        input_ids = full_enc["input_ids"]
        labels = input_ids.copy()
        # 对 prompt 部分掩码（设为 -100，不计算 loss）
        prompt_len = len(prompt_enc["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        # 对 padding 部分掩码
        for i in range(len(labels)):
            if input_ids[i] == tokenizer.pad_token_id:
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": full_enc["attention_mask"],
            "labels": labels,
        }

    dataset = Dataset.from_list(samples)
    dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
    print(f"[SFT] tokenize 完成，{len(dataset)} 条")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        seed=42,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\n[SFT] 开始训练 ({num_epochs} epochs)")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n[SFT] ✅ 完成，保存到 {output_dir}")
    return output_dir


# ============================================================
# 主入口
# ============================================================
def main():
    import yaml
    cfg_path = ROOT / "configs" / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", default=str(ROOT / "data" / "processed" / "train.jsonl"))
    ap.add_argument("--sft-output", default=str(ROOT / "data" / "processed" / "sft_data.jsonl"))
    ap.add_argument("--model-output", default=str(ROOT / "outputs" / "sft_cold_start"))
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--num-epochs", type=int, default=3)
    ap.add_argument("--skip-train", action="help", help="只生成数据，不训练")
    args = ap.parse_args()

    model_path = cfg["model"]["name"]

    # 1. 生成 SFT 数据
    print("=" * 60)
    print("  SFT 冷启动 · 数据生成")
    print("=" * 60)
    generate_sft_data(args.train_file, args.sft_output, args.num_samples)

    # 2. SFT 训练
    print("\n" + "=" * 60)
    print("  SFT 冷启动 · 训练")
    print("=" * 60)
    run_sft_train(args.sft_output, model_path, args.model_output,
                  num_epochs=args.num_epochs)


if __name__ == "__main__":
    main()
