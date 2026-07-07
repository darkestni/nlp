"""
evaluate.py
===========
讽刺检测模型评测：计算准确率、F1、格式合规率，并对比训前训后。

支持两种推理后端：
    1. 本地模型（--model 路径或 HF id）→ 需要 GPU，集群上用
    2. 离线模拟（--offline）→ 不加载模型，用规则模拟输出，Mac 上走通流程用

用法：
    # 集群：评测训后模型
    python eval/evaluate.py --model outputs/grpo_run --eval-file data/processed/eval.jsonl

    # 集群：对比训前 baseline
    python eval/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --eval-file data/processed/eval.jsonl

    # Mac 离线走通
    python eval/evaluate.py --offline --eval-file data/processed/eval.jsonl
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "data"))

from reward_fn import parse_output  # noqa: E402


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ============================================================
# 推理后端
# ============================================================
def generate_local(model_path, prompts, max_new_tokens=384, batch_size=8):
    """本地模型批量推理（集群用）。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="推理中"):
        batch = prompts[i:i + batch_size]
        msgs = [[{"role": "system", "content": _sys()}, {"role": "user", "content": p}] for p in batch]
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        outputs.extend([tokenizer.decode(g, skip_special_tokens=True) for g in gen])
    return outputs


def generate_offline(records):
    """离线模拟：用简单规则生成"看起来合理"的输出，走通评测流程用。"""
    outputs = []
    for r in records:
        text = r["text"]
        # 启发式：含明显讽刺词的标 sarcastic（仅用于走通流程，不代表真实能力）
        cues = ["真准", "太棒了", "真是", "简直是", "幸福得", "梦想", "当然", "完美"]
        is_sarc = any(c in text for c in cues) and any(
            w in text for w in ["停电", "挂了", "迟到", "暴雨", "丢了", "故障", "睡着", "早起"]
        )
        ans = "sarcastic" if is_sarc else "not_sarcastic"
        reasoning = f"<reasoning>\n1. 文本语境与字面表达存在{'冲突' if is_sarc else '一致'}\n</reasoning>\n"
        answer = f"<answer>\n{ans}\n</answer>"
        outputs.append(reasoning + answer)
    return outputs


def _sys():
    from prompt_template import SYSTEM_PROMPT
    return SYSTEM_PROMPT


# ============================================================
# 评测指标
# ============================================================
def evaluate(records, completions):
    """计算多项指标。"""
    n = len(records)
    golds = [r["label"] for r in records]

    parsed = [parse_output(c) for c in completions]
    preds = [p.answer_norm if p.answer_norm != "invalid" else "not_sarcastic" for p in parsed]

    # 准确率
    correct = sum(1 for g, p in zip(golds, preds) if g == p)
    acc = correct / n

    # F1（以 sarcastic 为正类）
    tp = sum(1 for g, p in zip(golds, preds) if g == "sarcastic" and p == "sarcastic")
    fp = sum(1 for g, p in zip(golds, preds) if g == "not_sarcastic" and p == "sarcastic")
    fn = sum(1 for g, p in zip(golds, preds) if g == "sarcastic" and p == "not_sarcastic")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # 格式合规率
    format_ok = sum(1 for p in parsed if p.has_reasoning and p.has_answer and p.answer_norm != "invalid")
    format_rate = format_ok / n

    # 推理链平均条数
    avg_lines = sum(p.reasoning_lines for p in parsed) / n

    return {
        "n": n,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "format_compliance": format_rate,
        "avg_reasoning_lines": avg_lines,
        "pred_distribution": dict(Counter(preds)),
        "gold_distribution": dict(Counter(golds)),
    }


def print_report(title, metrics, completions=None, records=None, show_cases=5):
    """打印评测报告。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  样本数:          {metrics['n']}")
    print(f"  准确率 Accuracy: {metrics['accuracy']:.2%}")
    print(f"  精确率 Precision:{metrics['precision']:.2%}  (sarcastic 为正)")
    print(f"  召回率 Recall:   {metrics['recall']:.2%}")
    print(f"  F1:              {metrics['f1']:.2%}")
    print(f"  格式合规率:      {metrics['format_compliance']:.2%}")
    print(f"  平均推理线索:    {metrics['avg_reasoning_lines']:.2f} 条")
    print(f"  预测分布:        {metrics['pred_distribution']}")
    print(f"  真实分布:        {metrics['gold_distribution']}")

    if completions and records and show_cases:
        print(f"\n  --- 前 {show_cases} 条样例 ---")
        for i, (r, c) in enumerate(zip(records[:show_cases], completions[:show_cases])):
            print(f"\n  [{i+1}] gold={r['label']}")
            print(f"      文本: {r['text'][:60]}...")
            print(f"      生成: {c.replace(chr(10), ' | ')[:120]}")


# ============================================================
# 主入口
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="模型路径或 HF id（不指定且非 offline 则报错）")
    ap.add_argument("--eval-file", default=str(ROOT / "data" / "processed" / "eval.jsonl"))
    ap.add_argument("--offline", action="store_true", help="离线模拟，不加载模型（Mac 走通用）")
    ap.add_argument("--baseline-model", default=None, help="同时评测 baseline 做对比")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output", default=None, help="结果保存为 json")
    args = ap.parse_args()

    records = load_jsonl(args.eval_file)
    print(f"[info] 加载评测集: {len(records)} 条")

    results = {}

    if args.offline:
        completions = generate_offline(records)
        m = evaluate(records, completions)
        print_report("离线模拟评测（仅走通流程，非真实能力）", m, completions, records)
        results["offline"] = m
    else:
        if not args.model:
            print("[fatal] 非 offline 模式必须指定 --model")
            sys.exit(1)
        prompts = [r["prompt"] for r in records]
        completions = generate_local(args.model, prompts, args.max_new_tokens, args.batch_size)
        m = evaluate(records, completions)
        print_report(f"模型评测: {args.model}", m, completions, records)
        results[args.model] = m

        if args.baseline_model:
            print(f"\n[info] 评测 baseline: {args.baseline_model}")
            base_completions = generate_local(args.baseline_model, prompts, args.max_new_tokens, args.batch_size)
            bm = evaluate(records, base_completions)
            print_report(f"Baseline: {args.baseline_model}", bm, base_completions, records)
            results[args.baseline_model] = bm

            # 对比
            print(f"\n{'=' * 60}")
            print("  训前 vs 训后 对比")
            print(f"{'=' * 60}")
            print(f"  {'指标':<16} {'Baseline':>12} {'After GRPO':>12} {'变化':>10}")
            for k in ["accuracy", "f1", "format_compliance"]:
                b = bm[k]
                a = m[k]
                d = a - b
                print(f"  {k:<16} {b:>12.2%} {a:>12.2%} {d:>+10.2%}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[done] 结果保存到 {args.output}")


if __name__ == "__main__":
    main()
