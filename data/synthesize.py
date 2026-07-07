"""
synthesize.py
=============
合成"带推理链的"冷启动数据 —— 用大模型 API 对难例生成高质量样例。

为什么要这个：GRPO 是 on-policy RL，对基座模型的初始推理能力有要求。
如果基座（0.5B）一开始就不会按 <reasoning>/<answer> 格式输出，
GRPO 很难从"格式都不对"的起点学起来。
解法：先用大模型 API 合成一批"格式规范、推理正确"的样例做 SFT 冷启动，
让基座先学会格式，再用 GRPO 优化质量。

对应 JD "合成数据生成 / 任务行为轨迹构建" 职责。

用法：
    # 需要设置 API
    export OPENAI_API_KEY=sk-...        # 或 DeepSeek 兼容
    export OPENAI_BASE_URL=https://api.deepseek.com   # DeepSeek 便宜

    python data/synthesize.py --input data/processed/train.jsonl \
        --output data/processed/sft_synth.jsonl --num 200
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prompt_template import normalize_label  # noqa: E402


SYNTH_SYSTEM = """你是讽刺检测数据标注专家。给定一条文本和它的正确标签，
请你生成一段"符合 <reasoning>/<answer> 格式、推理正确、答案与标签一致"的高质量示范。

输出格式严格为：
<reasoning>
1. 第一条线索（具体指出文本里的什么特征支持这个判断）
2. 第二条线索（可选，最多 4 条）
</reasoning>
<answer>
sarcastic 或 not_sarcastic
</answer>

要求：
- 推理必须真实指向答案，不得自相矛盾。
- 答案必须与给定的正确标签一致。
- 简洁，每条线索 1 句话。"""


def synth_one(client, model_name, text, gold):
    """调一次 API 合成一条样例。"""
    user_msg = (
        f"文本：{text}\n\n"
        f"正确标签：{gold}\n\n"
        f"请生成符合格式的示范推理与答案。"
    )
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYNTH_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/train.jsonl")
    ap.add_argument("--output", default="data/processed/sft_synth.jsonl")
    ap.add_argument("--num", type=int, default=200, help="合成多少条")
    ap.add_argument("--model", default="deepseek-chat", help="API 模型名")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    if not api_key:
        print("[fatal] 请先设置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("[fatal] 请先 pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    # 读输入
    with open(args.input, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    targets = records[: args.num]

    print(f"[info] 准备合成 {len(targets)} 条 → {args.output}")

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok, n_fail = 0, 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, r in enumerate(targets):
            try:
                completion = synth_one(client, args.model, r["text"], r["label"])
                # 写成 SFT 样本：prompt + 标准 completion
                sft_sample = {
                    "text": r["text"],
                    "label": r["label"],
                    "prompt": r["prompt"],
                    "completion": completion,
                    "source": "synth",
                }
                f.write(json.dumps(sft_sample, ensure_ascii=False) + "\n")
                n_ok += 1
                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{len(targets)}] 已合成 {n_ok} 条")
            except Exception as e:
                n_fail += 1
                print(f"  [{i+1}] 失败: {e}")

    print(f"\n[done] 合成完成: 成功 {n_ok} / 失败 {n_fail}")
    print(f"       输出: {args.output}")
    print(f"       下一步可用此文件做 SFT 冷启动，再接 GRPO。")


if __name__ == "__main__":
    main()
