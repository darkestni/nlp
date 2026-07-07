"""
prepare_data.py
===============
讽刺检测数据的下载、清洗、格式化、划分。

输出 JSONL，每行一条样本：
    {"text": "...", "label": "sarcastic"|"not_sarcastic", "prompt": "...", "messages": [...]}

三条路径（自动降级）：
    1. 在线下载 iSarcasm / SARC（集群上用）
    2. 本地 raw 文件（若 data/raw/ 下有自带数据）
    3. 内置小样本（网络不通时，Mac 上走通流程用）

用法：
    python data/prepare_data.py --source isarcasm --max-train 4000 --max-eval 500
    python data/prepare_data.py --source sample            # 内置小样本，离线可跑
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

# 让脚本无论从哪运行都能 import 同级 prompt_template
sys.path.insert(0, str(Path(__file__).parent))
from prompt_template import build_user_prompt, build_messages, normalize_label  # noqa: E402

DATA_DIR = Path(__file__).parent
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 路径 3：内置小样本（离线可跑，Mac 走通用）
# ============================================================
# 真实风格的讽刺/非讽刺样例，用于本地小规模走通流程。
# 集群训练请用 --source isarcasm 或 sarc 换成大规模真实数据。
SAMPLE_DATA = [
    # --- 讽刺 ---
    ("哇，周一早起上班真是人生最大的快乐，我简直幸福得要哭了。", "sarcastic"),
    ("太棒了，又停电了，完美的一天。", "sarcastic"),
    ("你这方案真是前无古人后无来者，把所有问题都解决了——除了你自己制造的那些。", "sarcastic"),
    ("当然，我特别乐意周末加班，这是我的梦想。", "sarcastic"),
    ("考试成绩出来了，又挂了三科，我真是太厉害了。", "sarcastic"),
    ("哦，你的代码又能跑了？那 bug 一定只是去度了个假。", "sarcastic"),
    ("谢谢你把我的咖啡全喝完了，真是体贴入微的好同事。", "sarcastic"),
    ("天气预报说今天有暴雨，结果出大太阳，预报真准。", "sarcastic"),
    ("这部电影太精彩了，我看了十分钟就睡着了。", "sarcastic"),
    ("哦太好了，地铁又故障了，我刚好喜欢站着等半小时。", "sarcastic"),
    ("你迟到的功力真是一流，开会就等你一个人，太有面子了。", "sarcastic"),
    ("行李箱在机场丢了，航空公司服务真好，给我带来了惊喜。", "sarcastic"),
    # --- 非讽刺 ---
    ("今天阳光很好，我们去了公园散步。", "not_sarcastic"),
    ("这本书详细介绍了机器学习的基本算法，适合初学者。", "not_sarcastic"),
    ("会议将在下午两点准时开始，请准时参加。", "not_sarcastic"),
    ("我喜欢这首歌的旋律，听起来很放松。", "not_sarcastic"),
    ("经过反复测试，这个方法显著提升了模型准确率。", "not_sarcastic"),
    ("他每天早上六点起床跑步，已经坚持了三年。", "not_sarcastic"),
    ("这家餐厅的菜分量很足，价格也合理。", "not_sarcastic"),
    ("新的地铁线路开通后，通勤时间缩短了二十分钟。", "not_sarcastic"),
    ("老师耐心地讲解了这道题的解题思路，我受益匪浅。", "not_sarcastic"),
    ("根据最新研究，规律运动有助于改善睡眠质量。", "not_sarcastic"),
    ("我把报告整理成三个部分，方便大家阅读。", "not_sarcastic"),
    ("秋天到了，树叶变黄，景色非常美丽。", "not_sarcastic"),
]


def load_sample_data(max_total: int = 1000):
    """内置小样本：复制扩充到指定规模（走通流程用）。"""
    rng = random.Random(42)
    samples = list(SAMPLE_DATA)
    out = []
    while len(out) < max_total:
        rng.shuffle(samples)
        out.extend(samples)
    return out[:max_total]


# ============================================================
# 路径 2：本地 raw 文件
# ============================================================
def load_local_raw():
    """若 data/raw/ 下放了 csv/json，按通用字段解析。"""
    raw_dir = DATA_DIR / "raw"
    if not raw_dir.exists():
        return None
    files = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.json"))
    if not files:
        return None
    import pandas as pd
    samples = []
    for f in files:
        if f.suffix == ".csv":
            df = pd.read_csv(f)
        else:
            df = pd.read_json(f)
        # 尝试常见列名
        text_col = next((c for c in ["text", "comment", "sentence", "content"] if c in df.columns), None)
        label_col = next((c for c in ["label", "sarcastic", "is_sarcastic", "class"] if c in df.columns), None)
        if text_col and label_col:
            for _, row in df.iterrows():
                samples.append((str(row[text_col]), row[label_col]))
    return samples if samples else None


# ============================================================
# 路径 1：在线下载（集群上用，需要联网）
# ============================================================
def load_isarcasm(max_total: int):
    """从 HuggingFace datasets 加载 iSarcasm（需要联网）。"""
    try:
        from datasets import load_dataset
        ds = load_dataset("osanseviero/iSarcasm", split="train")
        samples = []
        for ex in ds.select(range(min(max_total, len(ds)))):
            # iSarcasm 字段：sentence / sarcastic (1/0)
            text = ex.get("sentence") or ex.get("text") or ""
            label_raw = ex.get("sarcastic") or ex.get("label") or 0
            if text:
                samples.append((text, label_raw))
        return samples
    except Exception as e:
        print(f"[warn] 在线加载 iSarcasm 失败: {e}")
        return None


def load_sarc(max_total: int):
    """加载 SARC（Reddit 讽刺语料）。需要先下载到 data/raw/。"""
    raw = load_local_raw()
    if raw is None:
        print("[warn] SARC 需要预先下载到 data/raw/（csv/json）。详见 README。")
        return None
    return raw[:max_total]


# ============================================================
# 统一处理：清洗 → prompt 构造 → 写 JSONL
# ============================================================
def clean_text(text: str) -> str:
    """基础清洗：去多余空白、截断超长文本。"""
    text = str(text).replace("\r", " ").strip()
    text = " ".join(text.split())
    if len(text) > 500:
        text = text[:500]
    return text


def build_records(samples):
    """把 (text, label) 列表加工成完整记录（含 prompt / messages）。"""
    records = []
    for text, label in samples:
        text = clean_text(text)
        if not text:
            continue
        norm = normalize_label(label)
        records.append({
            "text": text,
            "label": norm,
            "prompt": build_user_prompt(text),
            "messages": build_messages(text),
        })
    return records


def split_and_save(records, test_size=0.1, max_train=None, max_eval=None):
    """划分 train/val 并写 JSONL。"""
    rng = random.Random(42)
    rng.shuffle(records)
    n_test = max(1, int(len(records) * test_size))
    test = records[:n_test]
    train = records[n_test:]
    if max_train:
        train = train[:max_train]
    if max_eval:
        test = test[:max_eval]

    train_path = PROCESSED_DIR / "train.jsonl"
    eval_path = PROCESSED_DIR / "eval.jsonl"
    for path, data in [(train_path, train), (eval_path, test)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 标签分布统计
    from collections import Counter
    train_dist = Counter(r["label"] for r in train)
    eval_dist = Counter(r["label"] for r in test)
    print(f"[done] train: {len(train)} 条 → {train_path}")
    print(f"       label 分布: {dict(train_dist)}")
    print(f"[done] eval:  {len(test)} 条 → {eval_path}")
    print(f"       label 分布: {dict(eval_dist)}")


# ============================================================
# 主入口
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sample",
                    choices=["sample", "isarcasm", "sarc", "local"],
                    help="数据来源（sample=离线小样本 / isarcasm / sarc / local）")
    ap.add_argument("--max-train", type=int, default=4000)
    ap.add_argument("--max-eval", type=int, default=500)
    args = ap.parse_args()

    print(f"[info] 数据来源: {args.source}")

    samples = None
    if args.source == "sample":
        samples = load_sample_data(max_total=args.max_train + args.max_eval)
    elif args.source == "isarcasm":
        samples = load_isarcasm(max_total=args.max_train + args.max_eval)
        if samples is None:
            print("[fallback] 在线加载失败，降级为内置小样本（仅走通流程，不可用于真实训练）")
            samples = load_sample_data(max_total=args.max_train + args.max_eval)
    elif args.source == "sarc":
        samples = load_sarc(max_total=args.max_train + args.max_eval)
        if samples is None:
            print("[fatal] SARC 数据未就绪，请先下载到 data/raw/。")
            sys.exit(1)
    elif args.source == "local":
        samples = load_local_raw()
        if samples is None:
            print("[fatal] data/raw/ 下无可识别的数据文件。")
            sys.exit(1)

    records = build_records(samples)
    print(f"[info] 清洗后有效样本: {len(records)}")
    split_and_save(records, max_train=args.max_train, max_eval=args.max_eval)


if __name__ == "__main__":
    main()
