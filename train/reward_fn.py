"""
reward_fn.py
============
GRPO 的多维奖励函数 —— 本项目的核心创新。

设计理念：
    传统 RLHF 只用"最终答案对错"作奖励，模型会学到"蒙答案"。
    本项目设计多维奖励，同时优化：
        1. 答案正确性（硬指标）
        2. 输出格式规范（软指标，保证 schema 可解析）
        3. 推理链实质内容（鼓励真推理，而非空壳）
        4. 推理链简洁性（惩罚冗余刷分）
        5. 推理-答案自洽性（防止"推理说讽刺、答案答非讽刺"）

每条样本的最终 reward = Σ (单项分 × 权重)，权重见 configs/default.yaml。

这套设计对应 JD 中"agentic / reasoning / coding 类任务的评测与奖励设计"职责。
"""
import re
from dataclasses import dataclass


# ---- 期望答案词 ----
POS = "sarcastic"
NEG = "not_sarcastic"


# ============================================================
# 解析模型输出
# ============================================================
@dataclass
class ParsedOutput:
    """模型输出的解析结果。"""
    raw: str
    has_reasoning: bool
    has_answer: bool
    reasoning_text: str
    answer_text: str          # 原始答案字符串
    answer_norm: str          # 规范化后的 sarcastic / not_sarcastic / invalid
    reasoning_lines: int      # 推理线索条数


def parse_output(text: str) -> ParsedOutput:
    """解析 <reasoning>...</reasoning><answer>...</answer> 结构。"""
    raw = text or ""

    # 抓 <reasoning>...</reasoning>
    m_reason = re.search(r"<reasoning>(.*?)</reasoning>", raw, re.DOTALL | re.IGNORECASE)
    reasoning_text = m_reason.group(1).strip() if m_reason else ""
    has_reasoning = bool(reasoning_text)

    # 抓 <answer>...</answer>
    m_answer = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL | re.IGNORECASE)
    answer_text = m_answer.group(1).strip() if m_answer else ""
    has_answer = bool(answer_text)

    # 规范化答案
    ans_low = answer_text.lower().replace("-", "_").strip()
    if ans_low == POS:
        answer_norm = POS
    elif ans_low == NEG:
        answer_norm = NEG
    else:
        answer_norm = "invalid"

    # 推理线索条数（按换行 / 分号 / 序号粗略计数）
    lines = [l for l in re.split(r"[\n;；]", reasoning_text) if l.strip()]
    # 去掉"线索："这类前缀后的空行
    reasoning_lines = len([l for l in lines if len(l.strip()) > 3])

    return ParsedOutput(
        raw=raw,
        has_reasoning=has_reasoning,
        has_answer=has_answer,
        reasoning_text=reasoning_text,
        answer_text=answer_text,
        answer_norm=answer_norm,
        reasoning_lines=reasoning_lines,
    )


# ============================================================
# 单项奖励
# ============================================================
def reward_answer_correct(parsed: ParsedOutput, gold: str) -> float:
    """答案正确性：+1 / -1 / 0(invalid)。"""
    if parsed.answer_norm == "invalid":
        return 0.0
    return 1.0 if parsed.answer_norm == gold else -1.0


def reward_format_valid(parsed: ParsedOutput) -> float:
    """格式规范：同时有合法 reasoning 和 answer 才给分。"""
    if parsed.has_reasoning and parsed.has_answer and parsed.answer_norm != "invalid":
        return 1.0
    return 0.0


def reward_reasoning_present(parsed: ParsedOutput) -> float:
    """推理链实质内容：至少 1 条线索给部分分，2~4 条满分，>4 不再加。"""
    n = parsed.reasoning_lines
    if n == 0:
        return 0.0
    if n == 1:
        return 0.5
    if 2 <= n <= 4:
        return 1.0
    return 1.0  # >4 由冗余惩罚项处理


def reward_reasoning_redundant(parsed: ParsedOutput) -> float:
    """冗余惩罚：线索过多（>4）扣分，鼓励简洁。返回负值。"""
    n = parsed.reasoning_lines
    if n <= 4:
        return 0.0
    return -min(1.0, (n - 4) * 0.25)  # 每多一条扣 0.25，最多 -1


def reward_contradiction(parsed: ParsedOutput, gold: str) -> float:
    """
    自洽性惩罚：检测推理倾向与答案是否矛盾。
    启发式：若推理文本里出现强讽刺词（"讽刺/反语/夸张/言不由衷"）但答案为 not，
    或推理全是中性/肯定词但答案为 sarcastic，判为矛盾，返回负值。
    """
    if parsed.answer_norm == "invalid" or not parsed.reasoning_text:
        return 0.0

    sarcasm_cues = ["讽刺", "反语", "反讽", "言不由衷", "说反话", "夸张", "sarcast", "irony"]
    literal_cues = ["字面", "直接", "literal", "如实", "陈述", "正常表达"]

    reasoning_low = parsed.reasoning_text.lower()
    cues_sarc = sum(1 for c in sarcasm_cues if c in reasoning_low)
    cues_literal = sum(1 for c in literal_cues if c in reasoning_low)

    # 推理明显倾向讽刺，但答 not
    if cues_sarc >= 1 and parsed.answer_norm == NEG and cues_literal == 0:
        return -1.0
    # 推理明显倾向字面，但答 sarcastic
    if cues_literal >= 1 and parsed.answer_norm == POS and cues_sarc == 0:
        return -1.0
    return 0.0


# ============================================================
# 组合：最终 reward
# ============================================================
DEFAULT_WEIGHTS = {
    "answer_correct": 1.0,
    "format_valid": 0.3,
    "reasoning_present": 0.3,
    "reasoning_redundant": -0.1,   # 注意：这本身就是惩罚方向，权重再乘一次
    "contradiction_penalty": -0.5,
}


def compute_reward(completion: str, gold: str, weights: dict = None) -> tuple:
    """
    计算单条样本的最终 reward。

    Args:
        completion: 模型生成的文本
        gold: 标准答案（sarcastic / not_sarcastic）
        weights: 各项权重，默认 DEFAULT_WEIGHTS

    Returns:
        (total_reward, breakdown_dict)  breakdown 便于训练时 log 细项
    """
    w = weights or DEFAULT_WEIGHTS
    parsed = parse_output(completion)

    items = {
        "answer_correct": reward_answer_correct(parsed, gold),
        "format_valid": reward_format_valid(parsed),
        "reasoning_present": reward_reasoning_present(parsed),
        "reasoning_redundant": reward_reasoning_redundant(parsed),
        "contradiction_penalty": reward_contradiction(parsed, gold),
    }

    total = sum(items[k] * w[k] for k in w)
    return total, items


# ============================================================
# TRL GRPOTrainer 期望的接口
# ============================================================
# TRL 的 GRPOTrainer 接受 reward_funcs: List[Callable[[prompts, completions, **kwargs], List[float]]]
# 下面提供两个工厂函数，适配 TRL 的签名。

def make_reward_funcs(weights: dict = None):
    """
    返回一个 reward 函数列表，供 GRPOTrainer 使用。
    TRL 会对每个函数分别计算并把结果相加。
    这里我们把多维奖励封装成 5 个独立函数。
    """
    w = weights or DEFAULT_WEIGHTS

    def _wrap(key, fn):
        def _f(prompts, completions, **kw):
            # TRL 传入 completions 是 List[str]，gold 在 dataset 的 "label" 列
            golds = kw.get("label") or kw.get("labels")
            if golds is None:
                # 没拿到 gold，无法算答案相关奖励，退化为 0
                return [0.0] * len(completions)
            out = []
            for comp, gold in zip(completions, golds):
                parsed = parse_output(comp)
                out.append(fn(parsed, gold) * w[key])
            return out
        return _f

    return [
        ("answer_correct", _wrap("answer_correct", reward_answer_correct)),
        ("format_valid", _wrap("format_valid", lambda p, g: reward_format_valid(p))),
        ("reasoning_present", _wrap("reasoning_present", lambda p, g: reward_reasoning_present(p))),
        ("reasoning_redundant", _wrap("reasoning_redundant", lambda p, g: reward_reasoning_redundant(p))),
        ("contradiction_penalty", _wrap("contradiction_penalty", reward_contradiction)),
    ]


# ============================================================
# 自测（python train/reward_fn.py 可直接跑）
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("reward_fn 自测")
    print("=" * 60)

    cases = [
        # (描述, 模型输出, gold, 期望倾向)
        ("完美：推理+正确答案", 
         "<reasoning>\n1. 字面说\"快乐\"但语境是周一早起\n2. \"幸福得要哭\"明显夸张\n</reasoning>\n<answer>\nsarcastic\n</answer>",
         "sarcastic", "高分"),
        ("格式错：缺 answer 标签",
         "<reasoning>\n这条明显是讽刺\n</reasoning>",
         "sarcastic", "低分"),
        ("答案错",
         "<reasoning>\n1. 正常陈述\n</reasoning>\n<answer>\nnot_sarcastic\n</answer>",
         "sarcastic", "低分"),
        ("矛盾：推理说讽刺，答案答 not",
         "<reasoning>\n明显是反语讽刺，言不由衷\n</reasoning>\n<answer>\nnot_sarcastic\n</answer>",
         "sarcastic", "低分（矛盾惩罚）"),
        ("冗余：推理 8 条",
         "<reasoning>\n" + "\n".join(f"{i}. 线索{i}" for i in range(1, 9)) + "\n</reasoning>\n<answer>\nsarcastic\n</answer>",
         "sarcastic", "中分（冗余扣分）"),
    ]

    for desc, comp, gold, expect in cases:
        total, items = compute_reward(comp, gold)
        print(f"\n【{desc}】 gold={gold} 预期={expect}")
        print(f"  reward = {total:+.3f}")
        for k, v in items.items():
            print(f"    {k:24s}: {v:+.2f} × {DEFAULT_WEIGHTS[k]:+.2f}")
