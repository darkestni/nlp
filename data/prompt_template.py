"""
prompt_template.py
==================
定义讽刺检测任务的 prompt schema。

核心设计：让模型输出「先推理、再判断」的结构化回复。
这是本项目的灵魂——把讽刺检测从"纯分类"升级为"reasoning-heavy"任务，
也正是 GRPO 要优化的对象。
"""

# 系统提示：告诉模型它的角色和输出格式
SYSTEM_PROMPT = """你是讽刺语言分析助手。对用户给出的文本，先进行结构化推理，再给出最终判断。

你必须严格按以下格式输出：

<reasoning>
逐条列出判断线索，例如：夸张、反语、语境矛盾、常识违背、语气与字面意思冲突等。
</reasoning>
<answer>
sarcastic 或 not_sarcastic
</answer>

规则：
- <reasoning> 中至少给出 1 条线索，最多 4 条，每条不超过 1 句话。
- <answer> 只能是 sarcastic 或 not_sarcastic 之一，不得有其他字符。
- 推理与答案必须自洽：推理倾向讽刺，答案就该是 sarcastic。"""

# 用户提示模板
USER_PROMPT_TEMPLATE = "请分析下面这条文本是否为讽刺：\n\n文本：{text}\n\n给出你的推理与判断。"


def build_user_prompt(text: str) -> str:
    """构造用户侧 prompt。"""
    return USER_PROMPT_TEMPLATE.format(text=text.strip())


def build_messages(text: str) -> list:
    """构造 chat 格式的 messages（Qwen-Instruct 系列）。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text)},
    ]


# ---- 期望答案的标签映射 ----
# 不同数据集的标签写法不一，统一映射到 schema 词
LABEL_MAP = {
    # iSarcasm / 通用
    "sarcastic": "sarcastic",
    "sarcasm": "sarcastic",
    "irony": "sarcastic",
    "1": "sarcastic",
    1: "sarcastic",
    True: "sarcastic",
    # 非讽刺
    "not_sarcastic": "not_sarcastic",
    "not-sarcastic": "not_sarcastic",
    "literal": "not_sarcastic",
    "regular": "not_sarcastic",
    "0": "not_sarcastic",
    0: "not_sarcastic",
    False: "not_sarcastic",
}


def normalize_label(label) -> str:
    """把数据集里五花八门的标签统一成 sarcastic / not_sarcastic。"""
    if isinstance(label, str):
        key = label.strip().lower()
    else:
        key = label
    return LABEL_MAP.get(key, "not_sarcastic")


# ---- 标准答案字符串（用于奖励函数比对）----
def gold_answer(label) -> str:
    """构造 <answer> 标签内的标准答案文本。"""
    return normalize_label(label)
