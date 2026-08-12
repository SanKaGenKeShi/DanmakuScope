"""
提示词构建器 - 注入社会语境 + 正字法判断指令
包含 tname、tags、register_hint 及正字法三类分类指令
"""

from typing import List, Optional
from dataclasses import dataclass

from .config import get_settings
from .llm_config import get_llm_settings


@dataclass
class PromptComponents:
    system_prompt: str
    user_prompt: str
    prompt_version: str


# 句类定义单一数据源：复杂任务系统提示词与简单任务提示词共用，变更只需改一处
SENTENCE_FUNCTION_TYPES = (
    ("assertion", "陈述句（陈述事实或观点）"),
    ("question", "疑问句（提出问题）"),
    ("exclamation", "感叹句（表达强烈情感）"),
    ("directive", "祈使句（发出指令或请求）"),
    ("fragment", "省略/片段（不完整的句子）"),
)
SENTENCE_FUNCTION_LABELS = " | ".join(label for label, _ in SENTENCE_FUNCTION_TYPES)


class PromptBuilder:
    
    def __init__(self):
        self.settings = get_settings()
        self.prompt_version = get_llm_settings().PROMPT_VERSION
    
    def build_system_prompt(self, tname: str, tags: List[str]) -> str:
        tag_context = ", ".join(tags[:5]) if tags else "无特定标签"
        
        register_hint = self.settings.REGISTER_HINTS.get(
            tname, 
            self.settings.DEFAULT_REGISTER_HINT
        )
        
        sf_lines = "\n".join(f"   - {label}: {desc}" for label, desc in SENTENCE_FUNCTION_TYPES)
        
        system_prompt = f"""你是一位严谨的社会语言学家。当前分析的弹幕语料出自B站【{tname}】分区。
该分区典型的语言风格特征为：{register_hint}。
该视频的社群标签倾向于：{tag_context}。

【重要】请在分析时充分考虑上述"语域（Register）"和"社群语境"。
同一条文本在不同分区下可能有完全不同的语用含义。

【附加任务：正字法规范性判断】
请基于当前【{tname}】分区的社群语境，区分该弹幕中的非规范书写属于哪一类：
1. standard：标准汉语。
2. community_variant：该分区/社群内公认的梗、谐音、缩写（如"yyds"、"栓Q"、"火钳刘明"），无论是否颠覆传统语法，只要圈内通用即算此类。
3. non_standard_typo：明显不符合任何社群习惯的偶然性笔误（如输入法导致的错字）。

重要准则：宁可误判为 "community_variant"，也不要将网络流行语误判为 "typo"。

【输出格式要求】
请严格按照以下 JSON 格式输出，不要添加任何额外文字：
```json
{{
    "emotion": {{"label": "positive | neutral | negative", "confidence": 0.95}},
    "cooperative_principle": {{"violated": false, "maxim": "quality | quantity | relation | manner"}},
    "interaction_type": {{"label": "check_in | identity_claim | mocking | info_request | expression | other", "confidence": 0.88}},
    "sentence_function": {{"label": "{SENTENCE_FUNCTION_LABELS}", "confidence": 0.92}},
    "orthography": {{
        "status": "standard | community_variant | non_standard_typo",
        "confidence": 0.98
    }}
}}
```

【分析维度说明】
1. emotion（情感倾向）：positive（积极）、neutral（中性）、negative（消极）
2. cooperative_principle（合作原则）：
   - violated: 是否违反合作原则
   - maxim: 违反了哪条准则（quality-质量、quantity-数量、relation-关联、manner-方式）
3. interaction_type（互动诉求类型）：
   - check_in: 打卡/签到
   - identity_claim: 身份认同
   - mocking: 嘲讽/调侃
   - info_request: 信息请求
   - expression: 情感表达
   - other: 其他
4. sentence_function（言语行为/句类）：
{sf_lines}
5. orthography（正字法状态）：
   - standard: 标准汉语
   - community_variant: 社群变体（梗、谐音、缩写等）
   - non_standard_typo: 非标准笔误
"""
        
        return system_prompt
    
    def build_user_prompt(
        self, 
        danmaku_content: str, 
        context_text: Optional[str] = None
    ) -> str:
        parts = []
        
        if context_text:
            parts.append(f"【微语境】\n{context_text}\n")
        
        parts.append(f"【待分析弹幕】\n{danmaku_content}")
        
        return "\n".join(parts)
    
    def build_sentence_function_prompt(self, danmaku_content: str) -> str:
        sf_block = "\n".join(f"- {label}: {desc}" for label, desc in SENTENCE_FUNCTION_TYPES)
        return f"""请判断以下弹幕的言语行为类型（句类）。

【待分析弹幕】
{danmaku_content}

【输出格式要求】
请严格按照以下 JSON 格式输出，不要添加任何额外文字：
```json
{{
    "sentence_function": {{
        "label": "{SENTENCE_FUNCTION_LABELS}",
        "confidence": 0.92
    }}
}}
```

【句类说明】
{sf_block}
"""
    
    def build_complex_prompt(
        self, 
        tname: str, 
        tags: List[str],
        danmaku_content: str,
        context_text: Optional[str] = None
    ) -> PromptComponents:
        system_prompt = self.build_system_prompt(tname, tags)
        user_prompt = self.build_user_prompt(danmaku_content, context_text)
        
        return PromptComponents(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_version=self.prompt_version,
        )

    def build_system_prompt_batch(self, tname: str, tags: List[str]) -> str:
        """段内批量模式系统提示词：社会语境与正字法指令与单条模式一致，输出格式改为按编号的 items 数组"""
        base = self.build_system_prompt(tname, tags)
        marker = "【输出格式要求】"
        head = base.split(marker)[0]
        sf_lines = "\n".join(f"   - {label}: {desc}" for label, desc in SENTENCE_FUNCTION_TYPES)
        return f"""{head}【输出格式要求（批量模式）】
输入为多条编号弹幕，请对每一条独立分析，并将结果按输入编号顺序放入 items 数组。
严格要求：items 元素个数必须与输入条数完全一致，不得遗漏或合并任何一条。
请严格按照以下 JSON 格式输出，不要添加任何额外文字：
```json
{{
    "items": [
        {{
            "emotion": {{"label": "positive | neutral | negative", "confidence": 0.95}},
            "cooperative_principle": {{"violated": false, "maxim": "quality | quantity | relation | manner"}},
            "interaction_type": {{"label": "check_in | identity_claim | mocking | info_request | expression | other", "confidence": 0.88}},
            "sentence_function": {{"label": "{SENTENCE_FUNCTION_LABELS}", "confidence": 0.92}},
            "orthography": {{
                "status": "standard | community_variant | non_standard_typo",
                "confidence": 0.98
            }}
        }}
    ]
}}
```

【分析维度说明】
1. emotion（情感倾向）：positive（积极）、neutral（中性）、negative（消极）
2. cooperative_principle（合作原则）：
   - violated: 是否违反合作原则
   - maxim: 违反了哪条准则（quality-质量、quantity-数量、relation-关联、manner-方式）
3. interaction_type（互动诉求类型）：
   - check_in: 打卡/签到
   - identity_claim: 身份认同
   - mocking: 嘲讽/调侃
   - info_request: 信息请求
   - expression: 情感表达
   - other: 其他
4. sentence_function（言语行为/句类）：
{sf_lines}
5. orthography（正字法状态）：
   - standard: 标准汉语
   - community_variant: 社群变体（梗、谐音、缩写等）
   - non_standard_typo: 非标准笔误
"""

    def build_complex_prompt_batch(
        self,
        tname: str,
        tags: List[str],
        items: List[tuple],
    ) -> PromptComponents:
        """段内批量复杂任务提示词：items 为 (弹幕内容, 微语境文本) 列表"""
        system_prompt = self.build_system_prompt_batch(tname, tags)
        blocks = []
        for idx, (content, context_text) in enumerate(items, 1):
            block = f"【第{idx}条】"
            if context_text:
                block += f"\n【微语境】\n{context_text}"
            block += f"\n【待分析弹幕】\n{content}"
            blocks.append(block)
        user_prompt = (
            f"以下共 {len(items)} 条待分析弹幕，请逐条分析并按编号顺序输出 items 数组：\n\n"
            + "\n\n".join(blocks)
        )
        return PromptComponents(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_version=self.prompt_version,
        )

    def build_simple_prompt(self, danmaku_content: str) -> PromptComponents:
        system_prompt = "你是一位语言学家，专门分析句子的言语行为类型。"
        user_prompt = self.build_sentence_function_prompt(danmaku_content)
        
        return PromptComponents(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            prompt_version=self.prompt_version,
        )

    def build_simple_prompt_batch(self, contents: List[str]) -> PromptComponents:
        """段内批量简单任务提示词：逐条编号输入，按编号输出 sentence_function 数组"""
        sf_block = "\n".join(f"- {label}: {desc}" for label, desc in SENTENCE_FUNCTION_TYPES)
        numbered = "\n".join(f"【第{idx}条】{content}" for idx, content in enumerate(contents, 1))
        user_prompt = f"""请逐条判断以下 {len(contents)} 条弹幕的言语行为类型（句类），每条独立判断，不得遗漏或合并。

{numbered}

【输出格式要求】
请严格按照以下 JSON 格式输出，items 元素个数必须与输入条数完全一致，顺序与编号对应，不要添加任何额外文字：
```json
{{
    "items": [
        {{
            "sentence_function": {{
                "label": "{SENTENCE_FUNCTION_LABELS}",
                "confidence": 0.92
            }}
        }}
    ]
}}
```

【句类说明】
{sf_block}
"""
        return PromptComponents(
            system_prompt="你是一位语言学家，专门分析句子的言语行为类型。",
            user_prompt=user_prompt,
            prompt_version=self.prompt_version,
        )

