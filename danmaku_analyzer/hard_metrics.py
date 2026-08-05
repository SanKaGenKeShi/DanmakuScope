"""
硬统计模块 - 纯硬统计分析（无句类判断，无硬错别字匹配）
包含词类、密度、变体正则等统计指标
支持 jieba+HMM 和 LLM 辅助分词两种模式
"""

import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import jieba
import jieba.posseg as pseg
import emoji
import regex

from .config import get_settings
from .utils.logger import get_logger

logger = get_logger(__name__)

_POS_ALIASES = {
    "noun": "n", "verb": "v", "adjective": "a", "adj": "a",
    "adverb": "d", "adv": "d", "pronoun": "r", "auxiliary": "v",
    "preposition": "p", "prep": "p", "conjunction": "c", "conj": "c",
    "particle": "u", "interjection": "e", "modal": "y",
    "onomatopoeia": "o", "proper_noun": "nz", "proper noun": "nz",
}


def _normalize_pos(pos: str) -> str:
    """LLM 返回英文词性标签时归一化为 jieba 单字母风格，未知标签原样保留"""
    cleaned = pos.strip()
    return _POS_ALIASES.get(cleaned.lower(), cleaned)


@dataclass
class HardMetricsResult:
    pos_distribution: Dict[str, float]  # 词性 -> 占比
    syllable_distribution: Dict[str, float]  # 单音节/双音节/三+音节 -> 占比
    avg_word_length: float  # 总字符数（去除标点）/ 总词数
    content_word_density: float  # (名词数 + 动词数 + 形容词数) / 总词数
    punctuation_emoji_rate: float  # 包含 !?~ 或 Emoji 的弹幕条数 / 总弹幕条数
    orthography_hard_metrics: Dict[str, float]  # 每千字中各类变体出现次数
    total_danmaku_count: int
    total_word_count: int
    total_char_count: int


class HardMetricsAnalyzer:
    
    def __init__(self):
        self.settings = get_settings()
        self._load_lexicons()
        
        self.enable_llm_tokenizer = self.settings.ENABLE_LLM_TOKENIZER
        self.llm_tokenizer_min_length = self.settings.LLM_TOKENIZER_MIN_LENGTH
        self.llm_tokenizer_concurrency = self.settings.LLM_TOKENIZER_CONCURRENCY
        
        self.llm_client = None
        if self.enable_llm_tokenizer:
            from .llm_config import get_llm_settings
            from .llm_factory import simple_sync_client
            llm_cfg = get_llm_settings()
            self.llm_client = simple_sync_client(timeout=30.0)
            self.llm_model = llm_cfg.SIMPLE_LLM_MODEL
            self.enable_thinking = llm_cfg.ENABLE_THINKING
            logger.info(f"LLM 分词已启用，模型: {self.llm_model}，最小触发长度: {self.llm_tokenizer_min_length}")
    
    def _load_lexicons(self):
        lexicon_dir = self.settings.LEXICON_DIR
        
        if not os.path.exists(lexicon_dir):
            logger.warning(f"词典目录不存在: {lexicon_dir}")
            return
        
        for filename in os.listdir(lexicon_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(lexicon_dir, filename)
                try:
                    jieba.load_userdict(filepath)
                    logger.info(f"加载词典: {filename}")
                except Exception as e:
                    logger.error(f"加载词典失败 {filename}: {e}")
    
    def _tokenize_batch(self, danmaku_list: List[str]) -> List[List[Tuple[str, str]]]:
        """批量分词：长文本并发走 LLM 分词（单条失败各自回退 jieba），其余直接 jieba"""
        results: List[Optional[List[Tuple[str, str]]]] = [None] * len(danmaku_list)
        llm_indices = []
        for i, text in enumerate(danmaku_list):
            if (self.enable_llm_tokenizer and
                self.llm_client is not None and
                len(text) >= self.llm_tokenizer_min_length):
                llm_indices.append(i)
            else:
                results[i] = list(pseg.cut(text))
        
        if llm_indices:
            texts = [danmaku_list[i] for i in llm_indices]
            max_workers = min(self.llm_tokenizer_concurrency, len(texts))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                llm_results = list(pool.map(self._llm_tokenize_safe, texts))
            for i, tokens in zip(llm_indices, llm_results):
                results[i] = tokens
        
        return results  # type: ignore[return-value]
    
    def _llm_tokenize_safe(self, text: str) -> List[Tuple[str, str]]:
        try:
            return self._llm_tokenize(text)
        except Exception as e:
            logger.warning(f"LLM 分词失败，回退到 jieba: {e}")
            return list(pseg.cut(text))
    
    def _llm_tokenize(self, text: str) -> List[Tuple[str, str]]:
        prompt = f"""请对以下中文文本进行分词和词性标注。

词性标注规范：
- n: 名词
- v: 动词
- a: 形容词
- d: 副词
- r: 代词
- p: 介词
- c: 连词
- u: 助词
- e: 叹词
- y: 语气词
- o: 拟声词
- nz: 其他专名（网络用语、梗等）

请以JSON数组格式返回，每个元素为 [词, 词性]。
只返回JSON，不要其他文字。

文本：{text}"""
        
        try:
            kwargs = {
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "extra_body": {"enable_thinking": self.enable_thinking},
            }
            response = self.llm_client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith('```'):
                content = content.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
            
            result = json.loads(content)
            
            if isinstance(result, list) and all(isinstance(item, list) and len(item) == 2 for item in result):
                return [(str(word), _normalize_pos(str(pos))) for word, pos in result]
            else:
                raise ValueError("返回格式不正确")
                
        except Exception as e:
            logger.warning(f"LLM 分词解析失败: {e}")
            raise
    
    def analyze(self, danmaku_list: List[str]) -> HardMetricsResult:
        if not danmaku_list:
            return self._empty_result()
        
        logger.info(f"开始硬统计分析，共 {len(danmaku_list)} 条弹幕")
        
        pos_counter = Counter()
        syllable_counter = Counter()
        total_words = 0
        total_chars = 0
        punctuation_emoji_count = 0
        
        uppercase_abbr_count = 0
        number_symbol_count = 0
        emoticon_count = 0
        
        uppercase_pattern = regex.compile(r'[A-Z]{2,}')
        number_pattern = regex.compile(r'\d{2,}')
        emoticon_pattern = regex.compile(r'[（(][\u4e00-\u9fff\w\s・ω･｡]+[）)]')
        punctuation_pattern = regex.compile(r'[!?~！？～]')
        
        tokenized_list = self._tokenize_batch(danmaku_list)
        
        for danmaku, words in zip(danmaku_list, tokenized_list):
            has_punctuation = bool(punctuation_pattern.search(danmaku))
            has_emoji = bool(emoji.emoji_count(danmaku) > 0)
            if has_punctuation or has_emoji:
                punctuation_emoji_count += 1
            
            for word, pos in words:
                if not word.strip():
                    continue
                
                pos_counter[pos] += 1
                total_words += 1
                
                word_len = len(word.strip())
                if word_len == 1:
                    syllable_counter['单音节'] += 1
                elif word_len == 2:
                    syllable_counter['双音节'] += 1
                else:
                    syllable_counter['三+音节'] += 1
                
                clean_word = regex.sub(r'[^\w\s]', '', word)
                total_chars += len(clean_word)
            
            # 正字法变体统计（基于整个弹幕）
            uppercase_abbr_count += len(uppercase_pattern.findall(danmaku))
            number_symbol_count += len(number_pattern.findall(danmaku))
            emoticon_count += len(emoticon_pattern.findall(danmaku))
        
        total_danmaku = len(danmaku_list)
        
        pos_distribution = {}
        for pos, count in pos_counter.items():
            pos_distribution[pos] = count / total_words if total_words > 0 else 0.0
        
        syllable_distribution = {}
        for syllable_type, count in syllable_counter.items():
            syllable_distribution[syllable_type] = count / total_words if total_words > 0 else 0.0
        
        avg_word_length = total_chars / total_words if total_words > 0 else 0.0
        
        content_word_count = 0
        for pos, count in pos_counter.items():
            if pos.startswith('n') or pos.startswith('v') or pos.startswith('a'):
                content_word_count += count
        content_word_density = content_word_count / total_words if total_words > 0 else 0.0
        
        punctuation_emoji_rate = punctuation_emoji_count / total_danmaku if total_danmaku > 0 else 0.0
        
        orthography_hard_metrics = {}
        if total_chars > 0:
            orthography_hard_metrics['uppercase_abbr_per_1000'] = (uppercase_abbr_count / total_chars) * 1000
            orthography_hard_metrics['number_symbol_per_1000'] = (number_symbol_count / total_chars) * 1000
            orthography_hard_metrics['emoticon_per_1000'] = (emoticon_count / total_chars) * 1000
        else:
            orthography_hard_metrics['uppercase_abbr_per_1000'] = 0.0
            orthography_hard_metrics['number_symbol_per_1000'] = 0.0
            orthography_hard_metrics['emoticon_per_1000'] = 0.0
        
        result = HardMetricsResult(
            pos_distribution=pos_distribution,
            syllable_distribution=syllable_distribution,
            avg_word_length=avg_word_length,
            content_word_density=content_word_density,
            punctuation_emoji_rate=punctuation_emoji_rate,
            orthography_hard_metrics=orthography_hard_metrics,
            total_danmaku_count=total_danmaku,
            total_word_count=total_words,
            total_char_count=total_chars,
        )
        
        logger.info(f"硬统计分析完成：{total_danmaku} 条弹幕，{total_words} 个词")
        return result
    
    def _empty_result(self) -> HardMetricsResult:
        return HardMetricsResult(
            pos_distribution={},
            syllable_distribution={},
            avg_word_length=0.0,
            content_word_density=0.0,
            punctuation_emoji_rate=0.0,
            orthography_hard_metrics={
                'uppercase_abbr_per_1000': 0.0,
                'number_symbol_per_1000': 0.0,
                'emoticon_per_1000': 0.0,
            },
            total_danmaku_count=0,
            total_word_count=0,
            total_char_count=0,
        )

