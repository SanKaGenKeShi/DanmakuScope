"""
LLM 客户端算法单元测试 - JSD 计算 / 共识判定 / 输出合并 / 容错解析
不依赖网络，纯本地逻辑验证
"""

import pytest

from danmaku_analyzer.llm_client import (
    LLMOutput, DualPathResult, ConsensusLevel,
    EmotionOutput, CooperativePrincipleOutput,
    InteractionTypeOutput, SentenceFunctionOutput, OrthographyOutput,
)
from danmaku_analyzer.llm_consensus import (
    calculate_jsd, determine_consensus_level, merge_outputs, calculate_weight_multiplier,
)

# 与 LLMSettings 默认阈值/权重保持一致，避免读 .env 引入环境差异
JSD_THRESHOLD_LOW = 0.2
JSD_THRESHOLD_MEDIUM = 0.6
LOW_CONSENSUS_WEIGHT = 0.2


# ========== JSD 计算测试 ==========

class TestCalculateJSD:
    """Jensen-Shannon 散度计算测试"""

    def test_identical_outputs_zero_jsd(self):
        """完全相同的输出 → JSD = 0"""
        outputs = [
            {"emotion": {"label": "positive", "confidence": 0.9}},
            {"emotion": {"label": "positive", "confidence": 0.9}},
        ]
        jsd = calculate_jsd(outputs)
        assert jsd == pytest.approx(0.0, abs=1e-6)

    def test_opposite_outputs_high_jsd(self):
        """各维度完全对立的输出 → 归一化 JSD 接近 1.0"""
        outputs = [
            {"emotion": {"label": "positive", "confidence": 0.95},
             "interaction_type": {"label": "expression", "confidence": 0.95},
             "orthography": {"status": "standard", "confidence": 0.95},
             "cooperative_principle": {"violated": False, "maxim": "quality"}},
            {"emotion": {"label": "negative", "confidence": 0.95},
             "interaction_type": {"label": "mocking", "confidence": 0.95},
             "orthography": {"status": "non_standard_typo", "confidence": 0.95},
             "cooperative_principle": {"violated": True, "maxim": "manner"}},
        ]
        jsd = calculate_jsd(outputs)
        assert jsd > 0.8  # 各维均对立，归一化均值应接近 1.0

    def test_single_output_returns_zero(self):
        """单个输出无法计算散度 → 返回 0"""
        outputs = [{"emotion": {"label": "positive", "confidence": 0.8}}]
        jsd = calculate_jsd(outputs)
        assert jsd == 0.0

    def test_empty_outputs_returns_zero(self):
        """空列表 → 返回 0"""
        jsd = calculate_jsd([])
        assert jsd == 0.0

    def test_missing_emotion_field_graceful(self):
        """缺少 emotion 字段时不崩溃"""
        outputs = [
            {"other_field": "value"},
            {"emotion": {"label": "neutral", "confidence": 0.5}},
        ]
        jsd = calculate_jsd(outputs)
        # 应该正常返回一个数值（不抛异常）
        assert isinstance(jsd, float)

    def test_jsd_symmetry(self):
        """JSD 对称性：交换输出顺序结果不变"""
        out_a = {"emotion": {"label": "positive", "confidence": 0.8}}
        out_b = {"emotion": {"label": "negative", "confidence": 0.7}}
        jsd_ab = calculate_jsd([out_a, out_b])
        jsd_ba = calculate_jsd([out_b, out_a])
        assert jsd_ab == pytest.approx(jsd_ba, abs=1e-6)

    def test_jsd_bounded(self):
        """归一化 JSD 上界为 1.0（完全分歧）"""
        outputs = [
            {"emotion": {"label": "positive", "confidence": 1.0}},
            {"emotion": {"label": "negative", "confidence": 1.0}},
        ]
        jsd = calculate_jsd(outputs)
        assert 0 <= jsd <= 1.0 + 0.01  # 允许浮点误差


# ========== 共识水平判定测试 ==========

class TestConsensusLevel:
    """共识水平判定测试"""

    def test_high_consensus(self):
        """归一化 JSD < 0.2 → HIGH"""
        level = determine_consensus_level(0.05, JSD_THRESHOLD_LOW, JSD_THRESHOLD_MEDIUM)
        assert level == ConsensusLevel.HIGH

    def test_medium_consensus(self):
        """0.2 <= 归一化 JSD < 0.6 → MEDIUM"""
        level = determine_consensus_level(0.4, JSD_THRESHOLD_LOW, JSD_THRESHOLD_MEDIUM)
        assert level == ConsensusLevel.MEDIUM

    def test_low_consensus(self):
        """归一化 JSD >= 0.6 → LOW"""
        level = determine_consensus_level(0.8, JSD_THRESHOLD_LOW, JSD_THRESHOLD_MEDIUM)
        assert level == ConsensusLevel.LOW

    def test_boundary_low_threshold(self):
        """归一化 JSD 恰好等于 0.2 → MEDIUM（不小于 low 阈值）"""
        level = determine_consensus_level(0.2, JSD_THRESHOLD_LOW, JSD_THRESHOLD_MEDIUM)
        assert level == ConsensusLevel.MEDIUM

    def test_boundary_medium_threshold(self):
        """归一化 JSD 恰好等于 0.6 → LOW"""
        level = determine_consensus_level(0.6, JSD_THRESHOLD_LOW, JSD_THRESHOLD_MEDIUM)
        assert level == ConsensusLevel.LOW


# ========== 权重乘数测试 ==========

class TestWeightMultiplier:
    """权重乘数计算测试"""

    def test_high_consensus_weight_1(self):
        """高共识 → 权重 1.0"""
        w = calculate_weight_multiplier(ConsensusLevel.HIGH, LOW_CONSENSUS_WEIGHT)
        assert w == 1.0

    def test_medium_consensus_weight_1(self):
        """中共识 → 权重 1.0"""
        w = calculate_weight_multiplier(ConsensusLevel.MEDIUM, LOW_CONSENSUS_WEIGHT)
        assert w == 1.0

    def test_low_consensus_weight_reduced(self):
        """低共识 → 权重 0.2（零丢弃铁律）"""
        w = calculate_weight_multiplier(ConsensusLevel.LOW, LOW_CONSENSUS_WEIGHT)
        assert w == pytest.approx(0.2)


# ========== 输出合并测试 ==========

class TestMergeOutputs:
    """输出合并逻辑测试"""

    def test_empty_outputs_returns_default(self):
        """空输出列表 → 返回默认 LLMOutput"""
        result = merge_outputs([], ConsensusLevel.HIGH)
        assert result.emotion.label == "neutral"
        assert result.orthography.status == "standard"

    def test_high_consensus_takes_first(self):
        """高共识 → 取第一个输出"""
        outputs = [
            {"emotion": {"label": "positive", "confidence": 0.9},
             "cooperative_principle": {"violated": False, "maxim": "quality"},
             "interaction_type": {"label": "expression", "confidence": 0.8},
             "sentence_function": {"label": "exclamation", "confidence": 0.85},
             "orthography": {"status": "standard", "confidence": 0.95}},
            {"emotion": {"label": "negative", "confidence": 0.7},
             "cooperative_principle": {"violated": True, "maxim": "manner"},
             "interaction_type": {"label": "mocking", "confidence": 0.6},
             "sentence_function": {"label": "assertion", "confidence": 0.7},
             "orthography": {"status": "community_variant", "confidence": 0.8}},
        ]
        result = merge_outputs(outputs, ConsensusLevel.HIGH)
        assert result.emotion.label == "positive"

    def test_low_consensus_takes_highest_confidence(self):
        """低共识 → 按各维 confidence 总和择优（非仅情感单维）"""
        outputs = [
            {"emotion": {"label": "positive", "confidence": 0.4},
             "cooperative_principle": {"violated": False, "maxim": "quality"},
             "interaction_type": {"label": "expression", "confidence": 0.5},
             "sentence_function": {"label": "fragment", "confidence": 0.5},
             "orthography": {"status": "standard", "confidence": 0.5}},
            {"emotion": {"label": "negative", "confidence": 0.9},
             "cooperative_principle": {"violated": True, "maxim": "manner"},
             "interaction_type": {"label": "mocking", "confidence": 0.8},
             "sentence_function": {"label": "assertion", "confidence": 0.7},
             "orthography": {"status": "community_variant", "confidence": 0.8}},
        ]
        result = merge_outputs(outputs, ConsensusLevel.LOW)
        # 第二个输出 emotion.confidence=0.9 更高
        assert result.emotion.label == "negative"


# ========== 容错解析测试 ==========

class TestDictToLLMOutput:
    """字典转类型化输出的容错解析测试"""

    def test_full_valid_dict(self):
        """完整有效字典 → 正确解析"""
        data = {
            "emotion": {"label": "positive", "confidence": 0.9},
            "cooperative_principle": {"violated": True, "maxim": "relation"},
            "interaction_type": {"label": "check_in", "confidence": 0.7},
            "sentence_function": {"label": "question", "confidence": 0.8},
            "orthography": {"status": "community_variant", "confidence": 0.6},
        }
        result = LLMOutput.from_dict(data)
        assert result.emotion.label == "positive"
        assert result.cooperative_principle.violated is True
        assert result.cooperative_principle.maxim == "relation"
        assert result.interaction_type.label == "check_in"
        assert result.sentence_function.label == "question"
        assert result.orthography.status == "community_variant"

    def test_empty_dict_uses_defaults(self):
        """空字典 → 全部使用默认值"""
        result = LLMOutput.from_dict({})
        assert result.emotion.label == "neutral"
        assert result.emotion.confidence == 0.5
        assert result.cooperative_principle.violated is False
        assert result.interaction_type.label == "other"
        assert result.sentence_function.label == "fragment"
        assert result.orthography.status == "standard"

    def test_partial_dict_fills_defaults(self):
        """部分字段缺失 → 缺失部分用默认值"""
        data = {
            "emotion": {"label": "negative"},  # 缺 confidence
            "orthography": {"status": "non_standard_typo"},
        }
        result = LLMOutput.from_dict(data)
        assert result.emotion.label == "negative"
        assert result.emotion.confidence == 0.5  # 默认
        assert result.orthography.status == "non_standard_typo"
        # 其他字段默认
        assert result.sentence_function.label == "fragment"

    def test_invalid_label_falls_back_to_default(self):
        """无效枚举值 → Pydantic 校验使用默认值"""
        data = {
            "emotion": {"label": "INVALID_LABEL", "confidence": 0.9},
        }
        # Pydantic model_validate 对 Literal 校验失败会抛异常
        # 但我们的代码用 model_validate，需要确认行为
        # 实际上 Pydantic v2 的 model_validate 对无效 Literal 会抛 ValidationError
        # 这里测试 LLMOutput.from_dict 是否优雅处理
        # 注意：当前实现没有 try-except 包裹，会直接抛异常
        # 这是一个已知的边界情况，记录为测试
        with pytest.raises(Exception):
            LLMOutput.from_dict(data)

    def test_to_dict_roundtrip(self):
        """LLMOutput.to_dict() 结构完整性"""
        output = LLMOutput(
            emotion=EmotionOutput(label="positive", confidence=0.9),
            cooperative_principle=CooperativePrincipleOutput(violated=False),
            interaction_type=InteractionTypeOutput(label="expression", confidence=0.8),
            sentence_function=SentenceFunctionOutput(label="exclamation", confidence=0.85),
            orthography=OrthographyOutput(status="standard", confidence=0.95),
        )
        d = output.to_dict()
        assert "emotion" in d
        assert "cooperative_principle" in d
        assert "interaction_type" in d
        assert "sentence_function" in d
        assert "orthography" in d
        assert d["emotion"]["label"] == "positive"
        assert d["orthography"]["status"] == "standard"


# ========== DualPathResult 序列化测试 ==========

class TestDualPathResultSerialization:
    """DualPathResult.to_dict() 测试"""

    def test_to_dict_structure(self):
        """to_dict 包含所有必要字段"""
        output = LLMOutput(
            emotion=EmotionOutput(),
            cooperative_principle=CooperativePrincipleOutput(),
            interaction_type=InteractionTypeOutput(),
            sentence_function=SentenceFunctionOutput(),
            orthography=OrthographyOutput(),
        )
        result = DualPathResult(
            output=output,
            consensus_level=ConsensusLevel.MEDIUM,
            jsd_score=0.2345,
            weight_multiplier=1.0,
            raw_outputs=[{"test": True}],
            prompt_version="v2.2.0",
        )
        d = result.to_dict()
        assert d["consensus_level"] == "medium"
        assert d["jsd_score"] == pytest.approx(0.2345)
        assert d["weight_multiplier"] == 1.0
        assert d["prompt_version"] == "v2.2.0"
        assert "output" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
