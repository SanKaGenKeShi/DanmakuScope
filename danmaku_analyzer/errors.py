"""系统异常层次：可预期的边界错误统一基类，供 CLI/TUI 分类捕获。

仅收口真正中止流程的边界错误；无凭证、LLM 未配置等属降级条件（告警后继续），
不在此列为异常，以免违背零丢弃与全失败保留的设计。
"""


class DanmakuScopeError(Exception):
    """基础异常"""


class InputParseError(DanmakuScopeError, ValueError):
    """输入无法解析为 BV/AV/URL；双继承 ValueError 保持既有捕获契约不变"""
