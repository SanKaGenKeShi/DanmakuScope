# DanmakuScope

B 站弹幕社会语言学分析命令行工具。采集弹幕及视频元数据，经硬统计与 LLM 软标签双通道分析后，按官方分区（tname）聚合输出交叉统计表，为社会语言学/语料库语言学实证研究提供可溯源、可复核的语料数据。

当前版本：**v0.1.5-beta**

---

## 功能概述

| 能力 | 说明 |
|------|------|
| 弹幕获取 | protobuf 分段接口（主路径）+ XML 兜底，支持 BV/AV/URL 输入 |
| 预处理 | 用户级去重、基于密度的时序动态切分（ruptures PELT） |
| 硬统计 | 词类分布、音节结构、实词密度、正字法变体率（纯正则） |
| LLM 软标签 | 情感、合作原则、互动类型、句类、正字法状态（双路推理 + JSD 共识） |
| 统计推断 | Wilson 置信区间 + 样本量校验，可选 Mann-Whitney U 探索性检验 |
| 分析报告 | 可选调用 LLM 生成社会语言学语料分析报告 |
| 产出打包 | 全部产出自动打包为 `[BV号]视频标题.zip` |

---

## 数据流

```
输入解析 → 爬取（带缓存）→ 社会变量提取 → 用户去重 → 时序切分
    → 硬统计 + LLM 分析（并发池）→ 聚合 → 统计验证 → 报告生成 → ZIP 打包
```

---

## 安装

**前置要求**：Python ≥ 3.10，兼容 OpenAI API 格式的 LLM 端点。

```bash
git clone https://github.com/SanKaGenKeShi/DanmakuScope.git
cd DanmakuScope
pip install -e .
# 开发环境：pip install -e ".[dev]"
```

---

## 配置

```bash
cd danmaku_analyzer
cp .env.example .env
```

**必填项**（LLM 双轨配置）：

| 变量 | 说明 |
|------|------|
| `COMPLEX_LLM_BASE_URL` | 复杂任务（情感/合作原则/互动/正字法）API 地址 |
| `COMPLEX_LLM_API_KEY` | 对应 API Key |
| `COMPLEX_LLM_MODEL` | 模型标识 |
| `SIMPLE_LLM_BASE_URL` | 简单任务（句类）API 地址 |
| `SIMPLE_LLM_API_KEY` | 对应 API Key |
| `SIMPLE_LLM_MODEL` | 模型标识 |

**关键可选项**（完整列表见 `.env.example`）：

| 变量 | 说明 |
|------|------|
| `ENABLE_LLM_ANALYSIS_REPORT` | 启用 LLM 社会语言学分析报告生成 |
| `DATA_ROOT` | 数据输出根目录，默认 `~/.danmaku-scope` |
| `ENABLE_THINKING` | 启用 LLM 思考模式（默认关闭） |

**B 站登录凭证**（推荐扫码登录，无需手动配置）：

```bash
danmaku-analyzer login      # 终端显示二维码，手机扫码后凭证自动保存
danmaku-analyzer account    # 验证凭证有效性
```

凭证保存至 `DATA_ROOT/credential.json`，`analyze`/`batch` 自动加载（优先级：`--credential` 文件 → 登录凭证 → `.env` 中的 `BILIBILI_SESSDATA`）。无有效凭证时仅能获取极少量弹幕。

---

## 使用

```bash
# 单视频分析
danmaku-analyzer analyze BV1xx411c7mD

# 指定输出目录
danmaku-analyzer analyze BV1xx411c7mD -o ./output

# 批量分析
danmaku-analyzer batch BV1xx411c7mD BV1yy411c7mE

# 频次排序采样（默认取每段前 N 条）
danmaku-analyzer analyze BV1xx411c7mD --freq-based --top-n 15

# 跳过缓存强制重新爬取
danmaku-analyzer analyze BV1xx411c7mD --no-cache

# 版本信息 / 当前配置
danmaku-analyzer version
danmaku-analyzer config

# 扫码登录 / 凭证状态
danmaku-analyzer login
danmaku-analyzer account
```

---

## 产出结构

每次分析生成一个 ZIP 包：

```
[BV号]视频标题.zip
├── table_lexical_by_partition.csv      # 词类占比 × 分区
├── table_orthography.csv               # 正字法统计
├── table_sentence_function.csv         # 言语行为分布
├── table_emotion.csv                   # 情感分布
├── table_interaction_type.csv          # 互动类型分布
├── table_consensus_stats.csv           # 共识水平统计
├── heatmap_data.json                   # 热力图数据
├── kappa_ready.csv                     # 编码员间一致性复核用
├── metadata.json                       # 元数据
└── sociolinguistic_analysis_report.md  # LLM 分析报告（可选）
```

---

## 目录结构

```
danmaku_analyzer/
├── cli.py                  # CLI 入口（click + rich）
├── pipeline.py             # 流程编排（阶段式）
├── config.py               # 业务配置中心（pydantic-settings）
├── llm_config.py           # LLM 配置中心
├── account.py              # B 站二维码登录与凭证管理
├── crawler.py              # B 站爬虫（protobuf + XML 兜底）
├── social_variables.py     # 社会变量锚定（tname 提取）
├── user_deduplicator.py    # 用户级去重
├── timeline_segmenter.py   # 时序切分（PELT / 等分）
├── hard_metrics.py         # 硬统计（词类/密度/正字法正则）
├── context_provider.py     # 微语境构建
├── prompt_builder.py       # Prompt 组装（社会语境注入）
├── llm_client.py           # 双路推理 + JSD 共识
├── report_generator.py     # LLM 分析报告生成器
├── aggregator.py           # 嵌套聚合（分区/热区）
├── statistical_validator.py # Wilson CI + 样本量校验
├── reporter.py             # 报告导出 + ZIP 打包
├── cache_manager.py        # 缓存管理（Pickle, 12h TTL）
├── utils/                  # 工具子包（日志/解析器/Token 计数）
└── lexicon/                # 自定义词典 + 报告规范
```

---

## 技术栈

- **CLI**：click, rich
- **配置**：pydantic-settings, python-dotenv
- **硬统计**：jieba（自定义词典）, regex, emoji
- **语义分析**：openai SDK, tenacity, tiktoken
- **统计**：scipy, numpy, ruptures
- **爬虫**：bilibili-api-python, httpx, qrcode（扫码登录）
- **数据处理**：pandas
- **日志**：loguru
- **测试**：pytest

---

## 测试

```bash
pytest
```

---

## License

GPL-3.0
