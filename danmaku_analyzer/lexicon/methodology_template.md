# 方法描述（格式参考模板）

> 本文件仅为 `methodology.md` 的**格式参考**，实际渲染由 `methodology.py` 按运行时配置以
> f-string + 条件拼接生成（条件块如"启用段内批量推理"才输出对应参数段）。请勿手工修改
> 生成产物以外的期望格式，改动本文件不会改变程序输出。

## 1. 数据收集

- 语料来源：哔哩哔哩（bilibili.com）视频弹幕，视频《{标题}》（{BV号}），官方分区「{tname}」。
- 发布时间：{pubdate}；播放量：{view_count}；获取弹幕：{danmaku_count} 条。
- 弹幕接口：protobuf 分段接口 / XML 兜底接口（截断样本告警）。
- 视频标签（前 5）：{tags}。

## 2. 预处理

- 用户级去重：按弹幕 CRC32 用户哈希去重。
- 时序切分：动态（ruptures PELT，密度信号）或固定等分；小段（< MIN_SEGMENT_SAMPLES）自动合并。

## 3. 采样策略

- LLM 标注样本：每段前 N 条 / 按频次排序取唯一弹幕，每段至多 TOP_N 条。
- 【条件】段内批量推理：同段采样弹幕合并为单次请求（请求数 = 段数 × 3）。

## 4. 硬统计（描写层）

- 分词：jieba + 自定义词典 / LLM 辅助分词（长文本）。
- 指标：词类分布、音节结构、实词密度、标点占比、正字法正则变体率。
- 句类判断全部由 LLM 软标签承担（硬性约定）。

## 5. LLM 软标签（语用层）

- Prompt 版本：{PROMPT_VERSION}。
- 复杂任务：{COMPLEX_LLM_MODEL}，双温度路径 + 归一化 JSD 共识判定（阈值 LOW/MEDIUM），低共识降权保留（零丢弃）。
- 简单任务（句类）：{SIMPLE_LLM_MODEL}，单路。
- 正字法三分类与"宁可误判 variant"准则；System Prompt 注入分区与标签语境。

## 6. 统计方法

- Wilson 置信区间（置信水平 {CONFIDENCE_LEVEL}），样本量 < {MIN_SEGMENT_SAMPLES} 标记 insufficient_sample。
- 语料库级：Kruskal-Wallis H + 逐对 Mann-Whitney U + Cliff's delta，未校正 p 值（不实施多重比较校正）。

## 7. 工具版本

- DanmakuScope v{pipeline_version}；运行参数快照见 metadata.json。
