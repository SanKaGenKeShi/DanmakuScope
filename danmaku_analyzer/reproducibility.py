"""
可复现 manifest 模块 - 记录完整运行环境供跨环境复核
配置快照仅收录带 reproducible 字段元数据标记的白名单项（凭证/路径/端点一律不入包）
"""

import importlib.metadata
import json
import os
import platform
import re
import sys
from datetime import datetime
from typing import Dict

from . import __version__
from .config import Settings, get_settings
from .llm_config import LLMSettings, get_llm_settings
from .utils.logger import get_logger

logger = get_logger(__name__)

REPRO_MANIFEST_FILENAME = "repro_manifest.json"


class ReproManifestBuilder:
    """可复现 manifest 构建：解释器/平台/依赖版本/白名单配置快照，写出 repro_manifest.json"""

    def build(self) -> Dict:
        """完整运行环境快照；流水线无随机采样组件，LLM 随机性由 config_snapshot 中的温度参数刻画"""
        return {
            "generated_at": datetime.now().isoformat(timespec='seconds'),
            "pipeline_version": __version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "package_versions": self.collect_package_versions(),
            "config_snapshot": self.reproducible_config_snapshot(),
        }

    def write(self, output_dir: str) -> str:
        filepath = os.path.join(output_dir, REPRO_MANIFEST_FILENAME)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.build(), f, ensure_ascii=False, indent=2)
        logger.info(f"可复现 manifest 已保存: {filepath}")
        return filepath

    @staticmethod
    def reproducible_config_snapshot() -> Dict:
        """按字段元数据白名单收集两个配置单例的可复现项；未标记字段默认排除"""
        snapshot: Dict = {}
        for cls, instance in ((Settings, get_settings()), (LLMSettings, get_llm_settings())):
            for name, field_info in cls.model_fields.items():
                extra = field_info.json_schema_extra
                if isinstance(extra, dict) and extra.get("reproducible"):
                    snapshot[name] = getattr(instance, name)
        return snapshot

    @staticmethod
    def collect_package_versions() -> Dict[str, str]:
        """本项目声明依赖的实际安装版本（未安装标 not-installed），含自身版本"""
        versions: Dict[str, str] = {"danmaku-analyzer": __version__}
        try:
            requires = importlib.metadata.requires("danmaku-analyzer") or []
        except importlib.metadata.PackageNotFoundError:
            logger.warning("未找到 danmaku-analyzer 发行元数据，manifest 依赖清单仅含自身版本")
            return versions
        for requirement in requires:
            name = re.split(r"[<>=!;(\[ ]", requirement, maxsplit=1)[0].strip()
            if not name:
                continue
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not-installed"
        return versions
