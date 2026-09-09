"""单视频报告的隔离工作目录、完整性校验与原子归档。"""

import json
import os
import tempfile
import zipfile
from typing import Iterable, Optional

from .utils.logger import get_logger

logger = get_logger(__name__)


class ReportArchive:
    """校验全部 ZIP 成员与单视频身份，仅在验证成功后替换正式归档。"""

    CORE_FILENAMES = frozenset({
        "metadata.json",
        "table_lexical_by_partition.csv",
        "table_orthography.csv",
        "table_sentence_function.csv",
        "table_emotion.csv",
        "table_interaction_type.csv",
        "table_consensus_stats.csv",
        "heatmap_data.json",
        "danmaku_raw.csv",
    })

    def __init__(self, zip_path: str, expected_bvid: Optional[str] = None):
        self.zip_path = os.path.abspath(zip_path)
        self.expected_bvid = expected_bvid
        self.temporary_path: Optional[str] = None

    @staticmethod
    def create_workspace(output_dir: str, bvid: str) -> str:
        return tempfile.mkdtemp(prefix=f".analysis-{bvid}-", dir=output_dir)

    def _read_verified(
        self,
        expected_names: Optional[Iterable[str]] = None,
        require_completed: bool = False,
    ) -> dict:
        with zipfile.ZipFile(self.zip_path, "r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if not names or len(set(names)) != len(names):
                raise ValueError("ZIP 为空或包含重复成员名")
            if expected_names is not None and set(names) != set(expected_names):
                raise ValueError("ZIP 成员与报告清单不一致")
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ValueError(f"ZIP 成员 CRC 校验失败: {corrupt}")
            if self.expected_bvid is None:
                return {}
            if not self.expected_bvid:
                raise ValueError("单视频校验缺少期望 BV 号")
            files = {member.filename for member in members if not member.is_dir()}
            missing = self.CORE_FILENAMES - files
            if missing:
                raise ValueError(f"ZIP 缺少核心文件: {', '.join(sorted(missing))}")
            metadata = json.loads(archive.read("metadata.json").decode("utf-8-sig"))
            if not isinstance(metadata, dict) or metadata.get("bvid") != self.expected_bvid:
                raise ValueError(f"ZIP 视频身份不符，期望 {self.expected_bvid}")
            if metadata.get("analysis_sample_count", 1) != 0 and "kappa_ready.csv" not in files:
                raise ValueError("ZIP 缺少核心文件: kappa_ready.csv")
            status = metadata.get("analysis_status", "ok")
            if status not in {"ok", "degraded", "failed"}:
                raise ValueError("ZIP 分析状态无效")
            if require_completed and status == "failed":
                raise ValueError("ZIP 仅保存失败分析档案，不能复用为完成结果")
            return metadata

    def validate(
        self,
        expected_names: Optional[Iterable[str]] = None,
        *,
        require_completed: bool = False,
    ) -> bool:
        try:
            self._read_verified(expected_names, require_completed)
        except Exception as error:
            logger.warning(f"报告归档校验失败: {self.zip_path} - {error}")
            return False
        return True

    def package(self, reports: dict) -> None:
        paths = list(reports.values())
        if not paths or any(not os.path.isfile(path) for path in paths):
            raise ValueError("报告清单包含缺失文件或为空")
        names = [os.path.basename(path) for path in paths]
        if len(set(names)) != len(names):
            raise ValueError("报告清单包含重复文件名")
        if self.zip_path in {os.path.abspath(path) for path in paths}:
            raise ValueError("ZIP 目标不能同时作为源报告")
        descriptor, self.temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(self.zip_path)}.",
            suffix=".tmp",
            dir=os.path.dirname(self.zip_path),
        )
        os.close(descriptor)
        with zipfile.ZipFile(self.temporary_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, name in zip(paths, names):
                archive.write(path, name)
        staged = ReportArchive(self.temporary_path, self.expected_bvid)
        metadata = staged._read_verified(expected_names=names)
        if (
            metadata.get("analysis_status") == "failed"
            and os.path.isfile(self.zip_path)
            and self.validate(require_completed=True)
        ):
            raise ValueError("保留已有成功 ZIP，本次失败分析的中间文件保留")
        os.replace(self.temporary_path, self.zip_path)
        self.temporary_path = None
