"""
缓存管理器模块 - 管理数据缓存
"""

import os
import hashlib
import pickle
from typing import Any, Optional
from datetime import datetime, timedelta

from .config import get_settings
from .utils.logger import get_logger

logger = get_logger(__name__)

# 缓存 schema 版本：Pydantic 模型字段变动（增/删/改名）时必须 +1，
# 旧版本缓存会在读取时被丢弃并重新拉取，避免 pickle 反序列化静默出错
CACHE_SCHEMA_VERSION = 1


class CacheManager:
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.settings = get_settings()
        raw_cache_dir = cache_dir or self.settings.CACHE_DIR
        self.cache_dir = self.settings.resolve_data_path(raw_cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> str:
        cache_key = self._get_cache_key(key)
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def get(self, key: str, max_age_hours: int = 24) -> Optional[Any]:
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return None
        
        file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - file_mtime > timedelta(hours=max_age_hours):
            logger.info(f"缓存已过期: {key}")
            self.delete(key)
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            if not isinstance(data, dict) or data.get("schema_version") != CACHE_SCHEMA_VERSION:
                logger.info(f"缓存 schema 版本不匹配，丢弃: {key}")
                self.delete(key)
                return None
            logger.debug(f"缓存命中: {key}")
            return data["payload"]
        except Exception as e:
            logger.error(f"读取缓存失败: {e}")
            return None
    
    def set(self, key: str, value: Any) -> bool:
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({"schema_version": CACHE_SCHEMA_VERSION, "payload": value}, f)
            logger.debug(f"缓存设置成功: {key}")
            return True
        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return True
        
        try:
            os.remove(cache_path)
            logger.debug(f"缓存删除成功: {key}")
            return True
        except Exception as e:
            logger.error(f"删除缓存失败: {e}")
            return False
    
    def clear(self, max_age_hours: Optional[int] = None) -> int:
        cleared_count = 0
        
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            
            if not os.path.isfile(filepath):
                continue
            
            if max_age_hours is not None:
                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                if datetime.now() - file_mtime <= timedelta(hours=max_age_hours):
                    continue
            
            try:
                os.remove(filepath)
                cleared_count += 1
            except Exception as e:
                logger.error(f"清理缓存失败: {filepath} - {e}")
        
        logger.info(f"清理缓存完成: {cleared_count} 个文件")
        return cleared_count


_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
