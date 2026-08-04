"""
缓存管理器模块 - 管理数据缓存
"""

import os
import json
import hashlib
import pickle
from typing import Any, Optional
from pathlib import Path
from datetime import datetime, timedelta

from .config import get_settings
from .utils.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.settings = get_settings()
        raw_cache_dir = cache_dir or self.settings.CACHE_DIR
        self.cache_dir = self.settings.resolve_data_path(raw_cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str, extension: str = ".pkl") -> str:
        cache_key = self._get_cache_key(key)
        return os.path.join(self.cache_dir, f"{cache_key}{extension}")
    
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
            logger.debug(f"缓存命中: {key}")
            return data
        except Exception as e:
            logger.error(f"读取缓存失败: {e}")
            return None
    
    def set(self, key: str, value: Any) -> bool:
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
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
    
    def get_json(self, key: str, max_age_hours: int = 24) -> Optional[dict]:
        cache_path = self._get_cache_path(key, extension=".json")
        
        if not os.path.exists(cache_path):
            return None
        
        file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - file_mtime > timedelta(hours=max_age_hours):
            logger.info(f"JSON 缓存已过期: {key}")
            self.delete_json(key)
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"JSON 缓存命中: {key}")
            return data
        except Exception as e:
            logger.error(f"读取 JSON 缓存失败: {e}")
            return None
    
    def set_json(self, key: str, value: dict) -> bool:
        cache_path = self._get_cache_path(key, extension=".json")
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
            logger.debug(f"JSON 缓存设置成功: {key}")
            return True
        except Exception as e:
            logger.error(f"设置 JSON 缓存失败: {e}")
            return False
    
    def delete_json(self, key: str) -> bool:
        cache_path = self._get_cache_path(key, extension=".json")
        
        if not os.path.exists(cache_path):
            return True
        
        try:
            os.remove(cache_path)
            logger.debug(f"JSON 缓存删除成功: {key}")
            return True
        except Exception as e:
            logger.error(f"删除 JSON 缓存失败: {e}")
            return False


_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """获取缓存管理器实例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def clear_all_caches():
    cache_manager = get_cache_manager()
    return cache_manager.clear()
