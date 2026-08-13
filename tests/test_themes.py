"""TUI 主题清单契约测试"""

from textual.theme import BUILTIN_THEMES

from danmaku_analyzer.tui.i18n import i18n
from danmaku_analyzer.tui.themes import CUSTOM_THEMES, DEFAULT_THEME, THEME_IDS


class TestThemeRegistry:
    def test_theme_ids_no_duplicates(self):
        assert len(THEME_IDS) == len(set(THEME_IDS))

    def test_theme_ids_available_in_runtime(self):
        available = set(BUILTIN_THEMES) | set(CUSTOM_THEMES)
        for theme_id in THEME_IDS:
            assert theme_id in available

    def test_default_theme_in_registry(self):
        assert DEFAULT_THEME in THEME_IDS

    def test_custom_theme_names_consistent(self):
        for theme_id, theme in CUSTOM_THEMES.items():
            assert theme.name == theme_id

    def test_i18n_covers_all_themes(self):
        names = i18n.raw("settings.theme_names")
        for theme_id in THEME_IDS:
            assert theme_id in names
