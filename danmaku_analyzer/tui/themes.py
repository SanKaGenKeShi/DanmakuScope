"""TUI 主题模块 - 自定义主题与可用主题清单单一数据源"""

from textual.theme import BUILTIN_THEMES, Theme

CUSTOM_THEMES: dict[str, Theme] = {
    "bilibili-pink": Theme(
        name="bilibili-pink",
        primary="#FB7299",
        secondary="#23ADE5",
        warning="#FFB027",
        error="#E54C4C",
        success="#3AD783",
        # accent 与 primary 同色时，模式按钮选中态（$primary 底 + $accent 字）粉上粉不可读，取哔哩哔哩蓝拉开对比
        accent="#23ADE5",
        foreground="#E8E6EA",
        background="#17181C",
        surface="#1F2026",
        panel="#282930",
        variables={"button-color-foreground": "#17181C"},
    ),
    "cyberpunk": Theme(
        name="cyberpunk",
        primary="#00E5FF",
        secondary="#FF2A6D",
        warning="#FCE300",
        error="#FF2A6D",
        success="#05FFA3",
        accent="#FF2A6D",
        foreground="#D1F4FF",
        background="#0D0221",
        surface="#160B33",
        panel="#241448",
        variables={"button-color-foreground": "#0D0221"},
    ),
    "kanagawa": Theme(
        name="kanagawa",
        primary="#7E9CD8",
        secondary="#957FB8",
        warning="#FF9E3B",
        error="#E82424",
        success="#76946A",
        accent="#E6C384",
        foreground="#DCD7BA",
        background="#1F1F28",
        surface="#2A2A37",
        panel="#363646",
        variables={"button-color-foreground": "#1F1F28"},
    ),
    "everforest": Theme(
        name="everforest",
        primary="#A7C080",
        secondary="#7FBBB3",
        warning="#DBBC7F",
        error="#E67E80",
        success="#A7C080",
        accent="#D699B6",
        foreground="#D3C6AA",
        background="#272E33",
        surface="#2E383C",
        panel="#374145",
        variables={"button-color-foreground": "#272E33"},
    ),
}

_BUILTIN_BASE = [
    "textual-dark",
    "textual-light",
    "tokyo-night",
    "dracula",
    "nord",
    "gruvbox",
    "monokai",
    "catppuccin-mocha",
    "catppuccin-latte",
    "catppuccin-frappe",
    "catppuccin-macchiato",
    "solarized-dark",
    "solarized-light",
    "rose-pine",
    "rose-pine-moon",
    "atom-one-dark",
]

_BUILTIN_EXTRA = ["flexoki", "rose-pine-dawn", "atom-one-light", "ansi-dark", "ansi-light"]

# 按已装 textual 实际提供的主题过滤，低版本缺项不列出也不崩溃
THEME_IDS = [
    *[t for t in _BUILTIN_BASE + _BUILTIN_EXTRA if t in BUILTIN_THEMES],
    *CUSTOM_THEMES,
]

DEFAULT_THEME = "textual-dark"
