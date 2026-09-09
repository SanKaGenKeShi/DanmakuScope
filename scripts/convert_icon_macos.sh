#!/bin/bash
# macOS 图标转换脚本：将 .png 文件集合转换为 .icns 文件
# 必须在 macOS 上运行
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RELEASE_DIR="$PROJECT_ROOT/Release"

echo "==> 检查 .png 图标文件"
if [ ! -f "$RELEASE_DIR/icon_512x512.png" ]; then
    echo "错误：未找到 .png 图标文件"
    echo "请先运行：python scripts/convert_icon.py Release/DanmakuScope.ico Release/DanmakuScope.icns"
    exit 1
fi

echo "==> 创建 .iconset 目录"
ICONSET_DIR="$RELEASE_DIR/DanmakuScope.iconset"
mkdir -p "$ICONSET_DIR"

echo "==> 复制 .png 文件到 .iconset"
for size in 16 32 64 128 256 512 1024; do
    cp "$RELEASE_DIR/icon_${size}x${size}.png" "$ICONSET_DIR/icon_${size}x${size}.png"
    # 为 Retina 显示器创建 2x 版本（如果存在）
    if [ "$size" -le 512 ] && [ -f "$RELEASE_DIR/icon_$((size*2))x$((size*2)).png" ]; then
        cp "$RELEASE_DIR/icon_$((size*2))x$((size*2)).png" "$ICONSET_DIR/icon_${size}x${size}@2x.png"
    fi
done

echo "==> 使用 iconutil 转换为 .icns"
iconutil -c icns "$ICONSET_DIR" -o "$RELEASE_DIR/DanmakuScope.icns"

echo "==> 清理 .iconset 目录"
rm -rf "$ICONSET_DIR"

echo "==> 清理临时 .png 文件"
rm -f "$RELEASE_DIR"/icon_*.png

echo "==> 完成：$RELEASE_DIR/DanmakuScope.icns"
ls -lh "$RELEASE_DIR/DanmakuScope.icns"
