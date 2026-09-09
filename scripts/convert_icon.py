#!/usr/bin/env python3
"""转换 .ico 到 .icns 格式（macOS 图标）"""
import os
import sys
from pathlib import Path

def convert_ico_to_icns(ico_path: str, icns_path: str) -> bool:
    """将 .ico 文件转换为 .icns 格式"""
    try:
        from PIL import Image
        
        # 打开 .ico 文件
        img = Image.open(ico_path)
        
        # 获取所有尺寸
        sizes = []
        for size in [16, 32, 64, 128, 256, 512, 1024]:
            resized = img.resize((size, size), Image.Resampling.LANCZOS)
            sizes.append(resized)
        
        # 保存为 .icns 格式
        # 注意：Pillow 不直接支持 .icns，需要使用其他方法
        # 在 macOS 上可以使用 iconutil 或 sips
        # 这里我们创建一个临时的 .png 集合，然后使用 iconutil
        
        print(f"原始图标尺寸: {img.size}")
        print(f"需要创建 .icns 文件: {icns_path}")
        
        # 由于 Pillow 不支持直接保存为 .icns，我们需要使用系统工具
        # 在 macOS 上，我们可以创建 .iconset 目录
        # 在 Windows/Linux 上，我们只能创建 .png 文件集合
        
        if sys.platform == "darwin":
            # macOS: 使用 iconutil
            iconset_dir = Path(icns_path).with_suffix('.iconset')
            iconset_dir.mkdir(exist_ok=True)
            
            for i, size in enumerate([16, 32, 64, 128, 256, 512, 1024]):
                resized = img.resize((size, size), Image.Resampling.LANCZOS)
                resized.save(iconset_dir / f"icon_{size}x{size}.png")
                # 为 Retina 显示器创建 2x 版本
                if size <= 512:
                    resized_2x = img.resize((size*2, size*2), Image.Resampling.LANCZOS)
                    resized_2x.save(iconset_dir / f"icon_{size}x{size}@2x.png")
            
            # 使用 iconutil 转换
            import subprocess
            result = subprocess.run([
                "iconutil", "-c", "icns", str(iconset_dir), "-o", icns_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"成功创建 .icns 文件: {icns_path}")
                # 清理 .iconset 目录
                import shutil
                shutil.rmtree(iconset_dir)
                return True
            else:
                print(f"iconutil 失败: {result.stderr}")
                return False
        else:
            # Windows/Linux: 保存为 .png 文件集合，用户需要手动转换
            print("注意：在非 macOS 系统上，无法直接创建 .icns 文件")
            print("已创建 .png 文件集合，您需要在 macOS 上使用 iconutil 转换")
            
            # 保存为多个 .png 文件
            for size in [16, 32, 64, 128, 256, 512, 1024]:
                resized = img.resize((size, size), Image.Resampling.LANCZOS)
                resized.save(f"icon_{size}x{size}.png")
            
            return False
            
    except ImportError:
        print("错误：需要安装 Pillow 库")
        print("请运行：pip install Pillow")
        return False
    except Exception as e:
        print(f"转换失败: {e}")
        return False

def main():
    # 检查参数
    if len(sys.argv) < 3:
        print("用法: python convert_icon.py <input.ico> <output.icns>")
        print("示例: python convert_icon.py DanmakuScope.ico DanmakuScope.icns")
        sys.exit(1)
    
    ico_path = sys.argv[1]
    icns_path = sys.argv[2]
    
    # 检查输入文件是否存在
    if not os.path.exists(ico_path):
        print(f"错误：输入文件不存在: {ico_path}")
        sys.exit(1)
    
    # 转换图标
    success = convert_ico_to_icns(ico_path, icns_path)
    
    if success:
        print("图标转换完成！")
        sys.exit(0)
    else:
        print("图标转换失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
