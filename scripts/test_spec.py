#!/usr/bin/env python3
"""测试 PyInstaller spec 文件是否能被正确加载"""
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_spec_file():
    """测试 spec 文件是否能被正确解析"""
    spec_file = os.path.join(project_root, "Release", "danmakuscope.spec")
    
    if not os.path.exists(spec_file):
        print(f"错误：spec 文件不存在: {spec_file}")
        return False
    
    print(f"检查 spec 文件: {spec_file}")
    
    # 尝试导入 spec 文件中的模块
    try:
        # 模拟 PyInstaller 的 SPECPATH 变量
        import types
        spec_module = types.ModuleType("spec_module")
        spec_module.SPECPATH = os.path.dirname(spec_file)
        
        # 读取并执行 spec 文件
        with open(spec_file, 'r', encoding='utf-8') as f:
            spec_content = f.read()
        
        # 替换 SPECPATH 引用
        spec_content = spec_content.replace('SPECPATH', f'"{spec_module.SPECPATH}"')
        
        # 执行 spec 文件内容
        exec(spec_content, spec_module.__dict__)
        
        # 检查是否生成了可执行文件
        if hasattr(spec_module, 'exe_tui'):
            print("✓ TUI 可执行文件配置正确")
        else:
            print("✗ 未找到 TUI 可执行文件配置")
            return False
            
        if hasattr(spec_module, 'exe_cli'):
            print("✓ CLI 可执行文件配置正确")
        else:
            print("✗ 未找到 CLI 可执行文件配置")
            return False
        
        print("✓ spec 文件语法正确")
        return True
        
    except Exception as e:
        print(f"✗ spec 文件解析失败: {e}")
        return False

def check_entry_files():
    """检查入口文件是否存在"""
    release_dir = os.path.join(project_root, "Release")
    
    entry_files = ["entry_cli.py", "entry_tui.py"]
    all_exist = True
    
    for entry_file in entry_files:
        file_path = os.path.join(release_dir, entry_file)
        if os.path.exists(file_path):
            print(f"✓ {entry_file} 存在")
        else:
            print(f"✗ {entry_file} 不存在")
            all_exist = False
    
    return all_exist

def main():
    print("=== PyInstaller 配置测试 ===\n")
    
    print("1. 检查入口文件:")
    entry_ok = check_entry_files()
    
    print("\n2. 检查 spec 文件:")
    spec_ok = test_spec_file()
    
    print("\n=== 测试结果 ===")
    if entry_ok and spec_ok:
        print("✓ 所有配置检查通过！")
        print("\n可以运行以下命令进行构建：")
        print("  Windows: pyinstaller Release\\danmakuscope.spec --distpath Release")
        print("  macOS/Linux: pyinstaller Release/danmakuscope.spec --distpath Release")
        return 0
    else:
        print("✗ 配置检查失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
