import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
reports_dir = os.path.join(project_root, "danmaku_analyzer", "data", "reports")
files = os.listdir(reports_dir)
print(f"报告目录共有 {len(files)} 个文件")
for f in sorted(files, key=lambda x: os.path.getmtime(os.path.join(reports_dir, x)), reverse=True)[:10]:
    print(f"  {f}")
# 检查是否有BV1NpXWBgEms相关文件
bv_files = [f for f in files if 'BV1NpXWBgEms' in f]
if bv_files:
    print(f"\n找到BV1NpXWBgEms相关文件: {bv_files}")
else:
    print("\n未找到BV1NpXWBgEms相关文件")
