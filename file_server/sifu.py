import os
import shutil
import subprocess
import sys
from pathlib import Path

def build_sifu_mod():
    """
    执行Sifu MOD构建流程
    """
    print("=" * 32)
    print("    开始执行 MOD 构建流程")
    print("=" * 32)
    print()

    # ================ 第一步：删除 Characters 中的 _Shared 和 Skeleton ================
    char_dir = Path(r"E:\blender\ue4\character\Saved\Cooked\WindowsNoEditor\new\Content\Characters")

    if (char_dir / "_Shared").exists():
        shutil.rmtree(char_dir / "_Shared")
        print("✅ 已删除 _Shared 文件夹")

    if (char_dir / "Skeleton").exists():
        shutil.rmtree(char_dir / "Skeleton")
        print("✅ 已删除 Skeleton 文件夹")

    print()

    # ================ 第二步：复制 Characters 到 pakchunk99-XXX-P\Sifu\Content\Characters ================
    source_dir = char_dir
    target_char_dir = Path(r"E:\blender\pakchunk99-XXX-P\Sifu\Content\Characters")

    if not source_dir.exists():
        print("❌ 错误：源目录不存在！")
        print(f"  {source_dir}")
        input("按任意键退出...")
        sys.exit(1)

    if target_char_dir.exists():
        shutil.rmtree(target_char_dir)
        print("✅ 已删除旧的目标 Characters 文件夹")

    print("正在复制 Characters 到 Sifu 项目...")
    try:
        shutil.copytree(source_dir, target_char_dir)
        print("✅ Characters 复制成功")
    except Exception as e:
        print(f"❌ 复制失败：{e}")
        input("按任意键退出...")
        sys.exit(1)

    print()

    # ================ 第三步：调用 UnrealPak 打包生成 .pak 文件 ================
    unreal_pak_script = Path(r"E:\blender\Sifu-MOD-TOOL\UnrealPak\UnrealPak-With-Compression.bat")
    pak_folder = Path(r"E:\blender\pakchunk99-XXX-P")

    if not unreal_pak_script.exists():
        print("❌ 错误：UnrealPak 打包脚本不存在！")
        print(f"  {unreal_pak_script}")
        input("按任意键退出...")
        sys.exit(1)

    print("正在调用 UnrealPak 打包...")
    try:
        subprocess.run([str(unreal_pak_script), str(pak_folder)], check=True)
    except subprocess.CalledProcessError:
        print("❌ 打包失败！")
        input("按任意键退出...")
        sys.exit(1)

    # 检查是否生成了 .pak 文件
    pak_file = Path(f"{pak_folder}.pak")
    if pak_file.exists():
        print("✅ .pak 文件已生成：")
        print(f"  {pak_file}")
    else:
        print("❌ 打包失败：未生成 .pak 文件！")
        input("按任意键退出...")
        sys.exit(1)

    print()

    # ================ 第四步：将 .pak 文件复制到游戏 MOD 目录 ================
    target_mod_dir = Path(r"G:\Sifu\Sifu\Content\Paks\~mods")
    target_pak = target_mod_dir / f"{pak_folder.name}.pak"

    # 确保 ~mods 目录存在
    if not target_mod_dir.exists():
        print("❌ 错误：MOD 目录不存在！请确认游戏路径正确。")
        print(f"  {target_mod_dir}")
        input("按任意键退出...")
        sys.exit(1)

    try:
        shutil.copy2(pak_file, target_pak)
        print("✅ 已替换 MOD 文件到：")
        print(f"  {target_pak}")
    except Exception as e:
        print(f"❌ 复制 MOD 文件失败：{e}")
        input("按任意键退出...")
        sys.exit(1)

    print()

    # ================ 第五步：启动游戏 ================
    game_exe = Path(r"G:\Sifu\Sifu.exe")

    if game_exe.exists():
        print("正在启动游戏...")
        print(f"  {game_exe}")
        subprocess.Popen([str(game_exe)])
        print("✅ 游戏已启动")
    else:
        print("❌ 错误：游戏主程序不存在！")
        print(f"  {game_exe}")
        print("请检查路径是否正确。")
        input("按任意键退出...")
        sys.exit(1)
        

    print()
    print("=" * 32)
    print("✅ 所有操作已完成！游戏已启动。")
    print("=" * 32)
    print()
    
    return {"status": "success", "result": "MOD build completed and game launched"}


if __name__ == "__main__":
    build_sifu_mod()