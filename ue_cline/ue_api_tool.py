import os
import subprocess
import time
import sys
import socket
import json
import shutil
from pathlib import Path

def activate_ue_window():
    """
    激活ue窗口（精确匹配窗口标题），如果没有运行则启动ue
    """
    try:
        import pygetwindow as gw
    except ImportError:
        print("pygetwindow未安装，请运行: pip install pygetwindow")
        return False
    
    try:
        # 获取所有窗口
        all_windows = gw.getAllWindows()
        ue_window = None
        
        for window in all_windows:
            window_title = window.title.strip()
            print(f"检查窗口: {window_title}")
            
            # 修改匹配逻辑以包含中文标题
            if ('虚幻编辑器' in window_title or 
                'Unreal Editor' in window_title or 
                'UE4Editor' in window_title or
                window_title == 'ue' or  # 基础标题
                (window_title.startswith('ue') and 
                 not any(exclude in window_title.lower() for exclude in ['vscode', 'visual studio', 'code']))):
                
                # 额外检查确保不是VSCode或其他编辑器
                if ' - ' not in window_title or 'ue.exe' in window_title.lower() or '虚幻编辑器' in window_title:
                    ue_window = window
                    break
        
        if ue_window:
            print(f"激活窗口: {ue_window.title}")
            
            try:
                import win32gui
                import win32con
                
                hwnd = ue_window._hWnd
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                
                print(f"窗口已放到最前端: {ue_window.title}")
                return True
            except ImportError:
                print("pywin32未安装，请运行: pip install pywin32")
                if hasattr(ue_window, 'isMinimized') and ue_window.isMinimized:
                    ue_window.restore()
                ue_window.activate()
                print("ue窗口已激活（使用pygetwindow方法）")
                return True
        else:
            print("未找到ue窗口，正在启动ue...")
            # 启动ue
            start_ue()
            return False

    except Exception as e:
        print(f"激活ue窗口时出错: {e}")
        return False

def send_python_code_request(code=None, host="localhost", port=8070):
    """
    发送Python代码执行请求到UE服务器
    """
    try:
        # 创建socket连接
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        
        # 准备数据
        payload = {
            "code": code or "unreal.SystemLibrary.print_string(None, 'Hello from remote execution!', True, True, unreal.LinearColor(0,1,0,1), 5.0)"
        }
        
        # 发送JSON数据
        json_data = json.dumps(payload)
        sock.send(json_data.encode('utf-8'))
        
        # 接收响应
        response = sock.recv(4096).decode('utf-8')
        print(f"服务器响应: {response}")
        
        sock.close()
        return {"status": "success", "result": response}
        
    except Exception as e:
        print(f"发送请求时出错: {str(e)}")
        return {"status": "error", "message": str(e)}

def send_fbx_import_request(host="localhost", port=8070):
    """
    发送FBX导入请求到服务器 - 使用固定路径
    """
    code = """
import unreal
import os

# 检查文件是否存在
fbx_path = "E:\\\\blender\\\\SK_W_MainChar_01.fbx"
destination_path = "/Game/Characters/MainChar/W/Meshes"

# 清空目标文件夹中的所有资产
print(f"正在扫描文件夹: {destination_path}")
asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()
filter = unreal.ARFilter(recursive_paths=True)
filter.package_paths.append(destination_path)

assets = asset_registry.get_assets(filter)
print(f"在 {destination_path} 中找到 {len(assets)} 个资产")

asset_paths = []
for asset_data in assets:
    print(f"找到资产: {asset_data.object_path}")
    asset_paths.append(str(asset_data.object_path))

deleted = []
for path in asset_paths:
    success = unreal.EditorAssetLibrary.delete_asset(path)
    if success:
        print(f"成功删除资产: {path}")
        deleted.append(path)
    else:
        print(f"删除资产失败: {path} - 可能被引用或锁定")

print(f"共删除 {len(deleted)} 个资产。")

if not os.path.exists(fbx_path):
    result = "文件路径无效: " + fbx_path
else:
    # 创建导入任务
    import_task = unreal.AssetImportTask()
    import_task.set_editor_property('filename', fbx_path)
    import_task.set_editor_property('destination_path', destination_path)
    import_task.set_editor_property('save', True)
    import_task.set_editor_property('automated', True)
    import_task.set_editor_property('replace_existing', True)
    
    # 执行导入任务
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    asset_tools.import_asset_tasks([import_task])
    
    # 获取文件名
    filename = os.path.basename(fbx_path)
    result = "成功导入文件: " + filename + " 到 " + destination_path

result
"""
    return send_python_code_request(code, host, port)

def delete_materials_in_folder():
    """
    删除指定文件夹中的材质文件

    Returns:
        dict: API响应结果
    """
    code = """
import unreal

folder_path = "/Game/Characters/MainChar/W/Meshes"
print(f"正在扫描文件夹: {folder_path}")

asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()
filter = unreal.ARFilter(recursive_paths=True)
filter.package_paths.append(folder_path)
filter.class_names.append("Material")
filter.class_names.append("MaterialInstanceConstant")

assets = asset_registry.get_assets(filter)
print(f"在 {folder_path} 中找到 {len(assets)} 个材质资产")

material_paths = []
for asset_data in assets:
    print(f"找到材质资产: {asset_data.object_path}")
    material_paths.append(str(asset_data.object_path))

deleted = []
for path in material_paths:
    success = unreal.EditorAssetLibrary.delete_asset(path)
    if success:
        print(f"成功删除材质: {path}")
        deleted.append(path)
    else:
        print(f"删除材质失败: {path} - 可能被引用或锁定")

print(f"共删除 {len(deleted)} 个材质资产。")

"""
    return send_python_code_request(code)

def move_and_rename_skeleton():
    """
    将meshes文件夹中的Skeleton资产移动到指定目标文件夹并重命名
    
    Returns:
        dict: API响应结果
    """
    code = '''
import unreal

# 配置路径和目标名称
delete_folder = "/Game/Characters/Skeleton"
mesh_folder = "/Game/Characters/MainChar/W/Meshes"
target_folder = delete_folder
target_name = "Base_skeleton"
target_asset_path = f"{target_folder}/{target_name}"

print(f"开始删除目录下所有Skeleton资产: {delete_folder}")

# 获取资产注册表
asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()

# 删除delete_folder下所有Skeleton资产
delete_filter = unreal.ARFilter(recursive_paths=True)
delete_filter.package_paths.append(delete_folder)
delete_filter.class_names.append("Skeleton")

skeletons_to_delete = asset_registry.get_assets(delete_filter)
deleted_count = 0
for asset_data in skeletons_to_delete:
    asset_path = str(asset_data.object_path)
    success = unreal.EditorAssetLibrary.delete_asset(asset_path)
    if success:
        print(f"已删除: {asset_path}")
        deleted_count += 1
    else:
        print(f"删除失败: {asset_path}（可能被引用或锁定）")
print(f"共删除 {deleted_count} 个Skeleton资产。")

# 扫描mesh_folder的Skeleton资产并移动重命名
print(f"开始扫描骨骼资产: {mesh_folder}")

mesh_filter = unreal.ARFilter(recursive_paths=True)
mesh_filter.package_paths.append(mesh_folder)
mesh_filter.class_names.append("Skeleton")

skeleton_assets = asset_registry.get_assets(mesh_filter)
print(f"在 {mesh_folder} 找到 {len(skeleton_assets)} 个Skeleton资产")

if len(skeleton_assets) == 0:
    print("未找到任何Skeleton资产，脚本结束。")
else:
    if unreal.EditorAssetLibrary.does_asset_exist(target_asset_path):
        print(f"目标路径 {target_asset_path} 已存在Skeleton资产，请先处理或改名，脚本中止。")
    else:
        asset_data = skeleton_assets[0]
        old_path = str(asset_data.object_path)
        print(f"尝试重命名并移动骨骼资产:\\n{old_path} -> {target_asset_path}")
        success = unreal.EditorAssetLibrary.rename_asset(old_path, target_asset_path)
        if success:
            print(f"成功重命名并移动Skeleton资产到：{target_asset_path}")
        else:
            print(f"重命名或移动失败，可能资产被引用或锁定。")
'''
    return send_python_code_request(code)

def create_material_instances_from_textures():
    """
    从纹理文件夹中的纹理创建材质实例，绑定到指定的主材质
    
    Returns:
        dict: API响应结果
    """
    code = '''
import unreal
 
master_material_path = "/Game/Characters/_Shared/Materials/Master/M_CharaMaster_Cloth"
textures_folder = "/Game/Characters/MainChar/W/Textures"
materials_folder = "/Game/Characters/MainChar/W/Materials"
base_color_param_name = "BaseColor"
 
print(f"开始清空材质实例目录: {materials_folder}")
assets_to_delete = unreal.EditorAssetLibrary.list_assets(materials_folder, recursive=True, include_folder=False)
for asset_path in assets_to_delete:
    unreal.EditorAssetLibrary.delete_asset(asset_path)
print(f"清空完成，共删除 {len(assets_to_delete)} 个资产。")
 
master_material = unreal.EditorAssetLibrary.load_asset(master_material_path)
if not master_material:
    raise RuntimeError(f"无法加载主材质: {master_material_path}")
 
texture_paths = unreal.EditorAssetLibrary.list_assets(textures_folder, recursive=True, include_folder=False)
texture_assets = []
for path in texture_paths:
    asset = unreal.EditorAssetLibrary.load_asset(path)
    if asset and isinstance(asset, unreal.Texture):
        texture_assets.append(asset)
print(f"找到 {len(texture_assets)} 个纹理用于创建材质实例。")
 
asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
for texture in texture_assets:
    instance_name = texture.get_name()
    instance_path = f"{materials_folder}/{instance_name}"
    if unreal.EditorAssetLibrary.does_asset_exist(instance_path):
        unreal.log(f"已存在材质实例，跳过：{instance_path}")
        continue
 
    mi = asset_tools.create_asset(
        asset_name=instance_name,
        package_path=materials_folder,
        asset_class=unreal.MaterialInstanceConstant,
        factory=unreal.MaterialInstanceConstantFactoryNew()
    )
    if not mi:
        unreal.log_error(f"创建失败：{instance_path}")
        continue
 
    mi.set_editor_property("parent", master_material)
 
    # 设置纹理参数，需使用 FMaterialParameterInfo
    parameter_info = unreal.MaterialParameterInfo()
    parameter_info.name = base_color_param_name
 
    # 用MaterialEditingLibrary接口设置参数更安全
    unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(mi, parameter_info.name, texture)
 
    unreal.EditorAssetLibrary.save_asset(instance_path)
    unreal.log(f"已创建材质实例: {instance_path}，BaseColor设置为纹理: {texture.get_path_name()}")
 
unreal.log(f"共创建 {len(texture_assets)} 个材质实例，路径: {materials_folder}")
'''
    return send_python_code_request(code)

def move_textures_to_folder():
    """
    将meshes文件夹中的纹理资产移动到textures文件夹中
    
    Returns:
        dict: API响应结果
    """
    code = """
import unreal

# 路径配置（无结尾斜杠）
textures_folder = "/Game/Characters/MainChar/W/Textures"
meshes_folder = "/Game/Characters/MainChar/W/Meshes"

# 1. 删除 textures_folder 目录下所有资产
print(f"开始删除目录下所有资产: {textures_folder}")
assets_to_delete = unreal.EditorAssetLibrary.list_assets(textures_folder, recursive=True, include_folder=False)

deleted_count = 0
for asset_path in assets_to_delete:
    success = unreal.EditorAssetLibrary.delete_asset(asset_path)
    if success:
        print(f"删除成功: {asset_path}")
        deleted_count += 1
    else:
        print(f"删除失败: {asset_path}（可能被引用或锁定）")
print(f"共删除了 {deleted_count} 个资产。")

# 2. 查找 meshes_folder 中所有纹理资产
print(f"\\n开始查找文件夹中的纹理资产: {meshes_folder}")
asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()

filter = unreal.ARFilter(recursive_paths=True)
filter.package_paths.append(meshes_folder)
filter.class_names.append("Texture2D")
filter.class_names.append("TextureCube")
filter.class_names.append("TextureRenderTarget2D")
filter.class_names.append("TextureRenderTargetCube")
filter.class_names.append("Texture")

texture_assets = asset_registry.get_assets(filter)
print(f"找到 {len(texture_assets)} 个纹理资产。")

# 3. 移动纹理资产到 textures_folder（重命名路径）
moved_count = 0
for asset_data in texture_assets:
    old_path = str(asset_data.object_path)
    asset_name = asset_data.asset_name
    new_path = f"{textures_folder}/{asset_name}"
    if unreal.EditorAssetLibrary.does_asset_exist(new_path):
        print(f"目标路径已存在，跳过: {new_path}")
        continue
    success = unreal.EditorAssetLibrary.rename_asset(old_path, new_path)
    if success:
        print(f"已移动: {old_path} -> {new_path}")
        moved_count += 1
    else:
        print(f"移动失败: {old_path} -> {new_path}（可能被引用或锁定）")

print(f"共移动 {moved_count} 个纹理资产到 {textures_folder}。")
"""
    return send_python_code_request(code)

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
        print(" 已删除 _Shared 文件夹")

    if (char_dir / "Skeleton").exists():
        shutil.rmtree(char_dir / "Skeleton")
        print(" 已删除 Skeleton 文件夹")

    print()

    # ================ 第二步：复制 Characters 到 pakchunk99-XXX-P\Sifu\Content\Characters ================
    source_dir = char_dir
    target_char_dir = Path(r"E:\blender\pakchunk99-XXX-P\Sifu\Content\Characters")

    if not source_dir.exists():
        print(" 错误：源目录不存在！")
        print(f"  {source_dir}")
        input("按任意键退出...")
        sys.exit(1)

    if target_char_dir.exists():
        shutil.rmtree(target_char_dir)
        print(" 已删除旧的目标 Characters 文件夹")

    print("正在复制 Characters 到 Sifu 项目...")
    try:
        shutil.copytree(source_dir, target_char_dir)
        print(" Characters 复制成功")
    except Exception as e:
        print(f" 复制失败：{e}")
        input("按任意键退出...")
        sys.exit(1)

    print()

    # ================ 第三步：调用 UnrealPak 打包生成 .pak 文件 ================
    unreal_pak_script = Path(r"E:\blender\Sifu-MOD-TOOL\UnrealPak\UnrealPak-With-Compression.bat")
    pak_folder = Path(r"E:\blender\pakchunk99-XXX-P")

    if not unreal_pak_script.exists():
        print(" 错误：UnrealPak 打包脚本不存在！")
        print(f"  {unreal_pak_script}")
        input("按任意键退出...")
        sys.exit(1)

    # 删除已存在的 .pak 文件
    pak_file = Path(f"{pak_folder}.pak")
    if pak_file.exists():
        print(f"正在删除已存在的 .pak 文件: {pak_file}")
        pak_file.unlink()
        print(" 已删除旧的 .pak 文件")

    print("正在调用 UnrealPak 打包...")
    try:
        # 使用 subprocess.run 并重定向输入输出以防止挂起
        result = subprocess.run(
            [str(unreal_pak_script), str(pak_folder)],
            stdin=subprocess.DEVNULL,  # 重定向标准输入
            stdout=subprocess.PIPE,    # 重定向标准输出
            stderr=subprocess.PIPE,    # 重定向标准错误
            text=True                  # 返回字符串而非字节
        )
        if result.returncode != 0:
            print(f" 打包失败！返回码: {result.returncode}")
            print(f"错误信息: {result.stderr}")
            input("按任意键退出...")
            sys.exit(1)
        else:
            print(" 打包成功！")
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f" 打包过程中出现错误：{e}")
        input("按任意键退出...")
        sys.exit(1)

    # 检查是否生成了 .pak 文件
    pak_file = Path(f"{pak_folder}.pak")
    if pak_file.exists():
        print(" .pak 文件已生成：")
        print(f"  {pak_file}")
    else:
        print(" 打包失败：未生成 .pak 文件！")
        input("按任意键退出...")
        sys.exit(1)

    print()

    # ================ 第四步：将 .pak 文件复制到游戏 MOD 目录 ================
    target_mod_dir = Path(r"G:\Sifu\Sifu\Content\Paks\~mods")
    target_pak = target_mod_dir / f"{pak_folder.name}.pak"

    # 确保 ~mods 目录存在
    if not target_mod_dir.exists():
        print(" 错误：MOD 目录不存在！请确认游戏路径正确。")
        print(f"  {target_mod_dir}")
        input("按任意键退出...")
        sys.exit(1)

    try:
        shutil.copy2(pak_file, target_pak)
        print(" 已替换 MOD 文件到：")
        print(f"  {target_pak}")
    except Exception as e:
        print(f" 复制 MOD 文件失败：{e}")
        input("按任意键退出...")
        sys.exit(1)

    print()



    print()
    print("=" * 32)
    print(" 所有操作已完成！游戏已启动。")
    print("=" * 32)
    print()
    
    return {"status": "success", "result": "MOD build completed and game launched"}

def import_fbx():
    """
    执行完整的FBX导入工作流并返回每个步骤的结果
    """
    results = []
    
    # 步骤1: 发送Python代码请求
    step_result = send_python_code_request()
    results.append({"step": "send_python_code_request", "result": step_result})
    
    # 步骤2: 发送FBX导入请求
    step_result = send_fbx_import_request()
    results.append({"step": "send_fbx_import_request", "result": step_result})
    
    # 步骤3: 删除材质
    step_result = delete_materials_in_folder()
    results.append({"step": "delete_materials_in_folder", "result": step_result})
    
    # 步骤4: 移动纹理
    step_result = move_textures_to_folder()
    results.append({"step": "move_textures_to_folder", "result": step_result})
    
    # 步骤5: 移动并重命名骨骼
    step_result = move_and_rename_skeleton()
    results.append({"step": "move_and_rename_skeleton", "result": step_result})
    
    # 步骤6: 创建材质实例
    step_result = create_material_instances_from_textures()
    results.append({"step": "create_material_instances_from_textures", "result": step_result})
    
    # 返回所有结果
    return {
        "status": "success",
        "workflow": "import_fbx",
        "steps_completed": len(results),
        "results": results
    }

def execute_tool(tool_name, *args):
    """
    根据工具名称执行对应的ue操作
    
    Args:
        tool_name (str): 工具名称
        *args: 工具参数
        
    Returns:
        dict: API响应结果
    """
    tool_functions = {
        "activate_ue": activate_ue_window,
        "import_fbx": import_fbx,
        "build_sifu_mod": build_sifu_mod,
    }
    
    if tool_name in tool_functions:
        if tool_name == "activate_ue":
            # 特殊处理：激活ue窗口不需要API调用
            result = tool_functions[tool_name]()
            return {"status": "success", "result": result} if result else {"status": "error", "message": "Failed to activate ue"}
      
        elif tool_name in ["delete_objects_by_name"] and args:
            # 处理带参数的删除功能
            return tool_functions[tool_name](args[0])
        else:
            # 其他工具调用API
            return tool_functions[tool_name]()
    else:
        print(f"错误: 未知的ue工具 '{tool_name}'")
        return {"status": "error", "message": f"未知的ue工具 '{tool_name}'"}

def start_ue():
    """
    启动ue（在后台进程启动，不阻塞当前线程）
    """
    ue_path = r"D:\UE_4.26\Engine\Binaries\Win64\UE4Editor.exe"
    
    if not os.path.exists(ue_path):
        print(f"错误: 找不到ue可执行文件: {ue_path}")
        return False
    
    try:
        # 使用CREATE_NEW_CONSOLE标志启动ue，使其在独立的进程中运行
        subprocess.Popen([ue_path], 
                        creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS)
        print(f"ue已在后台启动: {ue_path}")
        
        time.sleep(2)  # 短暂延迟确保进程已开始启动
        return True
    except Exception as e:
        print(f"启动ue时出错: {e}")
        return False

def main():
    # delete_objects_by_name("ObjectName")
    # return 
    tool_name = sys.argv[1]  
    
    # 检查是否有额外参数传递给工具
    args = sys.argv[2:]  # 获取工具名称之后的所有参数
    
    # 执行对应的工具
    response = execute_tool(tool_name, *args)
    
    if response:
        print(json.dumps(response))

if __name__ == "__main__":
    main()