import sys
import os
import requests
import json
import subprocess
import platform


def call_blender_api(endpoint, code):
    """
    调用Blender API执行代码
    """
    url = f"http://localhost:8080{endpoint}"
    
    payload = {
        "code": code
    }
    
    try:
        response = requests.post(url, json=payload)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None


def activate_blender_window():
    """
    激活Blender窗口（精确匹配窗口标题），如果没有运行则启动Blender
    """
    try:
        import pygetwindow as gw
    except ImportError:
        print("pygetwindow未安装，请运行: pip install pygetwindow")
        return False
    
    try:
        # 获取所有窗口
        all_windows = gw.getAllWindows()
        blender_window = None
        
        for window in all_windows:
            window_title = window.title.strip()
            
            # 多种可能的Blender标题格式
            if (window_title == 'Blender' or  # 基础标题
                window_title.startswith('Blender') and 
                not any(exclude in window_title.lower() for exclude in ['vscode', 'visual studio', 'code'])):
                
                # 额外检查确保不是VSCode或其他编辑器
                if ' - ' not in window_title or 'blender.exe' in window_title.lower():
                    blender_window = window
                    break
        
        if blender_window:
            print(f"激活窗口: {blender_window.title}")
            
            try:
                import win32gui
                import win32con
                
                hwnd = blender_window._hWnd
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                
                print(f"窗口已放到最前端: {blender_window.title}")
                return True
            except ImportError:
                print("pywin32未安装，请运行: pip install pywin32")
                if blender_window.isMinimized:
                    blender_window.restore()
                blender_window.activate()
                print("Blender窗口已激活（使用pygetwindow方法）")
                return True
        else:
            print("未找到Blender窗口，正在启动Blender...")
            # 启动Blender
            start_blender()
            return False

    except Exception as e:
        print(f"激活Blender窗口时出错: {e}")
        return False

def fix_model():
    """执行 Fix Model 操作"""
    code = '''
import bpy

# 执行 Fix Model 操作
bpy.ops.cats_armature.fix()
'''
    return call_blender_api('/api/exec', code)
def start_blender():
    """
    在命令提示符中启动Blender
    """
    blender_path = r"D:\blender\blender.exe"
    
    if not os.path.exists(blender_path):
        print(f"错误: 找不到Blender可执行文件: {blender_path}")
        return False
    
    try:
        # 在新的命令提示符窗口中启动Blender
        subprocess.Popen(['start', 'cmd', '/k', blender_path], shell=True)
        print(f"Blender已在新终端中启动: {blender_path}")
        
        import time
        time.sleep(5)
        return True
    except Exception as e:
        print(f"启动Blender时出错: {e}")
        return False

def import_pmx_file():
    """导入PMX文件"""
    code = '''
import bpy

def open_pmx_file_selector():  
    """打开PMX文件选择器"""  
    # 直接调用CATS的通用导入器，它会自动过滤PMX文件  
    bpy.ops.cats_importer.import_any_model('INVOKE_DEFAULT')  

# 在Blender中运行  
open_pmx_file_selector()
    '''
    return call_blender_api('/api/exec', code)


def delete_all_objects():
    """删除所有对象和集合"""
    code = '''
import bpy

# 选择所有对象
bpy.ops.object.select_all(action='SELECT')
# 删除选中的对象
bpy.ops.object.delete()

# 删除所有集合（除了默认的Master Collection）
# 删除场景中的所有集合
for collection in bpy.data.collections:
    bpy.data.collections.remove(collection)

# 删除所有空的集合（如果还有残留的话）
collections_to_remove = []
for collection in bpy.data.collections:
    collections_to_remove.append(collection)
    
for collection in collections_to_remove:
    bpy.data.collections.remove(collection)

print("所有对象和集合已删除")
'''
    return call_blender_api('/api/exec', code)
def parent_object_to_armature():
    """
    将选中的对象设置为骨骼绑定父级
    """
    code = '''
import bpy

# 确保至少有一个对象被选中
if bpy.context.selected_objects:
    # 执行骨骼绑定父级操作
    bpy.ops.object.parent_set(type='ARMATURE_NAME')
    print("对象已设置为骨骼绑定父级")
else:
    print("错误: 请先选择要绑定的对象")
'''
    return call_blender_api('/api/exec', code)

def set_blender_scale_settings():
    """
    设置Blender场景的单位比例和当前对象的缩放
    """
    code = '''
import bpy

# 设置场景单位比例
bpy.context.scene.unit_settings.scale_length = 0.01

# 设置当前选中对象的缩放（如果存在）
if bpy.context.object:
    bpy.context.object.scale[0] = 100
    bpy.context.object.scale[1] = 100
    bpy.context.object.scale[2] = 100

# 安全地设置3D视口的远裁剪平面
try:
    # 遍历所有屏幕区域，查找3D视口
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.clip_end = 300000
                    print("已设置3D视口远裁剪平面")
                    break
except Exception as e:
    print(f"设置3D视口裁剪平面时出错: {e}")

print("Blender缩放设置已应用")
'''
    return call_blender_api('/api/exec', code)
def delete_objects_by_name(name_pattern):
    """
    删除名称包含指定模式的物体
    Args:
        name_pattern (str): 要匹配的名称模式
    """
    code = f'''
import bpy

# 获取场景中所有名称包含指定模式的物体
objects_to_delete = []
for obj in bpy.context.scene.objects:
    if "{name_pattern}" in obj.name and obj.type == 'MESH':
        objects_to_delete.append(obj)

# 删除找到的物体并打印详细信息
deleted_count = 0
for obj in objects_to_delete:
    print("删除物体: 名称=" + obj.name + ", 类型=" + obj.type)
    
    bpy.data.objects.remove(obj, do_unlink=True)
    deleted_count += 1

print("已删除 "+str(deleted_count)+f"个包含 '{name_pattern}' 的物体")
'''
    return call_blender_api('/api/exec', code)
def clear_parent_keep_transform():
    """
    清除选中对象的父级关系，但保持变换（位置、旋转、缩放）
    """
    code = '''
import bpy

# 确保至少有一个对象被选中
if bpy.context.selected_objects:
    # 清除父级关系但保持变换
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    print("已清除选中对象的父级关系（保持变换）")
else:
    print("错误: 请先选择要清除父级的对象")
'''
    return call_blender_api('/api/exec', code)
def apply_armature_pose():
    """
    切换到姿态模式并应用选中的骨架
    """
    code = '''
import bpy

# 首先选择名称包含"ObjectName"的物体
found_object = None
for obj in bpy.context.scene.objects:
    if "ObjectName" in obj.name:
        found_object = obj
        break

if found_object is None:
    print("错误: 没有找到名称包含'ObjectName'的物体")
else:
    # 确保没有其他物体被选中
    bpy.ops.object.select_all(action='DESELECT')
    
    # 选择目标物体
    bpy.context.view_layer.objects.active = found_object
    found_object.select_set(True)
    
    print(f"已选择物体: {found_object.name}")
    
    # 切换到姿态模式
    bpy.ops.object.posemode_toggle()
    
    # 应用选中的骨架
    bpy.ops.pose.armature_apply(selected=True)
    
    print(f"已对物体 '{found_object.name}' 应用骨架姿态")
    
    # 切换回对象模式
    bpy.ops.object.posemode_toggle()
'''
    return call_blender_api('/api/exec', code)
def scale_objects_to_match_height():
    """
    将所有不包含"ObjectName"的物体缩放到与ObjectName物体相同的高度（三轴等比例缩放）
    """
    code = '''
import bpy



def get_object_height_z(obj):
    import mathutils
    """
    获取单个物体在Z轴方向的尺寸
    对于网格物体，使用边界框计算；对于其他类型物体，使用尺寸属性
    """
    if obj.type == 'MESH':
        # 对于网格物体，使用边界框计算Z方向尺寸
        bbox = obj.bound_box
        if bbox:
            # 将边界框顶点转换到世界坐标系
            world_bbox = [obj.matrix_world @ mathutils.Vector(co) for co in bbox]
            # 计算Z轴方向的最大最小值差
            z_coords = [v.z for v in world_bbox]
            height_z = max(z_coords) - min(z_coords)
            return height_z
    elif obj.type in ['CURVE', 'SURFACE', 'FONT', 'META']:
        # 对于其他类型的几何体，使用尺寸属性
        return obj.dimensions.z
    else:
        # 对于其他类型（如空对象），返回尺寸的Z分量
        return obj.dimensions.z
    
    return 0.0


def get_object_max_dimension(obj):
    import mathutils
    """
    获取物体的最大尺寸（X、Y、Z轴中的最大值）
    """
    if obj.type == 'MESH':
        bbox = obj.bound_box
        if bbox:
            world_bbox = [obj.matrix_world @ mathutils.Vector(co) for co in bbox]
            # 获取所有顶点的坐标
            x_coords = [v.x for v in world_bbox]
            y_coords = [v.y for v in world_bbox]
            z_coords = [v.z for v in world_bbox]
            
            # 计算各轴的尺寸
            x_size = max(x_coords) - min(x_coords)
            y_size = max(y_coords) - min(y_coords)
            z_size = max(z_coords) - min(z_coords)
            
            return max(x_size, y_size, z_size)
    else:
        # 对于非网格物体，使用dimensions属性
        return max(obj.dimensions.x, obj.dimensions.y, obj.dimensions.z)
    
    return 0.0


# 获取场景中所有物体的高度信息
object_heights = {}
object_max_dimensions = {}
for obj in bpy.context.scene.objects:
    if obj.type in ['MESH', 'CURVE', 'SURFACE', 'FONT', 'META', 'EMPTY']:
        height_z = get_object_height_z(obj)
        max_dimension = get_object_max_dimension(obj)
        object_heights[obj.name] = round(height_z, 6)
        object_max_dimensions[obj.name] = round(max_dimension, 6)

# 找到ObjectName物体的高度（可能有多个ObjectName开头的物体，取第一个）
target_objects = {name: height for name, height in object_heights.items() if 'ObjectName' in name}
if not target_objects:
    print("错误: 没有找到包含'ObjectName'的物体")
else:
    # 取第一个ObjectName物体的高度作为目标高度
    target_height = list(target_objects.values())[0]
    target_max_dimension = object_max_dimensions[list(target_objects.keys())[0]]
    
    # 缩放所有不包含'ObjectName'的物体到目标高度（三轴等比例缩放）
    for obj in bpy.context.scene.objects:
        if obj.type in ['MESH', 'CURVE', 'SURFACE', 'FONT', 'META', 'EMPTY'] and 'ObjectName' not in obj.name:
            current_height = get_object_height_z(obj)
            current_max_dimension = object_max_dimensions[obj.name]
            
            if current_max_dimension > 0:
                # 计算缩放因子，基于最大尺寸进行等比例缩放
                scale_factor = target_max_dimension / current_max_dimension
                
                # 三轴等比例缩放
                obj.scale[0] *= scale_factor
                obj.scale[1] *= scale_factor
                obj.scale[2] *= scale_factor
                
                print(f"缩放物体 '{obj.name}' 从最大尺寸 {round(current_max_dimension, 6)} 到 {round(target_max_dimension, 6)} (缩放因子: {round(scale_factor, 6)})")

    print(f"所有物体已缩放至与ObjectName物体相同的尺寸比例: {round(target_max_dimension, 6)}")
'''
    return call_blender_api('/api/exec', code)
def import_psk_file():
    """导入PSK文件（外部选择文件路径）"""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        try:
            import Tkinter as tk
            from tkFileDialog import askopenfilename
        except ImportError:
            print("错误: 无法导入tkinter模块")
            return None

    # 创建一个隐藏的tkinter根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 打开文件选择对话框
    filepath = filedialog.askopenfilename(
        title="选择PSK文件",
        filetypes=[("PSK files", "*.psk"), ("All files", "*.*")]
    )
    
    # 销毁tkinter根窗口
    root.destroy()
    print(f"选择的文件路径: {filepath}")
    if filepath and filepath.endswith('.psk'):
        # 构建执行PSK导入的代码，使用Blender操作符方式
        code = f'''
import bpy  
from io_scene_psk_psa.psk.reader import read_psk  
from io_scene_psk_psa.psk.importer import import_psk, PskImportOptions  
# 配置导入选项（需要添加 name 属性）  
options = PskImportOptions()  
options.name = 'ObjectName'  # 添加这个属性  
options.should_import_mesh = True  
options.should_import_skeleton = True  # 使用 skeleton 而不是 armature  
options.scale = 1.0  
  
# 读取PSK文件  
psk = read_psk('{filepath}')  
  
# 调用你的版本（3个参数）  
result = import_psk(psk, bpy.context, options)
'''
        return call_blender_api('/api/exec', code)
    else:
        print("未选择有效的PSK文件")
        return {"status": "error", "message": "未选择有效的PSK文件"}
def parent_object_to_bone():
    """
    将选中的对象设置为骨骼父级
    """
    code = '''
import bpy

# 确保至少有一个对象被选中，并且存在活动对象
if bpy.context.selected_objects and bpy.context.active_object:
    # 执行骨骼父级操作
    bpy.ops.object.parent_set(type='BONE')
    print("对象已设置为骨骼父级")
else:
    print("错误: 请先选择要绑定的对象并确保有一个活动对象")
'''
    return call_blender_api('/api/exec', code)
def add_data_transfer_modifier():
    """
    添加数据传输修改器并配置顶点组权重传输
    """
    code = '''
import bpy
bpy.ops.object.shape_key_remove(all=True)
# 添加数据传输修改器
bpy.ops.object.modifier_add(type='DATA_TRANSFER')

# 获取当前对象
obj = bpy.context.object

if obj and "DataTransfer" in obj.modifiers:
    modifier = obj.modifiers["DataTransfer"]
    
    # 配置修改器属性
    modifier.use_vert_data = True
    modifier.vert_mapping = 'POLYINTERP_NEAREST'
    modifier.data_types_verts = {'VGROUP_WEIGHTS'}
    
    print(f"数据传输修改器已添加并配置: {modifier.name}")
else:
    print("错误: 无法找到数据传输修改器")
'''
    return call_blender_api('/api/exec', code)

def print_response_result(tool_name, response):
    """
    打印API响应结果
    
    Args:
        tool_name (str): 工具名称
        response (dict): API响应
    """
    if response:
        if response['status'] == 'success':
            print(f"{tool_name} 执行成功!")
            print(f"返回结果: {response['result']}")
        else:
            print(f"{tool_name} 执行失败: {response['message']}")
    else:
        print("无法连接到Blender服务器")

def execute_tool(tool_name, *args):
    """
    根据工具名称执行对应的Blender操作
    
    Args:
        tool_name (str): 工具名称
        *args: 工具参数
        
    Returns:
        dict: API响应结果
    """
    tool_functions = {
        "import_pmx_file": import_pmx_file,
        "import_psk_file": import_psk_file,
        "scale_objects_to_match_height": scale_objects_to_match_height,
        "delete_all_objects": delete_all_objects,
        "activate_blender": activate_blender_window,
        "set_blender_scale": set_blender_scale_settings,
        "delete_objects_by_name": delete_objects_by_name,
        "fix_model": fix_model,
        "parent_object_to_armature": parent_object_to_armature,
        "apply_armature_pose": apply_armature_pose,
        "clear_parent_keep_transform": clear_parent_keep_transform,
        "parent_object_to_bone": parent_object_to_bone,  # 新增这一行
    }
    
    
    if tool_name in tool_functions:
        if tool_name == "activate_blender":
            # 特殊处理：激活Blender窗口不需要API调用
            result = tool_functions[tool_name]()
            return {"status": "success", "result": result} if result else {"status": "error", "message": "Failed to activate Blender"}
        elif tool_name in ["delete_objects_by_name"] and args:
            # 处理带参数的删除功能
            return tool_functions[tool_name](args[0])
        else:
            # 其他工具调用API
            return tool_functions[tool_name]()
    else:
        print(f"错误: 未知的Blender工具 '{tool_name}'")
        return None



def main():
    # delete_objects_by_name("ObjectName")
    # return 
    tool_name = sys.argv[1]  
    
    # 检查是否有额外参数传递给工具
    args = sys.argv[2:]  # 获取工具名称之后的所有参数
    
    # 执行对应的工具
    response = execute_tool(tool_name, *args)
    
    # 如果不是激活窗口操作，则打印响应结果
    if tool_name != "activate_blender":
        print_response_result(tool_name, response)


if __name__ == "__main__":
    main()