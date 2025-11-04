import clr
import sys
import os

# 尝试添加Tekla API DLL路径（根据实际安装路径调整）
tekla_path = r"F:\Program Files\Tekla Structures\2024.0\bin"  # 请根据实际版本调整路径
if tekla_path not in sys.path:
    sys.path.append(tekla_path)

def list_all_beams(model):
    """
    列出模型中的所有梁及其截面类型
    
    Args:
        model: Tekla Model 实例
    """
    try:
        # 获取模型中的所有对象
        all_objects = model.GetModelObjectSelector().GetAllObjects()
        
        beam_profiles = {}
        beam_count = 0
        
        # 遍历模型中的所有对象
        while all_objects.MoveNext():
            obj = all_objects.Current
            
            # 检查是否为梁对象且有Profile属性
            if isinstance(obj, Beam) and hasattr(obj, 'Profile'):
                beam_count += 1
                # 尝试多种方式获取截面名称
                profile_obj = obj.Profile
                
                # 方法1: 直接访问Profile对象的属性
                profile_name = None
                if hasattr(profile_obj, 'ProfileString'):
                    profile_name = profile_obj.ProfileString
                elif hasattr(profile_obj, 'Name'):
                    profile_name = profile_obj.Name
                else:
                    # 方法2: 使用ToString()方法
                    profile_name = profile_obj.ToString()
                
                # 如果以上方法都不行，就用repr查看对象内容
                if profile_name == "Tekla.Structures.Model.Profile" or profile_name is None:
                    profile_name = repr(profile_obj)
                
                # 统计每种截面类型的数量
                if profile_name in beam_profiles:
                    beam_profiles[profile_name] += 1
                else:
                    beam_profiles[profile_name] = 1
                    
                # 输出前几个梁的详细信息
                if beam_count <= 10:
                    # 使用正确的属性访问方式
                    identifier = obj.Identifier
                    print(f"型材 {beam_count}: ID={identifier.ID}, 截面={profile_name}")
        
        print(f"\n总共找到 {beam_count} 个梁构件")
        print("截面类型统计:")
        # 先将键转换为列表再排序
        sorted_profiles = sorted(beam_profiles.keys())
        for profile in sorted_profiles:
            print(f"  {profile}: {beam_profiles[profile]} 个")
            
    except Exception as e:
        print(f"列出所有梁时出错: {e}")
        import traceback
        traceback.print_exc()

def find_beams_by_profile(model, profile_name):
    """
    根据截面类型查找型材并进行处理
    
    Args:
        model: Tekla Model 实例
        profile_name (str): 要查找的截面类型名称
    """
    try:
        # 获取模型中的所有对象
        all_objects = model.GetModelObjectSelector().GetAllObjects()
        
        found_beams = []
        
        # 遍历模型中的所有对象
        while all_objects.MoveNext():
            obj = all_objects.Current
            
            # 检查是否为梁对象且截面类型匹配
            if isinstance(obj, Beam) and hasattr(obj, 'Profile'):
                # 获取真实的截面名称进行比较
                profile_obj = obj.Profile
                actual_profile_name = None
                if hasattr(profile_obj, 'ProfileString'):
                    actual_profile_name = profile_obj.ProfileString
                elif hasattr(profile_obj, 'Name'):
                    actual_profile_name = profile_obj.Name
                else:
                    actual_profile_name = profile_obj.ToString()
                
                if actual_profile_name == "Tekla.Structures.Model.Profile":
                    actual_profile_name = repr(profile_obj)
                
                if actual_profile_name == profile_name:
                    found_beams.append(obj)
                    
        if found_beams:
            print(f"找到 {len(found_beams)} 个截面类型为 '{profile_name}' 的型材")
            
            # 处理找到的型材（例如选中或高亮）
            for beam in found_beams:
                # 使用正确的属性访问方式
                identifier = beam.Identifier
                print(f"型材ID: {identifier.ID}")
                # 可以在这里添加更多操作，如选中、修改属性等
                
                # 示例：修改某个属性（请根据需要启用）
                # beam.Name = f"Modified_{profile_name}"
                # beam.Modify()
                
        else:
            print(f"未找到截面类型为 '{profile_name}' 的型材")
            
    except Exception as e:
        print(f"查找型材时出错: {e}")
        import traceback
        traceback.print_exc()

def highlight_beams_by_profile(model, profile_name):
    """
    根据截面类型高亮显示型材
    
    Args:
        model: Tekla Model 实例
        profile_name (str): 要高亮的截面类型名称
    """
    try:
        # 创建选择器来选择特定截面类型的构件
        selector = model.GetModelObjectSelector()
        
        # 这里可以实现更复杂的筛选逻辑
        # 比如通过过滤器选择特定截面类型的构件
        
        print(f"正在查找截面类型为 '{profile_name}' 的构件...")
        find_beams_by_profile(model, profile_name)
        
    except Exception as e:
        print(f"高亮显示构件时出错: {e}")

try:
    # 引用 Tekla Open API 的 DLL
    clr.AddReference('Tekla.Structures')
    clr.AddReference('Tekla.Structures.Model')

    from Tekla.Structures.Model import Model, Beam
    from Tekla.Structures import *
    
    # 连接到 Tekla Model
    model = Model()
    if model.GetConnectionStatus():
        print("模型名称:", model.GetInfo().ModelName)
        print("成功连接到Tekla模型")
        
        # 列出所有梁构件和它们的截面类型
        print("\n=== 模型中所有梁构件信息 ===")
        list_all_beams(model)
        
        # 如果你想查找特定截面类型，可以取消下面几行的注释并修改截面类型
        # print("\n=== 查找特定截面类型 ===")
        # profile_type = "HEA300"  # 修改为你想查找的实际截面类型
        # find_beams_by_profile(model, profile_type)
    else:
        print("无法连接到模型")
        
except Exception as e:
    print(f"Tekla API初始化失败: {e}")
    print("请确保Tekla Structures已正确安装，并检查DLL路径配置")