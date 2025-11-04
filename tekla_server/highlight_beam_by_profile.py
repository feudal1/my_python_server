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
        
    Returns:
        list: 找到的Beam对象列表
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
          
                if actual_profile_name == profile_name:
                    found_beams.append(obj)
                    
        if found_beams:
            print(f"找到 {len(found_beams)} 个截面类型为 '{profile_name}' 的型材")
            
            # 处理找到的型材（例如选中或高亮）
            for beam in found_beams:
                # 使用正确的属性访问方式
                identifier = beam.Identifier
               
                
        else:
            print(f"未找到截面类型为 '{profile_name}' 的型材")
            
        return found_beams
            
    except Exception as e:
        print(f"查找型材时出错: {e}")
        import traceback
        traceback.print_exc()
        return []

def highlight_beams_by_profile(model, profile_name):
    """
    根据截面类型高亮显示型材（不改变选择状态）
    
    Args:
        model: Tekla Model 实例
        profile_name (str): 要高亮显示的截面类型名称
    """
    try:
        print(f"正在查找并高亮显示截面类型为 '{profile_name}' 的构件...")
        found_beams = find_beams_by_profile(model, profile_name)
        
        if found_beams:
            # 创建.NET List<ModelObject>对象
            from System.Collections.Generic import List
            beam_list = List[ModelObject]()
            
            # 将找到的梁添加到.NET列表中
            for beam in found_beams:
                beam_list.Add(beam)
            
            # 使用.NET列表调用Highlight方法
            success = Operation.Highlight(beam_list)
            
            if success:
                print(f"已高亮显示 {len(found_beams)} 个截面类型为 '{profile_name}' 的构件")
                print("这些构件已在Tekla中高亮显示（不影响选择状态）")
            else:
                print("高亮显示操作失败")
        else:
            # 如果未找到构件，清除所有高亮显示
            from System.Collections.Generic import List
            empty_list = List[ModelObject]()
            Operation.Highlight(empty_list)  # 传入空列表以清除高亮
            print(f"未找到截面类型为 '{profile_name}' 的构件进行高亮显示")
            
    except Exception as e:
        print(f"高亮显示构件时出错: {e}")
        import traceback
        traceback.print_exc()

def select_beams_by_profile(model, profile_name):
    """
    根据截面类型选择型材并在Tekla中高亮显示
    
    Args:
        model: Tekla Model 实例
        profile_name (str): 要选择的截面类型名称
    """
    try:
        print(f"正在查找并选择截面类型为 '{profile_name}' 的构件...")
        found_beams = find_beams_by_profile(model, profile_name)
        
        if found_beams:
            # 清除当前选择
            
            # 选择找到的所有构件
            selected_count = 0
            for beam in found_beams:
                if beam.Select():
                    selected_count += 1
                
            print(f"已选择 {selected_count} 个截面类型为 '{profile_name}' 的构件")
            print("这些构件已在Tekla中高亮显示")
        else:
            print(f"未找到截面类型为 '{profile_name}' 的构件进行选择")
            
    except Exception as e:
        print(f"选择构件时出错: {e}")
        import traceback
        traceback.print_exc()

try:
    # 引用 Tekla Open API 的 DLL
    clr.AddReference('Tekla.Structures')
    clr.AddReference('Tekla.Structures.Model')

    from Tekla.Structures.Model import Model, Beam, ModelObject
    from Tekla.Structures.Model.Operations import Operation
    from Tekla.Structures import *
    
    # 连接到 Tekla Model
    model = Model()
    if model.GetConnectionStatus():
        print("模型名称:", model.GetInfo().ModelName)
        print("成功连接到Tekla模型")
        
        # 列出所有梁构件和它们的截面类型
        list_all_beams(model)
        
        # 高亮显示特定截面类型的构件 (例如PD89*3.5)
        print("\n=== 高亮显示特定截面类型 ===")
        profile_type = "RHS200*200*5"  # 修改为你想高亮显示的实际截面类型
        highlight_beams_by_profile(model, profile_type)
        
        # 如果需要也可以选择特定截面类型的构件
        # print("\n=== 选择特定截面类型 ===")
        # profile_type = "PD89*3.5"  # 修改为你想选择的实际截面类型
        select_beams_by_profile(model, profile_type)
    else:
        print("无法连接到模型")
        
except Exception as e:
    print(f"Tekla API初始化失败: {e}")
    print("请确保Tekla Structures已正确安装，并检查DLL路径配置")