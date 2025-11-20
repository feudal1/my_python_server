# modify_arc_length_fixed.py
from pyautocad import Autocad, APoint
import math

def get_selected_arcs(acad, doc):
    """
    获取用户选择的圆弧对象
    
    Args:
        acad: AutoCAD应用程序对象
        doc: AutoCAD文档对象
    
    Returns:
        list: 选中的圆弧对象列表
    """
    try:
        # 提示用户选择对象
        print("请选择圆弧对象...")
        
        # 创建唯一名称的选择集，避免命名冲突
        import uuid
        selection_set_name = f"ArcSelection_{str(uuid.uuid4())[:8]}"
        selection_set = doc.SelectionSets.Add(selection_set_name)
        selection_set.SelectOnScreen()
        
        # 筛选出圆弧对象
        arcs = []
        if selection_set.Count > 0:
            for i in range(selection_set.Count):
                try:
                    entity = selection_set.Item(i)
                    if entity.ObjectName == "AcDbArc":
                        arcs.append(entity)
                except Exception as e:
                    print(f"检查对象时出错: {e}")
                    continue
            
            print(f"共选择 {selection_set.Count} 个对象，其中 {len(arcs)} 个为圆弧")
        else:
            print("未选择任何对象")
        
        # 清理选择集
        selection_set.Delete()
        
        return arcs
        
    except Exception as e:
        print(f"选择对象时出错: {e}")
        return []

def calculate_current_arc_length(arc):
    """
    计算当前圆弧的弧长
    
    Args:
        arc: AutoCAD Arc对象
    
    Returns:
        float: 弧长
    """
    try:
        radius = arc.Radius
        start_angle = arc.StartAngle
        end_angle = arc.EndAngle
        
        # 计算角度差
        if end_angle >= start_angle:
            angle_diff = end_angle - start_angle
        else:
            angle_diff = (2 * math.pi - start_angle) + end_angle
            
        # 计算弧长
        arc_length = radius * angle_diff
        return arc_length
    except Exception as e:
        print(f"计算圆弧长度时出错: {e}")
        return None

def modify_arc_length(arc, new_length):
    """
    修改圆弧长度，保持中心位置不变
    
    Args:
        arc: AutoCAD Arc对象
        new_length: 新的弧长
    
    Returns:
        bool: 是否修改成功
    """
    try:
        # 获取当前属性
        center = arc.Center
        radius = arc.Radius
        current_start_angle = arc.StartAngle
        current_end_angle = arc.EndAngle
        
        # 计算当前弧长
        current_length = calculate_current_arc_length(arc)
        if current_length is None:
            return False
            
        doc = arc.Document
        doc.Utility.Prompt(f"调试信息 - 当前弧长: {current_length:.2f}, 目标弧长: {new_length:.2f}\n")
        doc.Utility.Prompt(f"调试信息 - 当前起始角: {math.degrees(current_start_angle):.2f}°, 终止角: {math.degrees(current_end_angle):.2f}°\n")
        
        # 计算新的角度差
        new_angle_diff = new_length / radius
        
        # 计算当前角度差和中间角度
        if current_end_angle >= current_start_angle:
            current_angle_diff = current_end_angle - current_start_angle
            mid_angle = (current_start_angle + current_end_angle) / 2
        else:
            current_angle_diff = (2 * math.pi - current_start_angle) + current_end_angle
            mid_angle = (current_start_angle + current_end_angle + 2 * math.pi) / 2
            if mid_angle >= 2 * math.pi:
                mid_angle -= 2 * math.pi
        
        doc.Utility.Prompt(f"调试信息 - 当前角度差: {math.degrees(current_angle_diff):.2f}°, 中间角: {math.degrees(mid_angle):.2f}°\n")
        
        # 计算角度变化量
        angle_change = new_angle_diff - current_angle_diff
        
        # 保持圆弧中心不变，两端等量缩放
        # 新的起始角和终止角（围绕中间角度对称调整）
        half_new_angle = new_angle_diff / 2
        new_start_angle = mid_angle - half_new_angle
        new_end_angle = mid_angle + half_new_angle
        
        # 角度标准化到 [0, 2π) 范围
        while new_start_angle < 0:
            new_start_angle += 2 * math.pi
        while new_start_angle >= 2 * math.pi:
            new_start_angle -= 2 * math.pi
            
        while new_end_angle < 0:
            new_end_angle += 2 * math.pi
        while new_end_angle >= 2 * math.pi:
            new_end_angle -= 2 * math.pi
        
        doc.Utility.Prompt(f"调试信息 - 新起始角: {math.degrees(new_start_angle):.2f}°, 新终止角: {math.degrees(new_end_angle):.2f}°\n")
        
        # 更新圆弧属性
        arc.StartAngle = new_start_angle
        arc.EndAngle = new_end_angle
        
        # 验证修改后的长度
        verified_length = calculate_current_arc_length(arc)
        doc.Utility.Prompt(f"调试信息 - 修改后验证长度: {verified_length:.2f}\n")
        
        return True
    except Exception as e:
        print(f"修改圆弧长度时出错: {e}")
        return False

def get_new_length_from_cad(doc, current_length):
    """
    从AutoCAD界面获取新的弧长值
    
    Args:
        doc: AutoCAD文档对象
        current_length: 当前弧长（用于显示给用户）
    
    Returns:
        float or None: 新的弧长值，如果输入无效则返回None
    """
    try:
        prompt_message = f"当前弧长为 {current_length:.2f}，请输入新的弧长值: "
        new_length = doc.Utility.GetReal(prompt_message)
        if new_length <= 0:
            doc.Utility.Prompt("弧长必须大于0\n")
            return None
        return new_length
    except Exception as e:
        # 用户可能按了ESC键或者输入了无效值
        doc.Utility.Prompt(f"获取输入时出错: {e}\n")
        return None

def process_single_arc(doc, arc, index):
    """
    处理单个圆弧的长度修改
    
    Args:
        doc: AutoCAD文档对象
        arc: AutoCAD Arc对象
        index: 圆弧索引号
    
    Returns:
        bool: 是否处理成功
    """
    try:
        # 计算当前弧长
        current_length = calculate_current_arc_length(arc)
        if current_length is None:
            doc.Utility.Prompt(f"无法计算第 {index} 个圆弧的当前长度\n")
            return False
        
        # 从AutoCAD界面获取新的弧长值
        new_length = get_new_length_from_cad(doc, current_length)
        if new_length is None:
            return False
        
        # 修改圆弧长度
        if modify_arc_length(arc, new_length):
            # 再次计算修改后的长度以显示准确信息
            final_length = calculate_current_arc_length(arc)
            if final_length is not None:
                doc.Utility.Prompt(f"成功将第 {index} 个圆弧的长度从 {current_length:.2f} 修改为 {final_length:.2f}\n")
            else:
                doc.Utility.Prompt(f"成功修改第 {index} 个圆弧的长度\n")
            return True
        else:
            doc.Utility.Prompt(f"修改第 {index} 个圆弧时出错\n")
            return False
            
    except Exception as e:
        doc.Utility.Prompt(f"处理第 {index} 个圆弧时发生错误: {e}\n")
        return False

def main():
    """
    主函数 - 为选中的圆弧修改长度
    """
    # 连接到正在运行的 AutoCAD
    try:
        acad = Autocad(create_if_not_exists=True)
        doc = acad.doc
        doc.Utility.Prompt(f"成功连接到 AutoCAD 文档: {doc.Name}\n")
    except Exception as e:
        print("无法连接到 AutoCAD:", e)
        return
    
    try:
        # 获取用户选择的圆弧
        selected_arcs = get_selected_arcs(acad, doc)
        
        if not selected_arcs:
            doc.Utility.Prompt("没有选择有效的圆弧对象\n")
            return
        
        # 统计成功修改的数量
        success_count = 0
        
        # 为每个选中的圆弧修改长度
        for i, arc in enumerate(selected_arcs, 1):
            doc.Utility.Prompt(f"\n--- 处理第 {i} 个圆弧 ---\n")
            
            # 处理单个圆弧
            if process_single_arc(doc, arc, i):
                success_count += 1
        
        doc.Utility.Prompt(f"\n总共成功修改了 {success_count} 个圆弧的长度\n")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}\n")


if __name__ == "__main__":
    main()