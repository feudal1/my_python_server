
# add_divide_arcs.py
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

def divide_arc_into_segments(doc, arc, num_segments=6):
    """
    将圆弧平均分成指定数量的小圆弧段
    
    Args:
        doc: AutoCAD文档对象
        arc: AutoCAD Arc对象
        num_segments: 分段数，默认为5
    
    Returns:
        bool: 是否分割成功
    """
    try:
        # 获取原始圆弧属性
        center = arc.Center
        radius = arc.Radius
        start_angle = arc.StartAngle
        end_angle = arc.EndAngle
        
        # 计算角度差
        if end_angle >= start_angle:
            angle_diff = end_angle - start_angle
        else:
            angle_diff = (2 * math.pi - start_angle) + end_angle
        
        # 计算每段的角度差
        segment_angle = angle_diff / num_segments
        
        # 创建模型空间对象
        model_space = doc.ModelSpace
        
        # 创建分割后的圆弧
        for i in range(num_segments):
            # 计算每段的起始和终止角度
            seg_start_angle = start_angle + i * segment_angle
            seg_end_angle = start_angle + (i + 1) * segment_angle
            
            # 角度标准化处理
            while seg_start_angle >= 2 * math.pi:
                seg_start_angle -= 2 * math.pi
            while seg_start_angle < 0:
                seg_start_angle += 2 * math.pi
                
            while seg_end_angle >= 2 * math.pi:
                seg_end_angle -= 2 * math.pi
            while seg_end_angle < 0:
                seg_end_angle += 2 * math.pi
            
            # 创建新的圆弧对象
            new_arc = model_space.AddArc(
                APoint(center[0], center[1]), 
                radius, 
                seg_start_angle, 
                seg_end_angle
            )
            
            doc.Utility.Prompt(f"创建第 {i+1} 段圆弧: 角度 {math.degrees(seg_start_angle):.2f}° 到 {math.degrees(seg_end_angle):.2f}°\n")
        
        doc.Utility.Prompt(f"成功将圆弧分割成 {num_segments} 段\n")
        return True
        
    except Exception as e:
        doc.Utility.Prompt(f"分割圆弧时出错: {e}\n")
        return False

def process_arc_division(doc, arc, index):
    """
    处理单个圆弧的分割
    
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
        
        doc.Utility.Prompt(f"第 {index} 个圆弧长度: {current_length:.2f}\n")
        
        # 分割圆弧
        if divide_arc_into_segments(doc, arc,6 ):
            doc.Utility.Prompt(f"成功将第 {index} 个圆弧分割成5段\n")
            return True
        else:
            doc.Utility.Prompt(f"分割第 {index} 个圆弧时出错\n")
            return False
            
    except Exception as e:
        doc.Utility.Prompt(f"处理第 {index} 个圆弧时发生错误: {e}\n")
        return False

def main():
    """
    主函数 - 将选中的圆弧平均分成5个小圆弧
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
        
        # 统计成功分割的数量
        success_count = 0
        
        # 为每个选中的圆弧进行分割
        for i, arc in enumerate(selected_arcs, 1):
            doc.Utility.Prompt(f"\n--- 处理第 {i} 个圆弧 ---\n")
            
            # 处理单个圆弧分割
            if process_arc_division(doc, arc, i):
                success_count += 1
        
        doc.Utility.Prompt(f"\n总共成功分割了 {success_count} 个圆弧\n")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}\n")

if __name__ == "__main__":
    main()