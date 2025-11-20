# total_arc_length_calculator.py
from pyautocad import Autocad, APoint
import math
import time
import os
from openpyxl import Workbook

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
        
        # 使用时间戳创建唯一的选择集名称
        selection_set_name = f"ArcSelection_{int(time.time())}"
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

def calculate_arc_length(arc):
    """
    计算单个圆弧的弧长
    
    Args:
        arc: AutoCAD Arc对象
    
    Returns:
        float: 圆弧的弧长
    """
    try:
        # 获取基本属性
        radius = arc.Radius
        start_angle = arc.StartAngle
        end_angle = arc.EndAngle
        
        # 计算弧长
        if end_angle >= start_angle:
            angle_diff = end_angle - start_angle
        else:
            angle_diff = (2 * math.pi - start_angle) + end_angle
            
        arc_length = radius * angle_diff
        return arc_length
    except Exception as e:
        print(f"计算圆弧弧长时出错: {e}")
        return 0

def calculate_total_arc_length(arcs):
    """
    计算所有圆弧的弧长总和
    
    Args:
        arcs: 圆弧对象列表
    
    Returns:
        tuple: (所有圆弧的弧长总和, 每个圆弧的详细信息列表)
    """
    total_length = 0
    arc_details = []
    
    for i, arc in enumerate(arcs, 1):
        length = calculate_arc_length(arc)
        total_length += length
        
        # 收集每个圆弧的详细信息
        detail = {
            '序号': i,
            '弧长': round(length, 2),
            '半径': round(arc.Radius, 2),
            '起始角度': round(math.degrees(arc.StartAngle), 2),
            '终止角度': round(math.degrees(arc.EndAngle), 2)
        }
        arc_details.append(detail)
        
        print(f"第 {i} 个圆弧弧长: {length:.2f}")
    
    return total_length, arc_details

def export_to_excel(arc_details, total_length, dxf_path):
    """
    将圆弧数据导出到Excel文件
    
    Args:
        arc_details: 圆弧详细信息列表
        total_length: 总弧长
        dxf_path: DXF文件路径
    """
    try:
        # 创建工作簿
        wb = Workbook()
        ws = wb.active
        ws.title = "圆弧弧长统计"
        
        # 写入表头
        headers = ['序号', '弧长', '半径', '起始角度(度)', '终止角度(度)']
        ws.append(headers)
        
        # 写入数据
        for detail in arc_details:
            row = [
                detail['序号'],
                detail['弧长'],
                detail['半径'],
                detail['起始角度'],
                detail['终止角度']
            ]
            ws.append(row)
        
        # 添加总计行
        ws.append([])
        ws.append(['总计', round(total_length, 2), '', '', ''])
        
        # 生成Excel文件路径（与DXF文件同目录）
        directory = os.path.dirname(dxf_path)
        filename = os.path.splitext(os.path.basename(dxf_path))[0]
        excel_path = os.path.join(directory, f"{filename}_弧长统计.xlsx")
        
        # 保存文件
        wb.save(excel_path)
        print(f"数据已导出到Excel文件: {excel_path}")
        
    except Exception as e:
        print(f"导出Excel文件时出错: {e}")

def main():
    """
    主函数 - 计算选中圆弧的弧长总和并导出到Excel
    """
    # 连接到正在运行的 AutoCAD
    try:
        acad = Autocad(create_if_not_exists=True)
        doc = acad.doc
        print(f"成功连接到 AutoCAD 文档: {doc.Name}")
    except Exception as e:
        print("无法连接到 AutoCAD:", e)
        return

    try:
        # 获取用户选择的圆弧
        selected_arcs = get_selected_arcs(acad, doc)
        
        if not selected_arcs:
            print("没有选择有效的圆弧对象")
            return
        
        # 计算所有圆弧的弧长总和及详细信息
        total_length, arc_details = calculate_total_arc_length(selected_arcs)
        
        print(f"\n总共选择了 {len(selected_arcs)} 个圆弧")
        print(f"所有圆弧的弧长总和为: {total_length:.2f}")
        
        # 导出到Excel
        dxf_path = doc.FullName
        if dxf_path:
            export_to_excel(arc_details, total_length, dxf_path)
        else:
            print("文档未保存，无法确定导出路径")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    main()