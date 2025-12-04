# total_line_length_calculator_excel.py
from pyautocad import Autocad, APoint
import math
import time
import os
from openpyxl import Workbook

def get_selected_lines(acad, doc):
    """
    获取用户选择的直线对象
    
    Args:
        acad: AutoCAD应用程序对象
        doc: AutoCAD文档对象
    
    Returns:
        list: 选中的直线对象列表
    """
    try:
        # 提示用户选择对象
        print("请选择直线对象...")
        
        # 使用时间戳创建唯一的选择集名称
        selection_set_name = f"LineSelection_{int(time.time())}"
        selection_set = doc.SelectionSets.Add(selection_set_name)
        selection_set.SelectOnScreen()
        
        # 筛选出直线对象
        lines = []
        if selection_set.Count > 0:
            for i in range(selection_set.Count):
                try:
                    entity = selection_set.Item(i)
                    if entity.ObjectName == "AcDbLine":
                        lines.append(entity)
                except Exception as e:
                    print(f"检查对象时出错: {e}")
                    continue
            
            print(f"共选择 {selection_set.Count} 个对象，其中 {len(lines)} 个为直线")
        else:
            print("未选择任何对象")
        
        # 清理选择集
        selection_set.Delete()
        
        return lines
        
    except Exception as e:
        print(f"选择对象时出错: {e}")
        return []

def calculate_line_length(line):
    """
    计算单个直线的长度
    
    Args:
        line: AutoCAD Line对象
    
    Returns:
        float: 直线的长度
    """
    try:
        # 获取起点和终点坐标
        start_point = line.StartPoint
        end_point = line.EndPoint
        
        # 计算两点间距离
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        dz = end_point[2] - start_point[2]
        
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        return length
    except Exception as e:
        print(f"计算直线长度时出错: {e}")
        return 0

def calculate_total_line_length(lines):
    """
    计算所有直线的长度总和
    
    Args:
        lines: 直线对象列表
    
    Returns:
        tuple: (所有直线的长度总和, 每个直线的详细信息列表)
    """
    total_length = 0
    line_details = []
    
    for i, line in enumerate(lines, 1):
        length = calculate_line_length(line)
        total_length += length
        
        # 收集每个直线的详细信息
        detail = {
            '序号': i,
            '长度': round(length, 2),
            '起点X': round(line.StartPoint[0], 2),
            '起点Y': round(line.StartPoint[1], 2),
            '终点X': round(line.EndPoint[0], 2),
            '终点Y': round(line.EndPoint[1], 2)
        }
        line_details.append(detail)
        
        print(f"第 {i} 条直线长度: {length:.2f}")
    
    return total_length, line_details

def export_to_excel(line_details, total_length, dxf_path):
    """
    将直线数据导出到Excel文件
    
    Args:
        line_details: 直线详细信息列表
        total_length: 总长度
        dxf_path: DXF文件路径
    """
    try:
        # 创建工作簿
        wb = Workbook()
        ws = wb.active
        ws.title = "直线长度统计"
        
        # 写入表头
        headers = ['序号', '长度', '起点X', '起点Y', '终点X', '终点Y']
        ws.append(headers)
        
        # 写入数据
        for detail in line_details:
            row = [
                detail['序号'],
                detail['长度'],
                detail['起点X'],
                detail['起点Y'],
                detail['终点X'],
                detail['终点Y']
            ]
            ws.append(row)
        
        # 添加总计行
        ws.append([])
        ws.append(['总计', round(total_length, 2), '', '', '', ''])
        
        # 生成Excel文件路径（与DXF文件同目录）
        directory = os.path.dirname(dxf_path)
        filename = os.path.splitext(os.path.basename(dxf_path))[0]
        excel_path = os.path.join(directory, f"{filename}_直线长度统计.xlsx")
        
        # 保存文件
        wb.save(excel_path)
        print(f"数据已导出到Excel文件: {excel_path}")
        
    except Exception as e:
        print(f"导出Excel文件时出错: {e}")

def main():
    """
    主函数 - 计算选中直线的长度总和并导出到Excel
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
        # 获取用户选择的直线
        selected_lines = get_selected_lines(acad, doc)
        
        if not selected_lines:
            print("没有选择有效的直线对象")
            return
        
        # 计算所有直线的长度总和及详细信息
        total_length, line_details = calculate_total_line_length(selected_lines)
        
        print(f"\n总共选择了 {len(selected_lines)} 条直线")
        print(f"所有直线的长度总和为: {total_length:.2f}")
        
        # 导出到Excel
        dxf_path = doc.FullName
        if dxf_path:
            export_to_excel(line_details, total_length, dxf_path)
        else:
            print("文档未保存，无法确定导出路径")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    main()