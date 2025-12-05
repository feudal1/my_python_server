# combined_cluster_print.py
import os
import subprocess
import time
import uuid
from pyautocad import Autocad, APoint
from dal_cluster_obb import (
    get_selection_or_model_space, 
    get_lines_from_entities, 
    cluster_lines, 
    get_points_from_cluster,
    get_oriented_bounding_box_approx,
    process_clusters
)

def create_output_directory(base_path, doc_name):
    """
    创建输出目录，名称为文档名+UUID
    
    Args:
        base_path (str): 基础路径
        doc_name (str): CAD文档名称
        
    Returns:
        str: 创建的目录路径
    """
    # 清理文档名称中的非法字符
    clean_doc_name = "".join(c for c in doc_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    # 添加UUID
    folder_name = f"{clean_doc_name}_{uuid.uuid4().hex}"
    output_dir = os.path.join(base_path, folder_name)
    
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"创建输出目录: {output_dir}")
    return output_dir

def get_cluster_boundaries(cluster):
    """
    获取聚类的边界坐标
    
    Args:
        cluster: 聚类对象
        
    Returns:
        tuple: (min_x, min_y, max_x, max_y)
    """
    points = get_points_from_cluster(cluster)
    if not points:
        return None
    
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

def get_cluster_x_position(cluster):
    """
    获取聚类的X轴位置（最小X坐标）
    
    Args:
        cluster: 聚类对象
        
    Returns:
        float: 聚类的最小X坐标值
    """
    points = get_points_from_cluster(cluster)
    if not points:
        return 0
    
    x_coords = [p[0] for p in points]
    return min(x_coords)

def print_cluster_to_pdf(acad, doc, cluster_id, boundaries, output_dir):
    """
    将单个聚类区域打印到PDF
    
    Args:
        acad: AutoCAD应用对象
        doc: AutoCAD文档对象
        cluster_id (int): 聚类ID
        boundaries (tuple): 边界坐标 (min_x, min_y, max_x, max_y)
        output_dir (str): 输出目录路径
    """
    try:
        min_x, min_y, max_x, max_y = boundaries
        
        # 计算中心点
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # 计算偏移量以确保边界
        offset = max(max_x - min_x, max_y - min_y) * 0.05  # 添加5%的边距
        
        # 定义打印窗口的两个角点
        point1 = acad.aDouble(min_x - offset, min_y - offset)
        point2 = acad.aDouble(max_x + offset, max_y + offset)
        
        # 计算宽度和高度
        width = abs(max_x - min_x)
        height = abs(max_y - min_y)
        
        # 设置打印配置
        doc.ActiveLayout.ConfigName = "DWG To PDF.pc3"
        doc.ActiveLayout.CenterPlot = True
        doc.ActiveLayout.CanonicalMediaName = "ISO_A4_(210.00_x_297.00_MM)"
        
        # 自动识别横竖方向
        if width > height:
            # 横向打印
            doc.ActiveLayout.PlotRotation = 1
            print(f"聚类 {cluster_id}: 检测为横向布局")
        else:
            # 纵向打印
            doc.ActiveLayout.PlotRotation = 0
            print(f"聚类 {cluster_id}: 检测为纵向布局")
        
        # 设置打印窗口
        doc.ActiveLayout.SetWindowToPlot(point1, point2)
        doc.ActiveLayout.PlotType = 4  # 打印窗口
        
        # 构造PDF文件路径
        pdf_filename = f"cluster_{cluster_id}.pdf"
        full_pdf_path = os.path.normpath(os.path.join(output_dir, pdf_filename))
        
        # 删除已存在的文件
        if os.path.exists(full_pdf_path):
            try:
                os.remove(full_pdf_path)
                print(f"删除已存在的文件: {full_pdf_path}")
            except Exception as e:
                print(f"无法删除已存在的文件: {e}")
        
        # 打印到文件
        doc.Plot.PlotToFile(full_pdf_path)
        print(f"聚类 {cluster_id}: 打印完成 -> {full_pdf_path}")
        
        return full_pdf_path
        
    except Exception as e:
        print(f"打印聚类 {cluster_id} 时出错: {e}")
        return None

def main():
    """主函数"""
    try:
        acad = Autocad(create_if_not_exists=True)
        doc = acad.doc
        print(f"成功连接到 AutoCAD 文档: {doc.Name}")
    except Exception as e:
        print("无法连接到 AutoCAD:", e)
        return
    
    try:
        # 获取选定对象
        entities = get_selection_or_model_space(acad, doc)
        
        if entities is None:
            print("获取对象过程中发生错误，程序退出")
            return
        
        if not entities:
            print("没有找到任何对象，程序退出")
            return
        
        print(f"处理 {len(entities)} 个对象")
        
        # 提取线段数据
        lines = get_lines_from_entities(entities)
        
        if len(lines) < 1:
            print("未找到有效线段")
            return
        
        # 对线段进行聚类
        clusters = cluster_lines(lines, distance_threshold=2000.0)
        print(f"\n总共识别到 {len(clusters)} 个聚类")
        
        if not clusters:
            print("未发现任何聚类，程序退出")
            return
        
        # 按照X轴坐标对聚类进行排序
        clusters.sort(key=get_cluster_x_position)
        print(f"\n已按X轴坐标对聚类进行排序")
        
        # 创建输出目录
        dxf_path = doc.FullName
        base_path = os.path.dirname(dxf_path)
        output_dir = create_output_directory(base_path, os.path.splitext(doc.Name)[0])
        
        # 处理每个聚类并打印
        print("\n开始逐个打印聚类...")
        for i, cluster in enumerate(clusters):
            print(f"\n处理聚类 {i}...")
            
            # 获取聚类边界
            boundaries = get_cluster_boundaries(cluster)
            if not boundaries:
                print(f"聚类 {i}: 无法确定边界，跳过")
                continue
            
            # 打印聚类到PDF
            pdf_path = print_cluster_to_pdf(acad, doc, i, boundaries, output_dir)
            if pdf_path:
                print(f"聚类 {i}: 成功保存到 {pdf_path}")
            else:
                print(f"聚类 {i}: 打印失败")
        
        print(f"\n所有聚类打印完成，文件保存在: {output_dir}")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")

if __name__ == "__main__":
    main()