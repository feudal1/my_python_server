import tkinter as tk
from tkinter import filedialog
import struct
import re

def parse_bin_file(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    
    print("文件大小:", len(data), "字节")
    
    # 查找所有的 LwPolyline 对象
    polylines = []
    pos = 0
    
    while pos < len(data):
        # 查找 "LwPolyline" 字符串
        lw_polyline_pos = data.find(b'LwPolyline', pos)
        if lw_polyline_pos == -1:
            break
            
        # 找到 LwPolyline 后，尝试提取相关信息
        # 根据观察，点数量信息在 LwPolyline 后面一定距离处
        offset = lw_polyline_pos - 20  # 回溯一些字节寻找对象头信息
        if offset < 0:
            offset = 0
            
        # 提取这一段数据用于分析
        segment = data[offset:min(offset+100, len(data))]
        
        # 查找点数量 - 在 LwPolyline 前后查找合理的数字
        point_count = 0
        # 通常在 LwPolyline 字符串之后第12个字节位置
        lw_rel_pos = segment.find(b'LwPolyline')
        if lw_rel_pos != -1 and lw_rel_pos + 12 < len(segment):
            potential_count = segment[lw_rel_pos + 12]
            # 判断是否为合理点数(一般不会超过100)
            if 0 < potential_count < 100:
                point_count = potential_count
        
        # 如果没有找到点数，则尝试其他方式确定
        if point_count == 0:
            # 查看下一个 LwPolyline 之间的数据量估算点数
            next_lw_pos = data.find(b'LwPolyline', lw_polyline_pos + 10)
            if next_lw_pos != -1:
                data_length = next_lw_pos - lw_polyline_pos
                # 假设每个点占用约24字节，反推点数
                estimated_points = data_length // 24
                if 0 < estimated_points < 100:
                    point_count = estimated_points
        
        polylines.append({
            'position': lw_polyline_pos,
            'point_count': point_count,
            'data_start': lw_polyline_pos + len(b'LwPolyline') + 5
        })
        
        pos = lw_polyline_pos + len(b'LwPolyline')
    
    print(f"找到 {len(polylines)} 个 LwPolyline 对象")
    
    # 解析每个 LwPolyline 对象中的点坐标
    for i, polyline in enumerate(polylines):
        print(f"\n--- LwPolyline {i+1} ---")
        print(f"位置: {polyline['position']}")
        print(f"点数量: {polyline['point_count']}")
        
        if polyline['point_count'] > 0:
            points = extract_points(data, polyline['data_start'], polyline['point_count'])
            if points:
                print("坐标点:")
                for j, (x, y) in enumerate(points):
                    print(f"  点{j+1}: ({x:.6f}, {y:.6f})")

def extract_points(data, start_pos, count):
    """
    从指定位置提取坐标点
    """
    points = []
    pos = start_pos
    
    for i in range(count):
        # 检查是否有足够的数据
        if pos + 16 > len(data):
            break
            
        try:
            # 尝试解包两个8字节的双精度浮点数作为坐标
            x = struct.unpack('<d', data[pos:pos+8])[0]
            y = struct.unpack('<d', data[pos+8:pos+16])[0]
            
            # 检查数值是否合理（过滤掉无效值）
            if -1e10 < x < 1e10 and -1e10 < y < 1e10:
                points.append((x, y))
            
            pos += 24  # 移动到下一个点（根据观察每点约占用24字节）
        except Exception as e:
            # 如果解包失败，尝试移动位置继续
            pos += 1
            
    return points

def select_and_parse():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    file_path = filedialog.askopenfilename(
        title="选择BIN文件",
        filetypes=[("BIN files", "*.bin")]
    )
    
    if file_path:
        print(f"正在解析文件: {file_path}")
        parse_bin_file(file_path)
    else:
        print("未选择文件")

if __name__ == "__main__":
    select_and_parse()