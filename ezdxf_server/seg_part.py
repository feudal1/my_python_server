import ezdxf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import tkinter as tk
from tkinter import filedialog

# 创建一个隐藏的根窗口用于文件对话框
root = tk.Tk()
root.withdraw()  # 隐藏主窗口

# 弹窗选择DXF文件
dxf_file_path = filedialog.askopenfilename(
    title="选择DXF文件",
    filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")]
)

# 检查用户是否选择了文件
if not dxf_file_path:
    print("未选择文件，程序退出")
    exit()

# 加载DXF文件
doc = ezdxf.readfile(dxf_file_path)
msp = doc.modelspace()  # 获取模型空间

# 创建一个图形对象和坐标轴对象
fig, ax = plt.subplots()

# 准备一个列表用于存储所有的线段
lines = []
        
# 遍历模型空间中的所有实体
for entity in msp:
    if entity.dxftype() == 'LINE':  # 如果是直线
        # 添加起点和终点到lines列表
        lines.append([(entity.dxf.start.x, entity.dxf.start.y), 
                      (entity.dxf.end.x, entity.dxf.end.y)])
    elif entity.dxftype() == 'SPLINE':  # 如果是样条曲线
        fit_points = entity.control_points
        # 将样条曲线的控制点首尾相连作为线段
        lines.append([(fit_points[0][0], fit_points[0][1]), 
                      (fit_points[-1][0], fit_points[-1][1])])
        # 在遍历实体的代码块中添加以下条件分支
    elif entity.dxftype() == 'LWPOLYLINE':  # 处理轻量多段线
        points = entity.get_points()  # 获取多段线的所有顶点
        # 将连续的点连接成线段
        for i in range(len(points) - 1):
            lines.append([(points[i][0], points[i][1]), 
                        (points[i+1][0], points[i+1][1])])
        # 如果多段线闭合，则连接最后一个点与第一个点
        if entity.is_closed:
            lines.append([(points[-1][0], points[-1][1]), 
                        (points[0][0], points[0][1])])
    elif entity.dxftype() == 'ARC':  # 如果是圆弧
        arc = entity
        center = np.array([arc.dxf.center.x, arc.dxf.center.y])
        radius = arc.dxf.radius
        start_angle = arc.dxf.start_angle
        end_angle = arc.dxf.end_angle
        if start_angle > end_angle:
            end_angle += 360  # 确保角度范围正确
        angle_step = (end_angle - start_angle) / 15  # 分割成小段
        angles = np.arange(start_angle, end_angle, angle_step)
        arc_points = [center + radius * np.array([np.cos(np.deg2rad(angle)), 
                      np.sin(np.deg2rad(angle))]) for angle in angles]
        # 将圆弧分割为多个线段
        for i in range(len(arc_points) - 1):
            lines.append([tuple(arc_points[i]), tuple(arc_points[i+1])])
    else:
        print('未处理的实体类型:', entity.dxftype())

# 聚类线段到不同的聚落
class Cluster:
    def __init__(self, lines=[], min_x=0, max_x=0, min_y=0, max_y=0):
        self.lines = lines
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

clusters= []
remaining_lines = list(lines)  # 创建副本用于处理

while remaining_lines:
    # 取出第一个线段作为种子
    seed = remaining_lines.pop(0)
    current_cluster = [seed]
    
    # 初始化聚落的边界
    min_x = min(seed[0][0], seed[1][0])
    max_x = max(seed[0][0], seed[1][0])
    min_y = min(seed[0][1], seed[1][1])
    max_y = max(seed[0][1], seed[1][1])
    
    while True:
        # 扩展边界
        expanded_min_x = min_x - 5
        expanded_max_x = max_x + 5
        expanded_min_y = min_y - 5
        expanded_max_y = max_y + 5
        
        # 寻找在扩展边界内的线段
        to_add = []
        for line in list(remaining_lines):  # 遍历副本避免问题
            in_cluster = False
            # 检查线段的两个端点是否在扩展后的边界内
            for point in line:
                if (expanded_min_x <= point[0] <= expanded_max_x and
                    expanded_min_y <= point[1] <= expanded_max_y):
                    in_cluster = True
                    break
            if in_cluster:
                to_add.append(line)
        
        # 如果没有找到，结束循环
        if not to_add:
            break
        
        # 将找到的线段加入当前聚落，并更新边界
        for line in to_add:
            current_cluster.append(line)
            remaining_lines.remove(line)
            # 更新当前聚落的边界
            line_min_x = min(p[0] for p in line)
            line_max_x = max(p[0] for p in line)
            line_min_y = min(p[1] for p in line)
            line_max_y = max(p[1] for p in line)
            min_x = min(min_x, line_min_x)
            max_x = max(max_x, line_max_x)
            min_y = min(min_y, line_min_y)
            max_y = max(max_y, line_max_y)
    for cluster in clusters:
        if min_x<=cluster.min_x and min_y<=cluster.min_y\
        and max_x>=cluster.max_x and max_y>=cluster.max_y:
         current_cluster.extend(cluster.lines)
         clusters.remove(cluster )
    # 将当前聚落加入结果列表
    current_cluster_c=Cluster(lines=current_cluster,min_x=min_x,max_x=max_x,min_y=min_y,max_y=max_y)
    clusters.append(current_cluster_c)

print(f"找到了{len(clusters)}个聚落")
# 为每个聚落分配颜色并绘制
colors = plt.cm.tab20.colors  # 使用预定义颜色
min_x_draw=0
max_x_draw=0
min_y_draw=0
max_y_draw=0
for i, cluster in enumerate(clusters):
    lines=cluster.lines
    color = colors[i % len(colors)]
    lc = LineCollection(lines, colors=color, linewidths=0.5)
    ax.add_collection(lc)  
 
 
 
   
    max_x=cluster.max_x
    min_x=cluster.min_x  
    max_y=cluster.max_y
    min_y=cluster.min_y
    min_x_draw= min_x if min_x<min_x_draw or i==0 else min_x_draw
    max_x_draw=max_x if max_x>max_x_draw  or i==0 else max_x_draw
    min_y_draw= min_y if min_y<min_y_draw  or i==0 else min_y_draw
    max_y_draw=max_y if max_y>max_y_draw  or i==0 else max_y_draw
    #画边界框，可注释
    '''
    lengthx= round(max_x  -min_x,1)
    lengthy= round(max_y -min_y,1)
    print(f'{i}视图左下角: ({round(min_x,1)}, {round(min_y,1)})')
    rect = plt.Rectangle((min_x, min_y), lengthx, lengthy, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    # 标记左下角序号
    ax.text(min_x, min_y, str(i), color='blue', fontsize=12)'
    '''
 
ax.set_xlim(min_x_draw, max_x_draw)  # x 轴范围从 -1 到 5
ax.set_ylim(min_y_draw, max_y_draw)  # y 轴范围从 -5 到 20
ax.set_aspect('equal')  # 固定比例
plt.show()