import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

# 隐藏窗口
Tk().withdraw()

print("请选择一张刀路图（S形，从下往上走）...")
file_path = askopenfilename(
    title="选择刀路图像",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

if not file_path:
    print("未选择图片，程序退出。")
    exit()

src = cv2.imread(file_path)
if src is None:
    print("无法读取图片。")
    exit()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
height, width = gray.shape

# === 1. 二值化：区分"点亮"和"未点亮" ===
# 根据你的图：点亮=高亮（>200），未点亮=中等亮度（100~200）
_, binary_lit = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)      # 点亮区域
_, binary_unlit_full = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)  # <=100 的暗区
# 但"未点亮"实际是 100~200，所以：
binary_unlit = cv2.bitwise_and(cv2.bitwise_not(binary_lit), binary_unlit_full)

# === 2. LSD 检测线段 ===
# 注意：OpenCV 的 createLineSegmentDetector 参数是 _refine
lsd = cv2.createLineSegmentDetector(refine=cv2.LSD_REFINE_ADV)
all_lines, _, _, _ = lsd.detect(gray)

if all_lines is None or len(all_lines) == 0:
    print("未检测到任何线段。")
    exit()

# === 3. 创建图结构表示线段连接关系 ===
def create_line_graph_with_networkx(lines, distance_threshold=20):
    """
    使用NetworkX创建线段图结构
    节点：线段
    边：线段之间的连接关系
    """
    # 创建无向图
    G = nx.Graph()
    
    # 将线段添加为节点
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        # 计算线段的起点、终点和中点
        line_info = {
            'start': (x1, y1),
            'end': (x2, y2),
            'mid': ((x1+x2)/2, (y1+y2)/2),
            'length': np.sqrt((x2-x1)**2 + (y2-y1)**2),
            'coords': (x1, y1, x2, y2)
        }
        G.add_node(i, **line_info)
    
    # 判断两点距离是否在阈值内
    def distance(p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    # 检查线段连接关系并添加边
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            line1 = G.nodes[i]
            line2 = G.nodes[j]
            
            # 检查线段的端点是否接近（连接）
            connections = []
            # 检查line1的起点和line2的起点/终点
            if distance(line1['start'], line2['start']) < distance_threshold:
                connections.append(('start', 'start'))
            if distance(line1['start'], line2['end']) < distance_threshold:
                connections.append(('start', 'end'))
            # 检查line1的终点和line2的起点/终点
            if distance(line1['end'], line2['start']) < distance_threshold:
                connections.append(('end', 'start'))
            if distance(line1['end'], line2['end']) < distance_threshold:
                connections.append(('end', 'end'))
            
            # 如果有连接关系，则在图中添加边
            if connections:
                # 添加边，权重为连接距离的倒数（距离越近，权重越大）
                min_dist = min([distance(line1['start'], line2['start']),
                                distance(line1['start'], line2['end']),
                                distance(line1['end'], line2['start']),
                                distance(line1['end'], line2['end'])])
                G.add_edge(i, j, weight=1.0/min_dist if min_dist > 0 else float('inf'))
    
    return G

# 创建线段图结构
line_graph = create_line_graph_with_networkx(all_lines)

# === 4. 按行分组线段 ===
row_height = 25  # 可调！根据你的图调整（你图中约 25px/行）
rows = {}
for i, line in enumerate(all_lines):
    x1, y1, x2, y2 = line[0]
    mid_y = (y1 + y2) / 2
    row_id = int(mid_y // row_height)
    if row_id not in rows:
        rows[row_id] = []
    rows[row_id].append((x1, y1, x2, y2, i))  # 添加索引i

# === 5. 从下往上找"当前行" ===
# 获取所有行ID并按从小到大排序（从下往上）
sorted_row_ids = sorted(rows.keys())
current_row_info = None

for row_id in sorted_row_ids:
    y_min = row_id * row_height
    y_max = (row_id + 1) * row_height

    has_lit = False
    has_unlit = False
    lit_side = None

    for (x1, y1, x2, y2, line_idx) in rows[row_id]:
        # 创建线段掩码
        seg_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.line(seg_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=2)

        # 计算该线段在点亮/未点亮区域中的覆盖率
        total_pixels = cv2.countNonZero(seg_mask)
        if total_pixels == 0:
            continue

        lit_pixels = cv2.countNonZero(cv2.bitwise_and(seg_mask, binary_lit))
        unlit_pixels = cv2.countNonZero(cv2.bitwise_and(seg_mask, binary_unlit))

        lit_ratio = lit_pixels / total_pixels
        unlit_ratio = unlit_pixels / total_pixels

        if lit_ratio > 0.3:   # 超过30%为点亮
            has_lit = True
            mid_x = (x1 + x2) / 2
            if mid_x < width / 2:
                lit_side = 'left'
            else:
                lit_side = 'right'
        if unlit_ratio > 0.3: # 超过30%为未点亮
            has_unlit = True

    if has_lit and has_unlit:
        current_row_info = {
            'row_id': row_id,
            'y_range': (y_min, y_max),
            'lit_side': lit_side
        }
        break

# === 6. 如果没找到"部分点亮"行，尝试找最下面的未点亮行（初始状态） ===
if current_row_info is None:
    # 检查是否全完成
    total_lit_area = cv2.countNonZero(binary_lit)
    if total_lit_area / (height * width) > 0.9:
        print("\n✅ 刀路已完成：所有区域均已点亮。")
    else:
        # 找最下面的有线段的行（作为起始行）
        if sorted_row_ids:
            bottom_row_id = sorted_row_ids[0]  # 最小 row_id（最下方）
            y_min = bottom_row_id * row_height
            y_max = (bottom_row_id + 1) * row_height
            current_row_info = {
                'row_id': bottom_row_id,
                'y_range': (y_min, y_max),
                'lit_side': None  # 全未点亮，假设从左开始
            }
            print("\nℹ️  刀路尚未开始，将从最下行左侧开始。")

# === 7. 输出结果 ===
if current_row_info:
    y_min, y_max = current_row_info['y_range']
    lit_side = current_row_info['lit_side']

    if lit_side is None:
        # 全未点亮：默认从左往右
        print("\n✅ 当前刀路状态分析:")
        print(f"- 当前行 Y 范围: {y_min:.1f} ~ {y_max:.1f}")
        print("- 当前行状态: 全未点亮（起始状态）")
        print("- 下一刀方向: 向右切削（从左侧开始）")
    else:
        side_str = "左边" if lit_side == 'left' else "右边"
        print("\n✅ 当前刀路状态分析:")
        print(f"- 当前行 Y 范围: {y_min:.1f} ~ {y_max:.1f}")
        print(f"- 当前行状态: 既有已点亮线段，也有未点亮线段")
        print(f"- 已点亮侧: {side_str}")
        
        if lit_side == 'left':
            print("- 下一刀方向: 向右切削")
        else:
            print("- 下一刀方向: 向左切削")

        if y_min < height * 0.05:  # 修改条件：如果当前行靠近底部，则提示即将开始
            print("⚠️ 注意：当前行靠近图像底部，可能即将开始或需换向。")

# === 8. 原始分析结果可视化 ===
def draw_result_image():
    img = src.copy()
    
    # 绘制所有检测线段（浅绿）
    for line in all_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 150), 1)
    
    if current_row_info:
        y_min, y_max = current_row_info['y_range']
        # 标记当前行区域（改为蓝色框，线宽为3）
        cv2.rectangle(img, (0, int(y_min)), (width, int(y_max)), (255, 0, 0), 3)  # 蓝色，更粗的线
        
        # 绘制当前行内线段状态
        row_id = current_row_info['row_id']
        if row_id in rows:
            for (x1, y1, x2, y2, line_idx) in rows[row_id]:
                mid_y = (y1 + y2) / 2
                if y_min <= mid_y <= y_max:
                    seg_mask = np.zeros_like(gray, dtype=np.uint8)
                    cv2.line(seg_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
                    lit_px = cv2.countNonZero(cv2.bitwise_and(seg_mask, binary_lit))
                    unlit_px = cv2.countNonZero(cv2.bitwise_and(seg_mask, binary_unlit))
                    
                    if lit_px > unlit_px:
                        color = (0, 255, 0)   # 绿：点亮
                    else:
                        color = (0, 100, 255) # 橙：未点亮
                    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # 标注文字
        text = f"当前行: {'左' if current_row_info['lit_side'] == 'left' else '右'}已点亮" if current_row_info['lit_side'] else "起始行"
        cv2.putText(img, text, (10, int(y_min) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img

result_img = draw_result_image()

# 显示原始分析结果图像
fig1, ax1 = plt.subplots(figsize=(14, 8))
ax1.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), origin='upper')
ax1.set_title('刀路状态分析结果')
ax1.set_xlabel('X 坐标 (像素)')
ax1.set_ylabel('Y 坐标 (像素)')

# 设置刻度标记，每50像素显示一个刻度
x_ticks = range(0, width + 1, max(1, width // 10))
y_ticks = range(0, height + 1, max(1, height // 10))
ax1.set_xticks(x_ticks)
ax1.set_yticks(y_ticks)

# 可选：添加网格线
ax1.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# === 9. 图结构可视化（单独显示）===
def draw_graph_structure():
    plt.figure(figsize=(14, 10))
    
    # 获取连通分量
    components = list(nx.connected_components(line_graph))
    
    # 为每个连通分量分配不同颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
    
    pos = {}
    node_colors = []
    node_sizes = []
    
    # 为每个节点设置位置（使用线段中点）
    for node in line_graph.nodes():
        x, y = line_graph.nodes[node]['mid']
        pos[node] = (x, y)
        # 根据节点所属的连通分量设置颜色
        for i, comp in enumerate(components):
            if node in comp:
                node_colors.append(colors[i])
                break
        node_sizes.append(line_graph.nodes[node]['length'] * 2)  # 根据线段长度设置节点大小
    
    # 绘制图结构
    nx.draw(line_graph, pos, 
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=True,
            font_size=8,
            font_weight='bold',
            edge_color='gray',
            alpha=0.7)
    
    plt.title('线段图结构 - 连通分量可视化')
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.axis('equal')  # 保持坐标轴比例一致
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 显示图结构
draw_graph_structure()

# === 10. 图结构分析和可视化（NetworkX自带布局）===
def draw_graph_with_layout():
    plt.figure(figsize=(14, 10))
    
    # 使用NetworkX的布局算法
    pos = nx.spring_layout(line_graph, k=1, iterations=50)
    
    # 获取连通分量用于着色
    components = list(nx.connected_components(line_graph))
    colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
    
    node_colors = []
    for node in line_graph.nodes():
        for i, comp in enumerate(components):
            if node in comp:
                node_colors.append(colors[i])
                break
    
    # 绘制图
    nx.draw(line_graph, pos, 
            node_color=node_colors,
            node_size=300,
            with_labels=True,
            font_size=8,
            font_weight='bold',
            edge_color='gray',
            alpha=0.7)
    
    plt.title('线段图结构 - 弹簧布局可视化')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 显示布局后的图结构
draw_graph_with_layout()

# === 11. 图结构分析 ===
print(f"\n📊 NetworkX 图结构分析:")
print(f"- 总线段数 (节点数): {line_graph.number_of_nodes()}")
print(f"- 连接关系数 (边数): {line_graph.number_of_edges()}")
print(f"- 连通分量数: {nx.number_connected_components(line_graph)}")

# 分析连通分量
components = list(nx.connected_components(line_graph))
print(f"- 连通组件数: {len(components)}")
for i, comp in enumerate(components[:5]):  # 只显示前5个组件的大小
    print(f"  - 组件 {i+1}: {len(comp)} 个线段")

# === 12. 调试信息 ===
print(f"\n📊 调试信息:")
print(f"- 图像尺寸: {width} x {height}")
print(f"- 检测线段数: {len(all_lines)}")
print(f"- 分组行数: {len(rows)}")
print(f"- 行高度: {row_height} 像素")

# === 13. NetworkX 图分析补充信息 ===
print(f"\n📈 NetworkX 详细分析:")
if line_graph.number_of_nodes() > 0:
    # 计算图的密度
    density = nx.density(line_graph)
    print(f"- 图密度: {density:.4f}")
    
    # 计算节点的度数统计
    degrees = [d for n, d in line_graph.degree()]
    if degrees:
        print(f"- 平均度数: {np.mean(degrees):.2f}")
        print(f"- 最大度数: {max(degrees)}")
        print(f"- 最小度数: {min(degrees)}")
    
    # 如果图是连通的，计算更多属性
    if nx.is_connected(line_graph):
        diameter = nx.diameter(line_graph)
        radius = nx.radius(line_graph)
        print(f"- 图直径: {diameter}")
        print(f"- 图半径: {radius}")