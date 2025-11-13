import pandas as pd
import ezdxf
from ezdxf.enums import TextEntityAlignment
import os
import re
from tkinter import Tk, simpledialog, messagebox, Canvas, Label, Frame, Button, Scrollbar, HORIZONTAL, VERTICAL
from tkinter.filedialog import askopenfilename
import threading
import time

# ==============================
# 弹窗输入板材尺寸
# ==============================
def input_sheet_size():
    root = Tk()
    root.withdraw()
    width = simpledialog.askinteger("板材尺寸设置", "请输入板材宽度 (mm):", initialvalue=1490, minvalue=100, maxvalue=50000)
    if width is None:
        messagebox.showinfo("提示", "未输入板材宽度,程序将使用默认值 1200mm")
        width = 1200
    height = simpledialog.askinteger("板材尺寸设置", "请输入板材高度 (mm):", initialvalue=12000, minvalue=100, maxvalue=50000)
    if height is None:
        messagebox.showinfo("提示", "未输入板材高度,程序将使用默认值 2400mm")
        height = 2400
    return width, height

SHEET_WIDTH, SHEET_HEIGHT = input_sheet_size()
MARGIN = 7
PART_CLEARANCE = 0.1

def select_input_file():
    root = Tk()
    root.withdraw()
    file_path = askopenfilename(title="选择零件数据文件", filetypes=[("Data files", "*.xlsx *.csv"), ("All files", "*.*")])
    return file_path

def read_data(file_path):
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.xlsx':
        df = pd.read_excel(file_path, usecols='A:D', header=0)
        df.columns = ['图号', 'OBB长度', 'OBB宽度', '数量']
    elif ext.lower() == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError("不支持的文件格式")

    required_columns = ['图号', 'OBB长度', 'OBB宽度', '数量']
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        if ext.lower() == '.csv' and len(df.columns) >= 4:
            df.columns = ['图号', 'OBB长度', 'OBB宽度', '数量'] + list(df.columns[4:])
        else:
            raise ValueError(f"缺少列: {missing}")
    df = df[required_columns].dropna()
    df['OBB长度'] = pd.to_numeric(df['OBB长度'], errors='coerce')
    df['OBB宽度'] = pd.to_numeric(df['OBB宽度'], errors='coerce')
    df['数量'] = pd.to_numeric(df['数量'], errors='coerce')
    df = df.dropna().astype({'OBB长度': int, 'OBB宽度': int, '数量': int})
    return df

file_path = select_input_file()
if not file_path:
    print("未选择文件,程序退出。")
    exit()

try:
    df = read_data(file_path)
    print(f"成功读取数据,共 {len(df)} 行")
except Exception as e:
    print(f"读取文件时出错: {e}")
    exit()

output_dir = os.path.dirname(file_path)

# ==============================
# 可视化类
# ==============================
class PackingVisualizer:
    def __init__(self, sheet_width, sheet_height):
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.scale = min(0.3, 400 / max(sheet_width, sheet_height))  # 自动缩放比例
        
        # 创建主窗口
        self.root = Tk()
        self.root.title("零件排版实时可视化")
        self.root.geometry("1200x700")
        
        # 创建控制面板
        control_frame = Frame(self.root)
        control_frame.pack(pady=5)
        
        self.start_button = Button(control_frame, text="开始排版", command=self.start_packing)
        self.start_button.pack(side="left", padx=5)
        
        self.pause_button = Button(control_frame, text="暂停", command=self.toggle_pause, state="disabled")
        self.pause_button.pack(side="left", padx=5)
        
        self.resume_button = Button(control_frame, text="继续", command=self.toggle_pause, state="disabled")
        self.resume_button.pack(side="left", padx=5)
        
        # 添加导出DXF按钮
        self.export_dxf_button = Button(control_frame, text="导出DXF", command=self.export_dxf, state="disabled")
        self.export_dxf_button.pack(side="left", padx=5)
        
        # 创建信息标签
        self.info_label = Label(self.root, text="准备开始排版...", font=("Arial", 12))
        self.info_label.pack(pady=5)
        
        # 创建画布和滚动条
        self.canvas_frame = Frame(self.root)
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 创建水平和垂直滚动条
        self.h_scrollbar = Scrollbar(self.canvas_frame, orient=HORIZONTAL)
        self.h_scrollbar.pack(side="bottom", fill="x")
        
        self.v_scrollbar = Scrollbar(self.canvas_frame, orient=VERTICAL)
        self.v_scrollbar.pack(side="right", fill="y")
        
        # 创建画布
        self.canvas = Canvas(
            self.canvas_frame, 
            bg="white",
            xscrollcommand=self.h_scrollbar.set,
            yscrollcommand=self.v_scrollbar.set
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # 配置滚动条
        self.h_scrollbar.config(command=self.canvas.xview)
        self.v_scrollbar.config(command=self.canvas.yview)
        
        # 初始化画布参数
        self.canvas_items = []  # 存储所有画布项目
        self.bin_canvases = []  # 存储每个板材的画布坐标
        
        # 绘制板材区域
        self.draw_initial_layout()
        
        # 状态变量
        self.paused = False
        self.packing_started = False
        self.bins_data = []  # 存储所有板材数据
        self.packing_completed = False  # 排版是否完成
        self.packer = None  # 用于存储packer对象引用
        
    def draw_initial_layout(self):
        """绘制初始布局"""
        # 设置滚动区域
        self.canvas.config(scrollregion=(0, 0, 2000, 800))
        
    def draw_sheet(self, bin_index):
        """为指定板材绘制边界"""
        # 计算板材位置
        x_offset = bin_index * (self.sheet_width * self.scale + 50)  # 50为板材间距
        y_offset = 50
        
        # 如果该板材索引已经存在，检查是否需要重新创建框架
        if bin_index < len(self.bin_canvases) and self.bin_canvases[bin_index] is not None:
            # 检查框架元素是否还存在
            canvas_info = self.bin_canvases[bin_index]
            framework_exists = True
            for element in canvas_info['framework_elements']:
                if element not in self.canvas_items:
                    framework_exists = False
                    break
            
            # 如果框架元素不存在，重新创建
            if not framework_exists:
                # 删除所有旧元素
                if 'elements' in canvas_info:
                    for element in canvas_info['elements']:
                        if element in self.canvas_items:
                            self.canvas.delete(element)
                            self.canvas_items.remove(element)
                
                # 重新创建框架
                self._create_sheet_framework(bin_index, x_offset, y_offset)
        else:
            # 创建新的板材框架
            self._create_sheet_framework(bin_index, x_offset, y_offset)
        
        # 确保bin_canvases数组足够大
        while len(self.bin_canvases) <= bin_index:
            self.bin_canvases.append(None)
            
        # 更新板材坐标信息（如果还没有初始化）
        if self.bin_canvases[bin_index] is None:
            self.bin_canvases[bin_index] = {
                'x_offset': x_offset,
                'y_offset': y_offset,
                'width': self.sheet_width * self.scale,
                'height': self.sheet_height * self.scale,
                'utilization_text': None,
                'framework_elements': [],  # 专门存储框架元素
                'elements': []  # 跟踪该板材的所有元素
            }
        
        # 更新滚动区域
        total_width = (bin_index + 1) * (self.sheet_width * self.scale + 50) + 100
        self.canvas.config(scrollregion=(0, 0, total_width, 800))
        
        return x_offset, y_offset
    
    def _create_sheet_framework(self, bin_index, x_offset, y_offset):
        """创建板材框架元素"""
        # 绘制板材边界
        rect = self.canvas.create_rectangle(
            x_offset, y_offset,
            x_offset + self.sheet_width * self.scale,
            y_offset + self.sheet_height * self.scale,
            outline="black", width=2
        )
        
        # 添加板材标签
        label = self.canvas.create_text(
            x_offset + (self.sheet_width * self.scale) / 2,
            y_offset - 20,
            text=f"板材 {bin_index + 1}",
            font=("Arial", 12, "bold")
        )
        
        # 添加利用率显示文本（初始为空）
        utilization_text = self.canvas.create_text(
            x_offset + (self.sheet_width * self.scale) / 2,
            y_offset + self.sheet_height * self.scale + 20,
            text="利用率: 0.00%",
            font=("Arial", 10)
        )
        
        # 保存框架元素
        framework_elements = [rect, label, utilization_text]
        
        # 更新板材画布信息
        if bin_index < len(self.bin_canvases) and self.bin_canvases[bin_index] is not None:
            self.bin_canvases[bin_index]['framework_elements'] = framework_elements
            self.bin_canvases[bin_index]['elements'] = framework_elements[:]
            self.bin_canvases[bin_index]['utilization_text'] = utilization_text
        else:
            # 确保数组足够大
            while len(self.bin_canvases) <= bin_index:
                self.bin_canvases.append(None)
                
            self.bin_canvases[bin_index] = {
                'x_offset': x_offset,
                'y_offset': y_offset,
                'width': self.sheet_width * self.scale,
                'height': self.sheet_height * self.scale,
                'utilization_text': utilization_text,
                'framework_elements': framework_elements,
                'elements': framework_elements[:]
            }
        
        self.canvas_items.extend(framework_elements)
        
    def draw_rectangle(self, bin_index, x, y, w, h, drawing_number, color="lightblue"):
        """在指定板材上绘制单个零件"""
        # 确保板材画布存在
        while len(self.bin_canvases) <= bin_index:
            self.draw_sheet(len(self.bin_canvases))
            
        # 获取板材偏移量
        bin_canvas = self.bin_canvases[bin_index]
        x_offset = bin_canvas['x_offset']
        y_offset = bin_canvas['y_offset']
        
        # 绘制矩形
        rect = self.canvas.create_rectangle(
            x_offset + x * self.scale, 
            y_offset + y * self.scale,
            x_offset + (x + w) * self.scale, 
            y_offset + (y + h) * self.scale,
            fill=color, outline="blue", width=1
        )
        
        # 添加零件编号文本
        text = self.canvas.create_text(
            x_offset + (x + w/2) * self.scale,
            y_offset + (y + h/2) * self.scale,
            text=drawing_number,
            font=("Arial", max(6, int(min(w, h) * self.scale / 6)))
        )
        
        # 将新元素添加到板材元素列表中（但不添加到框架元素中）
        self.bin_canvases[bin_index]['elements'].extend([rect, text])
        self.canvas_items.extend([rect, text])
        self.root.update()
        return rect
        
    def update_info(self, info_text):
        """更新信息标签"""
        self.info_label.config(text=info_text)
        self.root.update()
        
    def add_new_bin(self):
        """添加新板材"""
        bin_index = len(self.bins_data)
        self.bins_data.append({
            'rects': [],
            'used_area': 0
        })
        self.draw_sheet(bin_index)
        return bin_index
        
    def toggle_pause(self):
        """切换暂停状态"""
        self.paused = not self.paused
        if self.paused:
            self.pause_button.config(state="disabled")
            self.resume_button.config(state="normal")
            self.update_info("排版已暂停")
        else:
            self.pause_button.config(state="normal")
            self.resume_button.config(state="disabled")
            self.update_info("继续排版...")
            
    def start_packing(self):
        """开始排版"""
        self.start_button.config(state="disabled")
        self.pause_button.config(state="normal")
        self.packing_started = True
        self.update_info("开始排版...")
        
    def wait_for_start(self):
        """等待用户点击开始按钮"""
        while not self.packing_started:
            self.root.update()
            time.sleep(0.1)
            
    def check_pause(self):
        """检查是否需要暂停"""
        while self.paused:
            self.root.update()
            time.sleep(0.1)
            
    def show_completion(self):
        """显示完成信息"""
        self.update_info("排版完成!")
        self.pause_button.config(state="disabled")
        self.resume_button.config(state="disabled")
        self.export_dxf_button.config(state="normal")
        self.packing_completed = True

    def export_dxf(self):
        """导出DXF文件"""
        # 在新线程中执行导出操作，避免阻塞GUI
        self.export_dxf_button.config(state="disabled")
        self.update_info("正在导出DXF文件...")
        
        export_thread = threading.Thread(target=self._export_dxf_thread)
        export_thread.daemon = True
        export_thread.start()

    def _export_dxf_thread(self):
        """在单独线程中执行DXF导出"""
        try:
            if self.packer is None:
                self.root.after(0, lambda: self.update_info("导出失败: 未找到排版数据"))
                self.root.after(0, lambda: self.export_dxf_button.config(state="normal"))
                return
                
            # 获取排版数据
            bins_data = self.packer.bins
            
            # 执行导出操作
            saved_files = create_individual_dxf(bins_data, output_dir, self.sheet_width, self.sheet_height, MARGIN)
            
            # 统计信息
            total_parts = sum(len(b['rects']) for b in bins_data)
            total_bins = len(bins_data)
            total_used_area = sum(b['used_area'] for b in bins_data)
            effective_sheet_area = (self.sheet_width - 2*MARGIN) * (self.sheet_height - 2*MARGIN)
            overall_utilization = total_used_area / (total_bins * effective_sheet_area) * 100 if total_bins > 0 and effective_sheet_area > 0 else 0

            completion_info = (
                f"✅ DXF导出完成！\n"
                f"总共导出板材: {total_bins} 张\n"
                f"总共零件数: {total_parts} 个\n"
                f"整体材料利用率: {overall_utilization:.2f}%\n"
            )
            
            # 在主线程中更新GUI
            self.root.after(0, lambda: self.update_info(completion_info))
            self.root.after(0, lambda: self.export_dxf_button.config(state="normal"))
            
            print(f"DXF文件已保存到: {os.path.dirname(saved_files[0]['file_path']) if saved_files else '未知位置'}")
        except Exception as e:
            error_msg = f"导出失败: {str(e)}"
            print(error_msg)
            self.root.after(0, lambda: self.update_info(error_msg))
            self.root.after(0, lambda: self.export_dxf_button.config(state="normal"))

    def clear_visualization(self, bin_indices=None):
        """清除指定板材的可视化内容，如果不指定则清除所有"""
        if bin_indices is None:
            # 只清除零件，保留板材框架
            for bin_index in range(len(self.bin_canvases)):
                if self.bin_canvases[bin_index] is not None:
                    # 删除除了框架元素之外的所有元素
                    canvas_info = self.bin_canvases[bin_index]
                    if 'elements' in canvas_info and 'framework_elements' in canvas_info:
                        # 找出需要删除的元素（非框架元素）
                        remove_elements = []
                        for element in canvas_info['elements']:
                            if element not in canvas_info['framework_elements']:
                                remove_elements.append(element)
                        
                        # 删除需要移除的元素
                        for element in remove_elements:
                            if element in self.canvas_items:
                                self.canvas.delete(element)
                                self.canvas_items.remove(element)
                        
                        # 更新元素列表，只保留框架元素
                        canvas_info['elements'] = canvas_info['framework_elements'][:]
                        
                    # 重置板材数据
                    if bin_index < len(self.bins_data):
                        self.bins_data[bin_index] = {'rects': [], 'used_area': 0}
                    
                    # 更新利用率显示
                    self.update_bin_utilization(bin_index, 0.0)
        else:
            # 只清除指定板材的可视化内容
            for bin_index in sorted(bin_indices, reverse=True):  # 反向排序避免索引问题
                if bin_index < len(self.bin_canvases) and self.bin_canvases[bin_index] is not None:
                    # 删除除了框架元素之外的所有元素
                    canvas_info = self.bin_canvases[bin_index]
                    if 'elements' in canvas_info and 'framework_elements' in canvas_info:
                        # 找出需要删除的元素（非框架元素）
                        remove_elements = []
                        for element in canvas_info['elements']:
                            if element not in canvas_info['framework_elements']:
                                remove_elements.append(element)
                        
                        # 删除需要移除的元素
                        for element in remove_elements:
                            if element in self.canvas_items:
                                self.canvas.delete(element)
                                self.canvas_items.remove(element)
                        
                        # 更新元素列表，只保留框架元素
                        canvas_info['elements'] = canvas_info['framework_elements'][:]
                        
                    # 重置板材数据
                    if bin_index < len(self.bins_data):
                        self.bins_data[bin_index] = {'rects': [], 'used_area': 0}
                    
                    # 更新利用率显示
                    self.update_bin_utilization(bin_index, 0.0)
            
    def update_bin_utilization(self, bin_index, utilization):
        """更新指定板材的利用率显示"""
        if bin_index < len(self.bin_canvases) and self.bin_canvases[bin_index] is not None:
            bin_canvas = self.bin_canvases[bin_index]
            if bin_canvas['utilization_text']:
                utilization_text = f"利用率: {utilization:.2f}%"
                self.canvas.itemconfig(bin_canvas['utilization_text'], text=utilization_text)
                self.root.update()

# ==============================
# 装箱算法（简化版：只保留贪心算法）
# ==============================
class VisualBinPacking:
    def __init__(self, width, height, visualizer=None):
        self.width = width
        self.height = height
        self.effective_width = width - 2 * MARGIN
        self.effective_height = height - 2 * MARGIN
        self.bins = []
        self.visualizer = visualizer
        self.add_new_bin()

    def add_new_bin(self):
        self.bins.append({
            'rects': [],
            'used_area': 0,
            'total_area': self.effective_width * self.effective_height
        })
        
        # 如果有可视化对象，通知添加新板材
        if self.visualizer:
            self.visualizer.add_new_bin()

    def can_fit_in_bin(self, rect_width, rect_height, x, y, bin_data):
        if x + rect_width > self.effective_width or y + rect_height > self.effective_height:
            return False
        for r in bin_data['rects']:
            if (x < r['x'] + r['w'] + PART_CLEARANCE and x + rect_width + PART_CLEARANCE > r['x'] and
                y < r['y'] + r['h'] + PART_CLEARANCE and y + rect_height + PART_CLEARANCE > r['y']):
                return False
        return True

    def find_placement_in_bin(self, w, h, bin_data):
        if not bin_data['rects']:
            return 0, 0
        candidates = {(0, 0)}
        for r in bin_data['rects']:
            candidates.update([
                (r['x'] + r['w'] + PART_CLEARANCE, r['y']),
                (r['x'], r['y'] + r['h'] + PART_CLEARANCE),
                (r['x'] + r['w'] + PART_CLEARANCE, r['y'] + r['h'] + PART_CLEARANCE)
            ])
        best_x, best_y = None, None
        min_y = float('inf')
        for x, y in candidates:
            if 0 <= x <= self.effective_width - w and 0 <= y <= self.effective_height - h:
                if self.can_fit_in_bin(w, h, x, y, bin_data):
                    if y < min_y or (y == min_y and (best_x is None or x < best_x)):
                        best_x, best_y = x, y
                        min_y = y
        return best_x, best_y

    def place_rect(self, w, h, drawing_number, part_id, rotated=False):
        placed = False
        for bin_index, bin_data in enumerate(self.bins):
            x, y = self.find_placement_in_bin(w, h, bin_data)
            if x is not None:
                bin_data['rects'].append({'x': x, 'y': y, 'w': w, 'h': h, 'drawing_number': drawing_number, 'part_id': part_id, 'rotated': rotated})
                bin_data['used_area'] += w * h
                
                # 计算板材利用率
                utilization = (bin_data['used_area'] / bin_data['total_area']) * 100
                
                # 如果有可视化对象，则更新可视化界面
                if self.visualizer:
                    self.visualizer.check_pause()
                    
                    # 更新板材利用率显示
                    self.visualizer.update_bin_utilization(bin_index, utilization)
                    
                    # 绘制最新添加的零件
                    match = re.search(r'(\d+-\d+)', drawing_number)
                    display_drawing_number = match.group(1) if match else drawing_number
                    
                    self.visualizer.draw_rectangle(bin_index, x, y, w, h, display_drawing_number)
                    
                    # 更新信息
                    self.visualizer.update_info(
                        f"板材 {bin_index + 1}: "
                        f"已放置 {len(bin_data['rects'])} 个零件, 利用率: {utilization:.2f}%"
                    )
                    
                    # 短暂延迟以便观察
                    time.sleep(0.01)
                
                placed = True
                break
                
        if not placed:
            self.add_new_bin()
            current_bin = self.bins[-1]
            x, y = self.find_placement_in_bin(w, h, current_bin)
            if x is not None:
                current_bin['rects'].append({'x': x, 'y': y, 'w': w, 'h': h, 'drawing_number': drawing_number, 'part_id': part_id, 'rotated': rotated})
                current_bin['used_area'] += w * h
                
                # 计算板材利用率
                utilization = (current_bin['used_area'] / current_bin['total_area']) * 100
                
                # 如果有可视化对象，则更新可视化界面
                if self.visualizer:
                    self.visualizer.check_pause()
                    bin_index = len(self.bins) - 1
                    
                    # 更新板材利用率显示
                    self.visualizer.update_bin_utilization(bin_index, utilization)
                    
                    # 绘制最新添加的零件
                    match = re.search(r'(\d+-\d+)', drawing_number)
                    display_drawing_number = match.group(1) if match else drawing_number
                    
                    self.visualizer.draw_rectangle(bin_index, x, y, w, h, display_drawing_number)
                    
                    # 更新信息
                    self.visualizer.update_info(
                        f"板材 {bin_index + 1}: "
                        f"已放置 {len(current_bin['rects'])} 个零件, 利用率: {utilization:.2f}%"
                    )
                    
                    # 短暂延迟以便观察
                    time.sleep(0.01)
                
                return True
            return False
        return True

    def pack(self, parts):
        """标准贪心打包"""
        if self.visualizer:
            self.visualizer.wait_for_start()
            
        parts = parts.copy()
        parts['area'] = parts['OBB长度'] * parts['OBB宽度']
        parts = parts.sort_values('area', ascending=False).reset_index(drop=True)
        total_rows = len(parts)
        processed_rows = 0
        for idx, row in parts.iterrows():
            processed_rows += 1
            drawing_number = str(row['图号'])
            length, width, qty = row['OBB长度'], row['OBB宽度'], row['数量']
            match = re.search(r'(\d+-\d+)', drawing_number)
            display_drawing_number = match.group(1) if match else drawing_number
            
            info_text = f"正在处理第 {processed_rows}/{total_rows} 种零件 (图号: {display_drawing_number}, {qty} 个 {length}x{width} mm)"
            if self.visualizer:
                self.visualizer.update_info(info_text)
            print(info_text)
            
            for i in range(qty):
                part_id = f"Part_{idx}_{i+1}"
                if ((length <= self.effective_width and width <= self.effective_height) or
                    (width <= self.effective_width and length <= self.effective_height)):
                    if length <= self.effective_width and width <= self.effective_height:
                        if self.place_rect(length, width, drawing_number, part_id, rotated=False):
                            continue
                    if width <= self.effective_width and length <= self.effective_height:
                        if self.place_rect(width, length, drawing_number, part_id + "(R)", rotated=True):
                            continue
                    # 如果都失败，强制尝试（可能失败）
                    self.place_rect(length, width, drawing_number, part_id, rotated=False)
                else:
                    error_msg = f"错误：零件 {idx} 太大：{length}x{width}"
                    print(error_msg)
                    if self.visualizer:
                        self.visualizer.update_info(error_msg)

# ==============================
# 创建 DXF 输出（不变）
# ==============================
def create_individual_dxf(bins, base_output_dir, sheet_width, sheet_height, margin):
    base_folder_name = "sheet_dxf_files"
    dxf_folder = os.path.join(base_output_dir, base_folder_name)
    counter = 1
    while os.path.exists(dxf_folder):
        dxf_folder = os.path.join(base_output_dir, f"{base_folder_name}_{counter}")
        counter += 1
    os.makedirs(dxf_folder)
    effective_sheet_area = (sheet_width - 2*margin) * (sheet_height - 2*margin)
    saved_files_info = []
    for bin_idx, bin_data in enumerate(bins):
        bin_utilization = bin_data['used_area'] / effective_sheet_area * 100
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        for rect in bin_data['rects']:
            x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']
            x1 = round(x + MARGIN, 1)
            y1 = round(y + MARGIN, 1)
            x2 = round(x + w + MARGIN, 1)
            y2 = round(y + h + MARGIN, 1)
            msp.add_line((x1, y1), (x2, y1), dxfattribs={'color': 7})
            msp.add_line((x2, y1), (x2, y2), dxfattribs={'color': 7})
            msp.add_line((x2, y2), (x1, y2), dxfattribs={'color': 7})
            msp.add_line((x1, y2), (x1, y1), dxfattribs={'color': 7})
        msp.add_line((0, 0), (sheet_width, 0), dxfattribs={'color': 7})
        msp.add_line((sheet_width, 0), (sheet_width, sheet_height), dxfattribs={'color': 7})
        msp.add_line((sheet_width, sheet_height), (0, sheet_height), dxfattribs={'color': 7})
        msp.add_line((0, sheet_height), (0, 0), dxfattribs={'color': 7})
        for rect in bin_data['rects']:
            x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']
            match = re.search(r'(\d+-\d+)', rect['drawing_number'])
            display_drawing_number = match.group(1) if match else rect['drawing_number']
            text = msp.add_text(display_drawing_number, dxfattribs={'height': 15})
            text.set_placement((x + w / 2 + MARGIN, y + h / 2 + MARGIN), align=TextEntityAlignment.MIDDLE_CENTER)
        filename = f"sheet_{bin_idx + 1}_utilization_{bin_utilization:.1f}%.dxf"
        output_file = os.path.join(dxf_folder, filename)
        doc.saveas(output_file)
        saved_files_info.append({'file_path': output_file, 'utilization': bin_utilization, 'part_count': len(bin_data['rects'])})
    return saved_files_info

# ==============================
# 主程序（简化版）
# ==============================
if __name__ == "__main__":
    print(f"当前板材尺寸: {SHEET_WIDTH}mm x {SHEET_HEIGHT}mm")
    print(f"边缘预留: {MARGIN}mm, 零件间距: {PART_CLEARANCE}mm")

    # 创建可视化对象
    visualizer = PackingVisualizer(SHEET_WIDTH, SHEET_HEIGHT)
    
    # 创建支持可视化的装箱对象
    packer = VisualBinPacking(SHEET_WIDTH, SHEET_HEIGHT, visualizer)
    
    # 将packer对象引用传递给visualizer
    visualizer.packer = packer

    # 在新线程中执行排版任务，避免阻塞GUI
    def run_packing():
        # 直接使用贪心算法进行排版
        packer.pack(df)

        # 保存结果
        saved_files = create_individual_dxf(packer.bins, output_dir, SHEET_WIDTH, SHEET_HEIGHT, MARGIN)

        # 统计
        total_parts = sum(len(b['rects']) for b in packer.bins)
        total_bins = len(packer.bins)
        total_used_area = sum(b['used_area'] for b in packer.bins)
        effective_sheet_area = (SHEET_WIDTH - 2*MARGIN) * (SHEET_HEIGHT - 2*MARGIN)
        overall_utilization = total_used_area / (total_bins * effective_sheet_area) * 100

        completion_info = (
            f"\n✅ 排版完成！\n"
            f"总共使用板材: {total_bins} 张\n"
            f"总共放置零件: {total_parts} 个\n"
            f"整体材料利用率: {overall_utilization:.2f}%\n"
        )
        
        for i, b in enumerate(packer.bins):
            util = b['used_area'] / effective_sheet_area * 100
            completion_info += f"  板材 {i+1}: {util:.2f}% ({len(b['rects'])} 个零件)\n"
            
        print(completion_info)
        visualizer.update_info(completion_info)
        visualizer.show_completion()

    # 启动排版线程
    packing_thread = threading.Thread(target=run_packing)
    packing_thread.daemon = True
    packing_thread.start()

    # 启动GUI主循环
    visualizer.root.mainloop()