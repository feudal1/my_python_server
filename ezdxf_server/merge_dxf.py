import tkinter as tk
from tkinter import filedialog, messagebox
import ezdxf
import os
import logging
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def select_dxf_files():
    """
    使用弹窗选择多个DXF文件
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 弹出文件选择对话框，允许选择多个文件
    file_paths = filedialog.askopenfilenames(
        title="选择要合并的DXF文件",
        filetypes=[
            ("DXF files", "*.dxf"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return list(file_paths)

def is_supported_entity(entity):
    """
    检查实体是否为支持的类型（圆弧、直线、多段线、圆、文字）
    """
    supported_types = ['ARC', 'LINE', 'LWPOLYLINE', 'POLYLINE', 'CIRCLE', 'TEXT', 'MTEXT']
    return entity.dxftype() in supported_types

def extract_quantity_from_filename(filename):
    """
    从文件名中提取数量信息（提取=后面的数字）
    
    Args:
        filename: 文件名
        
    Returns:
        int: 提取的数量，默认为1
    """
    # 获取不带扩展名的文件名
    basename = os.path.splitext(os.path.basename(filename))[0]
    
    # 使用正则表达式查找=后面的数字
    match = re.search(r'=([0-9]+)', basename)
    if match:
        return int(match.group(1))
    return 1  # 没有找到则默认为1

def explode_all_blocks(msp):
    """
    分解模型空间中的所有块引用
    
    Args:
        msp: 模型空间对象
        
    Returns:
        tuple: (分解的块数量, 分解出的实体数量)
    """
    blocks_broken = 0
    exploded_entities = 0
    
    # 多次遍历直到没有更多的INSERT实体
    while True:
        # 收集所有需要分解的INSERT实体
        inserts = [entity for entity in msp if entity.dxftype() == 'INSERT']
        
        # 如果没有需要分解的实体，则退出循环
        if not inserts:
            break
            
        # 分解所有块引用
        for insert in inserts:
            try:
                exploded = insert.explode()
                blocks_broken += 1
                exploded_entities += len(exploded)
                logging.info(f"分解块 '{insert.dxf.name}'，获得 {len(exploded)} 个实体")
            except Exception as e:
                logging.warning(f"分解块 '{insert.dxf.name}' 时出错: {e}")
                
    return blocks_broken, exploded_entities

def calculate_dxf_extents(msp):
    """
    计算模型空间中所有实体的边界框
    
    Args:
        msp: 模型空间对象
        
    Returns:
        tuple: (min_x, min_y, max_x, max_y) 或 None(如果没有实体)
    """
    all_x_coords = []
    all_y_coords = []
    
    for entity in msp:
        try:
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                all_x_coords.extend([start[0], end[0]])
                all_y_coords.extend([start[1], end[1]])
                
            elif entity.dxftype() == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius
                all_x_coords.extend([center[0] - radius, center[0] + radius])
                all_y_coords.extend([center[1] - radius, center[1] + radius])
                
            elif entity.dxftype() == 'ARC':
                center = entity.dxf.center
                radius = entity.dxf.radius
                all_x_coords.extend([center[0] - radius, center[0] + radius])
                all_y_coords.extend([center[1] - radius, center[1] + radius])
                
            elif entity.dxftype() in ['TEXT', 'MTEXT']:
                insert = entity.dxf.insert
                all_x_coords.append(insert[0])
                all_y_coords.append(insert[1])
                
            elif entity.dxftype() == 'LWPOLYLINE':
                with entity.points() as points:
                    for point in points:
                        if len(point) >= 2:
                            all_x_coords.append(point[0])
                            all_y_coords.append(point[1])
                            
        except Exception as e:
            logging.warning(f"计算实体边界时出错: {e}")
            continue
    
    if not all_x_coords or not all_y_coords:
        return None
        
    return (min(all_x_coords), min(all_y_coords), max(all_x_coords), max(all_y_coords))

def copy_entities_with_offset(source_msp, target_msp, offset_x, offset_y):
    """
    将源模型空间中的实体复制到目标模型空间，并应用偏移
    
    Args:
        source_msp: 源模型空间
        target_msp: 目标模型空间
        offset_x: X轴偏移量
        offset_y: Y轴偏移量
    """
    entity_count = 0
    for entity in source_msp:
        try:
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                target_msp.add_line(
                    (start[0] + offset_x, start[1] + offset_y),
                    (end[0] + offset_x, end[1] + offset_y),
                    dxfattribs={
                        'color': entity.dxf.color,
                        'layer': entity.dxf.layer
                    }
                )
                
            elif entity.dxftype() == 'CIRCLE':
                center = entity.dxf.center
                target_msp.add_circle(
                    (center[0] + offset_x, center[1] + offset_y),
                    entity.dxf.radius,
                    dxfattribs={
                        'color': entity.dxf.color,
                        'layer': entity.dxf.layer
                    }
                )
                
            elif entity.dxftype() == 'ARC':
                center = entity.dxf.center
                target_msp.add_arc(
                    (center[0] + offset_x, center[1] + offset_y),
                    entity.dxf.radius,
                    entity.dxf.start_angle,
                    entity.dxf.end_angle,
                    dxfattribs={
                        'color': entity.dxf.color,
                        'layer': entity.dxf.layer
                    }
                )
                
            elif entity.dxftype() == 'TEXT':
                insert = entity.dxf.insert
                target_msp.add_text(
                    entity.dxf.text,
                    dxfattribs={
                        'insert': (insert[0] + offset_x, insert[1] + offset_y),
                        'height': entity.dxf.height,
                        'color': entity.dxf.color,
                        'layer': entity.dxf.layer
                    }
                )
                
            elif entity.dxftype() == 'MTEXT':
                insert = entity.dxf.insert
                target_msp.add_mtext(
                    entity.text,
                    dxfattribs={
                        'insert': (insert[0] + offset_x, insert[1] + offset_y),
                        'char_height': entity.dxf.char_height if hasattr(entity.dxf, 'char_height') else 0.5,
                        'color': entity.dxf.color,
                        'layer': entity.dxf.layer
                    }
                )
                
            elif entity.dxftype() == 'LWPOLYLINE':
                points = [(p[0] + offset_x, p[1] + offset_y) for p in entity.vertices()]
                target_msp.add_lwpolyline(
                    points,
                    dxfattribs={
                        'color': entity.dxf.color,
                        'layer': entity.dxf.layer,
                        'closed': entity.closed
                    }
                )
                
            entity_count += 1
                
        except Exception as e:
            logging.warning(f"复制实体时出错: {e}")
            continue
    
    return entity_count

def copy_multiple_entities_x_direction(source_msp, target_msp, base_offset_x, base_offset_y, count, x_spacing):
    """
    在X方向上复制多个相同的实体
    
    Args:
        source_msp: 源模型空间
        target_msp: 目标模型空间
        base_offset_x: 基础X偏移量
        base_offset_y: 基础Y偏移量
        count: 复制数量
        x_spacing: X方向间距
    """
    copied_count = 0
    for i in range(count):
        offset_x = base_offset_x + (i * x_spacing)
        copied_count += copy_entities_with_offset(source_msp, target_msp, offset_x, base_offset_y)
    return copied_count

def merge_dxf_files(input_files, output_file):
    """
    合并多个DXF文件，先分解块，然后垂直排列，相同零件在X方向重复排列
    
    Args:
        input_files: 输入的DXF文件路径列表
        output_file: 输出文件路径
    """
    if not input_files:
        raise ValueError("没有选择任何文件")
    
    # 创建新的DXF文档
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    
    # 记录处理状态
    processed_files = []
    errors = []
    
    # 初始化偏移量
    current_offset_y = 0
    
    # 遍历所有输入文件
    for i, file_path in enumerate(input_files):
        try:
            # 提取文件名中的数量信息
            quantity = extract_quantity_from_filename(file_path)
            
            # 读取每个DXF文件
            source_doc = ezdxf.readfile(file_path)
            source_msp = source_doc.modelspace()
            
            # 复制图层定义
            for layer in source_doc.layers:
                if layer.dxf.name not in doc.layers:
                    try:
                        doc.layers.new(
                            layer.dxf.name,
                            dxfattribs={
                                'color': layer.dxf.color,
                                'linetype': layer.dxf.linetype,
                            }
                        )
                    except Exception as e:
                        logging.warning(f"图层 {layer.dxf.name} 复制失败: {e}")
            
            # 复制线型定义
            for linetype in source_doc.linetypes:
                if linetype.dxf.name not in doc.linetypes:
                    try:
                        linetype_attribs = {
                            'description': getattr(linetype.dxf, 'description', ''),
                        }
                        
                        if hasattr(linetype.dxf, 'pattern'):
                            pattern = linetype.dxf.pattern
                            if isinstance(pattern, (list, tuple)) and len(pattern) > 0:
                                linetype_attribs['pattern'] = pattern
                        
                        doc.linetypes.new(
                            linetype.dxf.name,
                            dxfattribs=linetype_attribs
                        )
                    except Exception as e:
                        logging.warning(f"线型 {linetype.dxf.name} 复制失败: {e}")
                        try:
                            doc.linetypes.new(linetype.dxf.name)
                        except:
                            pass
            
            # 复制文字样式定义
            for style in source_doc.styles:
                if style.dxf.name not in doc.styles:
                    try:
                        doc.styles.new(
                            style.dxf.name,
                            dxfattribs={
                                'font': getattr(style.dxf, 'font', 'Arial'),
                            }
                        )
                    except Exception as e:
                        logging.warning(f"文字样式 {style.dxf.name} 复制失败: {e}")
            
            # 分解所有块
            blocks_broken, exploded_entities = explode_all_blocks(source_msp)
            logging.info(f"文件 {os.path.basename(file_path)} 分解了 {blocks_broken} 个块，得到 {exploded_entities} 个实体")
            
            # 重新获取模型空间（分解后）
            source_msp = source_doc.modelspace()
            
            # 计算边界框
            extents = calculate_dxf_extents(source_msp)
            if not extents:
                logging.warning(f"文件 {os.path.basename(file_path)} 没有有效实体")
                processed_files.append(f"{os.path.basename(file_path)} (0个实体)")
                continue
                
            min_x, min_y, max_x, max_y = extents
            
            # 计算X方向间距（使用零件宽度加上一定的间隔）
            x_spacing = (max_x - min_x) + 10
            
            # 设置基础偏移（左对齐，竖直堆叠）
            if i == 0:
                base_offset_x = -min_x
                base_offset_y = -min_y
            else:
                base_offset_x = -min_x
                base_offset_y = current_offset_y - min_y
            
            # 在X方向上复制指定数量的实体
            entity_count = copy_multiple_entities_x_direction(
                source_msp, msp, base_offset_x, base_offset_y, quantity, x_spacing)
            
            # 更新下一个文件的起始位置（只需要增加一行的高度）
            current_offset_y += max_y - min_y + 10  # 加间距
            
            processed_files.append(f"{os.path.basename(file_path)} ({entity_count}个实体, 数量:{quantity})")
            
        except Exception as e:
            error_msg = f"处理文件 {file_path} 时出错: {str(e)}"
            errors.append(error_msg)
            logging.error(error_msg)
    
    # 保存合并后的文件
    doc.saveas(output_file)
    
    return processed_files, errors

def main():
    """
    主函数：图形界面合并DXF文件，保存在第一个文件的目录下
    """
    try:
        # 选择多个DXF文件
        print("请选择要合并的DXF文件...")
        input_files = select_dxf_files()
        
        if not input_files:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("提示", "未选择任何文件")
            root.destroy()
            return
        
        print(f"已选择 {len(input_files)} 个文件:")
        for i, file in enumerate(input_files, 1):
            quantity = extract_quantity_from_filename(file)
            print(f"  {i}. {file} (数量: {quantity})")
        
        # 确定输出文件路径（在第一个文件的同一目录下）
        first_file_dir = os.path.dirname(input_files[0])
        output_file = os.path.join(first_file_dir, "merged_dxf_files.dxf")
        
        print(f"\n输出文件将保存到: {output_file}")
        
        # 执行合并操作
        print("\n正在合并文件...")
        processed_files, errors = merge_dxf_files(input_files, output_file)
        
        # 显示结果
        result_message = f"成功合并 {len(processed_files)} 个文件到:\n{output_file}\n\n"
        
        if processed_files:
            result_message += "已处理的文件:\n"
            for file_info in processed_files:
                result_message += f"  • {file_info}\n"
        
        if errors:
            result_message += f"\n错误信息 ({len(errors)} 个):\n"
            for error in errors[:5]:  # 只显示前5个错误
                result_message += f"  • {error}\n"
            if len(errors) > 5:
                result_message += f"  ... 还有 {len(errors) - 5} 个错误\n"
        
        # 显示结果对话框
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("合并完成", result_message)
        root.destroy()
        
        print(result_message)
        
    except Exception as e:
        error_msg = f"发生错误: {str(e)}"
        print(error_msg)
        
        # 显示错误对话框
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("错误", error_msg)
        root.destroy()

if __name__ == "__main__":
    main()