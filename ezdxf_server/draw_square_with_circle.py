import ezdxf  
import tkinter as tk  
from tkinter import filedialog  
import os  
import sys  
  
def draw_rectangle_with_circle(length, width, diameter):  
    """  
    创建一个包含长方形和中心圆的DXF文档,并添加标注  
      
    Args:  
        length (float): 长方形长度  
        width (float): 长方形宽度  
        diameter (float): 中心圆直径  
      
    Returns:  
        doc (ezdxf.document.Drawing): DXF文档对象  
    """  
    # 创建一个新的DXF文档,setup=True会初始化默认标注样式  
    doc = ezdxf.new("R2010", setup=True)  
    msp = doc.modelspace()  
      
    # 计算长方形的坐标  
    half_length = length / 2  
    half_width = width / 2  
    # 长方形四个角点坐标  
    points = [  
        (-half_length, -half_width),  # 左下角  
        (half_length, -half_width),   # 右下角  
        (half_length, half_width),    # 右上角  
        (-half_length, half_width)    # 左上角  
    ]  
      
    # 绘制长方形(闭合多段线)  
    msp.add_lwpolyline(points + [points[0]], close=True)  
      
    # 绘制中心圆  
    msp.add_circle(center=(0, 0), radius=diameter/2)  
      
    # 添加直径标注  
    add_diameter_dimension(msp, diameter)  
      
    # 添加长度标注  
    add_length_dimension(msp, length, width)  
      
    # 添加宽度标注  
    add_width_dimension(msp, length, width)  
      
    return doc  
  
def add_diameter_dimension(msp, diameter):  
    """  
    添加圆的直径标注  
      
    Args:  
        msp: 模型空间对象  
        diameter (float): 圆的直径  
    """  
    # 计算文字高度为直径值的1/7  
    text_height = diameter / 7  
      
    # 在X轴上添加直径标注,使用EZ_RADIUS样式  
    dim = msp.add_diameter_dim(  
        center=(0, 0),  
        radius=diameter/2,  
        angle=45,  # 45度角标注更美观  
        dimstyle="EZ_RADIUS",  
        override={  
            'dimlfac': 1.0,  # 设置线性缩放因子为1  
            'dimtxt': text_height,  # 文字高度为直径的1/7  
        },  
        dxfattribs={'layer': 'Dimensions'}  
    )  
    # 必须调用render()方法来创建实际的标注几何图形  
    dim.render()  
  
def add_length_dimension(msp, length, width):  
    """  
    添加长度方向的线性标注  
      
    Args:  
        msp: 模型空间对象  
        length (float): 长度  
        width (float): 宽度  
    """  
    half_length = length / 2  
    half_width = width / 2  
      
    # 计算标注位置偏移为长度的1/5  
    offset = length / 5  
    # 计算文字高度为长度值的1/7  
    text_height = length / 7  
      
    # 在长方形下方添加长度标注  
    dim = msp.add_linear_dim(  
        base=(0, -half_width - offset),  # 标注线位置为长度的1/5  
        p1=(-half_length, -half_width),  # 第一个测量点  
        p2=(half_length, -half_width),   # 第二个测量点  
        angle=0,  # 明确指定为水平标注  
        dimstyle="EZDXF",  
        override={  
            'dimlfac': 1.0,  # 设置线性缩放因子为1  
            'dimtxt': text_height,  # 文字高度为长度的1/7  
        },  
        dxfattribs={'layer': 'Dimensions'}  
    )  
    dim.render()  
  
def add_width_dimension(msp, length, width):  
    """  
    添加宽度方向的线性标注  
      
    Args:  
        msp: 模型空间对象  
        length (float): 长度  
        width (float): 宽度  
    """  
    half_length = length / 2  
    half_width = width / 2  
      
    # 计算标注位置偏移为宽度的1/5  
    offset = width / 5  
    # 计算文字高度为宽度值的1/7  
    text_height = width / 7  
      
    # 在长方形左侧添加宽度标注(垂直标注)  
    dim = msp.add_linear_dim(  
        base=(-half_length - offset, 0),   # 标注线位置为宽度的1/5  
        p1=(-half_length, -half_width),  # 第一个测量点  
        p2=(-half_length, half_width),   # 第二个测量点  
        angle=90,  # 明确指定为垂直标注  
        dimstyle="EZDXF",  
        override={  
            'dimlfac': 1.0,  # 设置线性缩放因子为1  
            'dimtxt': text_height,  # 文字高度为宽度的1/7  
        },  
        dxfattribs={'layer': 'Dimensions'}  
    )  
    dim.render()  
  
def save_dxf_file(doc):  
    """  
    弹出文件保存对话框并保存DXF文件  
      
    Args:  
        doc (ezdxf.document.Drawing): 要保存的DXF文档  
    """  
    # 创建隐藏的根窗口  
    root = tk.Tk()  
    root.withdraw()  # 隐藏主窗口  
    root.attributes('-topmost', True)  # 确保对话框在最前面  
      
    # 弹出保存文件对话框  
    file_path = filedialog.asksaveasfilename(  
        title="保存DXF文件",  
        defaultextension=".dxf",  
        filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")]  
    )  
      
    # 销毁根窗口  
    root.destroy()  
      
    # 如果用户选择了路径,则保存文件  
    if file_path:  
        try:  
            doc.saveas(file_path)  
            print(f"文件已保存至: {file_path}")  
            return True  
        except Exception as e:  
            print(f"保存文件时出错: {str(e)}")  
            return False  
    else:  
        print("操作已取消")  
        return False  
  
def get_user_input():  
    """  
    获取用户输入的参数  
    """  
    try:  
        length = float(input("请输入长方形长度: "))  
        width = float(input("请输入长方形宽度: "))  
        diameter = float(input("请输入中心圆直径: "))  
          
        # 验证输入值  
        if length <= 0 or width <= 0 or diameter <= 0:  
            print("错误:长度、宽度和直径必须为正数")  
            return None, None, None  
              
        # 检查圆是否适合在长方形内  
        if diameter >= min(length, width):  
            print("警告:圆的直径应小于长方形的最小边长以确保圆在长方形内")  
            confirm = input("是否继续?(y/n): ")  
            if confirm.lower() != 'y':  
                return None, None, None  
                  
        return length, width, diameter  
    except ValueError:  
        print("错误:请输入有效的数字")  
        return None, None, None  
    except KeyboardInterrupt:  
        print("\n操作已取消")  
        return None, None, None  
  
def main():  
    """  
    主函数:获取用户输入并执行绘图和保存操作  
    """  
    print("开始创建长方形和中心圆的DXF文件...")  
      
    # 获取用户输入  
    length, width, diameter = get_user_input()  
      
    # 检查输入是否有效  
    if length is None or width is None or diameter is None:  
        return  
      
    try:  
        # 创建图形  
        print("正在创建图形...")  
        doc = draw_rectangle_with_circle(length, width, diameter)  
          
        # 保存文件  
        print("准备保存文件...")  
        save_dxf_file(doc)  
          
    except Exception as e:  
        print(f"发生错误: {str(e)}")  
  
if __name__ == "__main__":  
    main()