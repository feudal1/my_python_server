import math
from collections import defaultdict
from pyautocad import Autocad, APoint

def connect_to_autocad():
    """连接到AutoCAD"""
    try:
        acad = Autocad(create_if_not_exists=True)
        return acad
    except Exception as e:
        print(f"连接失败: {e}")
        return None

def find_circle_like_objects(acad):
    """查找类圆形对象组合"""
    try:
        doc = acad.doc
        print(f"当前文档: {doc.FullName}")
        
        # 正确获取模型空间对象数量的方法
        try:
            model_space_count = doc.ModelSpace.Count
            print(f"模型空间对象数: {model_space_count}")
        except:
            print("无法获取模型空间对象数")
        
        # 分类对象
        arcs = []
        lines = []
        
        for obj in doc.ModelSpace:
            if not hasattr(obj, 'ObjectName'):
                continue
                
            try:
                if obj.ObjectName == 'AcDbArc':
                    # 验证圆弧对象的基本属性
                    _ = obj.Center
                    _ = obj.Radius
                    _ = obj.StartPoint
                    _ = obj.EndPoint
                    _ = obj.StartAngle
                    _ = obj.EndAngle
                    arcs.append(obj)
                elif obj.ObjectName == 'AcDbLine':
                    # 验证线对象的基本属性
                    _ = obj.StartPoint
                    _ = obj.EndPoint
                    lines.append(obj)
            except Exception:
                # 忽略不能访问必要属性的对象
                continue
        
        print(f"找到圆弧对象数: {len(arcs)}")
        print(f"找到直线对象数: {len(lines)}")
        
        # 将连接的对象分组
        return arcs, lines
        
    except Exception as e:
        print(f"获取对象时出错: {e}")
        return [],[]

# 主程序
if __name__ == "__main__":
    # 连接AutoCAD
    print("正在连接AutoCAD...")
    acad = connect_to_autocad()
    if not acad:
        print("无法连接到AutoCAD，退出程序")
        exit()
    
    print("AutoCAD连接成功")
    
    # 获取当前文档和模型空间
    try:
        doc = acad.doc
        print(f"活动文档: {doc.FullName}")
        model = doc.ModelSpace
        print("成功获取模型空间")
    except Exception as e:
        print(f"获取文档或模型空间时出错: {e}")
        exit()
    
    # 查找类圆形对象
    print("\n开始查找类圆形对象...")
    arcs, lines = find_circle_like_objects(acad)