# server.py
from pyautocad import Autocad
import win32gui
import win32ui
import win32con
import ctypes
from ctypes import wintypes
from flask import Flask, jsonify, request, Response
import sys
import os
import pythoncom
import pyautogui
import base64
from io import BytesIO
from PIL import Image
import time

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入删除区域功能
from delete_area import change_objects_color_by_window

# 定义PrintWindow常量
PW_CLIENTONLY = 1
PW_RENDERFULLCONTENT = 3

app = Flask(__name__)
# 在 server.py 文件末尾添加新的 API 端点
@app.route('/objects/all', methods=['GET'])
def get_all_objects():
    """
    获取AutoCAD中所有对象的类名
    
    Returns:
        JSON格式的对象类名列表
    """
    try:
        pythoncom.CoInitialize()
        acad = Autocad()
        
        # 获取模型空间
        model_space = acad.doc.ModelSpace
        
        # 收集所有对象的类名
        object_names = []
        
        for obj in model_space:
            if hasattr(obj, 'ObjectName'):
                object_names.append(obj.ObjectName)
            else:
                object_names.append('Unknown')
        
        # 统计各类对象数量
        name_count = {}
        for name in object_names:
            name_count[name] = name_count.get(name, 0) + 1
        
        return jsonify({
            'status': 'success',
            'object_names': object_names,
            'object_count': len(object_names),
            'class_statistics': name_count
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
def capture_autocad_region():
    """
    截取AutoCAD窗口中指定区域的截图，类似于C#中的Util.SavePng()方法
    
    Returns:
        PIL Image对象
    """
    try:
        pythoncom.CoInitialize()
        acad = Autocad()
        hwnd = acad.app.HWND
        
        # 获取窗口位置和大小
        rect = win32gui.GetWindowRect(hwnd)
        window_x, window_y, window_right, window_bottom = rect
        window_width = window_right - window_x
        window_height = window_bottom - window_y
        
        # 尝试使用PrintWindow API
        try:
            # 创建设备上下文
            wDC = win32gui.GetWindowDC(hwnd)
            dcObj = win32ui.CreateDCFromHandle(wDC)
            cDC = dcObj.CreateCompatibleDC()
            dataBitMap = win32ui.CreateBitmap()
            dataBitMap.CreateCompatibleBitmap(dcObj, window_width, window_height)
            cDC.SelectObject(dataBitMap)
            
            # 使用ctypes调用PrintWindow
            result = ctypes.windll.user32.PrintWindow(hwnd, cDC.GetSafeHdc(), PW_RENDERFULLCONTENT)
            
            # 转换为PIL图像
            bmpinfo = dataBitMap.GetInfo()
            bmpstr = dataBitMap.GetBitmapBits(True)
            img = Image.frombuffer(
                'RGB',
                (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                bmpstr, 'raw', 'BGRX', 0, 1)
            
            # 清理资源
            dcObj.DeleteDC()
            cDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, wDC)
            win32gui.DeleteObject(dataBitMap.GetHandle())
            
            if result == 0:
                raise Exception("PrintWindow failed")
                
        except Exception as e:
            # 如果PrintWindow失败，回退到屏幕截图方法
            # 先激活窗口
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.2)  # 给窗口更多时间来完全渲染
            
            # 使用pyautogui截取窗口区域
            screenshot = pyautogui.screenshot(region=(window_x, window_y, window_width, window_height))
            img = screenshot
        
        # 获取截图区域参数
        min_x = float(request.args.get('min_x',  window_width*0.2))
        min_y = float(request.args.get('min_y', window_height*0.2))
        max_x = float(request.args.get('max_x', window_width))
        max_y = float(request.args.get('max_y', window_height*0.9))
        
        min_point = (min_x, min_y)
        max_point = (max_x, max_y)
        
        # 计算相对坐标并确保坐标顺序正确
        left = int(min(min_point[0], max_point[0]))
        top = int(min(min_point[1], max_point[1]))
        right = int(max(min_point[0], max_point[0]))
        bottom = int(max(min_point[1], max_point[1]))
        
        # 检查坐标是否有效
        if left < 0:
            left = 0
        if top < 0:
            top = 0
        if right > window_width:
            right = window_width
        if bottom > window_height:
            bottom = window_height
            
        # 裁剪指定区域
        if right > left and bottom > top:
            cropped_img = img.crop((left, top, right, bottom))
            return cropped_img
        else:
            raise Exception("Invalid crop coordinates")
            
    except Exception as e:
        raise Exception(f"Error capturing AutoCAD region: {str(e)}")
# 在server.py文件中添加以下函数
@app.route('/edit/undo', methods=['GET'])
def undo_operation():
    """
    执行撤销操作（相当于Ctrl+Z）
    
    Returns:
        JSON格式的操作结果
    """
    try:
        pythoncom.CoInitialize()
        acad = Autocad()
        
        # 执行UNDO命令实现撤销操作
        acad.doc.SendCommand("_UNDO\n1\n")  # 1表示执行一次撤销
        
        return jsonify({
            'status': 'success',
            'message': 'Undo operation executed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
# 添加获取模型空间边界框的函数
def get_model_space_bounds():
    """
    获取模型空间的边界框
    :return: (min_x, min_y, max_x, max_y)
    """
    try:
        pythoncom.CoInitialize()
        acad = Autocad()
        
        # 获取模型空间
        model_space = acad.doc.ModelSpace
        
        # 获取模型空间的边界框
        extents = model_space.Extents
        
        # 返回边界框坐标
        min_point = extents.MinimumPoint
        max_point = extents.MaximumPoint
        
        return {
            'min_x': min_point.x,
            'min_y': min_point.y,
            'max_x': max_point.x,
            'max_y': max_point.y,
            'status': 'success'
        }
    except Exception as e:
        print(f"获取模型空间边界框出错: {e}")
        # 返回默认值
        return {
            'min_x': 0,
            'min_y': 0,
            'max_x': 1000,
            'max_y': 1000,
            'status': 'success'
        }

# 添加新的API端点
@app.route('/model/bounds', methods=['GET'])
def model_bounds():
    """
    获取模型空间边界框
    
    Returns:
        JSON格式的边界框信息
    """
    try:
        bounds = get_model_space_bounds()
        return jsonify(bounds)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
# 替换现有的 /command/echo 接口
@app.route('/command/echo', methods=['GET', 'POST'])
def echo_command():
    """
    在AutoCAD命令行显示文本信息
    
    For GET requests:
        Query Parameters:
            message: 要显示在命令行的文本消息
    
    For POST requests:
        Request Body:
            message: 要显示在命令行的文本消息
    
    Returns:
        JSON格式的操作结果
    """
    try:
        # 根据请求方法获取消息参数
        if request.method == 'GET':
            message = request.args.get('message', '')
        else:  # POST
            data = request.get_json()
            message = data.get('message', '') if data else ''
        
        if not message:
            return jsonify({
                'status': 'error',
                'message': 'Message parameter is required'
            }), 400
        
        # 初始化COM组件
        pythoncom.CoInitialize()
        acad = Autocad()
        
        # 使用Utility.Prompt方法在命令行显示消息
        acad.doc.Utility.Prompt(f"{message}\n")
        
        return jsonify({
            'status': 'success',
            'message': f'Message "{message}" sent to command line'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/screenshot/region', methods=['GET'])
def screenshot_region():
    """
    截取AutoCAD指定区域截图，类似于C#中的Util.SavePng()方法
    
    Query Parameters:
        min_x, min_y: 区域左下角点坐标
        max_x, max_y: 区域右上角点坐标
    
    Returns:
        PNG格式的图像数据
    """
    try:
        # 截取指定区域
        screenshot = capture_autocad_region()
        
        # 将截图转换为响应
        buffered = BytesIO()
        screenshot.save(buffered, format="PNG")
        buffered.seek(0)
        
        return Response(buffered.getvalue(), mimetype='image/png')
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/screenshot/region/base64', methods=['GET'])
def screenshot_region_base64():
    """
    截取AutoCAD指定区域截图并返回base64编码
    
    Query Parameters:
        min_x, min_y: 区域左下角点坐标
        max_x, max_y: 区域右上角点坐标
    
    Returns:
        JSON格式的截图数据(base64编码)
    """
    try:
        # 截取指定区域
        screenshot = capture_autocad_region()
        
        # 将截图转换为base64编码
        buffered = BytesIO()
        screenshot.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'status': 'success',
            'message': 'Region screenshot captured',
            'image': img_str,
            'format': 'png'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/delete-area', methods=['GET'])
def delete_area():
    """
    删除指定区域内的对象（GET方法便于调试）
    
    Query Parameters:
        min_x, min_y, min_z: 窗口左下角点坐标
        max_x, max_y, max_z: 窗口右上角点坐标
    
    Returns:
        JSON格式的操作结果
    """
    try:
        # 从查询参数获取坐标值，提供默认值
        min_x = float(request.args.get('min_x', 0))
        min_y = float(request.args.get('min_y', 0))
        min_z = 0
        max_x = float(request.args.get('max_x', 100))
        max_y = float(request.args.get('max_y', 100))
        max_z = 0
        
        # 构造坐标点
        min_point = (min_x, min_y, min_z)
        max_point = (max_x, max_y, max_z)
        
        # 执行删除操作
        count = change_objects_color_by_window(min_point, max_point)
        
        return jsonify({
            'status': 'success',
            'message': f'Deleted {count} objects',
            'deleted_count': count,
            'area': {
                'min_point': min_point,
                'max_point': max_point
            }
        })
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': 'Invalid coordinate values. Please provide numeric values.'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
# 在 server.py 文件中添加以下代码

# 在 server.py 文件中替换 /objects/perimeters 路由为以下代码：

# 替换 /objects/perimeters 路由为以下代码：

# 替换 /objects/perimeters 路由为以下代码：



# 替换 /objects/perimeters 路由为以下代码：

@app.route('/objects/perimeters', methods=['GET'])
def get_selected_objects_perimeters():
    """
    获取AutoCAD中选中对象的周长信息
    
    Returns:
        JSON格式的周长列表和详细信息
    """
    pythoncom.CoInitialize()
    try:
        acad = Autocad()
        
        # 检查是否存在活动文档
        try:
            doc = acad.doc
            if doc is None:
                return jsonify({
                    'status': 'error',
                    'message': '没有打开的AutoCAD文档'
                }), 400
        except:
            return jsonify({
                'status': 'error',
                'message': '无法连接到AutoCAD文档'
            }), 400
        
        # 获取所有已存在的选择集名称
        existing_selection_sets = []
        try:
            for i in range(doc.SelectionSets.Count):
                try:
                    selSetName = doc.SelectionSets.Item(i).Name
                    existing_selection_sets.append(selSetName)
                except:
                    continue
        except:
            pass
        
        # 生成唯一的选择集名称
        sel_set_name = "PerimeterSet_" + str(int(time.time() * 1000))
        # 确保名称唯一
        counter = 0
        original_name = sel_set_name
        while sel_set_name in existing_selection_sets:
            sel_set_name = original_name + "_" + str(counter)
            counter += 1
        
        # 创建选择集
        try:
            sel_set = doc.SelectionSets.Add(sel_set_name)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'无法创建选择集: {str(e)}'
            }), 500
        
        # 提示用户选择对象
        try:
            sel_set.SelectOnScreen()
        except Exception as e:
            # 清理选择集
            try:
                sel_set.Delete()
            except:
                pass
            return jsonify({
                'status': 'error',
                'message': f'选择对象时出错: {str(e)}'
            }), 500
        
        # 如果没有选中对象
        if sel_set.Count == 0:
            # 清理选择集
            try:
                sel_set.Delete()
            except:
                pass
            return jsonify({
                'status': 'success',
                'message': '未选择任何对象',
                'perimeters': [],
                'count': 0
            })
        
        # 存储对象详细信息
        objects_info = []
        
        # 遍历选中的对象
        for i in range(sel_set.Count):
            try:
                obj = sel_set.Item(i)
                
                # 直接获取对象的基本信息
                obj_info = {
                    'index': i,
                    'object_name': 'Unknown',
                    'entity_type': 'Unknown',
                    'handle': 'Unknown'
                }
                
                # 尝试获取基本信息
                try:
                    if hasattr(obj, 'ObjectName'):
                        obj_info['object_name'] = obj.ObjectName
                except:
                    pass
                    
                try:
                    if hasattr(obj, 'EntityType'):
                        obj_info['entity_type'] = obj.EntityType
                except:
                    pass
                    
                try:
                    if hasattr(obj, 'Handle'):
                        obj_info['handle'] = obj.Handle
                except:
                    pass
                
                # 获取所有可访问的属性（使用更保守的方法）
                accessible_attrs = {}
                perimeter = None
                
                # 尝试获取常见属性
                common_attributes = [
                    'Length', 'Circumference', 'Perimeter', 'Area',
                    'ObjectName', 'Handle', 'EntityType', 'Color',
                    'Layer', 'Linetype', 'Visible',"Closed"
                ]
                
                for attr_name in common_attributes:
                    try:
                        if hasattr(obj, attr_name):
                            value = getattr(obj, attr_name)
                            # 特殊处理点坐标
                            if hasattr(value, 'x') and hasattr(value, 'y'):
                                point_data = {
                                    'x': float(value.x) if hasattr(value, 'x') else None,
                                    'y': float(value.y) if hasattr(value, 'y') else None
                                }
                                if hasattr(value, 'z'):
                                    point_data['z'] = float(value.z)
                                accessible_attrs[attr_name] = point_data
                            else:
                                accessible_attrs[attr_name] = value
                                
                                # 检查是否是我们需要的周长相关属性
                                if attr_name in ['Length', 'Circumference', 'Perimeter'] and value is not None:
                                    try:
                                        # 确保是数值类型
                                        perimeter_value = float(value)
                                        if perimeter is None:  # 优先使用第一个找到的有效值
                                            perimeter = perimeter_value
                                    except (ValueError, TypeError):
                                        pass
                    except Exception as attr_err:
                        accessible_attrs[attr_name] = f"Error: {str(attr_err)}"
                
                # 如果还是没有找到周长，尝试使用AutoCAD命令方式获取
                if perimeter is None:
                    try:
                        # 使用AutoCAD的LIST命令获取对象信息
                        # 这里我们只记录需要通过命令获取的事实
                        accessible_attrs['RequiresCommandQuery'] = True
                    except:
                        pass
                
                obj_info.update({
                    'perimeter': perimeter,
                    'accessible_attributes': accessible_attrs
                })
                
                objects_info.append(obj_info)
                
            except Exception as obj_err:
                objects_info.append({
                    'index': i,
                    'error': str(obj_err),
                    'object_name': 'Error',
                    'perimeter': None
                })
        
        # 清理选择集
        try:
            sel_set.Delete()
        except:
            pass
        
        perimeters_list = [obj['perimeter'] for obj in objects_info]
        
        return jsonify({
            'status': 'success',
            'message': f'成功获取{len(objects_info)}个对象的详细信息',
            'objects_count': len(objects_info),
            'perimeters': perimeters_list,
            'valid_perimeter_count': len([p for p in perimeters_list if p is not None]),
            'objects_info': objects_info
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        pythoncom.CoUninitialize()



@app.route('/objects/texts', methods=['GET'])
def get_all_texts():
    """
    获取AutoCAD中所有文本对象的信息
    
    Returns:
        JSON格式的文本对象列表和详细信息
    """
    try:
        pythoncom.CoInitialize()
        acad = Autocad()
        
        # 获取模型空间
        model_space = acad.doc.ModelSpace
        
        # 存储文本对象信息
        texts_info = []
        
        # 遍历模型空间中的所有对象
        for i, obj in enumerate(model_space):
            try:
                # 检查对象是否为文本类型
                if hasattr(obj, 'ObjectName'):
                    object_name = obj.ObjectName
                    
                    # 处理不同类型的文本对象
                    text_info = {
                        'index': i,
                        'object_name': object_name,
                        'handle': getattr(obj, 'Handle', 'Unknown'),
                        'text_string': '',
                        'position': {},
                        'layer': getattr(obj, 'Layer', 'Unknown')
                    }
                    
                    # 处理单行文本 (AcDbText)
                    if object_name == 'AcDbText':
                        text_info['text_string'] = getattr(obj, 'TextString', '')
                        insertion_point = getattr(obj, 'InsertionPoint', None)
                        if insertion_point:
                            text_info['position'] = {
                                'x': insertion_point[0],
                                'y': insertion_point[1],
                                'z': insertion_point[2] if len(insertion_point) > 2 else 0
                            }
                    
                    # 处理多行文本 (AcDbMText)
                    elif object_name == 'AcDbMText':
                        text_info['text_string'] = getattr(obj, 'TextString', '')
                        insertion_point = getattr(obj, 'InsertionPoint', None)
                        if insertion_point:
                            text_info['position'] = {
                                'x': insertion_point[0],
                                'y': insertion_point[1],
                                'z': insertion_point[2] if len(insertion_point) > 2 else 0
                            }
                    
                    # 处理属性文本 (AcDbAttribute)
                    elif object_name == 'AcDbAttribute':
                        text_info['text_string'] = getattr(obj, 'TextString', '')
                        insertion_point = getattr(obj, 'InsertionPoint', None)
                        if insertion_point:
                            text_info['position'] = {
                                'x': insertion_point[0],
                                'y': insertion_point[1],
                                'z': insertion_point[2] if len(insertion_point) > 2 else 0
                            }
                    
                    # 如果找到了文本信息，则添加到结果列表
                    if text_info['text_string']:
                        texts_info.append(text_info)
                        
            except Exception as obj_err:
                # 记录错误但继续处理其他对象
                texts_info.append({
                    'index': i,
                    'error': str(obj_err),
                    'object_name': getattr(obj, 'ObjectName', 'Unknown') if 'obj' in locals() else 'Unknown'
                })
        
        return jsonify({
            'status': 'success',
            'message': f'成功获取{len(texts_info)}个文本对象',
            'texts_count': len(texts_info),
            'texts_info': texts_info
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        pythoncom.CoUninitialize()

@app.route('/document/name', methods=['GET'])
def get_document_name():
    """
    获取当前AutoCAD文档的文件名
    
    Returns:
        JSON格式的文档文件名信息
    """
    try:
        pythoncom.CoInitialize()
        acad = Autocad()
        
        # 检查是否存在活动文档
        try:
            doc = acad.doc
            if doc is None:
                return jsonify({
                    'status': 'error',
                    'message': '没有打开的AutoCAD文档'
                }), 400
        except:
            return jsonify({
                'status': 'error',
                'message': '无法连接到AutoCAD文档'
            }), 400
        
        # 获取文档名称
        doc_name = doc.Name
        doc_path = doc.FullName
        
        return jsonify({
            'status': 'success',
            'document_name': doc_name,
            'full_path': doc_path
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        pythoncom.CoUninitialize()

@app.route('/routes')
def show_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        # 获取端点函数的docstring
        endpoint_func = app.view_functions.get(rule.endpoint)
        docstring = endpoint_func.__doc__.strip() if endpoint_func and endpoint_func.__doc__ else "无描述"
        
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': str(rule),
            'description': docstring
        })
    return {'routes': routes}

import argparse

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=5001, help='Port to run the server on')
args = parser.parse_args()

# 使用指定的端口运行 Flask 应用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port, debug=True)