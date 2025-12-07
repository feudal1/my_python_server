# text_llm_to_clipboard.py
import sys
import os
import json
import clipboard  
from pyautocad import Autocad, APoint
from collections import Counter
import uuid
import re

# 添加项目路径到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from llm_server.llm_server import LLMService

llm_service = LLMService()

def get_selected_texts(acad, doc):
    """
    获取用户选择的文本对象（包括单行文本、多行文本和各种标注）
    
    Args:
        acad: AutoCAD应用程序对象
        doc: AutoCAD文档对象
    
    Returns:
        list: 选中的文本对象列表
    """
    try:
        # 提示用户选择对象
        print("请选择文本对象...")
        
        # 创建唯一名称的选择集，避免命名冲突
        selection_set_name = f"TextSelection_{str(uuid.uuid4())[:8]}"
        selection_set = doc.SelectionSets.Add(selection_set_name)
        selection_set.SelectOnScreen()
        
        # 定义有效的对象类型（包括各种标注类型）
        valid_object_names = [
            "AcDbText",              # 单行文本
            "AcDbMText",             # 多行文本
            "AcDbDimension",         # 基本标注
            
            "AcDbAlignedDimension",  # 对齐标注
            "AcDbRotatedDimension",  # 转角标注
          
            "AcDbOrdinateDimension",  # 坐标标注
        ]
        
        # 筛选出文本对象和标注对象
        texts = []
        if selection_set.Count > 0:
            for i in range(selection_set.Count):
                try:
                    entity = selection_set.Item(i)
                    # 检查是否为文本对象或标注对象
                    if entity.ObjectName in valid_object_names:
                        texts.append(entity)
                    else:
                        print(f"对象 {entity.ObjectName} 不是文本或标注对象")
                except Exception as e:
                    print(f"检查对象时出错: {e}")
                    continue
            
            print(f"共选择 {selection_set.Count} 个对象，其中 {len(texts)} 个为文本或标注")
        else:
            print("未选择任何对象")
        
        # 清理选择集
        selection_set.Delete()
        
        return texts
        
    except Exception as e:
        print(f"选择对象时出错: {e}")
        return []

def clean_autocad_text(text):
    """
    清理AutoCAD文本中的格式控制字符，提取纯文本内容
    
    Args:
        text (str): 包含格式控制字符的原始文本
        
    Returns:
        str: 清理后的纯文本
    """
    if not text:
        return text
    
    try:
        # 保存原始文本
        original_text = text
        
        # 处理换行符 \P
        text = text.replace('\\P', '\n')
        
        # 处理字体格式控制 {\f...;内容}
        # 提取其中的文本内容，而非直接删除
        text = re.sub(r'\{\\f[^;]*;([^}]*)\}', r'\1', text)
        
        # 处理颜色和其他格式控制符
        text = re.sub(r'\\\w\d+;', '', text)
        text = re.sub(r'\\[Cc]\d+;', '', text)
        
        # 处理其他RTF控制符
        text = re.sub(r'\\[pblit]\d+;', '', text)
        
        # 移除未配对的花括号
        text = re.sub(r'[{}]', '', text)
        
        # 清理多余的空格和换行
        text = re.sub(r'\n\s*\n', '\n', text)  # 合并多个空行
        text = re.sub(r' +', ' ', text)  # 合并多个空格
        
        return text.strip()
        
    except Exception as e:
        print(f"清理文本时出错: {e}")
        return text

def get_text_content(text_obj):
    """
    获取文本对象的内容（兼容单行文本、多行文本和各种标注）
    
    Args:
        text_obj: 文本对象
    
    Returns:
        str: 文本内容
    """
    try:
        object_name = text_obj.ObjectName
        
        # 修改后的 get_text_content 函数片段
        if object_name == "AcDbText":
            # 单行文本
            raw_text = text_obj.TextString
            return clean_autocad_text(raw_text)
        elif object_name == "AcDbMText":
            # 多行文本
            raw_text = text_obj.TextString
            return clean_autocad_text(raw_text)
        elif object_name in ["AcDbDimension", "AcDbAlignedDimension", "AcDbRotatedDimension",
                           "AcDbOrdinateDimension"]:
            # 各种标注对象
            # 如果标注有自定义文本且不为空，则返回自定义文本，否则返回测量值
            if (hasattr(text_obj, 'TextOverride') and 
                text_obj.TextOverride and 
                text_obj.TextOverride.strip() != "<>" and 
                text_obj.TextOverride.strip() != ""):
                return "尺寸：" + text_obj.TextOverride
            else:
                # 直接使用测量值
                measurement = text_obj.Measurement
                if measurement.is_integer():
                    return "尺寸：" + str(int(measurement))
                else:
                    return "尺寸：" + str(round(measurement, 1))
        else:
            return ""
    except Exception as e:
        print(f"获取文本内容时出错: {e}")
        return ""

def get_text_insertion_point(text_obj):
    """
    获取文本对象的插入点坐标
    
    Args:
        text_obj: 文本对象
    
    Returns:
        APoint: 插入点坐标
    """
    try:
        object_name = text_obj.ObjectName
        
        if object_name == "AcDbText":
            # 单行文本
            return APoint(text_obj.InsertionPoint)
        elif object_name == "AcDbMText":
            # 多行文本
            return APoint(text_obj.InsertionPoint)
        elif object_name in ["AcDbDimension", "AcDbAlignedDimension", "AcDbRotatedDimension",
                           "AcDbOrdinateDimension"]:
            # 标注对象
            return APoint(text_obj.TextPosition)
        else:
            # 默认返回原点
            return APoint(0, 0)
    except Exception as e:
        print(f"获取文本插入点时出错: {e}")
        return APoint(0, 0)

def add_processed_mark(acad, doc, insertion_point):
    """
    在指定位置添加"已处理"标记文本
    
    Args:
        acad: AutoCAD应用程序对象
        doc: AutoCAD文档对象
        insertion_point: 插入点坐标(APoint对象)
    """
    try:
        # 使用AddText创建单行文本，这是最稳定的方式
        text_entity = doc.ModelSpace.AddText("已处理", insertion_point, 200)
        
        # 设置文本颜色为红色 (ACI颜色索引为1代表红色)
        text_entity.Color = 1
        
        print("已在指定位置添加'已处理'标记文本")
        return text_entity
    except Exception as e:
        print(f"添加'已处理'标记时出错: {e}")
        return None

def preprocess_text_contents(text_contents):
    """
    预处理文本内容，提取关键信息
    
    Args:
        text_contents (list): 原始文本内容列表
    
    Returns:
        dict: 包含分类信息的字典
    """
    info = {
        "thickness_info": [],
        "dimension_info": [],
        "quantity_info": [],
        "other_info": []
    }
    
    for i, text in enumerate(text_contents, 1):
        # 查找厚度信息 (如 t20)
        thickness_match = re.search(r'[tT](\d+(?:\.\d+)?)', text)
        if thickness_match:
            info["thickness_info"].append({
                "index": i,
                "text": text,
                "thickness": float(thickness_match.group(1))
            })
        
        # 查找数量信息 (如 16件)
        quantity_match = re.search(r'(\d+(?:\.\d+)?)\s*件', text)
        if quantity_match:
            info["quantity_info"].append({
                "index": i,
                "text": text,
                "quantity": int(float(quantity_match.group(1)))
            })
        
        # 查找尺寸信息 (如 245.0, 560.0)
        if text.startswith("尺寸："):
            dimension_value = text.replace("尺寸：", "")
            try:
                dim_val = float(dimension_value)
                info["dimension_info"].append({
                    "index": i,
                    "text": text,
                    "value": dim_val
                })
            except ValueError:
                pass
                
        # 其他信息
        if not any([thickness_match, quantity_match]) and not text.startswith("尺寸："):
            info["other_info"].append({
                "index": i,
                "text": text
            })
            
    return info

def validate_extracted_specs(extracted_data, text_contents):
    """
    验证LLM提取的规格数据是否合理
    
    Args:
        extracted_data (dict): LLM提取的数据
        text_contents (list): 原始文本内容列表
        
    Returns:
        dict: 验证后的数据，如果有明显错误则返回空或其他标识
    """
    if not extracted_data or "items" not in extracted_data:
        return None
    
    # 提取原始文本中的所有数字，用于后续验证
    all_numbers = set()
    for text in text_contents:
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        all_numbers.update([float(n) for n in numbers])
    
    all_numbers = set(int(n) if float(n).is_integer() else float(n) for n in all_numbers)
    
    validated_items = []
    for item in extracted_data["items"]:
        spec = item.get("specification", "")
        quantity = item.get("quantity", 0)
        
        # 验证规格格式
        if spec.startswith("PL"):
            parts = spec.replace("PL", "").split("*")
            if len(parts) == 3:
                try:
                    thickness, width, length = [float(p) for p in parts]
                    
                    # 检查这些数值是否出现在原始文本中或可接受的默认值
                    check_values = [thickness, width, length]
                    valid = True
                    
                    for val in check_values:
                        if val != 0 and val not in all_numbers:
                            # 特殊情况：允许小数转整数的情况（例如560.0变成560）
                            matched = False
                            for orig_num in all_numbers:
                                if abs(orig_num - val) < 0.5:  # 几乎相等
                                    matched = True
                                    break
                                    
                            
                                    
                            if not matched:
                                print(f"警告：规格 {spec} 中数值 {val} 与原文不匹配")
                                valid = False
                                break
                    
                    # 验证数量是否合理
                    quantity_valid = True
                    # 检查数量是否在合理范围内（例如不能为负数或过大）
                    if quantity <= 0 or quantity > 10000:
                        print(f"警告：规格 {spec} 的数量 {quantity} 不在合理范围内")
                        quantity_valid = False
                    
                    if valid and quantity_valid:
                        validated_items.append(item)
                    else:
                        print(f"警告：规格 {spec} 未通过验证，可能存在幻觉现象")
                        
                except ValueError as e:
                    print(f"警告：规格 {spec} 格式不正确: {e}")
            else:
                print(f"警告：规格 {spec} 格式不正确")
        else:
            print(f"警告：规格 {spec} 格式不正确")
            
    return {"items": validated_items} if validated_items else None

def extract_specs_with_llm(text_contents):
    """
    使用LLM从文本内容中提取规格和数量信息
    
    Args:
        text_contents (list): 文本内容列表
    
    Returns:
        dict: 提取的结果
    """
    # 预处理文本内容
    processed_info = preprocess_text_contents(text_contents)
    
    # 合并所有文本内容
    combined_text = "\n".join([f"{i+1}. {text}" for i, text in enumerate(text_contents)])
    
    # 构建详细提示信息
    detailed_context = []
    if processed_info["thickness_info"]:
        detailed_context.append("厚度信息:")
        for item in processed_info["thickness_info"]:
            detailed_context.append(f"  [{item['index']}] {item['text']} (厚度: {item['thickness']}mm)")
            
    if processed_info["dimension_info"]:
        detailed_context.append("尺寸信息:")
        for item in processed_info["dimension_info"]:
            detailed_context.append(f"  [{item['index']}] {item['text']} (数值: {item['value']}mm)")
            
    if processed_info["quantity_info"]:
        detailed_context.append("数量信息:")
        for item in processed_info["quantity_info"]:
            detailed_context.append(f"  [{item['index']}] {item['text']} (数量: {item['quantity']}件)")

    context_str = "\n".join(detailed_context) if detailed_context else "无特殊结构化信息"
    
    messages = [
        {
            "role": "user",
            "content": f'''请从以下AutoCAD图纸文本中提取钢板规格和数量信息，并以严格的JSON格式返回。

提取规则：
1. 规格格式必须是"PL厚度*宽度*长度"的形式
2. PL代表钢板，厚度、宽度、长度单位均为毫米(mm)
3. 需要综合分析所有文本内容，关联相关信息
4. 数量单位为"件"
5. 返回格式如下：
{{
    "items": [
        {{
            "specification": "PL8*504*1000",
            "quantity": 5
        }},
        {{
            "specification": "PL10*600*1200",
            "quantity": 3
        }}
    ]
}}

重要分析指导：
- 当文本中包含"t数字"时表示厚度，如"t20"表示厚度20mm
- 当文本中包含"数字件"时表示数量，如"16件"表示数量16
- 对于形如"16(+16)件"或"16（+16）件"的数量表达式，应将括号内的数值与前面数值相加作为最终数量(即16+16=32件)
- "尺寸：数值"类型的文本通常表示长度或宽度
- 必须将分离的信息组合成完整规格，如厚度20、数量16、尺寸245和560应组合为"PL20*245*560"，数量16
- 没有尺寸的话规格填写PL0*0*0,反止误导
- **极其重要**：输出的所有数值必须严格来源于输入文本中的数字，不允许任何推测或计算得出的数值

特殊数量表达式处理规则：
- 如果数量信息显示为"A(+B)件"或"A（+B）件"的形式，实际数量应该是A+B的结果
- 示例："16(+16)件"表示实际数量为32件，"10(+5)件"表示实际数量为15件

已识别的关键信息：
{context_str}

需要分析的完整文本内容：
{combined_text}'''
        }
    ]
    
    try:
        result = llm_service.create(messages)
        print("LLM 返回:", result)
        
        # 解析LLM响应
        try:
            result_content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # 清理可能的代码块标记
            if result_content.startswith("```json"):
                result_content = result_content[7:]
            if result_content.startswith("```"):
                result_content = result_content[3:]
            if result_content.endswith("```"):
                result_content = result_content[:-3]
                
            result_content = result_content.strip()
            parsed_result = json.loads(result_content)
            
            return parsed_result
        except Exception as parse_error:
            print(f"解析LLM响应失败: {parse_error}")
            return {}
            
    except Exception as e:
        print(f"调用LLM服务失败: {e}")
        return {}

def format_results_for_clipboard(extracted_data):
    """
    将提取的数据格式化为剪贴板友好的文本
    
    Args:
        extracted_data (dict): 从LLM提取的数据
    
    Returns:
        str: 格式化的文本
    """
    if not extracted_data or "items" not in extracted_data:
        return ""
    
    items = extracted_data["items"]
    if not items:
        return ""
    
    # 格式化为纯数据形式，不包含表头
    lines = []
    for item in items:
        spec = item.get("specification", "")
        qty = item.get("quantity", "")
        if spec and qty != "":
            lines.append(f"{spec}\t{qty}")
    
    return "\n".join(lines)

def main():
    """ 
    主函数 - 获取用户选择的文本对象，发送给LLM提取规格和数量，并复制到剪贴板
    """
    try:
        acad = Autocad(create_if_not_exists=True)
        doc = acad.doc
        print(f"成功连接到 AutoCAD 文档: {doc.Name}")
        
    except Exception as e:
        print(f"无法连接到 AutoCAD: {e}")
        return
    
    try:
        # 获取用户选择的文本对象
        selected_texts = get_selected_texts(acad, doc)
        
        if not selected_texts:
            print("没有选择任何文本对象")
            return
            
        # 获取选中文本的内容
        texts_content = [get_text_content(text_obj) for text_obj in selected_texts]
        
        # 显示选中的文本内容
        print("\n选中的文本内容:")
        for i, text in enumerate(texts_content, 1):
            print(f"{i:2d}. {text}")
        
        # 使用LLM提取规格和数量
        print("\n正在使用LLM提取规格和数量信息...")
        extracted_data = extract_specs_with_llm(texts_content)
        
        if not extracted_data:
            print("未能从LLM获得有效响应")
            return
        
        # 验证提取结果
        print("\n正在验证提取结果...")
        validated_data = validate_extracted_specs(extracted_data, texts_content)
        
        if not validated_data:
            print("警告：提取结果未通过验证，可能存在幻觉现象")
            print("LLM原始输出:")
            print(json.dumps(extracted_data, indent=2, ensure_ascii=False))
            
            # 询问用户是否继续
            user_input = input("\n是否仍要将结果复制到剪贴板？输入'y'确认，其他键取消: ")
            if user_input.lower() != 'y':
                print("操作已取消")
                return
        else:
            extracted_data = validated_data
            print("提取结果已通过验证")
        
        # 格式化结果并复制到剪贴板
        formatted_result = format_results_for_clipboard(extracted_data)
        clipboard.copy(formatted_result)
        
        print("\n提取结果已复制到剪贴板:")
        print("-" * 40)
        print(formatted_result)
        print("-" * 40)
        
        # 添加"已处理"标记文本
        # 获取第一个选中文本的插入点作为标记位置
        if selected_texts:
            insertion_point = get_text_insertion_point(selected_texts[0])
            # 偏移一点位置，避免完全重叠，增加偏移量使标记更可见
            mark_point = APoint(insertion_point.x, insertion_point.y + 300)
            add_processed_mark(acad, doc, mark_point)
        
    except Exception as e:
        print(f"处理过程中出错: {e}")

if __name__ == "__main__":
    main()