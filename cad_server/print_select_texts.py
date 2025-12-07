from pyautocad import Autocad, APoint
from collections import Counter
import os
import uuid

def get_selected_texts(acad, doc):
    """
    获取用户选择的文本对象（包括单行文本和多行文本）
    
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
        
        # 筛选出文本对象（包括单行文本和多行文本）
        texts = []
        if selection_set.Count > 0:
            for i in range(selection_set.Count):
                try:
                    entity = selection_set.Item(i)
                    # 检查是否为文本对象（单行文本或多行文本）
                    if entity.ObjectName in ["AcDbText", "AcDbMText"]:
                        texts.append(entity)
                except Exception as e:
                    print(f"检查对象时出错: {e}")
                    continue
            
            print(f"共选择 {selection_set.Count} 个对象，其中 {len(texts)} 个为文本（单行文本或多行文本）")
        else:
            print("未选择任何对象")
        
        # 清理选择集
        selection_set.Delete()
        
        return texts
        
    except Exception as e:
        print(f"选择对象时出错: {e}")
        return []

def get_text_content(text_obj):
    """
    获取文本对象的内容（兼容单行文本和多行文本）
    
    Args:
        text_obj: 文本对象（AcDbText 或 AcDbMText）
    
    Returns:
        str: 文本内容
    """
    try:
        if text_obj.ObjectName == "AcDbText":
            # 单行文本
            return text_obj.TextString
        elif text_obj.ObjectName == "AcDbMText":
            # 多行文本
            return text_obj.TextString
        else:
            return ""
    except Exception as e:
        print(f"获取文本内容时出错: {e}")
        return ""

def list_all_texts(acad):
    """
    列出当前CAD文档中的所有文本内容（包括单行文本和多行文本）
    
    :param acad: Autocad实例
    """
    texts = []
    
    # 遍历 ModelSpace 中的所有对象
    for obj in acad.iter_objects(block=acad.model, dont_cast=True):
        try:
            # 检查是否为文本对象（单行文本或多行文本）
            if obj.ObjectName in ['AcDbText', 'AcDbMText']:
                # 获取文本内容
                texts.append(get_text_content(obj))
        except Exception:
            continue
    
    return texts

def list_all_object_types(acad):  
    """  
    列出当前CAD文档中的所有对象类型  
    
    :param acad: Autocad实例  
    """  
    object_types = []  
    
    # 遍历 ModelSpace 中的所有对象  
    for obj in acad.iter_objects(block=acad.model, dont_cast=True):  
        try:  
            object_types.append(obj.ObjectName)  
        except Exception:  
            continue  
    
    return object_types  

def export_selected_texts_to_file(acad, doc, selected_texts):
    """
    将选中的文本内容导出到CAD同目录下的txt文件中，文件名为CAD名称+uuid4
    
    :param acad: Autocad实例
    :param doc: AutoCAD文档对象
    :param selected_texts: 选中的文本对象列表
    """
    try:
        # 获取CAD文档路径和名称
        doc_path = doc.FullName
        doc_dir = os.path.dirname(doc_path)
        doc_name = os.path.splitext(os.path.basename(doc_path))[0]
        
        # 生成带uuid4的文件名
        unique_filename = f"{doc_name}_{uuid.uuid4()}.txt"
        txt_file_path = os.path.join(doc_dir, unique_filename)
        
        # 获取选中文本的内容
        texts_content = [get_text_content(text_obj) for text_obj in selected_texts]
        
        # 写入文件
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(texts_content, 1):
                f.write(f"{i:2d}. {text}\n")
        
        print(f"选中的文本已成功导出到: {txt_file_path}")
        return txt_file_path
        
    except Exception as e:
        print(f"导出文本时出错: {e}")
        return None

def main():  
    """  
    主函数 - 获取用户选择的文本对象并导出到文件
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
            
        # 显示选中的文本内容
        print("\n选中的文本内容:")
        for i, text_obj in enumerate(selected_texts, 1):
            content = get_text_content(text_obj)
            text_type = "单行文本" if text_obj.ObjectName == "AcDbText" else "多行文本"
            print(f"{i:2d}. [{text_type}] {content}")
        
        # 导出选中文本到文件
        export_selected_texts_to_file(acad, doc, selected_texts)
                
    except Exception as e:  
        print(f"处理对象时出错: {e}")  

if __name__ == "__main__":  
    main()