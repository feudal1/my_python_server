#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
from pyautocad import Autocad, APoint  
from collections import Counter  
  
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
  
def main():  
    """  
    主函数 - 列出所有对象类型  
    """  
    try:  
        acad = Autocad(create_if_not_exists=True)  
        print(f"成功连接到 AutoCAD 文档: {acad.doc.Name}")  
          
    except Exception as e:  
        print(f"无法连接到 AutoCAD: {e}")  
        return  
      
    try:  
        # 获取所有对象类型  
        object_types = list_all_object_types(acad)  
          
        if not object_types:  
            print("文档中没有找到任何对象")  
            return  
          
        # 统计每种对象类型的数量  
        type_counts = Counter(object_types)  
          
        print(f"\n文档中共有 {len(object_types)} 个对象:")  
        print("\n对象类型统计:")  
        for obj_type, count in sorted(type_counts.items()):  
            print(f"  {obj_type}: {count} 个")  
              
    except Exception as e:  
        print(f"处理对象时出错: {e}")  
  
if __name__ == "__main__":  
    main()