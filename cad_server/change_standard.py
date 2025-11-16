# -*- coding: utf-8 -*-
from pyautocad import Autocad

# 连接到AutoCAD
acad = Autocad(create_if_not_exists=True)


# 使用系统变量修改当前标注样式的属性
acad.doc.SetVariable("DIMDEC", 1)    # 主单位精度1位小数(0.0格式)
acad.doc.SetVariable("DIMTXT", 100)   # 文字高度50

