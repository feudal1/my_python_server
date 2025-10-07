import comtypes.client  
from pyautocad import APoint  
  
# 您的连接方式  
acad = comtypes.client.GetActiveObject('AutoCAD.Application', dynamic=True)  
  
# 获取活动文档和模型空间  
doc = acad.ActiveDocument  
model = doc.ModelSpace  
  
# 画圆  
p1 = APoint(0, 0)  # 圆心坐标  
model.AddCircle(p1, 10)  # 半径为10的圆