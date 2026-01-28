# e:\code\my_python_server\tools\yolo\detect_like_favorite.py
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import ImageGrab


def detect_like_favorite():
    """
    检测点赞和收藏按钮
    自动截取当前屏幕进行检测
    """
    # 获取项目根目录
    current_dir = Path(__file__).parent
    model_path = current_dir.parent / "my_models" / "like_favorite.pt"
    
    if not model_path.exists():
        return f"错误: 模型文件不存在 - {model_path}"
    
    try:
        # 加载模型
        model = YOLO(str(model_path))
        
        # 自动截取当前屏幕
        print("正在截取当前屏幕...")
        screenshot = ImageGrab.grab()
        image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        if image is None:
            return "错误: 无法读取图像"
        
        # 保存原始图像的副本用于绘制
        image_with_boxes = image.copy()
        
        # 进行预测
        results = model.predict(source=image, conf=0.1, save=False)
        
        detections = []
        result = results[0]
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标 [x1, y1, x2, y2]
            confs = result.boxes.conf.cpu().numpy()  # 置信度
            cls_ids = result.boxes.cls.cpu().numpy()  # 类别ID
            
            names = result.names if hasattr(result, 'names') else {}
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = confs[i]
                cls_id = int(cls_ids[i])
                class_name = names.get(cls_id, f"Class_{cls_id}")
                
                # 绘制边界框
                cv2.rectangle(image_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # 添加标签文本
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(image_with_boxes, label, (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 计算中心坐标
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                detections.append({
                    'class_name': class_name,
                    'center': [center_x, center_y],
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf
                })
        
        # 创建输出目录
        output_dir = current_dir / "output"
        output_dir.mkdir(exist_ok=True)
        
        import time
        timestamp = int(time.time())
        output_filename = f"result_auto_screenshot_{timestamp}.jpg"
        
        result_save_path = output_dir / output_filename
        
        # 保存带有边界框的图像
        cv2.imwrite(str(result_save_path), image_with_boxes)
        print(f"识别结果已保存到: {result_save_path}")
        
        if not detections:
            return "未检测到点赞或收藏按钮"
        
        result_str = f"检测到 {len(detections)} 个元素:\n"
        for det in detections:
            result_str += f"- {det['class_name']}: 位置({det['center'][0]:.1f}, {det['center'][1]:.1f}), 置信度{det['confidence']:.2f}\n"
        
        return result_str.strip()
        
    except Exception as e:
        return f"检测过程中出错: {str(e)}"


if __name__ == '__main__':
    print("检测点赞和收藏按钮")
    result = detect_like_favorite()
    print(result)