import tkinter as tk
from tkinter import filedialog, simpledialog
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from modelscope import AutoProcessor, AutoModelForZeroShotObjectDetection 

def visualize_results(image, results, text_labels):
    """
    可视化检测结果
    """
    draw = ImageDraw.Draw(image)
    
    # 从结果中获取检测框、标签和分数
    boxes = results[0]['boxes']
    labels = results[0]['text_labels'] if 'text_labels' in results[0] else results[0]['labels']
    scores = results[0]['scores']
    
    for i in range(len(boxes)):
        box = boxes[i].cpu().numpy()
        score = scores[i].item()
        label = labels[i]  # 现在是字符串，不需要 .item()
        
        # 只绘制置信度高的框
        if score > 0.4:
            x0, y0, x1, y1 = box
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            
            # 使用标签文本
            text_to_draw = f"{label} {score:.2f}"
            
            # 绘制标签
            font = ImageFont.load_default()
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((x0, y0), text_to_draw, font)
            else:
                w, h = draw.textsize(text_to_draw, font)
                bbox = (x0, y0, x0 + w, y0 + h)
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), text_to_draw, fill="white", font=font)
    
    return image

def select_image_and_classes():
    """
    弹窗选择图片和获取要识别的类名
    """
    # 创建根窗口并隐藏它
    root = tk.Tk()
    root.withdraw()
    
    # 弹窗选择图片文件
    image_path = filedialog.askopenfilename(
        title="选择要识别的图片",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
            ("All files", "*.*")
        ]
    )
    
    if not image_path:
        print("未选择图片，程序退出")
        return None, None
    
    # 弹窗输入要识别的类名
    class_names = simpledialog.askstring(
        "输入类名",
        "请输入要识别的类名（多个类名用逗号分隔）:\n例如: cat, dog, car, person"
    )
    
    if not class_names:
        print("未输入类名，程序退出")
        return None, None
    
    # 处理类名格式
    class_list = [name.strip().lower() + "." for name in class_names.split(",") if name.strip()]
    text = " ".join(class_list)
    
    return image_path, text
def detect_objects_with_text_transformers(base64_image, text_description):
    """
    使用Grounding DINO检测图像中的目标对象
    :param base64_image: base64编码的图像字符串
    :param text_description: 文本描述，用于检测的类别
    :return: 检测结果列表，包含边界框、标签和置信度
    """
    import base64
    from io import BytesIO
    
    # 将base64图像转换为PIL图像
    image_bytes = base64.b64decode(base64_image)
    image_buffer = BytesIO(image_bytes)
    image = Image.open(image_buffer)
    
    # 确保图像是RGB格式
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 设置模型和设备
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    
    # 处理文本描述格式
    text = text_description.strip().lower() + "."
    
    # 处理输入
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    
    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 后处理
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.input_ids,
        threshold=0.1,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    
    # 格式化结果
    detection_results = []
    if len(results) > 0:
        boxes = results[0]['boxes']
        labels = results[0]['text_labels'] if 'text_labels' in results[0] else results[0]['labels']
        scores = results[0]['scores']
        
        for i in range(len(boxes)):
            box = boxes[i].cpu().numpy()
            score = scores[i].item()
            label = labels[i]
            
            # 只包含置信度高的检测结果
            if score > 0.4:
                detection_results.append({
                    'bbox': box.tolist(),  # 转换为列表格式
                    'label': label,
                    'score': score
                })
    
    return detection_results
def main():
    # 加载模型
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # 选择图片和类名
    image_path, text = select_image_and_classes()
    
    if image_path is None or text is None:
        return
    
    # 加载图片并转换为RGB格式
    image = Image.open(image_path)
    
    # 确保图像是RGB格式，解决"Unable to infer channel dimension format"错误
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    print(f"选择的图片: {image_path}")
    print(f"要识别的类: {text}")
    
    # 处理输入
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    
    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 后处理
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.input_ids,
        threshold=0.1,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    
    # 打印结果
    print("\n检测结果:")
    for i, result in enumerate(results):
        print(f"Result {i}:")
        print(f"  Boxes shape: {result['boxes'].shape}")
        print(f"  Labels: {result['labels']}")
        print(f"  Scores: {result['scores']}")
        print(f"  Text Labels: {result.get('text_labels', 'N/A')}")
    
    # 可视化结果
    result_image = visualize_results(image.copy(), results, text)
    
    # 保存结果
    output_path = "result.jpg"
    result_image.save(output_path)
    print(f"\n结果已保存到 {output_path}")
    
    # 显示图片（可选）
    result_image.show()

if __name__ == "__main__":
    main()