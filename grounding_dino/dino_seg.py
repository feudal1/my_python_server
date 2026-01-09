import tkinter as tk
from tkinter import filedialog, simpledialog
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import cv2
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 全局变量用于缓存模型
_cached_sam_model = None
_sam_model_type = "vit_h"  # 可以是 "vit_h", "vit_l", "vit_b"
_sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"  # 根据模型类型变化
_grounding_dino_model = "IDEA-Research/grounding-dino-tiny"

def download_sam_model(model_type="vit_h"):
    """
    下载SAM模型权重
    """
    import urllib.request
    
    # 定义不同模型类型的下载链接
    model_urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    
    checkpoint_url = model_urls.get(model_type, model_urls["vit_h"])
    
    # 创建模型目录
    model_dir = os.path.expanduser("~/.cache/segment_anything")
    os.makedirs(model_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(model_dir, os.path.basename(checkpoint_url))
    
    if not os.path.exists(checkpoint_path):
        print(f"正在下载SAM模型: {checkpoint_url}")
        urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
        print(f"SAM模型已保存到: {checkpoint_path}")
    
    return checkpoint_path

def get_sam_model(model_type="vit_h"):
    """
    获取SAM模型，如果已缓存则直接返回，否则加载并缓存
    """
    global _cached_sam_model
    
    if _cached_sam_model is None:
        # 下载或获取本地模型路径
        checkpoint_path = download_sam_model(model_type)
        
        print(f"正在加载SAM模型: {model_type}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载SAM模型
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        
        _cached_sam_model = sam
        print("SAM模型加载完成！")
    else:
        print("使用已缓存的SAM模型")
    
    return _cached_sam_model

def detect_objects_with_grounding_dino(image, text_prompt):
    """
    使用Grounding DINO检测对象
    """
    from modelscope import AutoProcessor, AutoModelForZeroShotObjectDetection
    
    # 加载Grounding DINO模型
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # 处理文本提示
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 后处理
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.input_ids,
        threshold=0.3,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )
    
    return results[0]

def segment_with_sam(image, detection_boxes):
    """
    使用SAM对检测到的边界框进行分割
    """
    sam_model = get_sam_model()
    predictor = SamPredictor(sam_model)
    
    # 转换PIL图像为numpy数组
    image_np = np.array(image)
    predictor.set_image(image_np)
    
    masks = []
    for box in detection_boxes:
        # 将边界框坐标从xyxy格式转换为xywh格式
        x0, y0, x1, y1 = box
        input_box = np.array([x0, y0, x1, y1])
        
        # 预测掩码
        mask, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        masks.append(mask[0])  # 取第一个mask
    
    return masks

def visualize_segmentation_results(image, results, masks, text_labels):
    """
    可视化分割结果
    """
    image_np = np.array(image)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image_np)
    
    # 从结果中获取检测框、标签和分数
    boxes = results['boxes']
    labels = results['text_labels'] if 'text_labels' in results else results['labels']
    scores = results['scores']
    
    for i in range(len(boxes)):
        box = boxes[i].cpu().numpy()
        score = scores[i].item()
        label = labels[i]
        
        # 只显示置信度高的结果
        if score > 0.4 and i < len(masks):
            # 绘制边界框
            x0, y0, x1, y1 = box
            rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, 
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            # 添加标签
            ax.text(x0, y0-10, f'{label} {score:.2f}', 
                    bbox=dict(facecolor='red', alpha=0.5), fontsize=12, color='white')
            
            # 绘制分割掩码
            mask = masks[i]
            # 创建彩色掩码
            color = np.random.rand(3)
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4))  # RGBA
            colored_mask[:, :, :3] = color.reshape(1, 1, 3)
            colored_mask[:, :, 3] = mask * 0.3  # 设置透明度
            
            ax.imshow(colored_mask)
    
    ax.axis('off')
    
    # 将matplotlib图形转换为PIL图像 - 修复兼容性问题
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, pad_inches=0)
    buf.seek(0)
    pil_image = Image.open(buf)
    
    # 不要在这里关闭buf，让PIL图像自己管理
    plt.close(fig)
    return pil_image

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



def detect_objects_with_dino_seg(base64_image, text_description):
    """
    使用DINO-Seg检测并分割图像中的目标对象
    :param base64_image: base64编码的图像字符串
    :param text_description: 文本描述，用于检测的类别
    :return: 分割结果列表，包含边界框、标签、置信度和掩码
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
    
    # 使用Grounding DINO检测对象
    detection_results = detect_objects_with_grounding_dino(image, text_description)
    
    # 提取边界框
    boxes = detection_results['boxes'].cpu().numpy()
    
    if len(boxes) == 0:
        return []
    
    # 使用SAM对检测到的对象进行分割
    masks = segment_with_sam(image, boxes)
    
    # 格式化结果
    segmentation_results = []
    labels = detection_results['text_labels'] if 'text_labels' in detection_results else detection_results['labels']
    scores = detection_results['scores']
    
    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i].item()
        label = labels[i]
        mask = masks[i]
        
        # 只包含置信度高的检测结果
        if score > 0.4:
            segmentation_results.append({
                'bbox': box.tolist(),  # 转换为列表格式
                'label': label,
                'score': score,
                'mask': mask  # 掩码作为numpy数组
            })
    
    return segmentation_results
def main():
    # 选择图片和类名
    image_path, text = select_image_and_classes()
    
    if image_path is None or text is None:
        return

    # 加载图片并转换为RGB格式
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    print(f"选择的图片: {image_path}")
    print(f"要识别的类: {text}")

    # 使用Grounding DINO检测对象
    print("正在使用Grounding DINO检测对象...")
    detection_results = detect_objects_with_grounding_dino(image, text)
    
    # 提取边界框
    boxes = detection_results['boxes'].cpu().numpy()
    
    print(f"检测到 {len(boxes)} 个对象")
    
    if len(boxes) == 0:
        print("没有检测到任何对象")
        return

    # 使用SAM对检测到的对象进行分割
    print("正在使用SAM进行分割...")
    masks = segment_with_sam(image, boxes)
    
    # 可视化结果
    print("正在生成可视化结果...")
    result_image = visualize_segmentation_results(image, detection_results, masks, text)

    # 如果图像是RGBA模式，转换为RGB模式以支持JPEG保存
    if result_image.mode == 'RGBA':
        # 创建白色背景
        background = Image.new('RGB', result_image.size, (255, 255, 255))
        # 将RGBA图像粘贴到白色背景上
        background.paste(result_image, mask=result_image.split()[-1])  # 使用alpha通道作为蒙版
        result_image = background

    # 保存结果
    output_path = "dino_seg_result.jpg"
    result_image.save(output_path)
    print(f"\n结果已保存到 {output_path}")

    # 显示图片（可选）
    result_image.show()
if __name__ == "__main__":
    main()