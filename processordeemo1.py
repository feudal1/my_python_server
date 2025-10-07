# processordeemo1.py
import io
import os
import pickle
import tarfile
import urllib

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.transforms.functional as TF


# 加载DINOv3模型
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_NAME = "dinov3_vitl16_pretrain_lvd1689m"

# 加载预训练模型
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

pretrained_model_name = os.path.abspath("E:\\code\\dinov3\\model\\dinov3-convnext-tiny-pretrain-lvd1689m")
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(pretrained_model_name, device_map="auto")
model.cuda()

test_image_fpath = r"E:\code\dinov3\图片\white.png"

def load_image_from_path(path: str) -> Image:
    """支持本地和网络图片路径"""
    if path.startswith('http'):
        # 网络图片
        with urllib.request.urlopen(path) as f:
            return Image.open(f).convert("RGB")
    else:
        # 本地图片
        return Image.open(path).convert("RGB")

# 使用修改后的函数加载图片
test_image = load_image_from_path(test_image_fpath)

# 使用processor处理图像
inputs = processor(images=test_image, return_tensors="pt")

# 将输入数据移动到GPU
inputs['pixel_values'] = inputs['pixel_values'].to('cuda')

# 使用模型进行推理
with torch.no_grad():
    outputs = model(inputs['pixel_values'])
    features = outputs.last_hidden_state

# 获取处理后的图像用于可视化
test_image_processed = inputs['pixel_values'][0]  # 获取处理后的图像

# 可视化结果
plt.figure(figsize=(9, 3), dpi=300)
plt.subplot(1, 3, 1)
plt.axis('off')
plt.imshow(test_image)  # 显示原图
plt.title('original image')

plt.subplot(1, 3, 2)
plt.axis('off')
plt.imshow(TF.to_pil_image(test_image_processed))  # 显示处理后的图像
plt.title('processed image')




plt.tight_layout()
plt.show()

# 打印模型输出信息
print(f"Input shape: {inputs['pixel_values'].shape}")
print(f"Features shape: {features.shape}")