# 在 processordeemo1.py 基础上添加 Flask 服务
import io
import os
import pickle
import tarfile
import urllib
from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.transforms.functional as TF

# Flask 应用实例
app = Flask(__name__)

# 加载DINOv3模型
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_NAME = "dinov3_vitl16_pretrain_lvd1689m"

# 加载预训练模型
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

pretrained_model_name = os.path.abspath("E:\\code\\model\\dinov3-convnext-tiny-pretrain-lvd1689m")
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(pretrained_model_name, device_map="auto")
model.cuda()

def load_image_from_path(path: str) -> Image:
    """支持本地和网络图片路径"""
    if path.startswith('http'):
        # 网络图片
        with urllib.request.urlopen(path) as f:
            return Image.open(f).convert("RGB")
    else:
        # 本地图片
        return Image.open(path).convert("RGB")

@app.route('/process_image', methods=['GET'])
def process_image():
    # 从请求参数中获取图片路径
    test_image_fpath = request.args.get('image_path')
    
    if not test_image_fpath:
        return jsonify({"error": "Missing image_path parameter"}), 400
    
    try:
        # 加载图片
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
        test_image_processed = inputs['pixel_values'][0]
        
        # 创建内存缓冲区保存图像
        img_buffer = io.BytesIO()
        
        # 创建可视化结果
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
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close()  # 关闭图表以释放内存
        
        # 返回图像
        return send_file(img_buffer, mimetype='image/png')
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# 在您的 dino_server.py 中添加以下内容:

@app.route('/calculate_similarity', methods=['GET'])
def calculate_similarity():
    """
    计算两张图像的DINO特征相似度
    
    Query Parameters:
        image1_path: 第一张图像的路径
        image2_path: 第二张图像的路径
    
    Returns:
        JSON格式的相似度值
    """
    try:
        # 从请求参数获取图像路径
        image1_path = request.args.get('image1_path')
        image2_path = request.args.get('image2_path')
        
        if not image1_path or not image2_path:
            return jsonify({"error": "Missing image paths"}), 400
       
        
        # 加载两张图片
        image1 = load_image_from_path(image1_path)
        image2 = load_image_from_path(image2_path)
        
        # 使用processor处理图像
        inputs1 = processor(images=image1, return_tensors="pt")
        inputs2 = processor(images=image2, return_tensors="pt")
        
        # 将输入数据移动到GPU
        inputs1['pixel_values'] = inputs1['pixel_values'].to('cuda')
        inputs2['pixel_values'] = inputs2['pixel_values'].to('cuda')
        
        # 使用模型进行推理
        with torch.no_grad():
            outputs1 = model(inputs1['pixel_values'])
            outputs2 = model(inputs2['pixel_values'])
            
            # 获取特征向量（这里使用平均池化）
            features1 = outputs1.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
            features2 = outputs2.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
            
            # 计算余弦相似度
            cos_sim = torch.nn.functional.cosine_similarity(features1, features2)
            similarity = cos_sim.item()
        
        return jsonify({
            "similarity": similarity,
            "image1_path": image1_path,
            "image2_path": image2_path
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/image_info', methods=['GET'])
def image_info():
    # 从请求参数中获取图片路径
    test_image_fpath = request.args.get('image_path')
    
    if not test_image_fpath:
        return jsonify({"error": "Missing image_path parameter"}), 400
    
    try:
        # 加载图片
        test_image = load_image_from_path(test_image_fpath)
        
        # 使用processor处理图像
        inputs = processor(images=test_image, return_tensors="pt")
        
        # 将输入数据移动到GPU
        inputs['pixel_values'] = inputs['pixel_values'].to('cuda')
        
        # 使用模型进行推理
        with torch.no_grad():
            outputs = model(inputs['pixel_values'])
            features = outputs.last_hidden_state
        
        # 返回模型输出信息
        return jsonify({
            "input_shape": list(inputs['pixel_values'].shape),
            "features_shape": list(features.shape)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
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