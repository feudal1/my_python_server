import os
import json
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import pandas as pd
import time
import logging
from dotenv import load_dotenv
import http.client
from urllib.parse import urlparse
import base64
from flask import Flask, request, jsonify, send_file
import threading
import webbrowser
import io
import re
import tkinter as tk
from tkinter import filedialog

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFToImageConverter:
    @staticmethod
    def convert_pdf_to_pil_images(pdf_path):
        """
        将PDF文件转换为PIL图像对象列表（不保存到磁盘）
        
        Args:
            pdf_path (str): PDF文件路径
            
        Returns:
            list: PIL图像对象列表
        """
        logger.info(f"正在转换PDF为内存图像: {pdf_path}")
        
        try:
            # 打开PDF文件
            pdf_document = fitz.open(pdf_path)
            pil_images = []
            
            # 遍历每一页
            for page_num in range(len(pdf_document)):
                try:
                    # 获取页面
                    page = pdf_document[page_num]
                    
                    # 渲染页面为图像
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                    pix = page.get_pixmap(matrix=mat)
                    
                    # 直接转换为PIL图像而不保存到磁盘
                    img_data = pix.tobytes("ppm")
                    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    pil_images.append(pil_image)
                    
                    logger.info(f"已转换PDF第 {page_num + 1} 页为内存图像")
                except Exception as page_error:
                    logger.error(f"处理PDF第 {page_num + 1} 页时出错: {str(page_error)}")
                    continue  # 继续处理其他页面
            
            pdf_document.close()
            return pil_images
        except Exception as e:
            logger.error(f"打开PDF文件 {pdf_path} 时出错: {str(e)}")
            return []

class VLMClassifier:
    def __init__(self):
        """初始化VLM分类器"""
        dotenv_path = r'E:\code\apikey\.env'
        load_dotenv(dotenv_path)
        self.api_url = os.getenv('VLM_OPENAI_API_URL')
        self.model_name = os.getenv('VLM_MODEL_NAME')
        self.api_key = os.getenv('VLM_OPENAI_API_KEY')
        
        if not all([self.api_url, self.model_name, self.api_key]):
            raise ValueError("缺少VLM API配置，请检查.env文件")
    
    def encode_pil_image(self, pil_image):
        """
        将PIL图像编码为base64
        
        Args:
            pil_image (PIL.Image): PIL图像对象
            
        Returns:
            str: base64编码的图像
        """
        # 将PIL图像保存到内存中的字节流
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        # 编码为base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def classify_image(self, pil_image):
        """
        使用VLM对图像进行分类
        
        Args:
            pil_image (PIL.Image): PIL图像对象
            
        Returns:
            dict: 分类结果
        """
        logger.info("正在分类内存中的图像")
        
        # 编码图像
        base64_image = self.encode_pil_image(pil_image)
        
        # 构造消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "这张图是折弯,卷圆还是组装图，回答格式json{类别:折弯}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        # 解析URL
        parsed = urlparse(f"{self.api_url}/chat/completions")
        host, path = parsed.hostname, parsed.path
        
        # 创建连接
        conn = http.client.HTTPSConnection(host)
        
        # 构造请求体
        request_body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7
        }
        
        # 发送请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        conn.request(
            "POST",
            path,
            body=json.dumps(request_body),
            headers=headers
        )
        
        # 获取响应
        response = conn.getresponse()
        
        if response.status != 200:
            error_msg = response.read().decode('utf-8')
            conn.close()
            raise Exception(f"VLM服务器错误: {response.status} - {error_msg}")
        
        # 解析响应
        response_data = response.read().decode('utf-8')
        data = json.loads(response_data)
        conn.close()
        
        # 提取分类结果
        try:
            content = data['choices'][0]['message']['content']
            # 处理可能的Markdown包装
            if content.startswith("```json"):
                content = content[7:]  # 移除 ```json
                if content.endswith("```"):
                    content = content[:-3]  # 移除 ```
            
            result = json.loads(content)
            logger.info(f"分类结果: {result}")
            return result
        except Exception as e:
            logger.error(f"解析VLM响应失败: {e}")
            logger.error(f"原始响应: {content}")
            raise

class ExcelUpdater:
    @staticmethod
    def extract_drawing_number(filename):
        """
        从文件名中提取图号（由数字和横杠组成）
        
        Args:
            filename (str): 文件名
            
        Returns:
            str: 提取的图号，如果没有找到则返回原文件名（不含扩展名）
        """
        # 使用正则表达式查找由数字和横杠组成的图号
        match = re.search(r'[\d-]+', filename)
        if match:
            drawing_number = match.group()
            logger.info(f"从文件名 '{filename}' 中提取图号: {drawing_number}")
            return drawing_number
        # 如果没有找到匹配的图号，则返回文件名（不含扩展名）
        drawing_number = Path(filename).stem
        logger.info(f"未找到标准图号，使用文件名作为图号: {drawing_number}")
        return drawing_number
    
    @staticmethod
    def update_excel(excel_path, file_name, category):
        """
        更新Excel文件中的类别信息
        
        Args:
            excel_path (str): Excel文件路径
            file_name (str): PDF文件名(不含扩展名)
            category (str): 分类结果
        """
        logger.info(f"正在更新Excel: {excel_path}")
        
        # 提取PDF文件的图号
        pdf_drawing_number = ExcelUpdater.extract_drawing_number(file_name)
        logger.info(f"准备更新Excel，图号: {pdf_drawing_number}, 文件名: {file_name}, 类别: {category}")
        
        # 读取Excel文件
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            logger.info(f"成功读取Excel文件，现有 {len(df)} 行数据")
        else:
            # 如果文件不存在，创建新的DataFrame
            df = pd.DataFrame()
            logger.info("Excel文件不存在，创建新的DataFrame")
        
        # 确保有文件名列
        filename_column = None
        if len(df.columns) > 0:
            filename_column = df.columns[0]  # 总是使用第一列
            logger.info(f"使用第一列作为文件名列: {filename_column}")
        else:
            # 如果没有找到合适的文件名列，创建一个
            filename_column = '文件名'
            if df.empty:
                df[filename_column] = []
            logger.info(f"未找到文件名列，使用列名: {filename_column}")
        
        # 确保有类别列
        category_column = '类别'
        if category_column not in df.columns:
            df[category_column] = ''
            logger.info(f"创建类别列: {category_column}")
        else:
            logger.info(f"找到类别列: {category_column}")
        
        # 查找对应行 - 基于图号匹配而不是完整文件名匹配
        row_index = None
        for idx, row in df.iterrows():
            if filename_column in row:
                excel_filename = str(row[filename_column])
                # 提取Excel行中文件的图号
                excel_drawing_number = ExcelUpdater.extract_drawing_number(Path(excel_filename).stem)
                # 如果图号匹配，则认为是同一图纸的不同文件
                if excel_drawing_number == pdf_drawing_number:
                    row_index = idx
                    logger.info(f"在Excel中找到图号匹配行，索引: {idx}, 图号: {pdf_drawing_number}")
                    break
        
        # 如果找不到对应行，创建新行
        if row_index is None:
            new_row = pd.Series({filename_column: file_name + '.pdf', category_column: ''})
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            row_index = len(df) - 1
            logger.info(f"未找到匹配行，创建新行，索引: {row_index}")
        
        # 更新类别
        old_category = df.at[row_index, category_column]
        df.at[row_index, category_column] = category
        logger.info(f"更新类别信息: 行 {row_index}, 原类别: {old_category}, 新类别: {category}")
        
        # 保存Excel文件
        try:
            df.to_excel(excel_path, index=False)
            logger.info(f"已成功更新Excel文件，保存路径: {excel_path}")
        except Exception as e:
            logger.error(f"保存Excel文件失败: {str(e)}")
            raise
        logger.info(f"已更新Excel文件，行索引: {row_index}, 类别: {category}")

    @staticmethod
    def load_existing_categories(excel_path):
        """
        从Excel文件中加载已有的类别信息
        
        Args:
            excel_path (str): Excel文件路径
            
        Returns:
            dict: 图号到类别的映射
        """
        existing_categories = {}
        if not os.path.exists(excel_path):
            return existing_categories
            
        try:
            df = pd.read_excel(excel_path)
            if df.empty or len(df.columns) == 0:
                return existing_categories
                
            filename_column = df.columns[0]
            category_column = '类别'
            
            if category_column not in df.columns:
                return existing_categories
                
            for _, row in df.iterrows():
                if filename_column in row and pd.notna(row[filename_column]):
                    filename = str(row[filename_column])
                    # 提取图号
                    drawing_number = ExcelUpdater.extract_drawing_number(Path(filename).stem)
                    # 获取类别
                    category = row[category_column] if category_column in row and pd.notna(row[category_column]) else None
                    if category:
                        existing_categories[drawing_number] = category
                        logger.info(f"从Excel中读取已有类别: {drawing_number} -> {category}")
        except Exception as e:
            logger.error(f"读取Excel文件时出错: {str(e)}")
            
        return existing_categories


# 全局变量存储当前状态
current_state = {
    'pdf_files': [],
    'current_index': 0,
    'current_pdf': None,
    'current_image': None,
    'current_pil_image': None,
    'processing': False,
    'converter': None,
    'classifier': None,
    'updater': None,
    'pdf_directory': '',
    'excel_path': '',
    'image_paths': {},
    'pil_images_cache': {},
    'existing_categories': {}  # 存储从Excel中读取的现有类别
}

app = Flask(__name__)

@app.route('/')
def index():
    """Web界面主页"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF审核工具</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            button { padding: 12px 24px; margin: 10px; font-size: 16px; border: none; border-radius: 4px; cursor: pointer; transition: background-color 0.3s; }
            .btn-bend { background-color: #4CAF50; color: white; }
            .btn-bend:hover { background-color: #45a049; }
            .btn-roll { background-color: #2196F3; color: white; }
            .btn-roll:hover { background-color: #0b7dda; }
            .btn-assembly { background-color: #9C27B0; color: white; }
            .btn-assembly:hover { background-color: #7B1FA2; }
            .btn-custom { background-color: #FF9800; color: white; }
            .btn-custom:hover { background-color: #e68a00; }
            .btn-skip { background-color: #f44336; color: white; }
            .btn-skip:hover { background-color: #d32f2f; }
            .btn-reset { background-color: #607D8B; color: white; }
            .btn-reset:hover { background-color: #455A64; }
            button:disabled { background-color: #cccccc; cursor: not-allowed; }
            #image-container { margin-top: 20px; text-align: center; min-height: 300px; display: flex; align-items: center; justify-content: center; }
            img { max-width: 100%; max-height: 70vh; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
            .status { margin: 20px 0; padding: 15px; background-color: #e3f2fd; border-radius: 4px; }
            .controls { margin: 20px 0; text-align: center; }
            h1 { color: #333; text-align: center; }
            .progress-bar { height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 10px 0; }
            .progress-fill { height: 100%; background-color: #4CAF50; transition: width 0.3s; }
            .classification-result { 
                margin: 15px 0; 
                padding: 10px; 
                background-color: #fff3cd; 
                border: 1px solid #ffeaa7; 
                border-radius: 4px; 
                text-align: center;
                font-weight: bold;
                font-size: 18px;
            }
            /* 模态框样式 */
            .modal {
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.4);
            }
            .modal-content {
                background-color: #fefefe;
                margin: 15% auto;
                padding: 20px;
                border: 1px solid #888;
                width: 300px;
                border-radius: 5px;
                text-align: center;
            }
            .close {
                color: #aaa;
                float: right;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
            }
            .close:hover,
            .close:focus {
                color: black;
                text-decoration: none;
                cursor: pointer;
            }
            #custom-category {
                width: 100%;
                padding: 10px;
                margin: 10px 0;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-sizing: border-box;
            }
            #custom-category:focus {
                outline: none;
                border-color: #2196F3;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PDF审核工具</h1>
            <div class="status">
                <p>当前文件: <strong><span id="current-file">无</span></strong></p>
                <p>进度: <span id="progress">0/0</span></p>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-bar-fill" style="width: 0%"></div>
                </div>
            </div>
            
            <div class="classification-result" id="classification-result" style="display: none;">
                分类结果: <span id="classification-text"></span>
            </div>
            
            <div class="controls">
                <button onclick="classifyAs('折弯')" class="btn-bend" id="bend-btn" disabled>折弯</button>
                <button onclick="classifyAs('卷圆')" class="btn-roll" id="roll-btn" disabled>卷圆</button>
                <button onclick="classifyAs('组装图')" class="btn-assembly" id="assembly-btn" disabled>组装图</button>
                <button onclick="openCustomModal()" class="btn-custom" id="custom-btn" disabled>自定义</button>
                <button onclick="skipFile()" class="btn-skip" id="skip-btn" disabled>跳过</button>
                <button onclick="resetProcessing()" class="btn-reset" id="reset-btn">重置</button>
            </div>
            
            <!-- 自定义分类输入模态框 -->
            <div id="custom-modal" class="modal" style="display: none;">
                <div class="modal-content">
                    <span class="close" onclick="closeCustomModal()">&times;</span>
                    <h2>自定义分类</h2>
                    <input type="text" id="custom-category" placeholder="请输入分类名称">
                    <button onclick="confirmCustomCategory()" class="btn-custom">确认</button>
                </div>
            </div>
            
            <div id="image-container">
                <p id="no-image-msg">尚未开始处理</p>
                <img id="preview-image" src="" alt="PDF预览" style="display:none;">
            </div>
        </div>
        
        <script>
            let currentCustomCategory = '';
            
            // 页面加载后自动开始处理
            window.onload = function() {
                startProcessing();
            };
            
            function startProcessing() {
                fetch('/start_processing', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        updateStatus(data);
                        if (data.current_file) {
                            enableButtons();
                            loadImage();
                        }
                    })
                    .catch(error => {
                        alert('处理过程中出现错误: ' + error.message);
                    });
            }
            
            function classifyAs(category) {
                fetch('/confirm_and_next', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({classification: category})
                })
                    .then(response => response.json())
                    .then(data => {
                        updateStatus(data);
                        if (data.current_file) {
                            loadImage();
                        } else {
                            hideImage();
                            disableButtons();
                        }
                    })
                    .catch(error => {
                        alert('处理过程中出现错误: ' + error.message);
                    });
            }
            
            function skipFile() {
                fetch('/reject_and_next', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        updateStatus(data);
                        if (data.current_file) {
                            loadImage();
                        } else {
                            hideImage();
                            disableButtons();
                        }
                    })
                    .catch(error => {
                        alert('处理过程中出现错误: ' + error.message);
                    });
            }
            
            function resetProcessing() {
                fetch('/reset_processing', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        updateStatus(data);
                        hideImage();
                        disableButtons();
                        hideClassification();
                    });
            }
            
            function openCustomModal() {
                const modal = document.getElementById('custom-modal');
                const input = document.getElementById('custom-category');
                if (modal && input) {
                    modal.style.display = 'block';
                    input.focus();
                }
            }
            
            function closeCustomModal() {
                const modal = document.getElementById('custom-modal');
                const input = document.getElementById('custom-category');
                if (modal) {
                    modal.style.display = 'none';
                }
                if (input) {
                    input.value = '';
                }
            }
            
            function confirmCustomCategory() {
                const input = document.getElementById('custom-category');
                if (input) {
                    const customCategory = input.value.trim();
                    if (customCategory) {
                        classifyAs(customCategory);
                        closeCustomModal();
                        input.value = '';
                    } else {
                        alert('请输入分类名称');
                    }
                }
            }
            
            function loadImage() {
                const timestamp = new Date().getTime(); // 防止缓存
                const img = document.getElementById('preview-image');
                const msg = document.getElementById('no-image-msg');
                if (img && msg) {
                    img.src = '/image/' + timestamp;
                    img.style.display = 'inline';
                    msg.style.display = 'none';
                }
            }
            
            function hideImage() {
                const img = document.getElementById('preview-image');
                const msg = document.getElementById('no-image-msg');
                if (img && msg) {
                    img.style.display = 'none';
                    msg.style.display = 'block';
                    msg.textContent = '处理已完成';
                }
            }
            
            function updateStatus(data) {
                const currentFileEl = document.getElementById('current-file');
                const progressEl = document.getElementById('progress');
                const progressBarFill = document.getElementById('progress-bar-fill');
                
                if (currentFileEl) {
                    currentFileEl.textContent = data.current_file || '无';
                }
                if (progressEl) {
                    progressEl.textContent = data.progress || '0/0';
                }
                
                if (progressBarFill && data.total > 0) {
                    const percentage = (data.processed / data.total) * 100;
                    progressBarFill.style.width = percentage + '%';
                } else if (progressBarFill) {
                    progressBarFill.style.width = '0%';
                }
                
                // 显示分类结果
                if (data.classification) {
                    showClassification(data.classification);
                } else {
                    hideClassification();
                }
            }
            
            function showClassification(classification) {
                const textEl = document.getElementById('classification-text');
                const resultEl = document.getElementById('classification-result');
                if (textEl && resultEl) {
                    textEl.textContent = classification;
                    resultEl.style.display = 'block';
                }
            }
            
            function hideClassification() {
                const resultEl = document.getElementById('classification-result');
                if (resultEl) {
                    resultEl.style.display = 'none';
                }
            }
            
            function enableButtons() {
                const buttons = ['bend-btn', 'roll-btn', 'assembly-btn', 'custom-btn', 'skip-btn'];
                buttons.forEach(id => {
                    const btn = document.getElementById(id);
                    if (btn) btn.disabled = false;
                });
            }
            
            function disableButtons() {
                const buttons = ['bend-btn', 'roll-btn', 'assembly-btn', 'custom-btn', 'skip-btn'];
                buttons.forEach(id => {
                    const btn = document.getElementById(id);
                    if (btn) btn.disabled = true;
                });
            }
            
            // 添加键盘事件监听器，支持回车确认自定义分类
            document.addEventListener('keydown', function(event) {
                const modal = document.getElementById('custom-modal');
                if (event.key === 'Enter' && modal && modal.style.display === 'block') {
                    confirmCustomCategory();
                }
            });
        </script>
    </body>
    </html>
    '''

# 添加一个新的路由来提供图片，使用时间戳作为参数
@app.route('/image/<timestamp>')
def serve_image(timestamp):
    """提供当前图像文件"""
    logger.info(f"请求图片服务: current_pdf={current_state['current_pdf']}")
    
    if current_state['current_pdf'] and current_state['current_pil_image']:
        # 将PIL图像对象转换为响应
        img_io = io.BytesIO()
        current_state['current_pil_image'].save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
    
    logger.warning("没有当前图片可提供")
    return '', 404

# 添加缩略图路由
@app.route('/thumb/<filename>')
def serve_thumbnail(filename):
    """提供处理过的PDF文件的缩略图"""
    logger.info(f"请求缩略图: {filename}")
    
    try:
        # 重新生成缩略图而不是使用缓存的图像
        pdf_path = os.path.join(current_state['pdf_directory'], filename)
        if os.path.exists(pdf_path):
            # 转换PDF第一页为图像
            pil_images = PDFToImageConverter.convert_pdf_to_pil_images(pdf_path)
            if pil_images and len(pil_images) > 0:
                img_io = io.BytesIO()
                # 调整图像大小作为缩略图
                thumbnail = pil_images[0].copy()
                thumbnail.thumbnail((100, 100))
                thumbnail.save(img_io, 'PNG')
                img_io.seek(0)
                return send_file(img_io, mimetype='image/png')
            else:
                logger.warning(f"无法从PDF生成图像: {filename}")
        else:
            logger.warning(f"PDF文件不存在: {pdf_path}")
    except Exception as e:
        logger.error(f"生成缩略图时出错: {str(e)}")
    
    logger.warning(f"没有找到文件 {filename} 的缩略图")
    # 返回一个占位符图像
    return '', 404

@app.route('/start_processing', methods=['POST'])
def start_processing():
    """开始处理PDF文件"""
    global current_state
    
    # 初始化配置
    dotenv_path = r'E:\code\apikey\.env'
    load_dotenv(dotenv_path)
    
    # 创建隐藏的根窗口用于文件选择对话框
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.attributes('-topmost', True)  # 确保对话框在最前端
    
    # 弹窗选择PDF目录
    pdf_directory = filedialog.askdirectory(
        title="请选择包含PDF文件的目录"
    )
    if not pdf_directory:
        root.destroy()
        return jsonify({'error': '未选择PDF目录'})

    # 设置Excel文件路径为PDF目录下的固定文件名
    excel_path = os.path.join(pdf_directory, '分类结果.xlsx')
    
    current_state['pdf_directory'] = pdf_directory
    current_state['excel_path'] = excel_path
    
    # 销毁根窗口
    root.destroy()
    
    logger.info(f"开始处理，PDF目录: {current_state['pdf_directory']}")
    logger.info(f"Excel文件路径: {current_state['excel_path']}")
    
    # 清空之前的缓存
    current_state['pil_images_cache'] = {}
    current_state['image_paths'] = {}
    current_state['existing_categories'] = {}  # 清空现有类别缓存
    
    # 初始化组件
    current_state['converter'] = PDFToImageConverter()
    current_state['classifier'] = VLMClassifier()
    current_state['updater'] = ExcelUpdater()
    
    # 获取所有PDF文件
    if not os.path.exists(current_state['pdf_directory']):
        logger.error(f"PDF目录不存在: {current_state['pdf_directory']}")
        return jsonify({'error': f'PDF目录不存在: {current_state["pdf_directory"]}'})
        
    current_state['pdf_files'] = [f for f in os.listdir(current_state['pdf_directory']) if f.lower().endswith('.pdf')]
    current_state['current_index'] = 0
    
    logger.info(f"找到 {len(current_state['pdf_files'])} 个PDF文件")
    for i, pdf_file in enumerate(current_state['pdf_files']):
        logger.info(f"  {i+1}. {pdf_file}")
    
    if not current_state['pdf_files']:
        return jsonify({'error': '目录中没有找到PDF文件'})
    
    # 读取Excel中的现有类别信息
    current_state['existing_categories'] = ExcelUpdater.load_existing_categories(current_state['excel_path'])
    
    # 处理第一个文件
    return process_next_file()

@app.route('/confirm_and_next', methods=['POST'])
def confirm_and_next():
    """确认当前文件并处理下一个"""
    global current_state
    
    # 获取分类结果
    data = request.get_json()
    classification = data.get('classification', '未知')
    
    logger.info(f"确认分类: {classification}")
    
    # 更新Excel
    if current_state['current_pdf']:
        file_name = Path(current_state['current_pdf']).stem
        logger.info(f"准备更新Excel，文件名: {file_name}, 分类: {classification}")
        try:
            current_state['updater'].update_excel(current_state['excel_path'], file_name, classification)
            logger.info("Excel更新完成")
        except Exception as e:
            logger.error(f"更新Excel时出错: {str(e)}")
    
    # 移动到下一个文件
    current_state['current_index'] += 1
    return process_next_file()

@app.route('/reject_and_next', methods=['POST'])
def reject_and_next():
    """拒绝当前文件并处理下一个"""
    global current_state
    
    logger.info("跳过当前文件")
    
    # 移动到下一个文件（不执行任何操作）
    current_state['current_index'] += 1
    return process_next_file()

@app.route('/reset_processing', methods=['POST'])
def reset_processing():
    """重置处理状态"""
    global current_state
    
    logger.info("重置处理状态")
    
    current_state['pdf_files'] = []
    current_state['current_index'] = 0
    current_state['current_pdf'] = None
    current_state['current_image'] = None
    current_state['current_pil_image'] = None
    current_state['processing'] = False
    current_state['image_paths'] = {}
    current_state['pil_images_cache'] = {}
    current_state['existing_categories'] = {}
    
    return jsonify({
        'current_file': None,
        'progress': '0/0',
        'processed': 0,
        'total': 0
    })

def process_next_file():
    """处理下一个PDF文件"""
    global current_state
    
    while current_state['current_index'] < len(current_state['pdf_files']):
        # 获取当前PDF文件
        current_state['current_pdf'] = current_state['pdf_files'][current_state['current_index']]
        pdf_path = os.path.join(current_state['pdf_directory'], current_state['current_pdf'])
        
        # 提取图号
        file_name = Path(current_state['current_pdf']).stem
        drawing_number = ExcelUpdater.extract_drawing_number(file_name)
        
        # 检查Excel中是否已有类别
        existing_category = current_state['existing_categories'].get(drawing_number)
        if existing_category:
            logger.info(f"文件 {current_state['current_pdf']} 已有类别 '{existing_category}'，跳过处理")
            current_state['current_index'] += 1
            continue  # 跳过这个文件，处理下一个
        
        logger.info(f"开始处理PDF文件 ({current_state['current_index']+1}/{len(current_state['pdf_files'])}): {pdf_path}")
        logger.info(f"当前文件图号: {drawing_number}")
        
        try:
            # 检查是否已经转换过该PDF
            pdf_name = Path(current_state['current_pdf']).stem
            if pdf_name in current_state['pil_images_cache']:
                # 如果已经转换过，直接使用已有的图像对象
                pil_images = current_state['pil_images_cache'][pdf_name]
                logger.info(f"使用已缓存的图像对象，共 {len(pil_images)} 页")
            else:
                # 转换PDF为图像对象
                pil_images = current_state['converter'].convert_pdf_to_pil_images(pdf_path)
                
                # 缓存图像对象
                current_state['pil_images_cache'][pdf_name] = pil_images
                logger.info(f"新生成图像对象并缓存，共 {len(pil_images)} 页")
            
            classification = None
            # 使用第一页作为预览
            if pil_images and len(pil_images) > 0:
                current_state['current_pil_image'] = pil_images[0]
                logger.info("设置当前PIL图像对象用于预览")
                
                # 同时进行分类
                try:
                    result = current_state['classifier'].classify_image(current_state['current_pil_image'])
                    classification = result.get('类别', '未知')
                    logger.info(f"分类结果: {classification}")
                except Exception as e:
                    logger.error(f"分类失败: {str(e)}")
                    classification = "分类失败"
            else:
                current_state['current_pil_image'] = None
                logger.warning(f"未生成图像对象 for {current_state['current_pdf']}")
            
            return jsonify({
                'current_file': current_state['current_pdf'],
                'progress': f'{current_state["current_index"] + 1}/{len(current_state["pdf_files"])}',
                'processed': current_state["current_index"],
                'total': len(current_state["pdf_files"]),
                'classification': classification
            })
        except Exception as e:
            logger.error(f"处理文件 {current_state['current_pdf']} 时出错: {str(e)}")
            # 即使出错也继续处理下一个文件
            current_state['current_index'] += 1
    
    # 已经处理完所有文件
    logger.info("所有文件处理完成")
    current_state['current_pdf'] = None
    current_state['current_image'] = None
    current_state['current_pil_image'] = None
    return jsonify({
        'current_file': None,
        'progress': f'{len(current_state["pdf_files"])}/{len(current_state["pdf_files"])}',
        'processed': len(current_state["pdf_files"]),
        'total': len(current_state["pdf_files"])
    })

def run_web_interface():
    """运行Web界面"""
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

def open_browser():
    """打开浏览器"""
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
    print("启动PDF审核工具...")
    
    # 启动Web服务线程
    web_thread = threading.Thread(target=run_web_interface)
    web_thread.daemon = True
    web_thread.start()
    
    # 等待一秒确保服务器启动后再打开浏览器
    time.sleep(1)
    open_browser()
    
    print("Web界面已在 http://127.0.0.1:5000 启动")
    print("请在浏览器中查看界面")
    print("按 Ctrl+C 退出程序")
    
    # 主线程继续运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n程序已退出")