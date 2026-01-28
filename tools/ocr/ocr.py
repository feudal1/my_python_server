import tkinter as tk
from tkinter import filedialog
from PIL import ImageGrab
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
import io
from PIL import Image
import pyautogui
import sys
import os
from dotenv import load_dotenv
import base64
from datetime import datetime
import win32gui
import win32con
import time
import re

# 确保标准输出使用UTF-8编码
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
if sys.stderr.encoding != 'utf-8':
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

# 总是导入 pytesseract 作为备用
import pytesseract

# 尝试导入 EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    # 初始化 EasyOCR 阅读器（支持中英文）
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)  # 设置 gpu=True 以使用 GPU
except ImportError:
    EASYOCR_AVAILABLE = False
except Exception as e:
    EASYOCR_AVAILABLE = False


def image_to_base64(image: Image.Image) -> str:
    """将PIL图像转换为base64字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def preprocess_image(image: Image.Image) -> Image.Image:
    """预处理图像以提高OCR识别率"""
    from PIL import ImageEnhance, ImageFilter
    
    # 转换为灰度图
    img_gray = image.convert('L')
    
    # 提高对比度
    enhancer = ImageEnhance.Contrast(img_gray)
    img_enhanced = enhancer.enhance(3)
    
    # 提高亮度
    enhancer = ImageEnhance.Brightness(img_enhanced)
    img_enhanced = enhancer.enhance(1.2)
    
    # 应用中值滤波去噪
    img_denoised = img_enhanced.filter(ImageFilter.MedianFilter())
    
    # 二值化处理（调整阈值以适应中文文本）
    img_binary = img_denoised.point(lambda x: 0 if x < 128 else 255)
    
    return img_binary


def easyocr_ocr(image: Image.Image) -> Dict:
    """使用EasyOCR进行识别"""
    try:
        start_time = time.time()
        
        # 保存原始图像用于调试对比
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        original_debug_path = os.path.join(output_dir, f"debug_original_{int(time.time())}.png")
        image.save(original_debug_path)
        
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 保存处理后的图像用于调试
        debug_path = os.path.join(output_dir, f"debug_processed_{int(time.time())}.png")
        image.save(debug_path)
        print(f"[OCR] 处理后的图像已保存到: {debug_path}")
        
        # 使用EasyOCR进行识别
        print("[OCR] 使用 EasyOCR 进行识别")
        results = reader.readtext(img_array)
        
        ocr_result = []
        confidence_threshold = 0.6  # EasyOCR置信度阈值（0-1）
        print(f"[OCR] 使用置信度阈值: {confidence_threshold * 100}%")
        
        # 处理识别结果
        for (bbox, text, prob) in results:
            if prob >= confidence_threshold:
                # 计算边界框
                (top_left, top_right, bottom_right, bottom_left) = bbox
                x_min = int(top_left[0])
                y_min = int(top_left[1])
                x_max = int(bottom_right[0])
                y_max = int(bottom_right[1])
                
                width = x_max - x_min
                height = y_max - y_min
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                confidence = prob * 100  # 转换为百分比
                
                # 过滤掉太小的文本区域和异常大小的区域
                if width > 10 and height > 10 and width < 1000 and height < 500:
                    ocr_result.append({
                        'text': text,
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': width,
                        'height': height,
                        'confidence': confidence
                    })
                    print(f"[OCR] 识别到文本: '{text}' (置信度: {confidence:.1f}%, 位置: ({center_x},{center_y}), 大小: {width}x{height})")
        
        elapsed_time = time.time() - start_time
        print(f"[OCR] EasyOCR 完成，耗时: {elapsed_time:.2f}秒")
        print(f"[OCR] 识别到 {len(ocr_result)} 条文本")
        
        return {
            'success': True,
            'results': ocr_result,
            'elapsed_time': elapsed_time,
            'method': 'easyocr'
        }
    except Exception as e:
        print(f"[OCR] EasyOCR 失败: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def tesseract_ocr(image: Image.Image) -> Dict:
    """使用Tesseract OCR进行识别"""
    try:
        start_time = time.time()
        
        # 保存原始图像用于调试对比
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        original_debug_path = os.path.join(output_dir, f"debug_original_{int(time.time())}.png")
        image.save(original_debug_path)
        
        # 直接使用原始图像，不进行任何预处理
        img_processed = image
        
        # 保存处理后的图像用于调试
        debug_path = os.path.join(output_dir, f"debug_processed_{int(time.time())}.png")
        img_processed.save(debug_path)
        print(f"[OCR] 处理后的图像已保存到: {debug_path}")
        
        # 尝试使用中文语言包进行识别
        try:
            # 使用image_to_string直接获取文本
            text = pytesseract.image_to_string(img_processed, lang='chi_sim')  # 使用简体中文数据
            print("[OCR] 使用 Tesseract 简体中文语言包")
            print(f"[OCR] 识别结果: {text}")
            
            # 同时使用image_to_data获取详细信息
            custom_config = r'--oem 3 --psm 6'
            data = pytesseract.image_to_data(img_processed, output_type=pytesseract.Output.DICT,
                                           lang='chi_sim+eng', config=custom_config)
        except Exception as e:
            print(f"[OCR] 中文语言包使用失败: {e}")
            # 中文语言包不可用，尝试仅使用英文
            try:
                text = pytesseract.image_to_string(img_processed, lang='eng')
                data = pytesseract.image_to_data(img_processed, output_type=pytesseract.Output.DICT,
                                               lang='eng', config=r'--oem 3 --psm 6')
                print("[OCR] 中文语言包不可用，使用英文语言包")
            except Exception as e2:
                # 所有语言包都不可用，返回失败
                raise Exception(f"Tesseract 语言包不可用: {e2}")
        
        ocr_result = []
        confidence_threshold = 50  # Tesseract置信度阈值
        print(f"[OCR] 使用置信度阈值: {confidence_threshold}%")
        
        # 保存原始识别数据用于调试
        debug_data_path = os.path.join(os.path.dirname(__file__), "output", f"debug_ocr_data_{int(time.time())}.json")
        with open(debug_data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OCR] 原始 OCR 数据已保存到: {debug_data_path}")
        
        # 处理识别结果
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            confidence = int(data['conf'][i])
            
            # 过滤条件
            if (text and 
                confidence > confidence_threshold and
                len(text) > 0):
                
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                # 过滤掉太小的文本区域和异常大小的区域
                if w > 10 and h > 10 and w < 1000 and h < 500:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    ocr_result.append({
                        'text': text,
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': w,
                        'height': h,
                        'confidence': confidence
                    })
                    print(f"[OCR] 识别到文本: '{text}' (置信度: {confidence}%, 位置: ({x},{y}), 大小: {w}x{h})")
        
        elapsed_time = time.time() - start_time
        print(f"[OCR] Tesseract OCR 完成，耗时: {elapsed_time:.2f}秒")
        print(f"[OCR] 识别到 {len(ocr_result)} 条文本")
        
        return {
            'success': True,
            'results': ocr_result,
            'elapsed_time': elapsed_time,
            'method': 'tesseract'
        }
    except Exception as e:
        print(f"[OCR] Tesseract OCR 失败: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def local_ocr(image: Image.Image) -> Dict:
    """使用本地OCR进行识别（优先使用EasyOCR，回退到Tesseract）"""
    # 优先使用EasyOCR
    if EASYOCR_AVAILABLE:
        result = easyocr_ocr(image)
        if result['success']:
            return result
        print("[OCR] EasyOCR 失败，回退到 Tesseract")
    
    # 回退到Tesseract OCR
    return tesseract_ocr(image)


def save_ocr_image_with_boxes(image: Image.Image, ocr_result: str or Dict, image_path: str = None, regions: list = None):
    """
    保存带识别框和中心点的图片用于调试
    
    参数:
    - image: 原始图像
    - ocr_result: OCR 识别结果
    - image_path: 图像路径（用于生成输出文件名）
    - regions: 额外的区域框列表，格式为 [(x_min, y_min, x_max, y_max), ...]
    """
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用 PIL 库来绘制，支持中文
    from PIL import ImageDraw, ImageFont
    
    # 创建可绘制的图像副本
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # 尝试使用系统字体，支持中文
    try:
        # 尝试不同的字体路径
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/msyh.ttf",    # 微软雅黑
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
        ]
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 12)
                break
        
        # 如果没有找到系统字体，使用默认字体
        if not font:
            font = ImageFont.load_default()
    except Exception as e:
        print(f"[OCR] 加载字体失败: {e}")
        font = ImageFont.load_default()
    
    # 绘制额外的区域框
    if regions:
        for i, (x_min, y_min, x_max, y_max) in enumerate(regions):
            # 绘制区域框（蓝色实线，因为 PIL 不支持虚线）
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="blue", width=2)
            # 绘制区域编号
            region_text = f"区域 {i+1}"
            draw.text((x_min + 5, y_min - 15), region_text, fill="blue", font=font)
    
    if isinstance(ocr_result, dict) and 'results' in ocr_result:
        # 处理本地 OCR 结果
        for item in ocr_result['results']:
            center_x, center_y = item['center_x'], item['center_y']
            width, height = item['width'], item['height']
            text = item['text']
            
            # 绘制中心点
            draw.ellipse([(center_x-5, center_y-5), (center_x+5, center_y+5)], fill="red")
            
            # 绘制坐标文本
            coord_text = f"({center_x}, {center_y})"
            draw.text((center_x + 10, center_y - 10), coord_text, fill="red", font=font)
            
            # 绘制识别文本
            draw.text((center_x + 10, center_y + 10), text, fill="green", font=font)
            
            # 绘制包围框
            x1 = center_x - width // 2
            y1 = center_y - height // 2
            x2 = center_x + width // 2
            y2 = center_y + height // 2
            draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
    else:
        # 处理云端 OCR 结果
        import re
        rect_pattern = re.compile(r'"rotate_rect":   $ (\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+) $  ,\s*"text": "([^"]+)"')
        matches = rect_pattern.findall(str(ocr_result))
        
        for match in matches:
            center_x, center_y, width, height, angle, text = map(int, match[:5]) + (match[5],)
            
            # 绘制中心点
            draw.ellipse([(center_x-5, center_y-5), (center_x+5, center_y+5)], fill="red")
            
            # 绘制坐标文本
            coord_text = f"({center_x}, {center_y})"
            draw.text((center_x + 10, center_y - 10), coord_text, fill="red", font=font)
            
            # 绘制识别文本
            draw.text((center_x + 10, center_y + 10), text, fill="green", font=font)
            
            # 绘制包围框（简化，不旋转）
            if angle == 90:
                x1 = center_x - height // 2
                y1 = center_y - width // 2
                x2 = center_x + height // 2
                y2 = center_y + width // 2
            else:
                x1 = center_x - width // 2
                y1 = center_y - height // 2
                x2 = center_x + width // 2
                y2 = center_y + height // 2
            draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0] if image_path else "screenshot"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base_name}_ocr_{timestamp}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    
    # 保存图片
    draw_image.save(output_path, "JPEG")
    return output_path


def capture_screen_region(left: int = None, top: int = None, right: int = None, bottom: int = None) -> Image.Image:
    """捕获指定区域的屏幕"""
    if left is not None and top is not None and right is not None and bottom is not None:
        # 捕获指定区域
        image = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
        print(f"[OCR] 捕获区域截图: ({left}, {top}) -> ({right}, {bottom})")
    else:
        # 全屏截图
        image = pyautogui.screenshot()
        print("[OCR] 捕获全屏截图")
    return image


def find_text_coordinates(text_to_find: str, image_path: str = None,
                         left: int = None, top: int = None,
                         right: int = None, bottom: int = None):
    """
    查找指定文字的屏幕中心坐标。
    支持传入图片路径或自动截图（可指定区域）。
    返回 JSON 字符串，包含坐标、OCR 结果、调试图路径。
    """
    try:
        total_start_time = time.time()
        screen_width, screen_height = pyautogui.size()

        # 加载图像：优先使用指定路径，否则截图
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            print(f"[OCR] 使用指定图片: {image_path}")
        else:
            if image_path:
                print(f"[OCR] 图片路径不存在: {image_path}，使用屏幕截图")
            # 捕获指定区域或全屏
            image = capture_screen_region(left, top, right, bottom)

        screenshot_width, screenshot_height = image.size
        scale_x = screen_width / screenshot_width
        scale_y = screen_height / screenshot_height

        print(f"[OCR] 屏幕: ({screen_width}, {screen_height}), 截图: ({screenshot_width}, {screenshot_height})")
        print(f"[OCR] 缩放比例: x={scale_x:.4f}, y={scale_y:.4f}")

        # 步骤1：使用 Tesseract 快速扫描整个屏幕，找到相关字
        print("[OCR] 步骤1: 使用 Tesseract 快速扫描整个屏幕")
        tesseract_start = time.time()
        tesseract_result = tesseract_ocr(image)
        tesseract_time = time.time() - tesseract_start
        print(f"[OCR] Tesseract 扫描完成，耗时: {tesseract_time:.2f}秒")
        
        found = False
        center_x_screen = center_y_screen = None
        ocr_result = tesseract_result
        output_path = ""
        merged_regions = []

        if tesseract_result['success']:
            # 解析 Tesseract 结果
            tesseract_results = tesseract_result['results']
            
            # 方法1：直接查找完整文本
            for item in tesseract_results:
                if text_to_find in item['text']:
                    found = True
                    cx_img, cy_img = item['center_x'], item['center_y']
                    center_x_screen = int(cx_img * scale_x)
                    center_y_screen = int(cy_img * scale_y)
                    print(f"[OCR] Tesseract 直接找到 '{text_to_find}' → 屏幕坐标: ({center_x_screen}, {center_y_screen})")
                    break
            
            # 步骤2：如果方法1失败，使用混合策略：Tesseract 找相关字 + EasyOCR 精确识别
            relevant_regions = []
            merged_regions = []
            if not found and len(text_to_find) > 1 and EASYOCR_AVAILABLE:
                print(f"[OCR] 步骤2: 使用混合策略查找 '{text_to_find}'")
                # 提取目标文本的所有字符
                target_chars = set(text_to_find)
                
                # 找到包含任何目标字符的区域
                relevant_regions = []
                for item in tesseract_results:
                    item_text = item['text']
                    if any(char in item_text for char in target_chars):
                        # 计算区域边界，适当扩大范围
                        x_min = max(0, item['center_x'] - item['width'] - 50)
                        y_min = max(0, item['center_y'] - item['height'] - 50)
                        x_max = min(screenshot_width, item['center_x'] + item['width'] + 50)
                        y_max = min(screenshot_height, item['center_y'] + item['height'] + 50)
                        relevant_regions.append((x_min, y_min, x_max, y_max))
                
                if relevant_regions:
                    print(f"[OCR] 找到 {len(relevant_regions)} 个相关区域，使用 EasyOCR 进行精确识别")
                    
                    # 合并重叠的区域
                    merged_regions = _merge_overlapping_regions(relevant_regions)
                    print(f"[OCR] 合并后得到 {len(merged_regions)} 个区域")
                    
                    # 对每个合并后的区域使用 EasyOCR 进行精确识别
                    easyocr_start = time.time()
                    for i, (x_min, y_min, x_max, y_max) in enumerate(merged_regions):
                        # 截取区域
                        region = image.crop((x_min, y_min, x_max, y_max))
                        
                        # 使用 EasyOCR 识别该区域
                        region_result = easyocr_ocr(region)
                        
                        if region_result['success']:
                            # 在 EasyOCR 结果中查找完整文本
                            for item in region_result['results']:
                                if text_to_find in item['text']:
                                    # 计算原始图像中的坐标
                                    cx_region = item['center_x']
                                    cy_region = item['center_y']
                                    cx_img = x_min + cx_region
                                    cy_img = y_min + cy_region
                                    
                                    center_x_screen = int(cx_img * scale_x)
                                    center_y_screen = int(cy_img * scale_y)
                                    found = True
                                    print(f"[OCR] EasyOCR 在区域 {i+1} 中找到 '{text_to_find}' → 屏幕坐标: ({center_x_screen}, {center_y_screen})")
                                    # 更新 OCR 结果为 EasyOCR 的结果，并转换坐标为原始图像的绝对坐标
                                    # 复制 region_result 并转换所有坐标
                                    converted_result = {
                                        'success': region_result['success'],
                                        'results': [],
                                        'elapsed_time': region_result['elapsed_time'],
                                        'method': region_result['method']
                                    }
                                    for result_item in region_result['results']:
                                        converted_item = result_item.copy()
                                        # 转换坐标为原始图像的绝对坐标
                                        converted_item['center_x'] = result_item['center_x'] + x_min
                                        converted_item['center_y'] = result_item['center_y'] + y_min
                                        converted_result['results'].append(converted_item)
                                    ocr_result = converted_result
                                    break
                        if found:
                            break
                    
                    easyocr_time = time.time() - easyocr_start
                    print(f"[OCR] EasyOCR 精确识别完成，耗时: {easyocr_time:.2f}秒")
            
            # 步骤3：如果仍然失败，尝试组合相邻的汉字
            if not found and len(text_to_find) > 1:
                print(f"[OCR] 步骤3: 尝试组合相邻汉字查找 '{text_to_find}'")
                
                # 按 y 坐标排序，同一行的文字会排在一起
                sorted_results = sorted(tesseract_results, key=lambda x: (x['center_y'], x['center_x']))
                
                # 组合相邻的汉字
                combined_texts = []
                current_line = []
                current_y = None
                y_threshold = 20  # y坐标阈值，用于判断是否在同一行，增大阈值
                x_threshold = 40  # x坐标阈值，用于判断是否相邻，增大阈值
                
                for item in sorted_results:
                    cy = item['center_y']
                    
                    # 如果是新的一行，处理上一行并开始新行
                    if current_y is None or abs(cy - current_y) > y_threshold:
                        if current_line:
                            # 组合当前行的文字
                            combined = _combine_adjacent_texts(current_line, x_threshold)
                            combined_texts.extend(combined)
                        current_line = [item]
                        current_y = cy
                    else:
                        # 同一行，添加到当前行
                        current_line.append(item)
                
                # 处理最后一行
                if current_line:
                    combined = _combine_adjacent_texts(current_line, x_threshold)
                    combined_texts.extend(combined)
                
                # 打印组合后的文本，用于调试
                print("[OCR] 组合后的文本:")
                for item in combined_texts:
                    if len(item['text']) > 1:
                        print(f"  - '{item['text']}' (位置: ({item['center_x']},{item['center_y']}))")
                
                # 在组合后的文本中查找
                for item in combined_texts:
                    if text_to_find in item['text']:
                        found = True
                        cx_img, cy_img = item['center_x'], item['center_y']
                        center_x_screen = int(cx_img * scale_x)
                        center_y_screen = int(cy_img * scale_y)
                        print(f"[OCR] 组合汉字后找到 '{text_to_find}' → 屏幕坐标: ({center_x_screen}, {center_y_screen})")
                        break
        
        # 保存调试图（传递合并后的区域）
        output_path = save_ocr_image_with_boxes(image, ocr_result, image_path, merged_regions)

        total_elapsed_time = time.time() - total_start_time
        print(f"[OCR] 总耗时: {total_elapsed_time:.2f}秒")

        if found and center_x_screen is not None:
            # 找到目标文本时，确保返回结果中明确标识目标文本的坐标
            target_text_info = None
            # 尝试在ocr_result中找到目标文本的详细信息
            if ocr_result.get('success') and ocr_result.get('results'):
                for item in ocr_result['results']:
                    if text_to_find in item['text']:
                        target_text_info = item
                        break
            
            result = {
                "success": True,
                "text": text_to_find,
                "center_x": center_x_screen,
                "center_y": center_y_screen,
                "target_text_info": target_text_info,  # 明确的目标文本信息
                "ocr_result": ocr_result,
                "image_path": output_path,
                "elapsed_time": total_elapsed_time,
                "method": ocr_result.get('method', 'local'),
                "message": f"成功找到文本 '{text_to_find}'，坐标为 ({center_x_screen}, {center_y_screen})"
            }
        else:
            result = {
                "success": False,
                "error": f"未找到文字: {text_to_find}",
                "ocr_result": ocr_result,
                "image_path": output_path,
                "elapsed_time": total_elapsed_time,
                "method": ocr_result.get('method', 'local')
            }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": f"异常: {str(e)}"}, ensure_ascii=False, indent=2)


def _merge_overlapping_regions(regions):
    """
    合并重叠的区域
    """
    if not regions:
        return []
    
    # 按 x_min 排序
    sorted_regions = sorted(regions, key=lambda r: r[0])
    merged = [sorted_regions[0]]
    
    for current in sorted_regions[1:]:
        last = merged[-1]
        # 检查是否重叠
        if current[0] <= last[2] and current[1] <= last[3]:
            # 合并区域
            new_region = (
                min(last[0], current[0]),
                min(last[1], current[1]),
                max(last[2], current[2]),
                max(last[3], current[3])
            )
            merged[-1] = new_region
        else:
            merged.append(current)
    
    return merged


def _combine_adjacent_texts(line_items, x_threshold):
    """
    组合同一行中相邻的文本元素
    """
    if not line_items:
        return []
    
    # 按 x 坐标排序
    sorted_items = sorted(line_items, key=lambda x: x['center_x'])
    
    combined = []
    current_group = [sorted_items[0]]
    
    for item in sorted_items[1:]:
        prev_item = current_group[-1]
        # 计算两个文本元素之间的距离
        distance = item['center_x'] - prev_item['center_x']
        
        # 如果距离小于阈值，认为是相邻的
        if distance < x_threshold:
            current_group.append(item)
        else:
            # 组合当前组
            if len(current_group) == 1:
                # 单个文本元素，直接添加
                combined.append(current_group[0])
            else:
                # 组合多个文本元素
                combined_text = ''.join([i['text'] for i in current_group])
                # 计算组合后的中心坐标（取平均值）
                avg_x = sum([i['center_x'] for i in current_group]) / len(current_group)
                avg_y = sum([i['center_y'] for i in current_group]) / len(current_group)
                # 计算组合后的宽度
                min_x = min([i['center_x'] - i['width']//2 for i in current_group])
                max_x = max([i['center_x'] + i['width']//2 for i in current_group])
                combined_width = max_x - min_x
                # 使用最大的高度
                max_height = max([i['height'] for i in current_group])
                # 计算置信度（取平均值）
                avg_confidence = sum([i['confidence'] for i in current_group]) / len(current_group)
                
                # 创建组合后的文本元素
                combined_item = {
                    'text': combined_text,
                    'center_x': int(avg_x),
                    'center_y': int(avg_y),
                    'width': combined_width,
                    'height': max_height,
                    'confidence': avg_confidence
                }
                combined.append(combined_item)
            
            # 开始新的组
            current_group = [item]
    
    # 处理最后一组
    if len(current_group) == 1:
        combined.append(current_group[0])
    else:
        combined_text = ''.join([i['text'] for i in current_group])
        avg_x = sum([i['center_x'] for i in current_group]) / len(current_group)
        avg_y = sum([i['center_y'] for i in current_group]) / len(current_group)
        min_x = min([i['center_x'] - i['width']//2 for i in current_group])
        max_x = max([i['center_x'] + i['width']//2 for i in current_group])
        combined_width = max_x - min_x
        max_height = max([i['height'] for i in current_group])
        avg_confidence = sum([i['confidence'] for i in current_group]) / len(current_group)
        
        combined_item = {
            'text': combined_text,
            'center_x': int(avg_x),
            'center_y': int(avg_y),
            'width': combined_width,
            'height': max_height,
            'confidence': avg_confidence
        }
        combined.append(combined_item)
    
    return combined


def get_all_text(image_path: str = None, left: int = None, top: int = None, 
                right: int = None, bottom: int = None):
    """获取所有识别文本（辅助函数，非 MCP 工具）"""
    try:
        total_start_time = time.time()
        
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            print(f"[OCR] 使用指定图片: {image_path}")
        else:
            # 捕获指定区域或全屏
            image = capture_screen_region(left, top, right, bottom)

        # 只使用本地 OCR
        local_result = local_ocr(image)
        ocr_result = ""

        if local_result['success']:
            # 格式化本地 OCR 结果
            ocr_result = "\n".join([f"{item['text']} (置信度: {item['confidence']}%)" 
                                  for item in local_result['results']])
            print(f"[OCR] 本地 OCR 识别到 {len(local_result['results'])} 条文本")
        else:
            # 本地 OCR 失败
            print("[OCR] 本地 OCR 失败")
            ocr_result = f"本地 OCR 失败: {local_result.get('error', '未知错误')}"

        # 保存带识别框的图片
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 绘制并保存带框的图片
        output_path = save_ocr_image_with_boxes(image, local_result, image_path)
        print(f"[OCR] 带识别框的图片已保存: {output_path}")

        total_elapsed_time = time.time() - total_start_time
        print(f"[OCR] 总耗时: {total_elapsed_time:.2f}秒")

        method = local_result.get('method', 'unknown')
        method_name = 'EasyOCR' if method == 'easyocr' else 'Tesseract' if method == 'tesseract' else '未知'
        return f"识别结果:\n{ocr_result}\n\n耗时: {total_elapsed_time:.2f}秒\n方法: {method_name}"

    except Exception as e:
        return f"获取所有文本失败: {str(e)}"


def test_ocr_performance():
    """测试 OCR 性能"""
    print("=== OCR 性能测试 ===")
    
    # 测试全屏截图 + 本地 OCR
    print("\n1. 测试全屏截图 + 本地 OCR:")
    start_time = time.time()
    image = capture_screen_region()
    local_result = local_ocr(image)
    # 保存带识别框的图片
    if local_result['success']:
        output_path = save_ocr_image_with_boxes(image, local_result)
        print(f"带识别框的图片已保存: {output_path}")
    elapsed = time.time() - start_time
    print(f"耗时: {elapsed:.2f}秒")
    print(f"识别结果数量: {len(local_result['results']) if local_result['success'] else 0}")
    
    # 测试区域截图 + 本地 OCR
    print("\n2. 测试区域截图 + 本地 OCR:")
    # 截取屏幕中央区域
    screen_width, screen_height = pyautogui.size()
    left, top = screen_width // 4, screen_height // 4
    right, bottom = screen_width * 3 // 4, screen_height * 3 // 4
    
    start_time = time.time()
    image = capture_screen_region(left, top, right, bottom)
    local_result = local_ocr(image)
    # 保存带识别框的图片
    if local_result['success']:
        output_path = save_ocr_image_with_boxes(image, local_result)
        print(f"带识别框的图片已保存: {output_path}")
    elapsed = time.time() - start_time
    print(f"耗时: {elapsed:.2f}秒")
    print(f"识别结果数量: {len(local_result['results']) if local_result['success'] else 0}")
    
    print("\n=== 性能测试完成 ===")
    return "测试完成"


def test_ocr_image(image_path):
    """测试指定图片的OCR功能"""
    try:
        print(f"=== 测试指定图片的OCR功能 ===")
        print(f"图片路径: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"错误: 图片路径不存在: {image_path}")
            return f"错误: 图片路径不存在: {image_path}"
        
        # 加载图片
        start_time = time.time()
        image = Image.open(image_path).convert("RGB")
        print(f"图片加载成功，大小: {image.size}")
        
        # 执行OCR识别
        local_result = local_ocr(image)
        
        # 保存带识别框的图片
        output_path = save_ocr_image_with_boxes(image, local_result, image_path)
        print(f"带识别框的图片已保存: {output_path}")
        
        total_elapsed_time = time.time() - start_time
        print(f"总耗时: {total_elapsed_time:.2f}秒")
        
        if local_result['success']:
            print(f"识别到 {len(local_result['results'])} 条文本")
            # 格式化识别结果
            result_text = "识别结果:\n"
            for item in local_result['results']:
                text = item['text']
                confidence = item['confidence']
                center_x = item['center_x']
                center_y = item['center_y']
                result_text += f"- '{text}' (置信度: {confidence:.1f}%, 位置: ({center_x},{center_y}))\n"
            result_text += f"\n耗时: {total_elapsed_time:.2f}秒\n方法: {local_result.get('method', 'unknown')}\n输出路径: {output_path}"
        else:
            print(f"OCR识别失败: {local_result.get('error', '未知错误')}")
            result_text = f"OCR识别失败: {local_result.get('error', '未知错误')}\n输出路径: {output_path}"
        
        print(f"=== 测试完成 ===")
        return result_text
    except Exception as e:
        print(f"测试OCR功能失败: {e}")
        import traceback
        traceback.print_exc()
        return f"测试OCR功能失败: {str(e)}"

if __name__ == '__main__':
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        # 有命令行参数，测试指定图片
        image_path = sys.argv[1]
        result = test_ocr_image(image_path)
        print(result)
    else:
        # 无命令行参数，运行MCP服务
        mcp.run()