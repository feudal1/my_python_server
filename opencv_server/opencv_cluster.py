import os
import pyautogui
import time
from PIL import Image
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def ensure_dir(file_path):
    """
    确保目录存在，如果不存在则创建。
    :param file_path: 文件或目录路径
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def capture_screen_by_two_points(top_left, bottom_right):
    """
    根据给定的两个点（左上角和右下角）截图。
    
    :param top_left: 左上角点的位置 (x, y)
    :param bottom_right: 右下角点的位置 (x, y)
    :return: 截图得到的PIL.Image对象
    """
    # 计算截图区域的宽度和高度
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]

    # 检查截图区域是否有效
    if width <= 0 or height <= 0:
        raise ValueError(f"截图区域无效: 宽度={width}, 高度={height}。请确保右下角点在左上角点的右下方且区域有一定大小")

    try:
        screenshot = pyautogui.screenshot(region=(top_left[0], top_left[1], width, height))
        return screenshot
    except Exception as e:
        raise RuntimeError(f"截图失败: {str(e)}")

def process_image_for_clustering(img_path, output_dir):
    """
    对图像进行聚类分析，并保存带有聚类标注的结果。
    
    :param img_path: 输入图像路径
    :param output_dir: 输出目录
    """
    # 检查输入文件是否存在
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"输入图像文件不存在: {img_path}")
    
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像文件: {img_path}")
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用自适应阈值进行二值化处理
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 形态学操作去噪
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 过滤掉太小的轮廓
    min_area = 50  # 最小面积阈值
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # 如果没有找到合适的轮廓，则直接保存原图
    if len(filtered_contours) == 0:
        print("未检测到符合条件的轮廓")
        output_file = os.path.join(output_dir, 'cluster_output.png')
        ensure_dir(output_file)
        cv2.imwrite(output_file, img)
        print(f"原始图像已保存至 {output_file}")
        return

    # 计算每个轮廓的中心点
    centers = []
    valid_contours = []
    
    for cnt in filtered_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append([cx, cy])
            valid_contours.append(cnt)

    # 如果没有有效的中心点
    if len(centers) == 0:
        print("未找到有效的轮廓中心点")
        output_file = os.path.join(output_dir, 'cluster_output.png')
        ensure_dir(output_file)
        cv2.imwrite(output_file, img)
        print(f"原始图像已保存至 {output_file}")
        return

    centers = np.array(centers)
    
    # 使用DBSCAN进行聚类
    clustering = DBSCAN(eps=30, min_samples=2).fit(centers)
    labels = clustering.labels_

    unique_labels = set(labels)
    output_img = img.copy()

    # 创建单独保存每个聚类的目录
    clusters_dir = os.path.join(output_dir, 'clusters')
    ensure_dir(clusters_dir)

    # 绘制聚类结果并保存每个聚类的图像
    cluster_count = 0
    
    for label in unique_labels:
        # 获取当前标签的所有轮廓索引
        if label == -1:  # 噪声点
            cluster_indices = np.where(labels == label)[0]
            color = (0, 0, 255)  # 红色表示噪声
        else:
            cluster_indices = np.where(labels == label)[0]
            color = (0, 255, 0)  # 绿色表示聚类
        
        # 合并当前聚类的所有轮廓
        merged_contour = np.vstack([valid_contours[i] for i in cluster_indices])
        
        # 计算最小外接矩形
        rect = cv2.minAreaRect(merged_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)  # 使用np.intp替代np.int0
        
        # 绘制轮廓和边界框
        for i in cluster_indices:
            cv2.drawContours(output_img, [valid_contours[i]], -1, color, 2)
            
        cv2.drawContours(output_img, [box], -1, (255, 0, 0), 2)
        
        # 标注聚类编号
        center = tuple(map(int, rect[0]))
        label_text = "noise" if label == -1 else str(label)
        cv2.putText(output_img, label_text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 保存当前聚类的图像（仅保存有效聚类，不保存噪声点）
        if label != -1:
            # 获取OBB的边界
            x_vals = [point[0] for point in box]
            y_vals = [point[1] for point in box]
            min_x, max_x = min(x_vals), max(x_vals)
            min_y, max_y = min(y_vals), max(y_vals)
            
            # 确保边界在图像范围内
            min_x = max(0, min_x)
            max_x = min(img.shape[1], max_x)
            min_y = max(0, min_y)
            max_y = min(img.shape[0], max_y)
            
            # 裁剪并保存图像
            if max_x > min_x and max_y > min_y:
                cluster_img = img[min_y:max_y, min_x:max_x]
                cluster_filename = os.path.join(clusters_dir, f'cluster_{label}.png')
                ensure_dir(cluster_filename)  # 确保目录存在
                cv2.imwrite(cluster_filename, cluster_img)
                print(f"聚类 {label} 的图像已保存至 {cluster_filename}")
                cluster_count += 1

    output_file = os.path.join(output_dir, 'cluster_output.png')
    ensure_dir(output_file)
    cv2.imwrite(output_file, output_img)
    print(f"聚类结果已保存至 {output_file}")
    print(f"共保存了 {cluster_count} 个聚类图像")

def get_user_click_position(prompt):
    """
    提示用户点击并获取位置
    
    :param prompt: 提示信息
    :return: Point对象
    """
    print(prompt)
    print("3秒后开始记录位置...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    position = pyautogui.position()
    print(f"记录位置: {position}")
    return position

if __name__ == '__main__':
    try:
        # 获取截图区域
        top_left = get_user_click_position("请将鼠标移动到截图区域的左上角")
        bottom_right = get_user_click_position("请将鼠标移动到截图区域的右下角")
        
        # 截图
        screenshot = capture_screen_by_two_points(top_left, bottom_right)
        
        # 保存截图
        screenshot_filename = 'screenshot.png'
        ensure_dir(screenshot_filename)  # 确保目录存在
        screenshot.save(screenshot_filename)
        print(f"截图已保存至 {screenshot_filename}")
        
        # 处理图像
        output_dir = 'output/opencv_output/cluster_output'
        process_image_for_clustering(screenshot_filename, output_dir)
        
    except ValueError as ve:
        print(f"参数错误: {ve}")
    except FileNotFoundError as fe:
        print(f"文件错误: {fe}")
    except Exception as e:
        print(f"发生未知错误: {e}")