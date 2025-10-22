# draw_server.py
import os
import json
from ezdxf import new
import requests
from PIL import Image
import base64
import io

# 假设 llm_server.py 已运行在 http://localhost:5003
VLM_API_URL = "http://localhost:5003/vlm/chat"

def parse_vlm_response(response_data):
    """
    解析 VLM 返回的结构化数据，提取几何元素
    示例输出格式：
    {
        "lines": [{"start": [0,0], "end": [10,0]}],
        "circles": [{"center": [5,5], "radius": 2}],
        "text": [{"position": [1,1], "content": "T=6"}]
    }
    """
    try:
        # 假设返回的是 JSON 字符串
        result = response_data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        parsed = json.loads(result)
        return parsed
    except Exception as e:
        print(f"解析失败: {e}")
        return {}

def create_dxf_from_data(data, output_path="dxf_output/draw/drawing.dxf"):
    """
    使用 ezdxf 创建 DXF 文件
    """
    doc = new('R2010')
    msp = doc.modelspace()
    os.makedirs(r"dxf_output", exist_ok=True)
    os.makedirs(r"dxf_output/draw", exist_ok=True)

    # 添加直线
    for line in data.get("lines", []):
        start = line["start"]  # 直接使用列表/元组
        end = line["end"]      # 直接使用列表/元组
        msp.add_line(start, end)

    # 添加圆
    for circle in data.get("circles", []):
        center = circle["center"]  # 直接使用列表/元组
        radius = circle["radius"]
        msp.add_circle(center, radius)

    # 添加文本
    for text in data.get("text", []):
        pos = text["position"]  # 直接使用列表/元组
        msp.add_text(text["content"], dxfattribs={"insert": pos})

    # 保存文件
    doc.saveas(output_path)
    print(f"DXF 文件已保存至: {output_path}")

def analyze_image_and_generate_dxf(image_path):
    """
    主函数：调用 VLM 分析图像，并生成 DXF
    """
    # Step 1: 编码图像为 base64
    with open(image_path, "rb") as f:
        image_data = f.read()
   

    # Step 2: 构造请求消息
    messages = [
        {
            "role": "user",
            "content": '''请分析这张图纸，提取所有几何元素（线段、圆、文字），并以 JSON 格式返回。
示例输出格式：
{
    "lines": [{"start": [0,0], "end": [10,0]}],
    "circles": [{"center": [5,5], "radius": 2}],
    "text": [{"position": [1,1], "content": "T=6"}]
}'''
        }
    ]

    # Step 3: 调用 VLM API
    params = {
        "messages": json.dumps(messages),
        "image_source": image_path,
        "tools": None
    }

    try:
        response = requests.get(VLM_API_URL, params=params)
        if response.status_code == 200:
            result = response.json()
            print("VLM 返回:", result)
            # 解析响应
            extracted_data = parse_vlm_response(result)
            # 生成 DXF
            create_dxf_from_data(extracted_data)
        else:
            print("VLM 请求失败:", response.text)
    except Exception as e:
        print("错误:", str(e))

# 示例调用
if __name__ == "__main__":
    image_file = r"E:\code\temp\a.jpg"  # 你的图片路径
    analyze_image_and_generate_dxf(image_file)