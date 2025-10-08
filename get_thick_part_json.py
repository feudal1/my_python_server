#get_thick_part_json.py
import json
import re
import requests
import os
import logging
from flask import Flask, request, jsonify

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/process_excel', methods=['GET'])
def process_excel_api():
    """
    处理Excel文件并发送给AI服务的API接口
    ---
    parameters:
      - name: filepath
        in: query
        type: string
        required: true
        description: Excel文件路径
      - name: sheet_name
        in: query
        type: string
        required: false
        default: 零件清单
        description: 工作表名称
    responses:
      200:
        description: 成功处理
      400:
        description: 请求参数错误
      500:
        description: 服务器内部错误
    """
    # 获取查询参数
    filepath = request.args.get('filepath')
    sheet_name = request.args.get('sheet_name', '零件清单')
    import os
    filename_with_ext = os.path.basename(filepath) if filepath else ""
    filename = os.path.splitext(filename_with_ext)[0]
    
    # 检查必需参数
    if not filepath:
        return jsonify({"error": "缺少必需参数: filepath"}), 400
    
    try:
        # 第一步：从Excel获取数据
        excel_url = "http://localhost:5000/read_excel"
        
        # 对URL参数进行编码
        params = {
            "filepath": filepath,
            "sheet_name": sheet_name
        }
        
        # 获取Excel数据
        excel_response = requests.get(excel_url, params=params)
        excel_response.raise_for_status()
        excel_data = excel_response.json()
        
        # 提取板厚和构件信息
        extracted_info = excel_data
        
        # 在获取excel_memory之前，先从环境变量或配置中获取该信息
        excel_memory = os.environ.get('excel_memory', '')

        ai_message_content = f"""请从以下Excel数据中提取板厚,编号,数量和构件信息，然后整合成一个partname，以及需要平方米的板,以及有几种零件，并按照指定格式返回JSON，使用[START_JSON]和[END_JSON]标记框起JSON部分。特别注意：请将零件按照厚度进行分组。

        关于材料的一些记忆信息：{excel_memory}

        示例格式：
        [START_JSON]
        {{
        "thickness_groups": [
            {{
            "thickness": "10mm",
            "area_required": 2.5,
            "kind_member":"1",
            "components": [
                {{
                "part_name": "{filename }10-2PL10x570X4810=5个",
         
                "component": "角钢"
                 "thickness": "10mm",
                }}
            ]
            }}
        ]
        }}
        [END_JSON]
        Excel名称：{filename }
        Excel数据：
        {json.dumps(extracted_info, ensure_ascii=False)}"""
        
        # 第二步：使用POST请求发送数据给AI服务，避免URL过长问题
        ai_url = "http://localhost:5003/chat"
        ai_payload = {
            "messages": [{
                "role": "user",
                "content": ai_message_content
            }],
            "tools":  None
        }
        
        # 发送POST请求到AI服务
        ai_response = requests.post(ai_url, json=ai_payload)
        ai_response.raise_for_status()
        
        # 解析AI响应
        ai_response_data = ai_response.json()
        ai_content = ai_response_data['choices'][0]['message']['content']
                
        # 从AI响应中提取被标记框起的JSON
        json_pattern = r'\[START_JSON\]\s*(.*?)\s*\[END_JSON\]'
        match = re.search(json_pattern, ai_content, re.DOTALL)

        structured_data = None
        if match:
            try:
                json_content = match.group(1).strip()
                logger.info(f"提取到的JSON内容: {json_content}")
                
                # 将提取的JSON字符串解析为Python对象
                structured_data = json.loads(json_content)
                logger.info("JSON解析成功")
                
                # 保存解析出的JSON到文件
                save_path = os.path.join(os.getcwd(), 'ai_parsed_result.json')
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(structured_data, f, ensure_ascii=False, indent=2)
                logger.info(f"结果已保存至: {save_path}")
                    
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON解析失败: {json_err}")
                logger.error(f"尝试解析的内容: {json_content}")
                pass
            except Exception as save_err:
                logger.error(f"保存文件时出错: {save_err}")
                pass
        else:
            logger.warning("未在AI响应中找到JSON标记")
        
        # 返回处理后的数据
        return jsonify({
            "status": "success",

            # "excel_data": excel_data,
            "structured_data": structured_data  # 已解析的结构化数据
        })
        
    except requests.exceptions.RequestException as e:
        logger.error(f"请求出错: {str(e)}")
        return jsonify({"error": f"请求出错: {str(e)}"}), 500
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析出错: {str(e)}")
        return jsonify({"error": f"JSON解析出错: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"处理过程中出错: {str(e)}")
        return jsonify({"error": f"处理过程中出错: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    """
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)