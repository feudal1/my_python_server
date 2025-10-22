# extract_parts_from_drawing.py
import requests
import json
import os
import logging
from flask import Flask, jsonify, request

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/extract-parts', methods=['GET'])
def extract_parts():
    """
    从AutoCAD图纸中提取零件信息并结构化为JSON格式
    
    ---
    parameters:
      - name: cad_server_url
        in: query
        type: string
        required: false
        default: http://localhost:5300
        description: AutoCAD服务地址
      - name: llm_server_url
        in: query
        type: string
        required: false
        default: http://localhost:5003
        description: LLM服务地址
    responses:
      200:
        description: 成功处理
      500:
        description: 服务器内部错误
    """
    
    # 获取参数
    cad_server_url = request.args.get('cad_server_url', 'http://localhost:5300')
    llm_server_url = request.args.get('llm_server_url', 'http://localhost:5003')
    
    try:
        # 步骤1: 从AutoCAD获取文本对象
        logger.info("正在从AutoCAD获取文本对象...")
        texts_response = requests.get(f"{cad_server_url}/objects/texts")
        texts_response.raise_for_status()
        texts_data = texts_response.json()
        
        if texts_data.get('status') != 'success':
            raise Exception(f"获取AutoCAD文本失败: {texts_data.get('message')}")
            
        texts_info = texts_data.get('texts_info', [])
        
        # 提取文本字符串
        text_strings = [text_obj.get('text_string', '') for text_obj in texts_info]
        
        # 在获取cad_memory之前，先从环境变量或配置中获取该信息
        cad_memory = os.environ.get('cad_memory', '')
        
        # 步骤2: 准备发送给LLM的消息
        ai_message_content = f"""请从以下AutoCAD图纸文本中提取零件信息,然后整合成一个partname,名称里的*号要替换成x比如(PL10x570X4810),以及需要平方米的板,以及有几种零件，并按照指定格式返回JSON，使用[START_JSON]和[END_JSON]标记框起JSON部分。特别注意：请将零件按照厚度进行分组。
        圆管单重用这个公式算=((PI()/4)*((直径^2)-(直径-2*厚度)^2)/100)*7.85/1000*长度/1000
        槽钢单重用这个公式算=0.00785 * (高度*腰厚 + 2*(腿宽-腰厚)*腿厚)
       
        示例格式：
        [START_JSON]
        [
            {{
                "part_name": "PL10x570X4810=5件",
                "长度":"570",
                "宽度":"420",
                "编号":"H-12",
                "数量":"5",
                "类型":"钢板",
                "厚度": "10"
            }},
            {{
                "part_name": "L45x6x420=6件",
                "长度":"420",
                "编号":"Z-5",
                "数量":"6",
                "类型":"角钢",
                "厚度": "6"
            }},
            {{
                "part_name": "PIP50X3.5x420=6件",
                "长度":"420",
                "编号":"HO27",
                "数量":"6",
                "类型":"圆管",
                "厚度": "3.5"
            }}
        ]
        [END_JSON]

        AutoCAD图纸中的文本：
        {json.dumps(text_strings, ensure_ascii=False, indent=2)}"""
        
        # 步骤3: 调用LLM服务处理文本
        logger.info("正在调用LLM服务处理文本...")
        ai_payload = {
            "messages": [{
                "role": "user",
                "content": ai_message_content
            }],
            "tools": None
        }
        
        ai_response = requests.post(f"{llm_server_url}/chat", json=ai_payload)
        ai_response.raise_for_status()
        ai_response_data = ai_response.json()
        
        # 步骤4: 解析AI响应
        ai_content = ai_response_data['choices'][0]['message']['content']
        
        # 从AI响应中提取被标记框起的JSON
        import re
        json_pattern = r'\[START_JSON\]\s*(.*?)\s*\[END_JSON\]'
        match = re.search(json_pattern, ai_content, re.DOTALL)
        
        structured_data = []
        if match:
            try:
                json_content = match.group(1).strip()
                logger.info(f"提取到的JSON内容: {json_content}")
                
                # 将提取的JSON字符串解析为Python对象
                structured_data = json.loads(json_content)
                logger.info("JSON解析成功")
                
                # 保存解析出的JSON到文件
                os.makedirs('output', exist_ok=True)
                save_path = os.path.join(os.getcwd(), 'output', 'cad_parsed_result.json')
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(structured_data, f, ensure_ascii=False, indent=2)
                logger.info(f"结果已保存至: {save_path}")
                    
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON解析失败: {json_err}")
                logger.error(f"尝试解析的内容: {json_content}")
                raise Exception(f"JSON解析失败: {str(json_err)}")
        else:
            logger.warning("未在AI响应中找到JSON标记")
            raise Exception("未在AI响应中找到JSON标记")
        
        # 步骤5: 调用计算钢材参数的服务
        calculated_data = None
        try:
            # 调用/json_to_excel服务中的calculate_steel_data接口来计算钢材参数
            calculate_url = "http://localhost:5005/calculate_steel_data"
            calculate_payload = {"data": structured_data}
            
            calculate_response = requests.post(calculate_url, json=calculate_payload)
            
            if calculate_response.status_code == 200:
                calculate_result = calculate_response.json()
                calculated_data = calculate_result.get("data")
                logger.info("钢材参数计算完成")
                
                # 保存计算结果到文件
                calc_save_path = os.path.join(os.getcwd(), 'output', 'calculated_result.json')
                with open(calc_save_path, 'w', encoding='utf-8') as f:
                    json.dump(calculated_data, f, ensure_ascii=False, indent=2)
                logger.info(f"计算结果已保存至: {calc_save_path}")
            else:
                logger.error(f"计算钢材参数失败: {calculate_response.text}")
                
        except Exception as calc_err:
            logger.error(f"调用计算服务时出错: {str(calc_err)}")
            # 不中断流程，只是记录错误
            
        # 步骤6: 调用服务将计算结果保存为Excel文件
        saved_calculated_excel_path = None
        if calculated_data:
            try:
                # 获取AutoCAD文档名称
                doc_name = None
                try:
                    doc_name_response = requests.get(f"{cad_server_url}/document/name")
                    if doc_name_response.status_code == 200:
                        doc_name_data = doc_name_response.json()
                        if doc_name_data.get('status') == 'success':
                            doc_name = doc_name_data.get('document_name')
                            # 移除文件扩展名
                            if doc_name and '.' in doc_name:
                                doc_name = ".".join(doc_name.split('.')[:-1])
                except Exception as doc_name_err:
                    logger.warning(f"获取AutoCAD文档名称时出错: {str(doc_name_err)}")
                
                # 构造保存的文件名
                calculated_excel_filename = "output/calculated_data.xlsx"
                # 在实际使用前确保目录存在
                os.makedirs(os.path.dirname(calculated_excel_filename), exist_ok=True)
                
                # 保存带有计算结果的数据为Excel文件
                json_to_excel_url = "http://localhost:5005/save_structured_data"
                calculated_excel_params = {
                    "filename": calculated_excel_filename,
                    "dxf_filename": doc_name  # 传递文档名作为dxf_filename参数
                }
                calculated_excel_payload = {"structured_data": calculated_data}
                
                calculated_excel_response = requests.post(
                    json_to_excel_url,
                    params=calculated_excel_params,
                    json=calculated_excel_payload
                )
                
                if calculated_excel_response.status_code == 200:
                    calculated_excel_result = calculated_excel_response.json()
                    saved_calculated_excel_path = calculated_excel_result.get("file_path")
                    logger.info(f"计算结果已保存为Excel文件: {saved_calculated_excel_path}")
                else:
                    logger.error(f"保存计算结果Excel文件失败: {calculated_excel_response.text}")
                        
            except Exception as save_excel_err:
                logger.error(f"调用保存Excel服务时出错: {str(save_excel_err)}")
        
        return jsonify({
            "status": "success",
            "structured_data": structured_data,
            "calculated_data": calculated_data,
            "saved_calculated_excel_path": saved_calculated_excel_path
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