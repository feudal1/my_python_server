from typing import Any


class ContentExtractor:
    """从LLM响应中提取内容的工具类"""
    
    @staticmethod
    def extract_content_from_response(result: Any) -> str:
        """
        从LLM响应中安全提取纯文本内容，兼容多种格式
        
        Args:
            result: LLM返回的结果，可能是字符串或字典
            
        Returns:
            提取的文本内容
        """
        if isinstance(result, str):
            return result.strip()
        
        if isinstance(result, dict):
            choices = result.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                first_choice = choices[0]
                
                # 尝试从message中提取内容
                message = first_choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content.strip()
                    elif content is None:
                        return ""
                
                # 尝试从delta中提取内容（流式响应）
                delta = first_choice.get("delta")
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str):
                        return content.strip()
                
                # 如果有finish_reason但没有内容，返回空字符串
                if "finish_reason" in first_choice:
                    return ""
        
        return ""
