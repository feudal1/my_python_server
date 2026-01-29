from langchain_community.llms import VLLMOpenAI

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional, Dict, Any

openai_api_key="EMPTY",
openai_api_base="http://localhost:8001/v1"
model_name="/root/my_python_server/wsl/models/OpenBMB_MiniCPM-V-2_6-int4"
openai_api_key="d121416576624afca2902462fda1baff.Va2AyE6qDFzYTmZc",
openai_api_base="https://open.bigmodel.cn/api/paas/v4"
model_name="glm-4.6v-flash"
class VLMServiceWrapper(BaseChatModel):
    """使用LangChain的VLLMOpenAI包装器"""

    def __init__(self):
        """初始化VLM服务包装器"""
        # 初始化VLLMOpenAI模型
        self.llm = VLLMOpenAI(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            model_name=model_name,
            temperature=0.7,
            max_tokens=1000
        )

        print("VLM服务包装器初始化完成")

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        """生成响应"""
        # 将消息转换为文本格式
        prompt = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                prompt += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                prompt += f"Assistant: {message.content}\n"
            elif isinstance(message, SystemMessage):
                prompt += f"System: {message.content}\n"
        
        # 调用VLLMOpenAI生成响应
        response = self.llm.invoke(prompt, stop=stop)
        
        # 创建AIMessage
        ai_message = AIMessage(content=response)
        
        # 创建ChatGeneration
        generation = ChatGeneration(message=ai_message)
        
        # 创建ChatResult
        result = ChatResult(generations=[generation])
        return result

    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "vllm_openai"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回识别参数"""
        return {"service": "VLMServiceWrapper", "model": self.llm.model_name}
