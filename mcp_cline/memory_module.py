"""
游戏记忆模块 - 用于VLM游戏战术分析
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import threading
import re


class GameMemory:
    """游戏记忆管理类 - 记录玩家行为和游戏状态"""
    
    def __init__(self, memory_file: str = "game_memory.json"):
        self.memory_file = memory_file
        self.memories = []
        self.current_session = []
        self.lock = threading.Lock()
        self.load_memory()
    
    def load_memory(self):
        """从文件加载历史记忆"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():  # 只有文件不为空时才解析
                        data = json.loads(content)
                        self.memories = data.get('memories', [])
                    else:
                        self.memories = []
        except Exception as e:
            print(f"加载记忆文件失败: {e}")
            self.memories = []
    
    def save_memory(self):
        """保存记忆到文件"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'memories': self.memories,
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存记忆文件失败: {e}")
    
    def add_memory(self, action: str, context: str = "", analysis: str = ""):
        """
        添加一条游戏记忆
        
        Args:
            action: 执行的动作
            context: 当前上下文（如：敌我单位情况、资源状态等）
            analysis: AI的分析结果
        """
        memory = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'action': action,
            'context': context,
            'analysis': analysis
        }
        
        with self.lock:
            self.memories.append(memory)
            self.current_session.append(memory)
            # 限制内存中的记录数量，保留最近的100条
            if len(self.memories) > 100:
                self.memories = self.memories[-100:]
            self.save_memory()
    
    def get_recent_memories(self, count: int = 10) -> List[Dict]:
        """获取最近的记忆记录"""
        with self.lock:
            return self.memories[-count:] if self.memories else []
    
    def get_all_memories(self) -> List[Dict]:
        """获取所有记忆记录"""
        with self.lock:
            return self.memories.copy()
    
    def get_memories_by_action(self, keyword: str) -> List[Dict]:
        """根据关键词搜索相关记忆"""
        with self.lock:
            return [m for m in self.memories if keyword.lower() in m['action'].lower()]
    
    def clear_current_session(self):
        """清空当前会话记录（不清空历史）"""
        with self.lock:
            self.current_session = []
    
    def get_session_summary(self) -> str:
        """获取当前会话的摘要"""
        with self.lock:
            if not self.current_session:
                return "当前会暂无记录"
            
            summary_lines = [f"会话记录 (共{len(self.current_session)}条):"]
            for i, memory in enumerate(self.current_session[-20:], 1):  # 最多显示20条
                summary_lines.append(f"{i}. [{memory['timestamp']}] {memory['action']}")
                if memory['context']:
                    summary_lines.append(f"   上下文: {memory['context'][:100]}...")
                if memory['analysis']:
                    summary_lines.append(f"   分析: {memory['analysis'][:100]}...")
            return "\n".join(summary_lines)
    
    def get_context_for_prompt(self, include_count: int = 5) -> str:
        """
        获取用于prompt的上下文记忆
        返回格式化的记忆字符串，适合插入到AI对话中
        """
        recent_memories = self.get_recent_memories(include_count)
        if not recent_memories:
            return "暂无历史记录"
        
        context_lines = ["【历史游戏记录】"]
        for i, memory in enumerate(recent_memories, 1):
            context_lines.append(f"{i}. 时间: {memory['timestamp']}")
            context_lines.append(f"   行动: {memory['action']}")
            if memory['context']:
                context_lines.append(f"   当时情况: {memory['context']}")
            if memory['analysis']:
                context_lines.append(f"   战术分析: {memory['analysis']}")
            context_lines.append("")
        
        return "\n".join(context_lines)
    
    def analyze_memories(self) -> str:
        """对所有记忆进行战术分析总结"""
        with self.lock:
            if not self.memories:
                return "暂无足够的记录进行分析"
            
            # 统计常见行动
            actions = [m['action'] for m in self.memories]
            from collections import Counter
            common_actions = Counter(actions).most_common(5)
            
            analysis_lines = ["【战术分析总结】"]
            analysis_lines.append(f"总记录数: {len(self.memories)}")
            analysis_lines.append("\n最频繁的行动:")
            for action, count in common_actions:
                analysis_lines.append(f"- {action}: {count}次")
            
            # 显示最近的战略决策
            recent_with_analysis = [m for m in self.memories[-10:] if m['analysis']]
            if recent_with_analysis:
                analysis_lines.append("\n最近的重要分析:")
                for m in recent_with_analysis[-5:]:
                    analysis_lines.append(f"- [{m['timestamp']}] {m['analysis'][:150]}")
            
            return "\n".join(analysis_lines)


class MemoryPromptInjector:
    """记忆注入器 - 负责判断何时注入记忆到prompt中"""
    
    def __init__(self, memory: GameMemory):
        self.memory = memory
        # 工具调用命令前缀
        self.tool_command_prefixes = ['/r', '/R']
    
    def should_inject_memory(self, user_input: str) -> bool:
        """
        判断是否应该注入记忆
        
        规则:
        - /r 开头的命令（工具调用）不注入记忆
        - 普通对话注入记忆
        """
        user_input = user_input.strip()
        for prefix in self.tool_command_prefixes:
            if user_input.startswith(prefix):
                return False
        return True
    
    def inject_memory_to_prompt(self, user_input: str, memory_count: int = 5) -> str:
        """
        将记忆注入到用户输入中
        
        Args:
            user_input: 原始用户输入
            memory_count: 要注入的最近记忆条数
        
        Returns:
            注入记忆后的完整prompt
        """
        if not self.should_inject_memory(user_input):
            return user_input
        
        context = self.memory.get_context_for_prompt(memory_count)
        
        # 构造完整的prompt
        full_prompt = f"""{context}

【当前玩家问题】
{user_input}

请基于历史游戏记录和当前情况，给出战术分析和建议。"""
        
        return full_prompt
    
    def parse_ai_response(self, ai_response: str) -> tuple:
        """
        解析AI响应，提取行动和分析
        
        Returns:
            (action, analysis) 元组
        """
        action = ""
        analysis = ""
        
        # 尝试从响应中提取结构化信息
        # 假设AI回复格式包含"行动:"和"分析:"等关键词
        action_match = re.search(r'行动[：:]\s*(.+?)(?=\n|$)', ai_response)
        analysis_match = re.search(r'分析[：:]\s*(.+)', ai_response, re.DOTALL)
        
        if action_match:
            action = action_match.group(1).strip()
        else:
            # 如果没有明确的行动标记，取第一句话作为行动
            first_line = ai_response.split('\n')[0]
            action = first_line.strip() if first_line else ai_response[:100]
        
        if analysis_match:
            analysis = analysis_match.group(1).strip()
        else:
            # 剩余部分作为分析
            if action_match:
                analysis = ai_response[action_match.end():].strip()
            else:
                analysis = ai_response
        
        return action, analysis

