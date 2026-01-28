#!/usr/bin/env python3
"""
记忆工具MCP服务器 - 提供AI读写记忆的能力

支持功能：
- read_memory: 读取记忆文档
- write_memory: 写入记忆文档
- search_memory: 搜索记忆文档
- grep_memory: 在记忆文档中搜索模式
- clear_memory: 清空记忆文档
- get_memory_stats: 获取记忆统计信息
"""

import sys
import os

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from memory_manager import MemoryManager
from mcp.server.fastmcp import FastMCP

# 使用FastMCP创建服务器
mcp = FastMCP("memory-tool")

# 初始化记忆管理器
# 获取项目根目录
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
memory_manager = MemoryManager(base_dir)
print(f"[记忆服务器] 初始化完成，基础目录: {base_dir}")

@mcp.tool()
async def read_memory(memory_type: str = 'general'):
    """
    读取记忆文档
    
    Args:
        memory_type: 记忆类型 ('general' 或 'tool')
        
    Returns:
        记忆内容
    """
    content = memory_manager.read_memory(memory_type)
    return f"记忆内容 ({memory_type}):\n\n{content}"

@mcp.tool()
async def write_memory(content: str, memory_type: str = 'general', append: bool = True):
    """
    写入记忆文档
    
    Args:
        content: 要写入的内容
        memory_type: 记忆类型 ('general' 或 'tool')
        append: 是否追加到文件末尾 (默认: true)
        
    Returns:
        操作结果
    """
    if not content:
        return "错误：内容不能为空"
    
    return memory_manager.write_memory(content, memory_type, append)

@mcp.tool()
async def search_memory(query: str, memory_type: str = 'general', max_results: int = 5):
    """
    搜索记忆文档
    
    Args:
        query: 搜索关键词
        memory_type: 记忆类型 ('general' 或 'tool')
        max_results: 最大返回结果数 (默认: 5)
        
    Returns:
        搜索结果
    """
    if not query:
        return "错误：查询词不能为空"
    
    results = memory_manager.search_memory(query, memory_type, max_results)
    
    # 格式化结果
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(
            f"{i}. [行{result['line_number']}] {result['content'][:100]}...\n"
            f"   匹配分数: {result['match_score']:.2f}"
        )
    
    output = f"搜索结果 (查询: '{query}', 类型: {memory_type}):\n\n"
    output += "\n\n".join(formatted_results) if formatted_results else "未找到匹配结果"
    
    return output

@mcp.tool()
async def grep_memory(pattern: str, memory_type: str = 'general', context_lines: int = 2):
    """
    在记忆文档中搜索模式（类似grep命令）
    
    Args:
        pattern: 正则表达式模式
        memory_type: 记忆类型 ('general' 或 'tool')
        context_lines: 上下文行数 (默认: 2)
        
    Returns:
        匹配结果
    """
    if not pattern:
        return "错误：模式不能为空"
    
    results = memory_manager.grep_memory(pattern, memory_type, context_lines)
    
    # 格式化结果
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(
            f"{i}. [行{result['line_number']}] {result['matched_line'][:80]}...\n"
            f"   上下文:\n{result['context']}"
        )
    
    output = f"Grep结果 (模式: '{pattern}', 类型: {memory_type}):\n\n"
    output += "\n\n".join(formatted_results) if formatted_results else "未找到匹配结果"
    
    return output

@mcp.tool()
async def clear_memory(memory_type: str = 'general'):
    """
    清空记忆文档
    
    Args:
        memory_type: 记忆类型 ('general' 或 'tool')
        
    Returns:
        操作结果
    """
    return memory_manager.clear_memory(memory_type)

@mcp.tool()
async def get_memory_stats(memory_type: str = 'general'):
    """
    获取记忆统计信息
    
    Args:
        memory_type: 记忆类型 ('general' 或 'tool')
        
    Returns:
        统计信息
    """
    stats = memory_manager.get_memory_stats(memory_type)
    
    # 格式化统计信息
    output = f"记忆统计 ({memory_type}):\n\n"
    output += f"存在: {'是' if stats.get('exists') else '否'}\n"
    output += f"大小: {stats.get('size', 0)} 字节\n"
    output += f"行数: {stats.get('lines', 0)}\n"
    output += f"最后修改: {stats.get('last_modified', '无')}\n"
    
    return output

if __name__ == "__main__":
    mcp.run()