#!/usr/bin/env python3
"""
技能加载工具 - 提供专业领域知识的技能加载功能
"""

import os
from typing import Any, Dict, List
from langchain_core.tools import tool


# 技能目录路径
SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))


@tool
def load_skill(skill_name: str) -> str:
    """
    根据技能名称加载对应的专家提示（prompt）。
    支持的技能：ue_mod制作助手, blender_mod制作助手, 网页_mod制作助手
    """
    # 加载目录中的技能
    skills = {}
    
    if os.path.exists(SKILLS_DIR):
        for skill_folder in os.listdir(SKILLS_DIR):
            skill_path = os.path.join(SKILLS_DIR, skill_folder)
            if os.path.isdir(skill_path):
                skills_file = os.path.join(skill_path, "skills.md")
                if os.path.exists(skills_file):
                    try:
                        with open(skills_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        # 解析skills.md文件
                        lines = content.strip().split('\n')
                        skill_name_line = next((line for line in lines if line.startswith('name:')), None)
                        if skill_name_line:
                            loaded_skill_name = skill_name_line.split(':', 1)[1].strip()
                            skills[loaded_skill_name] = content
                    except Exception as e:
                        print(f"加载技能 {skill_folder} 时出错: {e}")
    
    if skill_name not in skills:
        available = ", ".join(skills.keys())
        return f"错误：技能 '{skill_name}' 不存在。可用技能：{available}"
    
    return skills[skill_name]


@tool
def list_skills() -> str:
    """
    列出所有可用的技能
    """
    # 加载目录中的技能
    skills = {}
    
    if os.path.exists(SKILLS_DIR):
        for skill_folder in os.listdir(SKILLS_DIR):
            skill_path = os.path.join(SKILLS_DIR, skill_folder)
            if os.path.isdir(skill_path):
                skills_file = os.path.join(skill_path, "skills.md")
                if os.path.exists(skills_file):
                    try:
                        with open(skills_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        # 解析skills.md文件
                        lines = content.strip().split('\n')
                        skill_name_line = next((line for line in lines if line.startswith('name:')), None)
                        desc_line = next((line for line in lines if line.startswith('description:')), None)
                        if skill_name_line:
                            loaded_skill_name = skill_name_line.split(':', 1)[1].strip()
                            description = desc_line.split(':', 1)[1].strip() if desc_line else "无描述"
                            skills[loaded_skill_name] = description
                    except Exception as e:
                        print(f"加载技能 {skill_folder} 时出错: {e}")
    
    result = "可用技能：\n"
    for skill_name, description in skills.items():
        result += f"- {skill_name}: {description}\n"
    
    return result.strip()


if __name__ == "__main__":
    print("技能加载工具")
    print("可用函数:")
    print("- load_skill(skill_name)")
    print("- list_skills()")
    
    # 测试内部逻辑
    print("\n测试技能加载逻辑:")
    
    # 测试列出技能
    from langchain_core.tools import StructuredTool
    
    # 模拟 list_skills 工具的逻辑
    def test_list_skills():
        skills = {}
        
        if os.path.exists(SKILLS_DIR):
            for skill_folder in os.listdir(SKILLS_DIR):
                skill_path = os.path.join(SKILLS_DIR, skill_folder)
                if os.path.isdir(skill_path):
                    skills_file = os.path.join(skill_path, "skills.md")
                    if os.path.exists(skills_file):
                        try:
                            with open(skills_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            lines = content.strip().split('\n')
                            skill_name_line = next((line for line in lines if line.startswith('name:')), None)
                            desc_line = next((line for line in lines if line.startswith('description:')), None)
                            if skill_name_line:
                                loaded_skill_name = skill_name_line.split(':', 1)[1].strip()
                                description = desc_line.split(':', 1)[1].strip() if desc_line else "无描述"
                                skills[loaded_skill_name] = description
                        except Exception as e:
                            print(f"加载技能 {skill_folder} 时出错: {e}")
        
        result = "可用技能：\n"
        for skill_name, description in skills.items():
            result += f"- {skill_name}: {description}\n"
        
        return result.strip()
    
    print(test_list_skills())
    
    print("\n" + "="*50)
    print("技能系统测试完成！")
