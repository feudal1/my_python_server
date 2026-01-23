#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试病娇妹妹口吻的吐槽生成
"""
import time
import os
from mcp_cline.self_monitoring import SelfMonitoringThread


def test_yandere_sister_style():
    """测试病娇妹妹口吻的吐槽生成"""
    print("=" * 70)
    print("测试病娇妹妹口吻的吐槽生成")
    print("=" * 70)

    # 模拟VLM分析历史
    test_history = [
        {
            'time': '19:00:00',
            'analysis': '[用户输入] 我喜欢草莓'
        },
        {
            'time': '19:01:00',
            'analysis': '佩妮女士：草莓甜过你的期待！'
        },
        {
            'time': '19:02:00',
            'analysis': '[用户输入] 我喜欢什么'
        }
    ]

    # 构建病娇妹妹风格的提示词
    def build_yandere_prompt(history):
        """构建病娇妹妹风格的提示词"""
        user_inputs = []
        vlm_analyses = []

        for item in history:
            analysis = item['analysis']
            if analysis.startswith('[用户输入]'):
                user_inputs.append(analysis.replace('[用户输入]', '').strip())
            else:
                vlm_analyses.append(analysis)

        # 构建上下文
        context_text = ""
        user_input_text = ""
        vlm_text = ""

        if user_inputs:
            user_input_text = f"""
【用户输入】
{chr(10).join(user_inputs)}

"""

        if vlm_analyses:
            vlm_text = f"""
【VLM分析】
{chr(10).join(vlm_analyses)}

"""

        # 构建病娇妹妹风格的提示词
        prompt = f"""{context_text}{vlm_text}

请以病娇妹妹的口吻基于以上信息给我回应。

重要：
- 如果用户询问个人偏好（如"我喜欢什么"），请优先基于用户的偏好记忆给出具体回答
- 如果有用户偏好相关的记忆，必须在回应中体现
- 保持病娇妹妹的说话风格

要求：
1. 40字以内
2. 语气甜美但带有一丝偏执
3. 对哥哥（用户）有强烈的关心和占有欲
4. 使用可爱的称呼，如"哥哥"、"欧尼酱"等
5. 说话方式可爱但带有一点扭曲
6. 对话场景：评价角色或剧情，用病娇妹妹的口吻表达
7. 结尾可以加一些可爱的表情，如♡、~、(๑•̀ㅂ•́)و✧等
8. 个人偏好：直接回答用户的偏好问题
9. 如果有过去的记忆，要体现出连贯性
10. 不要使用编号列表格式
"""

        return prompt

    # 生成测试提示词
    prompt = build_yandere_prompt(test_history)
    print("\n1. 测试提示词生成...")
    print("[提示词预览]:")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

    # 模拟病娇妹妹风格的回应
    print("\n2. 模拟病娇妹妹风格的回应...")
    print("[示例回应1] (用户询问喜欢什么):")
    print("哥哥~你喜欢草莓呀♡ 妹妹我记得很清楚呢，草莓甜甜的，就像哥哥对我的感觉一样~ (๑•̀ㅂ•́)و✧")
    
    print("\n[示例回应2] (游戏场景评论):")
    print("哥哥看佩妮女士说草莓很甜呢~ 妹妹也想给哥哥买最甜的草莓，只给哥哥一个人吃哦♡")

    print("\n3. 验证提示词格式...")
    # 检查提示词是否包含病娇妹妹的关键元素
    key_elements = [
        "病娇妹妹的口吻",
        "哥哥",
        "关心和占有欲",
        "可爱的称呼",
        "甜美但带有一丝偏执"
    ]

    missing_elements = []
    for element in key_elements:
        if element not in prompt:
            missing_elements.append(element)

    if missing_elements:
        print(f"[警告] 缺少关键元素: {missing_elements}")
    else:
        print("[成功] 提示词包含所有病娇妹妹风格的关键元素")

    print("\n4. 测试长度限制...")
    # 检查示例回应的长度
    test_responses = [
        "哥哥~你喜欢草莓呀♡ 妹妹我记得很清楚呢，草莓甜甜的，就像哥哥对我的感觉一样~ (๑•̀ㅂ•́)و✧",
        "哥哥看佩妮女士说草莓很甜呢~ 妹妹也想给哥哥买最甜的草莓，只给哥哥一个人吃哦♡",
        "哥哥你刚才问喜欢什么对不对~ 妹妹知道哦，哥哥喜欢草莓，对不对？♡"
    ]

    for i, response in enumerate(test_responses, 1):
        length = len(response)
        status = "✓" if length <= 40 else "✗"
        print(f"[回应{i}] 长度: {length}字 {status}")
        print(f"    内容: {response}")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
    print("\n修改内容总结:")
    print("1. 将吐槽口吻改为病娇妹妹风格")
    print("2. 添加病娇妹妹特有的语气和特点")
    print("3. 要求使用可爱的称呼，如'哥哥'、'欧尼酱'等")
    print("4. 保持甜美但带有一丝偏执的语气")
    print("5. 强调对哥哥的关心和占有欲")
    print("6. 允许使用可爱的表情符号")


def main():
    """主测试函数"""
    test_yandere_sister_style()


if __name__ == "__main__":
    main()
