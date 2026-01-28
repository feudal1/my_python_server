#!/usr/bin/env python3
"""
鼠标和键盘操作工具 - 适配executor.py的格式
参数: 函数名 参数列表
"""
import sys
import pyautogui
from PIL import Image
import base64
import io
import time

# 禁用pyautogui的安全限制（在生产环境中请谨慎使用）
pyautogui.FAILSAFE = False


def click_mouse(x=None, y=None, click_type='left', clicks=1, interval=0.0):
    """
    鼠标点击操作
    """
    try:
        if x is not None and y is not None:
            pyautogui.click(x=int(float(x)), y=int(float(y)), clicks=int(float(clicks)), interval=float(interval), button=click_type)
            return f"点击成功: ({x}, {y}), 类型: {click_type}, 次数: {clicks}"
        else:
            pyautogui.click(clicks=int(float(clicks)), interval=float(interval), button=click_type)
            return f"点击当前位置成功, 类型: {click_type}, 次数: {clicks}"
    except Exception as e:
        return f"点击失败: {str(e)}"

def get_position():
    """
    获取当前鼠标位置
    """
    try:
        x, y = pyautogui.position()
        return f"当前位置: ({x}, {y})"
    except Exception as e:
        return f"获取位置失败: {str(e)}"

def move_mouse(x, y, duration=0.0):
    """
    移动鼠标到指定位置
    """
    try:
        pyautogui.moveTo(x=int(float(x)), y=int(float(y)), duration=float(duration))
        return f"移动鼠标成功: ({x}, {y})"
    except Exception as e:
        return f"移动鼠标失败: {str(e)}"

def scroll_mouse(units, x, y):
    """
    鼠标滚轮滚动操作，必须指定坐标位置
    :param units: 滚动单位，正数向上滚动，负数向下滚动
    :param x: 滚动操作的x坐标
    :param y: 滚动操作的y坐标
    """
    try:
        units = int(float(units))
        x = int(float(x))
        y = int(float(y))
        
        # 移动到指定坐标再滚动
        pyautogui.moveTo(x=x, y=y)
        pyautogui.scroll(units)
        return f"滚动成功: {units}单位，位置: ({x}, {y})"
    except Exception as e:
        return f"滚动失败: {str(e)}"

def press_key(*keys):
    """
    按下键盘按键
    :param keys: 要按下的键，可以是单个键或组合键
    """
    try:
        if len(keys) == 1:
            # 单个按键
            pyautogui.press(keys[0])
            return f"按下按键成功: {keys[0]}"
        else:
            # 组合键
            pyautogui.hotkey(*keys)
            return f"按下组合键成功: {'+'.join(keys)}"
    except Exception as e:
        return f"按键操作失败: {str(e)}"

def key_down(*keys):
    """
    按住键盘按键（不释放）
    :param keys: 要按住的键
    """
    try:
        pyautogui.keyDown(*keys)
        return f"按住按键成功: {'+'.join(keys)}"
    except Exception as e:
        return f"按住按键失败: {str(e)}"

def key_up(*keys):
    """
    释放键盘按键
    :param keys: 要释放的键
    """
    try:
        pyautogui.keyUp(*keys)
        return f"释放按键成功: {'+'.join(keys)}"
    except Exception as e:
        return f"释放按键失败: {str(e)}"

def type_text(text, interval=0.0):
    """
    输入文本
    :param text: 要输入的文本
    :param interval: 每个字符之间的输入间隔时间
    """
    try:
        pyautogui.typewrite(text, interval=float(interval))
        return f"输入文本成功: {text}"
    except Exception as e:
        return f"输入文本失败: {str(e)}"

def parse_args(args):
    """
    解析参数，支持位置参数和 key=value 格式
    """
    positional_args = []
    named_args = {}
    
    for arg in args:
        if '=' in arg and not arg.startswith('='):
            key, value = arg.split('=', 1)
            named_args[key] = value
        else:
            positional_args.append(arg)
    
    return positional_args, named_args

def safe_float_convert(value, default=0.0):
    """安全转换为浮点数，默认返回默认值"""
    try:
        if value in ['None', 'null', '']:
            return default
        return float(value)
    except ValueError:
        return default

def safe_int_convert(value, default=0):
    """安全转换为整数，默认返回默认值"""
    try:
        if value in ['None', 'null', '']:
            return default
        return int(float(value))
    except ValueError:
        return default


if __name__ == '__main__':
    print("鼠标和键盘操作工具")
    print("可用函数:")
    print("- click_mouse(x=None, y=None, click_type='left', clicks=1, interval=0.0)")
    print("- get_position()")
    print("- move_mouse(x, y, duration=0.0)")
    print("- scroll_mouse(units, x, y)")
    print("- press_key(*keys)")
    print("- key_down(*keys)")
    print("- key_up(*keys)")
    print("- type_text(text, interval=0.0)")