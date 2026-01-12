import time
import math
from pynput import mouse, keyboard

class MovementController:
    """
    移动控制器类，用于处理前进、后退、左转和右转动作
    使用pynput库实现游戏友好的输入控制
    """
    
    def __init__(self):
        self.keyboard_controller = keyboard.Controller()
        self.mouse_controller = mouse.Controller()
        
    def move_forward(self, distance=1, duration=0.1):
        """
        向前移动指定距离 - 按W键
        :param distance: 移动距离，默认为1
        :param duration: 按键持续时间，默认为0.1秒
        """
        for _ in range(int(distance)):
            self._press_key_with_duration('w', duration)
            time.sleep(0.1)
        return f"向前移动 {distance} 单位"
        
    def move_backward(self, distance=1, duration=0.1):
        """
        向后移动指定距离 - 按S键
        :param distance: 移动距离，默认为1
        :param duration: 按键持续时间，默认为0.1秒
        """
        for _ in range(int(distance)):
            self._press_key_with_duration('s', duration)
            time.sleep(0.1)
        return f"向后移动 {distance} 单位"
        
    def turn_left(self, angle=90, duration=0.5):
        """
        左转指定角度 - 鼠标向左移动
        :param angle: 转动角度，默认为90度
        :param duration: 转动持续时间，默认为0.5秒
        """
        # 通过相对鼠标移动来实现视角转动
        # 调整系数，根据实际游戏灵敏度调整
        adjusted_movement = int(angle * 2)  # 调整系数使转动更精确
        self._move_mouse_relative_with_duration(-adjusted_movement, 0, duration)
        return f"左转 {angle} 度"
        
    def turn_right(self, angle=90, duration=0.5):
        """
        右转指定角度 - 鼠标向右移动
        :param angle: 转动角度，默认为90度
        :param duration: 转动持续时间，默认为0.5秒
        """
        # 通过相对鼠标移动来实现视角转动
        # 调整系数，根据实际游戏灵敏度调整
        adjusted_movement = int(angle * 2)  # 调整系数使转动更精确
        self._move_mouse_relative_with_duration(adjusted_movement, 0, duration)
        return f"右转 {angle} 度"
        
    def strafe_left(self, distance=1, duration=0.1):
        """
        左平移 - 按A键
        :param distance: 移动距离，默认为1
        :param duration: 按键持续时间，默认为0.1秒
        """
        for _ in range(int(distance)):
            self._press_key_with_duration('a', duration)
            time.sleep(0.1)
        return f"左平移 {distance} 单位"
        
    def strafe_right(self, distance=1, duration=0.1):
        """
        右平移 - 按D键
        :param distance: 移动距离，默认为1
        :param duration: 按键持续时间，默认为0.1秒
        """
        for _ in range(int(distance)):
            self._press_key_with_duration('d', duration)
            time.sleep(0.1)
        return f"右平移 {distance} 单位"
        
    def move_sequence(self, commands):
        """
        执行移动序列
        :param commands: 移动命令列表，支持 w(前), s(后), a(左平移), d(右平移) 和数字的组合
        例如: [("w", 2), ("d", 90), ("s", 1), ("a", 90)]
        """
        for command in commands:
            action, value = command[0], command[1]
            if action.lower() == "w":
                self.move_forward(value)
            elif action.lower() == "s":
                self.move_backward(value)
            elif action.lower() == "a":
                self.strafe_left(value)
            elif action.lower() == "d":
                self.strafe_right(value)
            elif action.lower() == "turn_left":
                self.turn_left(value)
            elif action.lower() == "turn_right":
                self.turn_right(value)
            time.sleep(0.5)  # 短暂延迟，模拟移动时间
        return f"移动序列执行完成"
            
    def get_position(self):
        """
        获取当前位置
        :return: 当前位置 [x, y]
        """
        current_pos = self.mouse_controller.position
        return [current_pos[0], current_pos[1]]
        
    def get_direction(self):
        """
        获取当前方向
        :return: 当前方向 (角度)
        """
        return "方向追踪已禁用"

    def _press_key_with_duration(self, key, duration=0.1):
        """
        内部方法：模拟按键，支持持续时间
        :param key: 要按下的键
        :param duration: 按键持续时间
        """
        try:
            self.keyboard_controller.press(key)
            time.sleep(duration)
            self.keyboard_controller.release(key)
            return f"按键 {key} 成功，持续时间 {duration} 秒"
        except Exception as e:
            return f"执行按键操作时出错: {str(e)}"

    def _move_mouse_relative_with_duration(self, x_offset, y_offset, duration=0.1):
        """
        内部方法：相对移动鼠标，支持持续时间
        :param x_offset: X轴偏移量
        :param y_offset: Y轴偏移量
        :param duration: 移动持续时间
        """
        try:
            # 获取当前鼠标位置
            current_x, current_y = self.mouse_controller.position
            
            # 计算目标位置
            target_x = current_x + x_offset
            target_y = current_y + y_offset
            
            # 实现平滑移动
            steps = max(abs(int(x_offset)), abs(int(y_offset)))
            if steps > 0 and duration > 0:
                dx = x_offset / steps
                dy = y_offset / steps
                delay = duration / steps
                for i in range(steps):
                    self.mouse_controller.move(int(dx), int(dy))
                    time.sleep(delay)
            else:
                # 如果没有步数或持续时间为0，直接移动
                self.mouse_controller.move(int(x_offset), int(y_offset))
                
            return f"鼠标移动成功: ({current_x}, {current_y}) -> ({target_x}, {target_y})"
        except Exception as e:
            return f"执行鼠标移动操作时出错: {str(e)}"

    def _press_key(self, key):
        """
        内部方法：模拟按键（单次按键）
        :param key: 要按下的键
        """
        try:
            self.keyboard_controller.press(key)
            self.keyboard_controller.release(key)
            return f"按键 {key} 成功"
        except Exception as e:
            return f"执行按键操作时出错: {str(e)}"

    def _move_mouse_relative(self, x_offset, y_offset):
        """
        内部方法：相对移动鼠标（无持续时间）
        :param x_offset: X轴偏移量
        :param y_offset: Y轴偏移量
        """
        try:
            self.mouse_controller.move(int(x_offset), int(y_offset))
            current_x, current_y = self.mouse_controller.position
            return f"鼠标移动成功: 相对移动({x_offset}, {y_offset}), 当前位置({current_x}, {current_y})"
        except Exception as e:
            return f"执行鼠标移动操作时出错: {str(e)}"


# 全局函数接口，方便调用
def send_key_event(key, duration=0.1):
    """
    使用pynput发送按键事件，适用于游戏
    :param key: 按键
    :param duration: 按键持续时间
    :return: 执行结果
    """
    try:
        kb_controller = keyboard.Controller()
        kb_controller.press(key)
        time.sleep(duration)
        kb_controller.release(key)
        return f"使用pynput发送按键: {key} 成功，持续时间 {duration} 秒"
    except Exception as e:
        return f"使用pynput发送按键事件时出错: {str(e)}"


def send_mouse_move_relative(x_offset, y_offset, duration=0.1):
    """
    使用pynput相对移动鼠标，适用于游戏视角控制
    :param x_offset: X轴偏移量
    :param y_offset: Y轴偏移量
    :param duration: 移动持续时间
    :return: 执行结果
    """
    try:
        mouse_controller = mouse.Controller()
        current_x, current_y = mouse_controller.position
        
        # 计算目标位置
        target_x = current_x + x_offset
        target_y = current_y + y_offset
        
        # 实现平滑移动
        steps = max(abs(int(x_offset)), abs(int(y_offset)))
        if steps > 0 and duration > 0:
            dx = x_offset / steps
            dy = y_offset / steps
            delay = duration / steps
            for i in range(steps):
                mouse_controller.move(int(dx), int(dy))
                time.sleep(delay)
        else:
            mouse_controller.move(int(x_offset), int(y_offset))
            
        return f"使用pynput相对移动鼠标: ({x_offset}, {y_offset}) 成功"
    except Exception as e:
        return f"使用pynput移动鼠标时出错: {str(e)}"