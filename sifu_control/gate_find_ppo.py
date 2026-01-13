import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import cv2
from PIL import Image
import base64
from collections import deque
import logging
from pathlib import Path
from ultralytics import YOLO
import pyautogui  # 添加pyautogui库用于按键操作


# 添加当前目录到Python路径，确保能正确导入同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入现有的移动控制器
from control_api_tool import ImprovedMovementController

# 添加项目根目录到路径，以便导入其他模块
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# 配置日志
def setup_logging():
    """设置日志配置"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = os.path.join(log_dir, f"gate_find_ppo_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    return logging.getLogger(__name__)


class TargetSearchEnvironment:
    """
    目标搜索环境 - 适配更大网络的版本
    """
    def __init__(self, target_description="gate"):
        self.controller = ImprovedMovementController()
        self.target_description = target_description
        self.step_count = 0
        self.max_steps = 60  # 增加最大步数，给更大网络更多探索时间
        self.last_detection_result = None
        self.last_center_distance = float('inf')
        self.last_area = 0  # 新增：跟踪上一帧目标区域
        self.logger = logging.getLogger(__name__)
        
        # 记录探索历史，帮助判断是否在原地打转
        self.position_history = []
        self.max_history_length = 15  # 适度增加历史长度
        self.yolo_model = self._load_yolo_model()
        # 预先初始化检测模型，确保在执行任何动作前模型已加载
        self._warm_up_detection_model()
        
        # 调整成功条件的阈值 - 保持与中等规模网络兼容
        self.MIN_GATE_AREA = 250000  # 适度调整阈值
        self.CENTER_THRESHOLD = 180  # 适度收紧居中要求
        

    def reset_to_origin(self):
        """
        重置到原点操作：按下ESC键，按Q键，按下回车键，检测门，若无门则等待1秒按回车再次检测，最后再按一次回车
        """
        self.logger.info("执行重置到原点操作")
        print("执行重置到原点操作...")
        
        # 按键操作序列
        pyautogui.press('esc')
        time.sleep(0.2)  # 减少等待时间
        pyautogui.press('q')
        time.sleep(0.2)  # 减少等待时间
        pyautogui.press('enter')
        time.sleep(0.2)  # 减少等待时间
        pyautogui.press('enter')
        time.sleep(0.2)  # 减少等待时间
        # 检测门是否存在
        gate_detected = False

        while not gate_detected :  # 限制尝试次数
            # 重新截图并检测门
            new_state = self.capture_screen()
            detection_results = self.detect_target(new_state)
            
            # 检查是否检测到了门
            for detection in detection_results:
                if detection['label'].lower() == 'gate' or 'gate' in detection['label'].lower():
                    gate_detected = True
                    time.sleep(0.2)
                    pyautogui.press('enter')
                    time.sleep(0.3)
                    pyautogui.press('enter')
                    break
            
            if not gate_detected:
                print(f"未检测到门，等待1秒后按回车重新检测...")
                time.sleep(1)
                pyautogui.press('enter')
                time.sleep(0.2)  # 减少等待时间
               
                        
        # 最后再按一次回车
        pyautogui.press('enter')
        time.sleep(0.2)  # 减少等待时间
        pyautogui.press('enter')
        time.sleep(1.0)  # 减少等待时间
        print("重置到原点操作完成")
        self.logger.info("重置到原点操作完成")

    def _load_yolo_model(self):
        """
        加载YOLO模型
        """
        # 获取项目根目录
        current_dir = Path(__file__).parent
        model_path = current_dir.parent / "models" / "find_gate.pt"
        
        if not model_path.exists():
            self.logger.error(f"YOLO模型文件不存在: {model_path}")
            return None
        
        try:
            model = YOLO(str(model_path))
            self.logger.info(f"成功加载YOLO模型: {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"加载YOLO模型失败: {e}")
            return None

    def _warm_up_detection_model(self):
        """
        预热检测模型，确保在训练开始前模型已加载到内存
        """
        self.logger.info("正在预热检测模型，确保模型已加载...")
        try:
            # 在环境初始化时立即加载模型
            dummy_image = self.capture_screen()
            if dummy_image is not None and dummy_image.size > 0:
                # 对当前屏幕截图进行检测，触发模型加载
                dummy_result = self.detect_target(dummy_image)
                self.logger.info("检测模型已预热完成")
            else:
                self.logger.warning("无法获取初始截图进行模型预热")
        except Exception as e:
            self.logger.warning(f"模型预热过程中出现错误: {e}")
    
    def capture_screen(self):
        """
        截取当前屏幕画面
        """
        # 使用现有的浏览器截图功能，修改为截取名称包含"sifu"的窗口
        try:
            from computer_server.prtsc import capture_window_by_title
            # 截取标题包含"sifu"的窗口
            result = capture_window_by_title("sifu", "sifu_window_capture.png")
            if result:
                # 读取保存的图片
                from PIL import Image
                screenshot = Image.open("sifu_window_capture.png")
                # 转换为numpy数组
                import numpy as np
                screenshot = np.array(screenshot)
                # 由于PIL读取的图像格式是RGB，而OpenCV使用BGR，所以需要转换
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                return screenshot
            else:
                self.logger.warning("未找到包含 'sifu' 的窗口，使用全屏截图")
                import pyautogui
                screenshot = pyautogui.screenshot()
                screenshot = np.array(screenshot)
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                return screenshot
        except ImportError:
            # 如果没有截图功能，模拟返回一张图片
            self.logger.warning("截图功能不可用，使用模拟图片")
            return np.zeros((480, 640, 3), dtype=np.uint8_)
    
    def detect_target(self, image):
        """
        使用YOLO检测目标
        """
        if self.yolo_model is None:
            self.logger.error("YOLO模型未加载，无法进行检测")
            return []
        
        try:
            # 进行预测 - 降低置信度阈值以增加检测敏感性
            results = self.yolo_model.predict(
                source=image,
                conf=0.6,  # 适度降低置信度阈值，提高检测敏感性
                save=False,
                verbose=False
            )
            
            # 获取检测结果
            detections = []
            result = results[0]  # 获取第一个结果
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # 获取边界框坐标 [x1, y1, x2, y2]
                confs = result.boxes.conf.cpu().numpy()  # 获取置信度
                cls_ids = result.boxes.cls.cpu().numpy()  # 获取类别ID
                
                # 获取类别名称（如果模型有类别名称）
                names = result.names if hasattr(result, 'names') else {}
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    conf = confs[i]
                    cls_id = int(cls_ids[i])
                    class_name = names.get(cls_id, f"Class_{cls_id}")
                    
                    # 计算边界框中心坐标
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # 计算边界框宽度和高度
                    width = x2 - x1
                    height = y2 - y1
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],  # 左上角和右下角坐标
                        'label': class_name,
                        'score': conf,
                        'width': width,
                        'height': height
                    })
            
            self.logger.debug(f"YOLO检测到 {len(detections)} 个目标")
            return detections
        except Exception as e:
            self.logger.error(f"YOLO检测过程中出错: {e}")
            return []
    

    def calculate_reward(self, detection_results, prev_distance, action_taken=None, prev_area=None):
        """
        改进的奖励函数：检测到门就给正反馈，基于目标绝对大小给予奖励
        """
        reward = 0.0
        
        # 基于探索的奖励 - 即使没有检测到门也给一定奖励
        exploration_bonus = 0.01  # 进一步减少探索奖励
        
        if not detection_results or len(detection_results) == 0:
            # 没有检测到目标，给予轻微惩罚但加上探索奖励
            reward = -0.03 + exploration_bonus
            self.logger.debug(f"未检测到目标，奖励: {reward:.2f}")
            return reward, 0
        
        # 找到最近的检测框和最大面积
        min_distance = float('inf')
        max_area = 0
        
        for detection in detection_results:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            img_width = detection.get('img_width', 640)
            img_height = detection.get('img_height', 480)
            
            img_center_x = img_width / 2
            img_center_y = img_height / 2
            distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
            
            area = detection['width'] * detection['height']
            if area > max_area:
                max_area = area
        
        # 基础奖励：只要检测到门就给予正反馈
        base_detection_reward = 0.2  # 适度调整基础检测奖励
        reward += base_detection_reward
        self.logger.debug(f"检测到门，基础奖励: {base_detection_reward}")
        
        # 基于目标绝对大小的奖励：目标越大，奖励越高
        size_based_reward = max_area / 12000  # 调整大小奖励比例
        
        reward += size_based_reward
        self.logger.debug(f"基于目标大小的奖励: {size_based_reward}, 目标面积: {max_area}")
        
        # 基于接近目标的奖励
        if min_distance < prev_distance:
            approach_reward = 0.15  # 适度调整接近奖励
            reward += approach_reward
            self.logger.debug(f"接近目标奖励: {approach_reward}")
        
        # 基于远离中心的惩罚
        if min_distance > prev_distance:
            distance_penalty = -0.03
            reward += distance_penalty
            self.logger.debug(f"远离目标惩罚: {distance_penalty}")
        
        # 探索奖励
        reward += exploration_bonus
        
        return reward, max_area

    def step(self, move_action, turn_action, move_forward_step=2, turn_angle=30):
        """
        执行动作并返回新的状态、奖励和是否结束
        move_action: 0-forward, 1-backward, 2-strafe_left, 3-strafe_right
        turn_action: 0-turn_left, 1-turn_right
        """
        move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
        turn_action_names = ["turn_left", "turn_right"]
        
        self.logger.debug(f"执行动作: 移动-{move_action_names[move_action]}, 转头-{turn_action_names[turn_action]}, 步长: {move_forward_step}, 角度: {turn_angle}")
        
        # 根据动作类型动态计算等待时间
        move_wait_time = 0
        turn_wait_time = 0
        
        # 执行移动动作
        if move_action == 0 and move_forward_step > 0:  # forward
            self.controller.move_forward(duration=move_forward_step*2)
            move_wait_time = move_forward_step * 0.4  # 0.4秒/单位移动距离
        elif move_action == 1 and move_forward_step > 0:  # backward
            self.controller.move_backward(duration=move_forward_step*2)
            move_wait_time = move_forward_step * 0.4  # 0.4秒/单位移动距离
        elif move_action == 2 and move_forward_step > 0:  # strafe_left
            self.controller.strafe_left(duration=move_forward_step*2)
            move_wait_time = move_forward_step * 0.4  # 0.4秒/单位移动距离
        elif move_action == 3 and move_forward_step > 0:  # strafe_right
            self.controller.strafe_right(duration=move_forward_step*2)
            move_wait_time = move_forward_step * 0.4  # 0.4秒/单位移动距离
        
        # 执行转头动作
        if turn_action == 0 and turn_angle > 0:  # turn_left
            self.controller.turn_left(turn_angle*8, duration=turn_angle*0.70)
            turn_wait_time = turn_angle * 0.015  # 0.015秒/度
        elif turn_action == 1 and turn_angle > 0:  # turn_right
            self.controller.turn_right(turn_angle*8, duration=turn_angle*0.70)
            turn_wait_time = turn_angle * 0.015  # 0.015秒/度
        
        # 取两个等待时间的最大值
        wait_time = max(move_wait_time, turn_wait_time, 0.2)  # 至少0.2秒
        
        # 确保等待时间不会太长
        wait_time = min(wait_time, 1.5)  # 限制最大等待时间为1.5秒
       

        # 获取新状态（截图）
        new_state = self.capture_screen()
        
        # 检测目标
        detection_results = self.detect_target(new_state)
        
        # 计算奖励 - 添加当前目标区域
        current_distance = self.last_center_distance
        current_area = self.last_area  # 获取上一帧的目标区域
        area = 0
        if detection_results:
            # 计算当前最近目标到中心的距离
            min_distance = float('inf')
            max_area = 0
            for detection in detection_results:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2  # 修复：应该是bbox[3]而不是bbox[2]
                img_center_x = new_state.shape[1] / 2
                img_center_y = new_state.shape[0] / 2
                # 添加检测信息到detection字典
                detection['img_width'] = new_state.shape[1]
                detection['img_height'] = new_state.shape[0]
                
                distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                
                # 计算目标区域
                area = detection['width'] * detection['height']
                if area > max_area:
                    max_area = area
            current_distance = min_distance
            current_area = max_area

        reward, new_area = self.calculate_reward(detection_results, self.last_center_distance, (move_action, turn_action), current_area)
        self.last_center_distance = current_distance
        self.last_area = new_area  # 保存当前目标区域
        self.last_detection_result = detection_results
        
        # 更新步数
        self.step_count += 1
        
        # 检查是否成功找到门（新条件）
        gate_found_and_close = (
            detection_results 
            and current_area > self.MIN_GATE_AREA
        )
        
        # 检查是否结束
        done = self.step_count >= self.max_steps or gate_found_and_close
        
        # 如果成功找到门，给予额外奖励
        if gate_found_and_close:
            # 计算快速完成的额外奖励：基于剩余步数给予额外奖励
            remaining_steps = self.max_steps - self.step_count
            speed_bonus = remaining_steps * 5  # 适度调整速度奖励
            reward += 80 + speed_bonus  # 适度调整成功奖励
            self.logger.info(f"成功找到目标！基础奖励: 80.0, 速度奖励: {speed_bonus}, 当前面积: {current_area}")
        
        # 输出每步得分
        detected_targets = len(detection_results) if detection_results else 0
        print(f"Step {self.step_count}, Area: {current_area:.2f}, Reward: {reward:.2f}, "
            f"Targets Detected: {detected_targets}, Distance to Center: {current_distance:.2f}, "
            f"Done: {done}, Move Action: {move_action_names[move_action]}, Turn Action: {turn_action_names[turn_action]}, "
            f"Move Step: {move_forward_step}, Turn Angle: {turn_angle}, Wait Time: {wait_time:.2f}")
        
        # 更新位置历史，记录当前状态的特征（如检测结果数量）
        state_feature = len(detection_results)  # 这里用检测到的对象数量作为状态特征
        self.position_history.append(state_feature)
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        
        if done:
            if gate_found_and_close:
                self.logger.info(f"在第 {self.step_count} 步成功找到目标，面积: {current_area}")
            else:
                self.logger.info(f"达到最大步数 {self.max_steps}")
        
        return new_state, reward, done, detection_results
    
    def reset(self):
        """
        重置环境
        """
        self.logger.debug("重置环境")
        self.step_count = 0
        self.last_center_distance = float('inf')
        self.last_area = 0  # 重置目标区域
        self.last_detection_result = None
        self.position_history = []  # 重置位置历史
        initial_state = self.capture_screen()
        return initial_state



def execute_ppo_tool(tool_name, *args):
    """
    根据工具名称执行对应的PPO操作
    
    Args:
        tool_name (str): 工具名称
        *args: 工具参数
        
    Returns:
        dict: API响应结果
    """
    logger = logging.getLogger(__name__)
    
    # 动态导入函数以避免循环导入
    if tool_name == "find_gate_with_ppo":
        from ppo_networks import find_gate_with_ppo_medium
        func = find_gate_with_ppo_medium
    elif tool_name == "train_gate_search_ppo_agent":
        from ppo_networks import train_gate_search_ppo_agent_medium
        func = train_gate_search_ppo_agent_medium
    elif tool_name == "evaluate_trained_ppo_agent":
        from ppo_networks import evaluate_trained_ppo_agent_medium
        func = evaluate_trained_ppo_agent_medium
    elif tool_name == "load_and_test_ppo_agent":
        from ppo_networks import load_and_test_ppo_agent_medium
        func = load_and_test_ppo_agent_medium
    else:
        logger.error(f"错误: 未知的PPO工具 '{tool_name}'")
        print(f"错误: 未知的PPO工具 '{tool_name}'")
        return {"status": "error", "message": f"未知的PPO工具 '{tool_name}'"}

    try:
        # 特殊处理带参数的函数
        if tool_name == "train_gate_search_ppo_agent":
            episodes = int(args[0]) if args else 30  # 默认训练episode
            model_path = args[1] if len(args) > 1 else "gate_search_ppo_model_medium.pth"
            target_desc = args[2] if len(args) > 2 else "gate"
            result = func(episodes, model_path, target_desc)
        elif tool_name == "evaluate_trained_ppo_agent":
            model_path = args[0] if args else "gate_search_ppo_model_medium.pth"
            episodes = int(args[1]) if len(args) > 1 else 5
            target_desc = args[2] if len(args) > 2 else "gate"
            result = func(model_path, episodes, target_desc)
        elif tool_name == "load_and_test_ppo_agent":
            model_path = args[0] if args else "gate_search_ppo_model_medium.pth"
            target_desc = args[1] if len(args) > 1 else "gate"
            result = func(model_path, target_desc)
        elif tool_name == "find_gate_with_ppo":
            target_desc = args[0] if args else "gate"
            result = func(target_desc)
        else:
            result = func()
        logger.info(f"PPO工具执行成功: {tool_name}")
        return {"status": "success", "result": str(result)}
    except Exception as e:
        logger.error(f"执行PPO工具时出错: {str(e)}")
        import traceback
        traceback.print_exc()  # 添加详细的错误追踪
        return {"status": "error", "message": f"执行PPO工具时出错: {str(e)}"}


def main():
    """
    主函数，用于直接运行此脚本
    """
    # 设置日志
    logger = setup_logging()
    
    if len(sys.argv) < 2:
        print("导入成功！")
        print("\n可用的功能:")
        print("1. 训练门搜索智能体: train_gate_search_ppo_agent")
        print("2. 寻找门（包含训练）: find_gate_with_ppo")
        print("3. 评估已训练模型: evaluate_trained_ppo_agent")
        print("4. 加载并测试模型: load_and_test_ppo_agent")
        
        # 运行快速训练
        print("\n=== 开始中等规模训练门搜索智能体 ===")
        try:
            from ppo_networks import train_gate_search_ppo_agent_medium
            agent = train_gate_search_ppo_agent_medium(
                episodes=30, 
                model_path="./model/medium_gate_search_ppo_model.pth",
                target_description="gate"
            )
            print("\n中等规模训练完成！")
        except Exception as e:
            logger.error(f"中等规模训练出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return

    tool_name = sys.argv[1]
    args = sys.argv[2:]
    
    # 执行对应的工具
    response = execute_ppo_tool(tool_name, *args)
    print(response)


if __name__ == "__main__":
    main()