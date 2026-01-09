"""
门搜索DQN实现
使用DQN算法在虚拟环境中搜索目标
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from PIL import Image
import base64
from collections import deque
import logging

# 添加当前目录到Python路径，确保能正确导入同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入现有的移动控制器
from sifu_control.control_api_tool import ImprovedMovementController

# 添加项目根目录到路径，以便导入其他模块
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# 导入现有的目标检测功能
from grounding_dino.dino import detect_objects_with_text_transformers


# 配置日志
def setup_logging():
    """设置日志配置"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = os.path.join(log_dir, f"gate_find_dqn_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
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
    目标搜索环境
    """
    def __init__(self, target_description="gate"):
        self.controller = ImprovedMovementController()
        self.target_description = target_description
        self.step_count = 0
        self.max_steps = 50
        self.last_detection_result = None
        self.last_center_distance = float('inf')
        self.logger = logging.getLogger(__name__)
        
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
            return np.zeros((480, 640, 3), dtype=np.uint8)  
    
    def detect_target(self, image):
        """
        使用Grounding DINO检测目标
        """
        # 将numpy数组转换为base64
        _, buffer = cv2.imencode('.jpg', image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # 调用现有的检测函数
        try:
            detection_results = detect_objects_with_text_transformers(base64_image, self.target_description)
            self.logger.debug(f"检测到 {len(detection_results)} 个目标")
            return detection_results
        except Exception as e:
            self.logger.error(f"检测过程中出错: {e}")
            return []
    
    def calculate_reward(self, detection_results, prev_distance):
        """
        计算奖励值
        """
        if not detection_results:
            # 没有检测到目标，给予小的负奖励
            self.logger.debug("未检测到目标，给予-1.0奖励")
            return -1.0
        
        # 找到最近的检测框
        min_distance = float('inf')
        for detection in detection_results:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            # 计算到图像中心的距离
            img_center_x = 320  # 假设图像宽度为640
            img_center_y = 240  # 假设图像高度为480
            distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
        
        # 如果距离比之前更近，给予正奖励
        if prev_distance != float('inf'):  # 只有在之前有有效距离的情况下才比较
            if min_distance < prev_distance:
                # 避免除零错误，当prev_distance接近0时给予固定奖励
                if prev_distance > 0:
                    reward = 10.0 * (prev_distance - min_distance) / max(prev_distance, 1)
                else:
                    reward = 10.0  # 当前一次距离为0时的奖励
                self.logger.debug(f"距离变近，奖励: {reward:.2f}")
            else:
                reward = -5.0  # 距离变远，给予负奖励
                self.logger.debug(f"距离变远，奖励: {reward:.2f}")
        else:
            # 第一次检测到目标时的奖励
            reward = 5.0 - (min_distance / 100.0)  # 距离越近奖励越高
            self.logger.debug(f"首次检测到目标，奖励: {reward:.2f}")
        
        # 如果目标在中心附近，给予额外奖励
        if min_distance < 50:
            reward += 20.0  # 找到目标，给予大奖励
            self.logger.info(f"目标接近中心，额外奖励，总奖励: {reward:.2f}")
        
        return reward
    
    def step(self, action):
        """
        执行动作并返回新的状态、奖励和是否结束
        动作空间: 0-forward, 1-backward, 2-turn_left, 3-turn_right, 4-strafe_left, 5-strafe_right
        """
        action_names = ["forward", "backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]
        self.logger.debug(f"执行动作: {action_names[action]}")
        
        # 执行动作
        if action == 0:  # forward
            self.controller.move_forward(1)
        elif action == 1:  # backward
            self.controller.move_backward(1)
        elif action == 2:  # turn_left
            self.controller.turn_left(30)
        elif action == 3:  # turn_right
            self.controller.turn_right(30)
        elif action == 4:  # strafe_left
            self.controller.strafe_left(1)
        elif action == 5:  # strafe_right
            self.controller.strafe_right(1)
        
        time.sleep(0.5)  # 等待动作完成
        
        # 获取新状态（截图）
        new_state = self.capture_screen()
        
        # 检测目标
        detection_results = self.detect_target(new_state)
        
        # 计算奖励
        current_distance = self.last_center_distance
        if detection_results:
            # 计算当前最近目标到中心的距离
            min_distance = float('inf')
            for detection in detection_results:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                img_center_x = new_state.shape[1] / 2
                img_center_y = new_state.shape[0] / 2
                distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
            current_distance = min_distance
        
        reward = self.calculate_reward(detection_results, self.last_center_distance)
        self.last_center_distance = current_distance
        self.last_detection_result = detection_results
        
        # 更新步数
        self.step_count += 1
        
        # 检查是否结束
        done = self.step_count >= self.max_steps or (detection_results and current_distance < 50)
        
        if done:
            if detection_results and current_distance < 50:
                self.logger.info(f"在第 {self.step_count} 步找到目标")
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
        self.last_detection_result = None
        initial_state = self.capture_screen()
        return initial_state

class DQNNetwork(nn.Module):
    """
    深度Q网络
    """
    def __init__(self, input_channels=3, action_space=6):
        super(DQNNetwork, self).__init__()
        
        # 卷积层处理图像输入
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 计算卷积输出大小
        conv_out_size = self._get_conv_output((input_channels, 480, 640))
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )
    
    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv_layers(x)
            return int(np.prod(x.size()))
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x

class TargetSearchDQNAgent:
    """
    目标搜索DQN智能体
    """
    def __init__(self, action_space=6, learning_rate=1e-4, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        # 首先初始化logger
        self.logger = logging.getLogger(__name__)
        
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 添加历史动作记录，防止连续重复相同动作
        self.action_history = deque(maxlen=5)  # 记录最近5个动作
        
        # 神经网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(action_space=action_space).to(self.device)
        self.target_network = DQNNetwork(action_space=action_space).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        
        # 更新目标网络
        self.update_target_network()
    
    def update_target_network(self):
        """
        更新目标网络权重
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.logger.debug("目标网络已更新")
    
    def remember(self, state, action, reward, next_state, done):
        """
        存储经验
        """
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) % 1000 == 0:
            self.logger.debug(f"经验回放池大小: {len(self.memory)}")
    
    def act(self, state):
        """
        根据当前策略选择动作
        """
        # 检查是否需要避免重复动作
        if len(self.action_history) >= 3:
            # 检查最近3个动作是否相同
            recent_actions = list(self.action_history)[-3:]
            if len(set(recent_actions)) == 1:  # 如果最近3个动作都相同
                # 返回一个不同的动作
                forbidden_action = recent_actions[0]
                valid_actions = [a for a in range(self.action_space) if a != forbidden_action]
                action = random.choice(valid_actions)
                self.logger.debug(f"避免重复动作，选择了: {action}")
                self.action_history.append(action)
                return action
        
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_space)
            self.logger.debug(f"随机选择动作: {action}")
            self.action_history.append(action)
            return action
        
        # 预处理状态
        state_tensor = self._preprocess_state(state)
        state_tensor = state_tensor.unsqueeze(0).to(self.device)  # 添加批次维度
        
        q_values = self.q_network(state_tensor)
        action = np.argmax(q_values.cpu().data.numpy())
        self.logger.debug(f"选择动作: {action}")
        self.action_history.append(action)
        return action
    
    def _preprocess_state(self, state):
        """
        预处理状态（图像）
        """
        # 将BGR转为RGB
        if len(state.shape) == 3:
            state_rgb = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
        else:
            state_rgb = state
        
        # 转换为tensor并归一化
        state_tensor = torch.FloatTensor(state_rgb).permute(2, 0, 1) / 255.0
        return state_tensor
    
    def replay(self, batch_size=32):
        """
        经验回放训练
        """
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.stack([self._preprocess_state(e[0]) for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.stack([self._preprocess_state(e[3]) for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.logger.debug(f"经验回放训练完成，损失: {loss.item():.4f}")

def train_gate_search_agent(episodes=200, model_path="gate_search_dqn_model.pth", target_description="gate"):
    """
    训练门搜索智能体
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练门搜索智能体，目标: {target_description}")
    
    # 确保模型保存目录存在
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"创建模型保存目录: {model_dir}")
    
    env = TargetSearchEnvironment(target_description)
    agent = TargetSearchDQNAgent()
    
    scores = deque(maxlen=100)
    total_rewards = deque(maxlen=100)
    
    for episode in range(episodes):  # 训练200轮
        state = env.reset()
        # 重置智能体的动作历史
        agent.action_history.clear()
        total_reward = 0
        step_count = 0
        
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, detection_results = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done:
                logger.info(f"Episode: {episode}, Score: {step_count}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
                scores.append(step_count)
                total_rewards.append(total_reward)
                break
        
        if len(agent.memory) > 32:
            agent.replay(32)
        
        # 每100轮更新一次目标网络
        if episode % 100 == 0:
            agent.update_target_network()
        
        # 每50轮打印一次平均分数
        if episode % 50 == 0 and episode > 0:
            avg_score = np.mean(scores)
            avg_total_reward = np.mean(total_rewards)
            logger.info(f"Episode {episode}, Average Score: {avg_score:.2f}, Average Total Reward: {avg_total_reward:.2f}")
    
    logger.info("训练完成！")
    
    # 保存模型
    torch.save(agent.q_network.state_dict(), model_path)
    logger.info(f"模型已保存为 {model_path}")
    
    return agent


def evaluate_trained_agent(episodes=10, model_path="gate_search_dqn_model.pth", target_description="gate"):
    """
    评估已训练好的模型性能
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始评估模型: {model_path}")
    
    # 创建智能体
    agent = TargetSearchDQNAgent()
    
    # 加载已保存的模型
    if os.path.exists(model_path):
        agent.q_network.load_state_dict(torch.load(model_path, map_location=agent.device))
        agent.q_network.eval()  # 设置为评估模式
        logger.info(f"模型已从 {model_path} 加载")
    else:
        logger.warning(f"模型文件 {model_path} 不存在，使用随机模型")
    
    # 禁用epsilon-greedy策略的随机性
    agent.epsilon = 0.0
    
    evaluation_results = []
    
    for episode in range(episodes):
        env = TargetSearchEnvironment(target_description)
        state = env.reset()
        agent.action_history.clear()
        
        total_reward = 0
        step_count = 0
        success = False
        
        done = False
        while not done and step_count < 50:
            action = agent.act(state)
            
            # 打印动作
            action_names = ["forward", "backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]
            logger.info(f"Episode {episode+1}, Step {step_count}: Taking action - {action_names[action]}")
            
            state, reward, done, detection_results = env.step(action)
            total_reward += reward
            step_count += 1
            
            if detection_results:
                logger.info(f"Episode {episode+1}: Detected {len(detection_results)} instances of {target_description}")
                
            if detection_results and env.last_center_distance < 50:
                success = True
                logger.info(f"Episode {episode+1}: 成功找到目标！")
        
        result = {
            'episode': episode + 1,
            'steps': step_count,
            'total_reward': total_reward,
            'success': success
        }
        evaluation_results.append(result)
        
        logger.info(f"Episode {episode+1} 完成 - Steps: {step_count}, Total Reward: {total_reward}, Success: {success}")
    
    # 计算总体统计信息
    successful_episodes = sum(1 for r in evaluation_results if r['success'])
    avg_steps = np.mean([r['steps'] for r in evaluation_results])
    avg_reward = np.mean([r['total_reward'] for r in evaluation_results])
    
    stats = {
        'total_episodes': episodes,
        'successful_episodes': successful_episodes,
        'success_rate': successful_episodes / episodes if episodes > 0 else 0,
        'avg_steps': avg_steps,
        'avg_reward': avg_reward,
        'details': evaluation_results
    }
    
    logger.info(f"评估完成 - 总体成功率: {stats['success_rate']*100:.2f}% ({successful_episodes}/{episodes})")
    logger.info(f"平均步数: {avg_steps:.2f}, 平均奖励: {avg_reward:.2f}")
    
    return stats

def load_and_test_agent(model_path="gate_search_dqn_model.pth", target_description="gate"):
    """
    加载已训练的模型并进行测试
    """
    logger = logging.getLogger(__name__)
    logger.info(f"加载模型并测试: {model_path}")
    
    # 创建智能体
    agent = TargetSearchDQNAgent()
    
    # 加载已保存的模型
    if os.path.exists(model_path):
        agent.q_network.load_state_dict(torch.load(model_path, map_location=agent.device))
        agent.q_network.eval()  # 设置为评估模式
        logger.info(f"模型已从 {model_path} 加载")
    else:
        logger.error(f"模型文件 {model_path} 不存在")
        return {"status": "error", "message": f"模型文件 {model_path} 不存在"}
    
    # 禁用epsilon-greedy策略的随机性
    agent.epsilon = 0.0
    
    # 测试环境
    env = TargetSearchEnvironment(target_description)
    state = env.reset()
    agent.action_history.clear()
    
    total_reward = 0
    step_count = 0
    done = False
    
    logger.info("开始测试智能体...")
    
    while not done and step_count < 50:
        action = agent.act(state)
        
        # 打印动作
        action_names = ["forward", "backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]
        logger.info(f"Step {step_count}: Taking action - {action_names[action]}")
        
        state, reward, done, detection_results = env.step(action)
        total_reward += reward
        step_count += 1
        
        if detection_results:
            logger.info(f"Detected {len(detection_results)} instances of {target_description}")
        
        if done:
            if detection_results and env.last_center_distance < 50:
                logger.info("成功找到目标！")
                return {"status": "success", "result": f"成功找到{target_description}！", "steps": step_count, "total_reward": total_reward}
            else:
                logger.info("未能找到目标")
                return {"status": "partial_success", "result": f"未能找到{target_description}", "steps": step_count, "total_reward": total_reward}
    
    result = {"status": "timeout", "result": f"测试完成但未找到目标 - Steps: {step_count}, Total Reward: {total_reward}"}
    logger.info(result["result"])
    return result

def find_gate_with_dqn(target_description="gate"):
    """
    使用DQN算法寻找门（训练+测试流程）
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练并寻找目标: {target_description}")
    
    # 确保模型保存目录存在
    model_path = "./model/gate_search_dqn_model.pth"
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"创建模型保存目录: {model_dir}")
    
    # 训练智能体
    trained_agent = train_gate_search_agent(target_description=target_description, model_path=model_path)
    
    # 测试智能体
    env = TargetSearchEnvironment(target_description)
    state = env.reset()
    # 重置智能体的动作历史
    trained_agent.action_history.clear()
    
    total_reward = 0
    step_count = 0
    done = False
    
    logger.info("开始测试智能体...")
    
    while not done and step_count < 50:
        action = trained_agent.act(state)
        
        # 打印动作
        action_names = ["forward", "backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]
        logger.info(f"Step {step_count}: Taking action - {action_names[action]}")
        
        state, reward, done, detection_results = env.step(action)
        total_reward += reward
        step_count += 1
        
        if detection_results:
            logger.info(f"Detected {len(detection_results)} instances of {target_description}")
        
        if done:
            if detection_results and env.last_center_distance < 50:
                logger.info("成功找到目标！")
                return "成功找到目标！"
            else:
                logger.info("未能找到目标")
                return "未能找到目标"
    
    result = f"测试完成 - Steps: {step_count}, Total Reward: {total_reward}"
    logger.info(result)
    return result

def execute_tool(tool_name, *args):
    """
    根据工具名称执行对应的DQN操作
    
    Args:
        tool_name (str): 工具名称
        *args: 工具参数
        
    Returns:
        dict: API响应结果
    """
    logger = logging.getLogger(__name__)
    
    tool_functions = {
        "find_gate_with_dqn": find_gate_with_dqn,
        "train_gate_search_agent": train_gate_search_agent,
        "evaluate_trained_agent": evaluate_trained_agent,
        "load_and_test_agent": load_and_test_agent
    }
    
    if tool_name in tool_functions:
        try:
            # 特殊处理带参数的函数
            if tool_name == "train_gate_search_agent":
                episodes = int(args[0]) if args else 200
                model_path = args[1] if len(args) > 1 else "gate_search_dqn_model.pth"
                target_desc = args[2] if len(args) > 2 else "gate"
                result = tool_functions[tool_name](episodes, model_path, target_desc)
            elif tool_name == "evaluate_trained_agent":
                model_path = args[0] if args else "gate_search_dqn_model.pth"
                episodes = int(args[1]) if len(args) > 1 else 10
                target_desc = args[2] if len(args) > 2 else "gate"
                result = tool_functions[tool_name](model_path, episodes, target_desc)
            elif tool_name == "load_and_test_agent":
                model_path = args[0] if args else "gate_search_dqn_model.pth"
                target_desc = args[1] if len(args) > 1 else "gate"
                result = tool_functions[tool_name](model_path, target_desc)
            elif tool_name == "find_gate_with_dqn":
                target_desc = args[0] if args else "gate"
                result = tool_functions[tool_name](target_desc)
            else:
                result = tool_functions[tool_name]()
            logger.info(f"工具执行成功: {tool_name}")
            return {"status": "success", "result": str(result)}
        except Exception as e:
            logger.error(f"执行工具时出错: {str(e)}")
            return {"status": "error", "message": f"执行工具时出错: {str(e)}"}
    else:
        logger.error(f"错误: 未知的DQN工具 '{tool_name}'")
        print(f"错误: 未知的DQN工具 '{tool_name}'")
        return {"status": "error", "message": f"未知的DQN工具 '{tool_name}'"}

def main():
    """
    主函数，用于直接运行此脚本
    """
    # 设置日志
    logger = setup_logging()
    
    if len(sys.argv) < 2:
        logger.info("用法: python gate_find_dqn.py <tool_name> [args...]")
        logger.info("可用工具: find_gate_with_dqn, train_gate_search_agent, evaluate_trained_agent, load_and_test_agent")
        print("用法: python gate_find_dqn.py <tool_name> [args...]")
        print("可用工具:")
        print("  find_gate_with_dqn [target_desc] - 训练并寻找目标")
        print("  train_gate_search_agent [episodes] [model_path] [target_desc] - 训练智能体")
        print("  evaluate_trained_agent [model_path] [episodes] [target_desc] - 评估已训练模型")
        print("  load_and_test_agent [model_path] [target_desc] - 加载并测试模型")
        return

    tool_name = sys.argv[1]
    args = sys.argv[2:]
    
    # 执行对应的工具
    response = execute_tool(tool_name, *args)
    print(response)

if __name__ == "__main__":
    main()