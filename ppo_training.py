def train_with_visualization():
    """
    带可视化的训练函数
    """
    # 创建可视化器
    visualizer = RealTimeVisualizer()
    
    # 为环境和智能体添加可视化功能
    add_visualization_to_environment(TargetSearchEnvironment, visualizer)
    
    # 初始化环境和智能体
    env = TargetSearchEnvironment()
    
    state_dim = (3, CONFIG['IMAGE_HEIGHT'], CONFIG['IMAGE_WIDTH'])
    move_action_dim = 4
    turn_action_dim = 2
    agent = PPOAgent(state_dim, move_action_dim, turn_action_dim)
    
    # 为智能体添加可视化功能
    add_visualization_to_agent(PPOAgent, visualizer)
    
    # 在后台线程运行训练
    def run_training():
        for episode in range(0, 2001):
            print(f"\n=== Episode {episode} Started ===")
            
            env._current_episode = episode
            
            state = env.reset()
            memory = Memory()
            total_reward = 0
            
            for t in range(env.max_steps):
                # 智能体执行动作
                move_action, turn_action, move_step, turn_angle, debug_info = agent.act(
                    state, memory, return_debug_info=True
                )
                
                # 执行环境步骤
                next_state, reward, done, detections = env.step(
                    move_action, turn_action, move_step, turn_angle
                )
                
                # 更新记忆
                memory.rewards[-1] = reward
                memory.is_terminals[-1] = done
                
                total_reward += reward
                state = next_state
                
                # 更新可视化信息
                visualizer.update_agent_info({
                    'episode': episode,
                    'step': t,
                    'total_reward': total_reward,
                    'reward': reward,
                    'done': done,
                    'move_action': move_action,
                    'turn_action': turn_action,
                    'move_step': round(move_step, 2),
                    'turn_angle': round(turn_angle, 2)
                })
                
                if done:
                    break
            
            # 更新策略
            agent.update(memory)
            
            # 记录训练信息
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Steps: {t+1}")
            
            # 保存检查点
            if episode % 10 == 0:
                agent.save_checkpoint(f'ppo_model_checkpoint_ep{episode}.pth')
    
    # 启动训练线程
    import threading
    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()
    
    # 在主线程运行可视化界面
    visualizer.run()


class RealTimeVisualizer:
    """
    实时可视化器，使用Tkinter显示截图、YOLO检测框和智能体状态
    """
    def __init__(self, window_name="PPO Agent Visualizer"):
        self.window_name = window_name
        self.current_image = None
        self.detections = []
        self.agent_info = {}
        self.image_lock = Lock()
        self.info_queue = queue.Queue(maxsize=10)  # 限制队列大小
        
        # 标记是否在主线程中初始化
        self.root = None
        self.image_frame = None
        self.canvas = None
        self.display_image = None
        self.image_tk = None
        self.timer_id = None
        
        # 使用队列在线程间传递数据
        self.gui_queue = queue.Queue()

    def init_gui(self):
        """在主线程中初始化GUI"""
        if self.root is None:
            self.root = tk.Tk()
            self.root.title(self.window_name)
            self.root.geometry("800x600")  # 减小窗口尺寸
            
            # 创建图像显示框架
            self.image_frame = ttk.Frame(self.root)
            self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # 创建Canvas用于显示图像
            self.canvas = tk.Canvas(self.image_frame, bg="black")
            self.canvas.pack(fill=tk.BOTH, expand=True)

    def update_image_and_detections(self, image, detections):
        """
        更新图像和检测结果，并立即触发界面更新
        """
        with self.image_lock:
            self.current_image = image.copy() if image is not None else None
            self.detections = detections.copy() if detections is not None else []

        # 将更新请求放入队列
        try:
            self.gui_queue.put(('update_image', self.current_image, self.detections), block=False)
        except queue.Full:
            pass

    def process_gui_updates(self):
        """
        处理GUI更新，应在主线程中定期调用
        """
        try:
            while True:
                # 非阻塞地从队列获取GUI更新请求
                msg_type, *args = self.gui_queue.get_nowait()
                
                if msg_type == 'update_image':
                    image, detections = args
                    self._update_image_display(image, detections)
                elif msg_type == 'update_info':
                    info = args[0]
                    self._update_info_on_image(info)
        except queue.Empty:
            pass  # 队列为空，正常情况
        
        # 继续安排下一次处理
        if self.root:
            self.root.after(50, self.process_gui_updates)  # 每50ms检查一次更新

    def _update_image_display(self, image, detections):
        """
        更新图像显示
        """
        if image is not None:
            # 绘制检测框
            display_img = self._draw_detections(image.copy())
            
            # 转换为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
            
            # 调整图像大小以适应canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                pil_image = pil_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            # 转换为PhotoImage对象
            self.image_tk = ImageTk.PhotoImage(pil_image)
            
            # 更新canvas上的图像
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=self.image_tk)

    def _update_info_on_image(self, info):
        """
        在图像上更新信息
        """
        if self.current_image is not None:
            # 复制当前图像
            img_with_info = self.current_image.copy()
            
            # 设置字体和颜色
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            text_color = (255, 255, 255)  # 白色文字
            text_thickness = 2
            
            # 计算起始位置
            start_y = 20
            line_height = 30
            
            # Move Probabilities
            move_probs = info.get('move_probs', 'N/A')
            cv2.putText(img_with_info, f"Move Probabilities: {move_probs}", (10, start_y), font, font_scale, text_color, text_thickness)
            start_y += line_height
            
            # Turn Probabilities  
            turn_probs = info.get('turn_probs', 'N/A')
            cv2.putText(img_with_info, f"Turn Probabilities: {turn_probs}", (10, start_y), font, font_scale, text_color, text_thickness)
            start_y += line_height
            
            # Action Params
            action_params = info.get('action_params', 'N/A')
            cv2.putText(img_with_info, f"Action Params: {action_params}", (10, start_y), font, font_scale, text_color, text_thickness)
            start_y += line_height
            
            # Value Estimation - 安全处理数值格式化
            value = info.get('value', 'N/A')
            if isinstance(value, (int, float)):
                cv2.putText(img_with_info, f"Value Estimation: {value:.2f}", (10, start_y), font, font_scale, text_color, text_thickness)
            else:
                cv2.putText(img_with_info, "Value Estimation: N/A", (10, start_y), font, font_scale, text_color, text_thickness)
            start_y += line_height
            
            # Reward - 安全处理数值格式化
            reward = info.get('reward', 'N/A')
            if isinstance(reward, (int, float)):
                cv2.putText(img_with_info, f"Reward: {reward:.2f}", (10, start_y), font, font_scale, text_color, text_thickness)
            else:
                cv2.putText(img_with_info, "Reward: N/A", (10, start_y), font, font_scale, text_color, text_thickness)
            start_y += line_height
            
            # 其他文本
            cv2.putText(img_with_info, f"Step: {info.get('step', 'N/A')}", (10, start_y), font, font_scale, text_color, text_thickness)
            start_y += line_height
            cv2.putText(img_with_info, f"Episode: {info.get('episode', 'N/A')}", (10, start_y), font, font_scale, text_color, text_thickness)
            start_y += line_height
            cv2.putText(img_with_info, f"Move Action: {info.get('move_action', 'N/A')}", (10, start_y), font, font_scale, text_color, text_thickness)
            start_y += line_height
            cv2.putText(img_with_info, f"Turn Action: {info.get('turn_action', 'N/A')}", (10, start_y), font, font_scale, text_color, text_thickness)
            
            # 更新当前图像
            self.current_image = img_with_info
            self.update_image_and_detections(self.current_image, self.detections)

    def _draw_detections(self, image):
        """
        在图像上绘制检测框
        """
        if self.detections:
            for detection in self.detections:
                bbox = detection['bbox']
                label = detection['label']
                score = detection['score']
                
                # 转换边界框坐标为整数
                x1, y1, x2, y2 = map(int, bbox)
                
                # 绘制矩形框
                color = (0, 255, 0)  # 绿色框
                thickness = 2
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                
                # 添加标签和置信度文本
                text = f"{label}: {score:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                text_color = (255, 255, 255)  # 白色文字
                text_thickness = 1
                
                # 获取文本框的尺寸
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
                
                # 绘制文本背景
                cv2.rectangle(image, (x1, y1 - text_height - 10), 
                            (x1 + text_width, y1), color, -1)
                
                # 在图像上绘制文本
                cv2.putText(image, text, (x1, y1 - 5), font, 
                           font_scale, text_color, text_thickness)
        
        return image

    def run(self):
        """
        运行Tkinter主循环
        """
        self.init_gui()
        # 设置定期处理GUI更新
        self.root.after(50, self.process_gui_updates)
        self.root.mainloop()


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet特征提取器
    """
    def __init__(self, input_channels=3, block_channels=[32, 64, 128]):
        super(ResNetFeatureExtractor, self).__init__()
        
        # 使用配置文件中的图像尺寸
        height = CONFIG.get('IMAGE_HEIGHT', 480)
        width = CONFIG.get('IMAGE_WIDTH', 640)
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(input_channels, block_channels[0], 
                              kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(block_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet层
        layers = []
        # 第一层
        layers.append(ResidualBlock(block_channels[0], block_channels[0]))
        layers.append(ResidualBlock(block_channels[0], block_channels[0]))
        
        # 第二层 - 下采样
        layers.append(ResidualBlock(block_channels[0], block_channels[1], stride=2))
        layers.append(ResidualBlock(block_channels[1], block_channels[1]))
        
        # 第三层 - 下采样
        layers.append(ResidualBlock(block_channels[1], block_channels[2], stride=2))
        layers.append(ResidualBlock(block_channels[2], block_channels[2]))
        
        self.layers = nn.Sequential(*layers)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_features = nn.Linear(block_channels[2], 256)
        
        # 添加更强的正则化以防止特征坍塌
        self.dropout = nn.Dropout(p=0.5)  # 增加dropout率
        self.dropout2 = nn.Dropout(p=0.3)  # 添加额外的dropout层

        # 初始化权重
        self._initialize_weights()

        # 添加特征输出记录
        self.feature_outputs = []

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layers(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(x)  # 应用第一个dropout
        x = self.fc_features(x)
        x = self.dropout2(x)  # 应用第二个dropout
        
        # 记录特征输出用于检查特征是否多样
        self.feature_outputs.append(x.clone().detach())
        if len(self.feature_outputs) > 100:  # 只保留最近100个特征
            self.feature_outputs.pop(0)
            
        return x
        
    def _initialize_weights(self):
        """初始化权重以避免梯度消失/爆炸"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # 使用Xavier初始化以获得更好的梯度流
                nn.init.constant_(m.bias, 0)


class PolicyNetwork(nn.Module):
    """
    策略网络
    """
    def __init__(self, state_dim, move_action_dim, turn_action_dim, hidden_size=256):
        super(PolicyNetwork, self).__init__()
        
        self.feature_extractor = ResNetFeatureExtractor(3)
        feature_size = 256  # 根据上面的修改调整
        
        # Actor heads - 添加更多的隐藏层和Dropout来增加复杂性和正则化
        self.move_actor = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),  # 添加dropout层
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加dropout层
            nn.Linear(hidden_size // 2, move_action_dim)
        )
        
        self.turn_actor = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),  # 添加dropout层
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加dropout层
            nn.Linear(hidden_size // 2, turn_action_dim)
        )
        
        # Action parameter head
        self.action_param_head = nn.Sequential(
            nn.Linear(feature_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加dropout层
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加dropout层
            nn.Linear(64, 2),
            nn.Tanh()
        )
        
        # Value network
        self.critic = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),  # 添加dropout层
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加dropout层
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 初始化logger
        self.logger = setup_logging()

    def forward(self, state, return_debug_info=False):
        features = self.feature_extractor(state)
        
        # Actor outputs
        move_logits = self.move_actor(features)
        turn_logits = self.turn_actor(features)
        
        # Action parameters
        action_params = torch.tanh(self.action_param_head(features))
        
        # Critic output
        value = self.critic(features)
        
        if return_debug_info:
            debug_info = {
                'move_logits': move_logits.detach().cpu().numpy(),
                'turn_logits': turn_logits.detach().cpu().numpy(),
                'action_params': action_params.detach().cpu().numpy(),
                'value': value.detach().cpu().numpy(),
                'features_shape': features.shape
            }
            
            # 打印模型输出信息
            self.logger.info(f"Move: {[round(float(x), 2) for x in debug_info['move_logits'][0]]}")
            self.logger.info(f"Turn: {[round(float(x), 2) for x in debug_info['turn_logits'][0]]}")
            self.logger.info(f"参数: {[round(float(x), 2) for x in debug_info['action_params'][0]]}")
            self.logger.info(f"价值: {round(float(debug_info['value'][0][0]), 2)}")
            
            return (
                F.softmax(move_logits, dim=-1),
                F.softmax(turn_logits, dim=-1),
                action_params,
                value,
                debug_info
            )
        else:
            return (
                F.softmax(move_logits, dim=-1),
                F.softmax(turn_logits, dim=-1),
                action_params,
                value
            )


class PPOAgent:
    """
    PPO智能体
    """
    def __init__(self, state_dim, move_action_dim, turn_action_dim):
        config = CONFIG
        
        self.lr = config['LEARNING_RATE']
        self.betas = (0.9, 0.999)
        self.gamma = config['GAMMA']
        self.K_epochs = config['K_EPOCHS']
        self.eps_clip = config['EPS_CLIP']
        
        # 添加logger
        self.logger = setup_logging()
        
        # Create policy networks
        self.policy = PolicyNetwork(
            state_dim, move_action_dim, turn_action_dim
        )
        self.optimizer = torch.optim.AdamW(  # 改用AdamW优化器
            self.policy.parameters(), 
            lr=self.lr, 
            betas=self.betas,
            weight_decay=1e-4  # 添加权重衰减以防止过拟合
        )
        self.policy_old = PolicyNetwork(
            state_dim, move_action_dim, turn_action_dim
        )
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        # 添加梯度检查相关变量
        self.gradient_norms = []
        self.parameter_norms = []

    def act(self, state, memory, return_debug_info=False):
        # 预处理状态
        state_tensor = self._preprocess_state(state)
        
        # 使用旧策略获取动作概率
        with torch.no_grad():
            if return_debug_info:
                move_probs, turn_probs, action_params, state_val, debug_info = self.policy_old(state_tensor.unsqueeze(0), return_debug_info=True)
            else:
                move_probs, turn_probs, action_params, state_val = self.policy_old(state_tensor.unsqueeze(0))
                
            move_dist = Categorical(move_probs)
            turn_dist = Categorical(turn_probs)
            
            # 采样动作
            move_action = move_dist.sample()
            turn_action = turn_dist.sample()
            
            # 计算对数概率
            move_logprob = move_dist.log_prob(move_action)
            turn_logprob = turn_dist.log_prob(turn_action)
            logprob = move_logprob + turn_logprob
        
        # 处理动作参数
        move_forward_step_raw = action_params[0][0].item()  # [-1,1]范围
        turn_angle_raw = action_params[0][1].item()         # [-1,1]范围
        
        # 使用配置文件中的参数范围
        MOVE_STEP_MIN = CONFIG.get('MOVE_STEP_MIN', 0.0)
        MOVE_STEP_MAX = CONFIG.get('MOVE_STEP_MAX', 1.0)
        TURN_ANGLE_MIN = CONFIG.get('TURN_ANGLE_MIN', 5.0)
        TURN_ANGLE_MAX = CONFIG.get('TURN_ANGLE_MAX', 60.0)
        
        # 将 [-1, 1] 映射到实际范围
        move_forward_step = ((move_forward_step_raw + 1.0) / 2.0) * (MOVE_STEP_MAX - MOVE_STEP_MIN) + MOVE_STEP_MIN
        turn_angle = ((turn_angle_raw + 1.0) / 2.0) * (TURN_ANGLE_MAX - TURN_ANGLE_MIN) + TURN_ANGLE_MIN
        
        # 存储到记忆中
        memory.append(
            state_tensor,
            move_action.item(),
            turn_action.item(),
            logprob.item(),
            0,  # 奖励稍后更新
            False,  # 是否结束稍后更新
            [move_forward_step, turn_angle]  # 存储参数
        )
        
        if return_debug_info:
            return move_action.item(), turn_action.item(), move_forward_step, turn_angle, debug_info
        else:
            return move_action.item(), turn_action.item(), move_forward_step, turn_angle

    def update(self, memory):
        # 计算折扣奖励
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # 标准化奖励
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 转换为张量 - 注意logprobs需要特殊处理
        old_states = torch.stack(memory.states).detach()
        old_actions_move = torch.LongTensor(memory.move_actions).detach()
        old_actions_turn = torch.LongTensor(memory.turn_actions).detach()
        
        # 修改这一行：将logprobs转换为tensor
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).detach()
        
        # 修改这一行：正确处理action_params
        if memory.action_params:
            # 将列表转换为tensor
            action_params_tensor = torch.tensor(memory.action_params, dtype=torch.float32)
            old_values = action_params_tensor.detach()
        else:
            old_values = None

        # PPO更新
        for _ in range(self.K_epochs):
            # 前向传播
            move_probs, turn_probs, action_params, state_vals = self.policy(old_states)

            # 计算对数概率
            move_dist = Categorical(move_probs)
            turn_dist = Categorical(turn_probs)
            
            logprobs = move_dist.log_prob(old_actions_move) + turn_dist.log_prob(old_actions_turn)
            
            # 计算比率
            ratios = torch.exp(logprobs - old_logprobs)

            # 计算优势 - 使用广义优势估计(GAE)以获得更好的性能
            advantages = rewards - state_vals.squeeze().detach()

            # PPO损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            critic_loss = self.MseLoss(state_vals.squeeze(), rewards)

            # 熵损失 - 提高探索性
            move_entropy = move_dist.entropy().mean()
            turn_entropy = turn_dist.entropy().mean()
            entropy_loss = move_entropy + turn_entropy

            # 特征多样性的损失项 - 添加对比损失以鼓励特征多样化
            feature_extractor = self.policy.feature_extractor
            if len(feature_extractor.feature_outputs) > 1:
                # 获取最近的特征向量
                recent_features = torch.stack(feature_extractor.feature_outputs[-10:], dim=0) if len(feature_extractor.feature_outputs) >= 10 else torch.stack(feature_extractor.feature_outputs, dim=0)
                
                # 计算特征之间的相似度矩阵
                normalized_features = F.normalize(recent_features, p=2, dim=1)
                similarity_matrix = torch.mm(normalized_features, normalized_features.t())
                
                # 排除对角线元素（自己与自己的相似度）
                mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
                similarity_matrix = similarity_matrix.masked_fill(mask, 0)
                
                # 计算平均相似度，希望最小化它
                diversity_loss = similarity_matrix.mean()
            else:
                diversity_loss = torch.tensor(0.0, device=old_states.device)

            # 总损失 - 调整权重平衡
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss + 0.1 * diversity_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 检查梯度
            total_norm = 0
            param_norm = 0
            for p in self.policy.parameters():
                if p.grad is not None:
                    param_norm += p.norm(2).item() ** 2
                    total_norm += p.grad.norm(2).item() ** 2
            
            param_norm = param_norm ** 0.5
            total_norm = total_norm ** 0.5
            
            self.gradient_norms.append(total_norm)
            self.parameter_norms.append(param_norm)
            
            # 梯度裁剪以稳定训练
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            
            self.optimizer.step()

        # 更新旧策略 - 使用软更新来提高稳定性
        with torch.no_grad():
            for old_param, new_param in zip(self.policy_old.parameters(), self.policy.parameters()):
                old_param.data.copy_(0.995 * old_param.data + 0.005 * new_param.data)

        # 清空记忆
        memory.clear_memory()  


def run_episode(env, ppo_agent, visualizer, episode_num, total_episodes, training_mode=True, print_debug=False):
    """
    运行单个episode，确保在结束时重置环境
    """
    # 设置当前episode号
    env._current_episode = episode_num
    
    state = env.reset()  # 重置环境
    total_reward = 0
    step_count = 0
    done = False
    final_area = 0
    success_flag = False
    
    # 为每个episode创建独立的记忆
    episode_memory = Memory()
    
    # 定义动作名称映射
    move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
    turn_action_names = ["turn_left", "turn_right"]
    
    # 添加最大步数限制，防止无限循环
    max_steps = CONFIG.get('MAX_STEPS_PER_EPISODE', 1000)
    
    while not done and step_count < max_steps:
        if training_mode:
            # 训练模式：使用act方法
            if print_debug:
                move_action, turn_action, move_forward_step, turn_angle, debug_info = ppo_agent.act(
                    state, episode_memory, return_debug_info=True)
                
                # 检查特征多样性
                feature_extractor = ppo_agent.policy.feature_extractor
                if len(feature_extractor.feature_outputs) >= 2:
                    # 计算最后两个特征之间的相似度
                    last_features = feature_extractor.feature_outputs[-1]
                    prev_features = feature_extractor.feature_outputs[-2]
                    
                    # 计算余弦相似度
                    cos_sim = torch.nn.functional.cosine_similarity(last_features, prev_features, dim=1)
                    avg_cos_sim = cos_sim.mean().item()
                    
                    ppo_agent.logger.info(f"特征相似度: {avg_cos_sim:.4f}")
                else:
                    # 即使在第一个步骤也输出特征相似度（虽然会是默认值）
                    ppo_agent.logger.info(f"特征相似度: N/A (初始状态)")
                
                # 打印调试信息
                if 'move_probs' in debug_info:
                    print(f"move_probs shape: {debug_info['move_probs'].shape}")
                    print(f"turn_probs shape: {debug_info['turn_probs'].shape}")
                    print(f"move_action: {move_action}, turn_action: {turn_action}")
            else:
                move_action, turn_action, move_forward_step, turn_angle = ppo_agent.act(
                    state, episode_memory)
        else:
            # 评估模式：使用确定性动作
            move_action, turn_action, move_forward_step, turn_angle = get_deterministic_action(ppo_agent, state)
        
        # 执行环境步骤
        next_state, reward, done, detection_results = env.step(
            move_action, turn_action, move_forward_step, turn_angle)
        
        # 更新episode_memory中的奖励和终端状态
        if len(episode_memory.rewards) > 0:
            episode_memory.rewards[-1] = reward
            episode_memory.is_terminals[-1] = done
        
        # 更新状态
        state = next_state
        total_reward += reward
        step_count += 1
        
        # 更新可视化信息
        visualizer.update_agent_info({
            'episode': episode_num,  # 明确传递episode号
            'step': step_count,      # 传递当前step数
            'total_reward': total_reward,
            'reward': reward,        # 传递当前reward
            'done': done,
            'success': done and any(
                detection['label'].lower() == 'climb' or 'climb' in detection['label'].lower()
                for detection in detection_results
            ),
            'move_action': move_action,
            'turn_action': turn_action,
            'move_step': round(move_forward_step, 2),
            'turn_angle': round(turn_angle, 2)
        })
        
        # 记录最终检测面积
        if detection_results:
            final_area = max(d['width'] * d['height'] for d in detection_results 
                           if 'width' in d and 'height' in d)
    
    # 检查是否成功找到目标
    if detection_results:
        climb_detected = any(
            detection['label'].lower() == 'climb' or 'climb' in detection['label'].lower()
            for detection in detection_results
        )
        success_flag = climb_detected
    
    # 在训练模式下，需要更新智能体
    if training_mode and len(episode_memory.rewards) > 0:
        # 更新智能体
        ppo_agent.update(episode_memory)
        
        # 输出梯度和参数范数信息
        if ppo_agent.gradient_norms:
            avg_grad_norm = sum(ppo_agent.gradient_norms[-10:]) / len(ppo_agent.gradient_norms[-10:])
            avg_param_norm = sum(ppo_agent.parameter_norms[-10:]) / len(ppo_agent.parameter_norms[-10:])
            
            ppo_agent.logger.info(f"梯度范数: {avg_grad_norm:.4f}, 参数范数: {avg_param_norm:.4f}")
    
    # 获取最近的检测图像（如果需要的话）
    recent_detection_images = getattr(env, 'get_recent_detection_images', lambda: [])()
    
    # 确保在episode结束时重置环境
    # 注意：env.reset()已经在env.step()中被调用了，所以这里不需要再次调用
    # env.reset()
    
    # 清空特征输出记录，以便下一episode重新开始记录
    ppo_agent.policy.feature_extractor.feature_outputs = []
    
    return {
        'total_reward': total_reward,
        'step_count': step_count,
        'final_area': final_area,
        'success_flag': success_flag,
        'detection_results': detection_results,
        'recent_detection_images': recent_detection_images
    }


def main():
    """
    主函数，用于直接运行此脚本
    """
    import sys
    if len(sys.argv) < 2:
        print("默认运行带可视化的训练...")
        train_with_visualization()  # 调用带可视化的训练函数
        print("导入成功！")
        print("\n可用的功能:")
        print("1. 训练门搜索智能体: python ppo_agents.py train_new_ppo_agent [model_path]")
        print("2. 继续训练门搜索智能体: python ppo_agents.py continue_train_ppo_agent [model_path]")
        print("3. 评估已训练模型: python ppo_agents.py evaluate_trained_ppo_agent [model_path]")
        print("4. 带可视化训练: python ppo_agents.py visualize_train")
        
        return

    tool_name = sys.argv[1]
    args = sys.argv[2:]
    
    # 如果指定可视化训练
    if tool_name == "visualize_train":
        train_with_visualization()
        return
    
    # 执行对应的工具
    response = execute_ppo_tool(tool_name, *args)
    print(str(response))
