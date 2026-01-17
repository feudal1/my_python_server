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
