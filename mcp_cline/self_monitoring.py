"""
自我监控模块 - 自动截图、VLM分析和吐槽生成
集成向量记忆系统,支持长期记忆检索
"""
import os
import sys
import threading
import time
import requests
import urllib3
from datetime import datetime
from typing import List, Optional
from PIL import ImageGrab, Image
import win32gui
import win32ui
import win32con
import win32api
from ctypes import windll

# 添加当前目录到路径,支持直接运行
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 尝试导入向量记忆系统
try:
    from vector_memory import VectorMemory
    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False
    print("[自我监控] 警告: 向量记忆系统未可用,将运行在无记忆模式")


class SelfMonitoringThread(threading.Thread):
    """自我监控线程 - 自动截图、VLM分析和吐槽生成"""

    def __init__(self, vlm_service, llm_service, callback_analysis=None, callback_commentary=None, verbose=False, enable_memory=False, enable_memory_retrieval=False, callback_memory_retrieved=None, callback_memory_saved=None, callback_hide_windows=None, callback_show_windows=None, blocked_windows=None):
        """
        初始化自我监控线程

        Args:
            vlm_service: VLM服务实例
            llm_service: LLM服务实例
            callback_analysis: VLM分析结果回调函数
            callback_commentary: 吐槽结果回调函数
            verbose: 是否输出详细日志到控制台
            enable_memory: 是否启用向量记忆系统
            enable_memory_retrieval: 是否启用记忆检索功能
            callback_memory_retrieved: 记忆检索回调函数
            callback_memory_saved: 记忆保存回调函数
            callback_hide_windows: 截图前隐藏窗口回调函数
            callback_show_windows: 截图后显示窗口回调函数
            blocked_windows: 要屏蔽的窗口标题列表
        """
        super().__init__(daemon=True)
        self.vlm_service = vlm_service
        self.llm_service = llm_service
        self.callback_analysis = callback_analysis
        self.callback_commentary = callback_commentary
        self.callback_memory_retrieved = callback_memory_retrieved
        self.callback_memory_saved = callback_memory_saved
        self.callback_hide_windows = callback_hide_windows
        self.callback_show_windows = callback_show_windows
        self.verbose = True  # 强制输出详细日志，便于调试
        print(f"[自我监控] 初始化参数: enable_memory={enable_memory}, enable_memory_retrieval={enable_memory_retrieval}, HAS_MEMORY={HAS_MEMORY}")
        self.enable_memory = enable_memory and HAS_MEMORY
        self.enable_memory_retrieval = enable_memory_retrieval and self.enable_memory
        print(f"[自我监控] 最终设置: self.enable_memory={self.enable_memory}, self.enable_memory_retrieval={self.enable_memory_retrieval}")
        self.blocked_windows = blocked_windows or []

        self.running = False
        self.paused = False

        # 监控参数
        self.monitor_interval = 5  # 每5秒执行一次监控周期
        self.screenshots_per_cycle = 5  # 每个周期截图5张
        self.screenshot_interval = 1  # 每张截图间隔1秒

        # 截图清理参数
        self.max_screenshots = 50  # 最多保留最新的50张截图（约10分钟）

        # 吐槽阈值
        self.commentary_threshold = 2  # 每2个VLM分析就触发吐槽，提高频率
        self.vlm_analysis_history = []  # VLM分析历史
        self.user_input_history = []  # 用户输入历史

        # 截图目录
        self.screenshots_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'screenshots'
        )
        os.makedirs(self.screenshots_dir, exist_ok=True)

        # 初始化向量记忆系统
        self.vector_memory = None
        if self.enable_memory:
            try:
                self.vector_memory = VectorMemory()
                print(f"[自我监控] 向量记忆系统已启用")
            except Exception as e:
                print(f"[自我监控] 向量记忆系统初始化失败: {e}")
                self.vector_memory = None

        if self.verbose:
            print("[自我监控] 线程初始化完成")

    def _log(self, message):
        """输出日志（始终输出关键信息）"""
        # 关键信息始终输出
        if "VLM分析完成" in message or "触发吐槽" in message or "吐槽生成完成" in message:
            print(message)
        elif self.verbose:
            print(message)

    def start_monitoring(self):
        """启动监控"""
        if not self.running:
            self.running = True
            self.paused = False
            self.start()
            self._log("[自我监控] 监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        self._log("[自我监控] 监控已停止")

    def add_user_input(self, user_input):
        """
        添加用户输入到历史记录（作为VLM分析的一部分，一起触发吐槽）

        Args:
            user_input: 用户输入文本
        """
        # 添加到用户输入历史（用于上下文）
        self.user_input_history.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'input': user_input
        })

        # 将用户输入也作为VLM分析添加到历史记录中，这样会触发吐槽
        self.vlm_analysis_history.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'analysis': f"[用户输入] {user_input}"
        })
        self._log(f"[自我监控] 已添加用户输入到VLM历史记录: {user_input[:30]}...")

        # 用户输入时也检索记忆
        if self.vector_memory and self.enable_memory_retrieval:
            try:
                self._log(f"[用户输入] 使用用户输入检索记忆: {user_input[:20]}...")
                # 只检索主记忆（type=monitoring）
                relevant_memories = self.vector_memory.retrieve_memory(
                    query_text=user_input,
                    top_k=3,
                    memory_type="monitoring"
                )
                if relevant_memories:
                    # 显示检索到的记忆内容（前10个字符）
                    memory_previews = []
                    for mem in relevant_memories:
                        doc = mem.get('document', '')
                        # 显示主记忆的前10个字符
                        preview = doc[:10] if len(doc) > 10 else doc
                        memory_previews.append(preview)

                    if memory_previews:
                        self._log(f"[用户输入] 检索到记忆: {', '.join(memory_previews)}")
                    else:
                        self._log("[用户输入] 检索到相关记忆")
                else:
                    self._log("[用户输入] 未找到相关记忆")

                # 调用记忆检索回调（传入检索类型）
                if self.callback_memory_retrieved:
                    self.callback_memory_retrieved("用户输入", user_input, relevant_memories)
            except Exception as e:
                print(f"[用户输入] 检索记忆失败: {e}")

    def pause_monitoring(self):
        """暂停监控"""
        self.paused = True
        self._log("[自我监控] 监控已暂停")

    def resume_monitoring(self):
        """恢复监控"""
        self.paused = False
        self._log("[自我监控] 监控已恢复")

    def run(self):
        """主线程循环"""
        self._log("[自我监控] 进入主循环")

        while self.running:
            try:
                if not self.paused:
                    self._log(f"[自我监控] 开始新的监控周期 - {datetime.now().strftime('%H:%M:%S')}")

                    # 1. 截取5张截图
                    screenshots = self._capture_screenshots()
                    self._log(f"[自我监控] 已捕获 {len(screenshots)} 张截图")

                    # 1.5 清理旧截图
                    self._cleanup_old_screenshots()

                    if screenshots:
                        # 2. VLM分析截图
                        vlm_analysis = self._analyze_with_vlm(screenshots)
                        # 修复：当vlm_analysis为None时，避免尝试切片操作
                        analysis_preview = vlm_analysis[:50] if vlm_analysis else '无'
                        self._log(f"[自我监控] VLM分析完成: {analysis_preview}...")

                        if vlm_analysis:
                            # 3. 从向量记忆中检索相关记忆 (每回合都检索)
                            relevant_memories = []
                            # 调用记忆检索回调，无论记忆检索是否启用
                            if self.callback_memory_retrieved:
                                try:
                                    # 尝试从向量记忆中检索相关记忆
                                    relevant_memories = []
                                    if self.vector_memory and self.enable_memory_retrieval:
                                        try:
                                            # 分别检索不同类型的记忆
                                            user_inputs = [item['input'] for item in self.user_input_history]
                                            user_inputs_str = " ".join(user_inputs)
                                            
                                            # 1. 首先检索用户偏好相关记忆（如果有用户输入）
                                            preference_memories = []
                                            if user_inputs_str:
                                                # 单独使用用户输入检索偏好记忆
                                                preference_query = user_inputs_str
                                                preference_memories = self.vector_memory.retrieve_memory(
                                                    query_text=preference_query,
                                                    top_k=2,
                                                    memory_type="monitoring"
                                                )
                                                self._log(f"[自我监控] 使用用户输入单独检索偏好记忆: {preference_query[:30]}...")
                                                if preference_memories:
                                                    previews = [mem.get('document', '')[:10] for mem in preference_memories]
                                                    self._log(f"[自我监控] 检索到偏好记忆: {', '.join(previews)}")
                                            
                                            # 2. 检索游戏相关记忆
                                            game_query = vlm_analysis
                                            game_memories = self.vector_memory.retrieve_memory(
                                                query_text=game_query,
                                                top_k=2,
                                                memory_type="monitoring"
                                            )
                                            self._log(f"[自我监控] 使用VLM分析检索游戏记忆: {game_query[:30]}...")
                                            if game_memories:
                                                previews = [mem.get('document', '')[:10] for mem in game_memories]
                                                self._log(f"[自我监控] 检索到游戏记忆: {', '.join(previews)}")
                                            
                                            # 3. 合并记忆结果（偏好记忆优先）
                                            relevant_memories = preference_memories + game_memories
                                            # 去重，保留前3条
                                            seen_docs = set()
                                            unique_memories = []
                                            for mem in relevant_memories:
                                                doc = mem.get('document', '')
                                                if doc not in seen_docs:
                                                    seen_docs.add(doc)
                                                    unique_memories.append(mem)
                                                    if len(unique_memories) >= 3:
                                                        break
                                            relevant_memories = unique_memories
                                            
                                            if relevant_memories:
                                                previews = [mem.get('document', '')[:10] for mem in relevant_memories]
                                                self._log(f"[自我监控] 最终检索到记忆: {', '.join(previews)}")
                                            else:
                                                self._log("[自我监控] 未找到相关记忆")
                                        except Exception as e:
                                            print(f"[自我监控] 检索记忆失败: {e}")
                                            self._log("[自我监控] 未找到相关记忆")
                                    else:
                                        if not self.enable_memory_retrieval:
                                            self._log("[自我监控] 记忆检索已禁用")
                                        elif not self.vector_memory:
                                            self._log("[自我监控] 向量记忆系统不可用")
                                        self._log("[自我监控] 未找到相关记忆")

                                    # 调用记忆检索回调
                                    query_type = "混合检索"
                                    combined_query = f"{vlm_analysis} {user_inputs_str}".strip() if user_inputs_str else vlm_analysis
                                    self.callback_memory_retrieved(query_type, combined_query, relevant_memories)
                                except Exception as e:
                                    print(f"[自我监控] 记忆检索回调失败: {e}")

                            # 4. 添加到历史记录（去重）
                            # 检查是否已存在相同的分析结果
                            analysis_exists = any(
                                item['analysis'] == vlm_analysis 
                                for item in self.vlm_analysis_history
                            )
                            if not analysis_exists:
                                self.vlm_analysis_history.append({
                                    'time': datetime.now().strftime('%H:%M:%S'),
                                    'analysis': vlm_analysis
                                })
                                # 限制历史记录长度，最多保留10条
                                if len(self.vlm_analysis_history) > 10:
                                    self.vlm_analysis_history = self.vlm_analysis_history[-10:]

                            # 5. 调用VLM分析回调
                            if self.callback_analysis:
                                try:
                                    self.callback_analysis(vlm_analysis)
                                except Exception as e:
                                    print(f"[自我监控] VLM分析回调失败: {e}")
                                    import traceback
                                    traceback.print_exc()

                            # 6. 检查是否需要生成吐槽
                            if len(self.vlm_analysis_history) >= self.commentary_threshold:
                                self._log(f"[自我监控] 已收集{len(self.vlm_analysis_history)}个VLM分析，触发吐槽")
                                commentary = self._generate_commentary(relevant_memories)
                                self._log(f"[自我监控] 吐槽生成完成: {commentary[:50] if commentary else '无'}...")

                                # 7. 调用吐槽回调
                                if self.callback_commentary and commentary:
                                    try:
                                        self.callback_commentary(commentary)
                                    except Exception as e:
                                        print(f"[自我监控] 吐槽回调失败: {e}")

                                # 8. 保存记忆到向量数据库（包含用户输入、VLM分析、LLM吐槽）
                                memory_id = None
                                if self.vector_memory:
                                    try:
                                        # 收集所有VLM分析历史
                                        vlm_analyses = [item['analysis'] for item in self.vlm_analysis_history]

                                        # 收集用户输入历史
                                        user_inputs = [item['input'] for item in self.user_input_history] if self.user_input_history else []

                                        memory_id = self.vector_memory.save_memory(
                                            vlm_analysis=vlm_analysis,
                                            llm_commentary=commentary,
                                            metadata={
                                                "user_inputs": user_inputs,
                                                "vlm_analyses": vlm_analyses
                                            }
                                        )
                                        self._log("[自我监控] 已保存到向量记忆库")

                                        # 调用记忆保存回调
                                        if self.callback_memory_saved and memory_id:
                                            try:
                                                self.callback_memory_saved(memory_id, vlm_analysis, commentary)
                                            except Exception as e:
                                                print(f"[自我监控] 记忆保存回调失败: {e}")
                                    except Exception as e:
                                        print(f"[自我监控] 保存记忆失败: {e}")

                                # 9. 清空历史记录
                                self.vlm_analysis_history = []
                                self.user_input_history = []  # 同时清空用户输入历史
                            else:
                                self._log("[自我监控] 截图失败，跳过本次周期")

                    self._log(f"[自我监控] 等待 {self.monitor_interval} 秒后开始下一个周期...")

                # 等待下一个周期
                time.sleep(self.monitor_interval)

            except Exception as e:
                print(f"[自我监控] 主循环异常: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)  # 出错后等待5秒再继续

        self._log("[自我监控] 主循环退出")

    def _cleanup_old_screenshots(self):
        """清理旧的截图，只保留最新的max_screenshots张"""
        try:
            # 获取所有自我监控截图文件
            files = []
            for filename in os.listdir(self.screenshots_dir):
                if filename.startswith("self_monitor_") and filename.endswith(".png"):
                    filepath = os.path.join(self.screenshots_dir, filename)
                    # 获取文件修改时间
                    mtime = os.path.getmtime(filepath)
                    files.append((filepath, mtime))

            # 按修改时间排序（最新的在前）
            files.sort(key=lambda x: x[1], reverse=True)

            # 删除超出限制的旧文件
            if len(files) > self.max_screenshots:
                files_to_delete = files[self.max_screenshots:]
                deleted_count = 0
                for filepath, _ in files_to_delete:
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                    except Exception as e:
                        if self.verbose:
                            print(f"[清理] 删除失败 {filepath}: {e}")

                if deleted_count > 0:
                    self._log(f"[清理] 已删除 {deleted_count} 张旧截图（保留最新{self.max_screenshots}张）")

        except Exception as e:
            if self.verbose:
                print(f"[清理] 清理失败: {e}")

    def _capture_window_by_title(self, window_title):
        """
        根据窗口标题截取指定窗口内容，使用PrintWindow API

        Args:
            window_title: 窗口标题（部分匹配）

        Returns:
            PIL.Image对象，如果失败返回None
        """
        # 查找窗口句柄
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if window_title.lower() in window_text.lower():
                    windows.append(hwnd)
            return True

        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)

        if not windows:
            return None

        # 选择第一个匹配的窗口
        hwnd = windows[0]

        # 获取窗口位置和大小
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top

        # 获取设备上下文
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        # 创建位图
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)

        # 使用 PrintWindow API 截取窗口
        result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)

        if result:
            # 转换为PIL图像
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)

            im = Image.frombuffer(
                'RGB',
                (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                bmpstr, 'raw', 'BGRX', 0, 1)
        else:
            im = None

        # 清理资源
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)

        return im

    def _capture_screenshots(self) -> List[str]:
        """
        截取5张截图 - 始终截取Endfield窗口

        Returns:
            截图文件路径列表
        """
        screenshots = []
        target_window = "Endfield"  # 目标窗口名称

        for i in range(self.screenshots_per_cycle):
            try:
                # 截图前隐藏窗口
                if self.callback_hide_windows:
                    try:
                        self.callback_hide_windows()
                        # 等待窗口隐藏（增加等待时间确保所有UI操作完成，包括清空吐槽内容）
                        time.sleep(0.8)
                    except Exception as e:
                        print(f"[截图] 隐藏窗口失败: {e}")

                # 使用PrintWindow API截取Endfield窗口
                screenshot = self._capture_window_by_title(target_window)

                if screenshot:
                    self._log(f"[截图] 已截取窗口: {target_window}")
                else:
                    # 如果无法截取窗口，截取整个屏幕
                    screenshot = ImageGrab.grab()
                    self._log(f"[截图] 未找到{target_window}窗口，截取整个屏幕")

                # 截图后显示窗口
                if self.callback_show_windows:
                    try:
                        self.callback_show_windows()
                    except Exception as e:
                        print(f"[截图] 显示窗口失败: {e}")

                # 生成唯一文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"self_monitor_{timestamp}.png"

                # 保存截图
                filepath = os.path.join(self.screenshots_dir, filename)
                screenshot.save(filepath, "PNG")

                screenshots.append(filepath)
                self._log(f"[截图] 第 {i+1}/{self.screenshots_per_cycle} 张已保存: {filename}")

                # 等待间隔（最后一张不需要等待）
                if i < self.screenshots_per_cycle - 1:
                    time.sleep(self.screenshot_interval)

            except ImportError:
                # 如果没有安装win32gui模块，截取整个屏幕
                # 截图前隐藏窗口
                if self.callback_hide_windows:
                    try:
                        self.callback_hide_windows()
                        # 等待窗口隐藏（增加等待时间确保所有UI操作完成，包括清空吐槽内容）
                        time.sleep(0.8)
                    except Exception as e:
                        print(f"[截图] 隐藏窗口失败: {e}")

                screenshot = ImageGrab.grab()
                self._log("[截图] 未安装win32gui模块，截取整个屏幕")
                
                # 截图后显示窗口
                if self.callback_show_windows:
                    try:
                        self.callback_show_windows()
                    except Exception as e:
                        print(f"[截图] 显示窗口失败: {e}")

                # 生成唯一文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"self_monitor_{timestamp}.png"

                # 保存截图
                filepath = os.path.join(self.screenshots_dir, filename)
                screenshot.save(filepath, "PNG")

                screenshots.append(filepath)
                self._log(f"[截图] 第 {i+1}/{self.screenshots_per_cycle} 张已保存: {filename}")

                # 等待间隔（最后一张不需要等待）
                if i < self.screenshots_per_cycle - 1:
                    time.sleep(self.screenshot_interval)
            except Exception as e:
                print(f"[截图] 第 {i+1} 张截图失败: {e}")

        return screenshots

    def _analyze_with_vlm(self, screenshots: List[str]) -> Optional[str]:
        """
        使用VLM分析截图（带重试机制）

        Args:
            screenshots: 截图文件路径列表

        Returns:
            VLM分析结果
        """
        if not screenshots:
            return None

        # 最大重试次数
        max_retries = 3
        retry_delay = 5  # 秒

        for attempt in range(max_retries):
            try:
                # 构建VLM提示词 - 简短对话提取
                prompt = """请简要分析这5张截图，只返回一个综合结果：

如果是游戏/对话场景：
- 如果有对话，提取对话（角色A：对话内容 / 角色B：对话内容）
- 如果没有对话，概括场景（20字以内）
- 忽略所有工具窗口、UI元素、字幕和分析结果
- 不要分析工具、不要分析环境
- 不要分析蓝色的、黄色的和绿色的字

如果是工作场景（建模/办公/开发）：
- 识别软件类型和当前操作
- 20字以内概括当前状态
- 不要分析蓝色的和黄色的字

重要要求：
- 只返回一个综合结果，不要分别分析每张图
- 严格控制在30字以内
- 对话只提取文本，不添加任何描述
- 不要多余的标点符号"""

                # 调用VLM分析
                vlm_messages = [{"role": "user", "content": prompt}]

                # 尝试使用多图分析
                try:
                    if attempt > 0:
                        self._log(f"[VLM] 第{attempt+1}次重试...")

                    vlm_result = self.vlm_service.create_with_multiple_images(
                        vlm_messages,
                        image_sources=screenshots
                    )

                    # 提取分析结果
                    analysis_text = self._extract_content_from_vlm_result(vlm_result)

                    if analysis_text:
                        self._log(f"[VLM] 分析成功 (第{attempt+1}次尝试)")
                        return analysis_text
                    else:
                        raise ValueError("VLM返回空结果")

                except AttributeError:
                    # 如果不支持多图分析，fallback到单图分析（使用最后一张）
                    self._log("[VLM] 不支持多图分析，使用最后一张截图分析")
                    single_prompt = """简要描述这个截图（20字以内）：

对话场景只提取对话内容，工作场景只说明在做什么

要求：
- 忽略所有工具窗口、UI元素、字幕和分析结果
- 不要分析工具、不要分析环境
- 不要分析蓝色的、黄色的和绿色的字
- 严格控制在20字以内
- 直接描述，不要多余的标点符号"""

                    vlm_result = self.vlm_service.create_with_image(
                        [{"role": "user", "content": single_prompt}],
                        image_source=screenshots[-1]
                    )

                    analysis_text = self._extract_content_from_vlm_result(vlm_result)

                    if analysis_text:
                        self._log(f"[VLM] 分析成功 (第{attempt+1}次尝试)")
                        return analysis_text
                    else:
                        raise ValueError("VLM返回空结果")

            except (ConnectionError, ConnectionResetError, requests.exceptions.ConnectionError,
                    requests.exceptions.SSLError, urllib3.exceptions.ProtocolError) as e:
                if attempt < max_retries - 1:
                    self._log(f"[VLM警告] 网络连接错误: {type(e).__name__}，{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    self._log(f"[VLM错误] 网络连接失败，已达最大重试次数")
                    return None

            except Exception as e:
                self._log(f"[VLM错误] 分析失败: {e}")
                import traceback
                traceback.print_exc()
                return None

        return None

    def _extract_content_from_vlm_result(self, result) -> Optional[str]:
        """
        从VLM结果中提取文本内容

        Args:
            result: VLM返回结果

        Returns:
            提取的文本内容
        """
        try:
            if isinstance(result, str):
                return result.strip()

            if isinstance(result, dict):
                choices = result.get("choices")
                if isinstance(choices, list) and len(choices) > 0:
                    first_choice = choices[0]
                    message = first_choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str):
                            return content.strip()
                        elif isinstance(content, list):
                            # 处理可能的多媒体内容
                            for item in content:
                                if isinstance(item, dict) and "text" in item:
                                    return item["text"].strip()

            return None
        except Exception as e:
            if self.verbose:
                print(f"[VLM] 提取内容失败: {e}")
            return None

    def _generate_commentary(self, relevant_memories: List = None) -> Optional[str]:
        """
        基于VLM分析历史和相关记忆生成吐槽

        Args:
            relevant_memories: 从向量数据库检索到的相关记忆

        Returns:
            吐槽文本
        """
        if not self.vlm_analysis_history:
            return None

        try:
            # 构建分析历史摘要（包含用户输入和VLM分析）
            analysis_summary = []
            user_inputs = []
            vlm_analyses = []

            for item in self.vlm_analysis_history:
                analysis = item['analysis']
                if analysis.startswith('[用户输入]'):
                    user_inputs.append(analysis.replace('[用户输入]', '').strip())
                else:
                    vlm_analyses.append(analysis)

            # 构建上下文
            context_text = ""
            if relevant_memories and self.vector_memory:
                memory_context = self.vector_memory.format_memories_for_context(relevant_memories, max_count=3)
                context_text = f"""
【过去的类似场景和吐槽】
{memory_context}

"""
                self._log("[吐槽] 已注入相关记忆到上下文")

            # 添加用户输入到上下文
            user_input_text = ""
            if user_inputs:
                user_input_text = f"""
【用户输入】
{chr(10).join(user_inputs)}

"""
                self._log("[吐槽] 已注入用户输入到上下文")

            # 添加VLM分析到上下文
            vlm_text = ""
            if vlm_analyses:
                vlm_text = f"""
【VLM分析】
{chr(10).join(vlm_analyses)}

"""
                self._log("[吐槽] 已注入VLM分析到上下文")

            # 构建吐槽提示词
            commentary_prompt = f"""{context_text}{user_input_text}{vlm_text}

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
10. 不要使用编号列表格式"""

            # 调用LLM生成吐槽
            llm_result = self.llm_service.create([{"role": "user", "content": commentary_prompt}])

            # 提取吐槽文本
            commentary_text = self._extract_content_from_llm_result(llm_result)
            return commentary_text

        except Exception as e:
            if self.verbose:
                print(f"[吐槽] 生成失败: {e}")
                import traceback
                traceback.print_exc()
            return None

    def _extract_content_from_llm_result(self, result) -> Optional[str]:
        """
        从LLM结果中提取文本内容

        Args:
            result: LLM返回结果

        Returns:
            提取的文本内容
        """
        try:
            if isinstance(result, str):
                return result.strip()

            if isinstance(result, dict):
                choices = result.get("choices")
                if isinstance(choices, list) and len(choices) > 0:
                    first_choice = choices[0]
                    message = first_choice.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str):
                            return content.strip()

            return None
        except Exception as e:
            if self.verbose:
                print(f"[LLM] 提取内容失败: {e}")
            return None
