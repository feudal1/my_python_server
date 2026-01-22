"""
自我监控模块 - 自动截图、VLM分析和吐槽生成
集成向量记忆系统,支持长期记忆检索
"""
import os
import sys
import threading
import time
from datetime import datetime
from typing import List, Optional
from PIL import ImageGrab

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

    def __init__(self, vlm_service, llm_service, callback_analysis=None, callback_commentary=None, verbose=False, enable_memory=True, callback_memory_retrieved=None, callback_memory_saved=None):
        """
        初始化自我监控线程

        Args:
            vlm_service: VLM服务实例
            llm_service: LLM服务实例
            callback_analysis: VLM分析结果回调函数
            callback_commentary: 吐槽结果回调函数
            verbose: 是否输出详细日志到控制台
            enable_memory: 是否启用向量记忆系统
            callback_memory_retrieved: 记忆检索回调函数
            callback_memory_saved: 记忆保存回调函数
        """
        super().__init__(daemon=True)
        self.vlm_service = vlm_service
        self.llm_service = llm_service
        self.callback_analysis = callback_analysis
        self.callback_commentary = callback_commentary
        self.callback_memory_retrieved = callback_memory_retrieved
        self.callback_memory_saved = callback_memory_saved
        self.verbose = verbose  # 控制是否输出详细日志
        self.enable_memory = enable_memory and HAS_MEMORY

        self.running = False
        self.paused = False

        # 监控参数
        self.monitor_interval = 10  # 每10秒执行一次监控周期
        self.screenshots_per_cycle = 5  # 每个周期截图5张
        self.screenshot_interval = 2  # 每张截图间隔2秒

        # 截图清理参数
        self.max_screenshots = 50  # 最多保留最新的50张截图（约10分钟）

        # 吐槽阈值
        self.commentary_threshold = 1  # 每次1个VLM分析就触发吐槽
        self.vlm_analysis_history = []  # VLM分析历史

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
                        self._log(f"[自我监控] VLM分析完成: {vlm_analysis[:50] if vlm_analysis else '无'}...")

                        if vlm_analysis:
                            # 3. 从向量记忆中检索相关记忆 (每回合都检索)
                            relevant_memories = []
                            if self.vector_memory:
                                try:
                                    relevant_memories = self.vector_memory.retrieve_memory(
                                        query_text=vlm_analysis,
                                        top_k=3
                                    )
                                    if relevant_memories:
                                        self._log(f"[自我监控] 检索到 {len(relevant_memories)} 条相关记忆")

                                    # 调用记忆检索回调
                                    if self.callback_memory_retrieved:
                                        try:
                                            self.callback_memory_retrieved(vlm_analysis, relevant_memories)
                                        except Exception as e:
                                            print(f"[自我监控] 记忆检索回调失败: {e}")
                                except Exception as e:
                                    print(f"[自我监控] 检索记忆失败: {e}")

                            # 4. 添加到历史记录
                            self.vlm_analysis_history.append({
                                'time': datetime.now().strftime('%H:%M:%S'),
                                'analysis': vlm_analysis
                            })

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

                                # 8. 保存记忆到向量数据库
                                memory_id = None
                                if self.vector_memory:
                                    try:
                                        memory_id = self.vector_memory.save_memory(
                                            vlm_analysis=vlm_analysis,
                                            llm_commentary=commentary,
                                            metadata={"timestamp": time.time()}
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

    def _capture_screenshots(self) -> List[str]:
        """
        截取5张截图

        Returns:
            截图文件路径列表
        """
        screenshots = []

        for i in range(self.screenshots_per_cycle):
            try:
                # 截取整个屏幕
                screenshot = ImageGrab.grab()

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
        使用VLM分析截图

        Args:
            screenshots: 截图文件路径列表

        Returns:
            VLM分析结果
        """
        if not screenshots:
            return None

        try:
            # 构建VLM提示词 - 简短对话提取
            prompt = """请简要分析这5张截图：

如果是游戏/对话场景：
- 只提取角色的对话内容（20字以内）
- 不要分析工具、不要分析环境
- 格式：角色A：对话内容 / 角色B：对话内容

如果是工作场景（建模/办公/开发）：
- 识别软件类型和当前操作
- 20字以内概括当前状态

要求：
- 严格控制在20字以内
- 对话只提取文本，不添加任何描述
- 不要多余的标点符号"""

            # 调用VLM分析
            vlm_messages = [{"role": "user", "content": prompt}]

            # 尝试使用多图分析
            try:
                vlm_result = self.vlm_service.create_with_multiple_images(
                    vlm_messages,
                    image_sources=screenshots
                )

                # 提取分析结果
                analysis_text = self._extract_content_from_vlm_result(vlm_result)
                return analysis_text

            except AttributeError:
                # 如果不支持多图分析，fallback到单图分析（使用最后一张）
                self._log("[VLM] 不支持多图分析，使用最后一张截图分析")
                single_prompt = """简要描述这个截图（20字以内）：

对话场景只提取对话内容，工作场景只说明在做什么"""

                vlm_result = self.vlm_service.create_with_image(
                    [{"role": "user", "content": single_prompt}],
                    image_source=screenshots[-1]
                )

                analysis_text = self._extract_content_from_vlm_result(vlm_result)
                return analysis_text

        except Exception as e:
            print(f"[VLM] 分析失败: {e}")
            import traceback
            traceback.print_exc()
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
            # 获取最近的VLM分析
            latest_analysis = self.vlm_analysis_history[-1]['analysis']

            # 构建上下文,包含相关记忆
            context_text = ""
            if relevant_memories and self.vector_memory:
                memory_context = self.vector_memory.format_memories_for_context(relevant_memories, max_count=3)
                context_text = f"""
【过去的类似场景和吐槽】
{memory_context}

"""
                self._log("[吐槽] 已注入相关记忆到上下文")

            # 构建吐槽提示词 - 包含记忆上下文
            commentary_prompt = f"""{context_text}
【当前场景分析】
{latest_analysis}

请基于以上信息给我一句话建议或吐槽。

要求：
1. 40字以内
2. 简洁有力
3. 对话场景：评价角色或剧情，延续之前的吐槽风格
4. 工作场景：给简短建议
5. 如果有过去的记忆，要体现出连贯性（比如延续外号、引用之前的吐槽）
6. 不要编号列表
7. 直接一句话"""

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
