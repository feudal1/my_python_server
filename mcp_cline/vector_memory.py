"""
向量记忆系统 - 基于向量数据库的长期记忆存储和检索
支持存储VLM分析、LLM吐槽等信息,并提供语义检索功能
"""
import os
import time
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv


# 尝试导入完整版向量记忆系统
FULL_MEMORY_AVAILABLE = False
DASHSCOPE_AVAILABLE = False

# 加载环境变量（使用与llm_server相同的方式）
dotenv_path = r'E:\code\my_python_server\my_python_server_private\.env'
load_dotenv(dotenv_path)

# 尝试导入dashscope
try:
    import dashscope
    from http import HTTPStatus
    # 从环境变量获取API密钥
    api_key = os.getenv('VLM_OPENAI_API_KEY')
    if api_key:
        dashscope.api_key = api_key
        DASHSCOPE_AVAILABLE = True
        print("[记忆系统] Dashscope库可用，API密钥已配置")
    else:
        print("[记忆系统] Dashscope库可用，但API密钥未配置")
except ImportError:
    print("[记忆系统] Dashscope库不可用, 将使用默认编码")
except Exception as e:
    print(f"[记忆系统] 初始化Dashscope失败: {e}, 将使用默认编码")

# 尝试导入chromadb
try:
    import chromadb
    from chromadb.config import Settings
    FULL_MEMORY_AVAILABLE = True
except ImportError:
    print("[记忆系统] 完整版依赖未找到, 将使用简化版记忆系统")
except Exception as e:
    print(f"[记忆系统] 初始化完整版失败: {e}, 将使用简化版记忆系统")


# 导入简化版记忆系统
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_vector_memory import SimpleVectorMemory


class VectorMemory:
    """向量记忆系统 - 自动选择完整版或简化版"""

    def __init__(self, persist_directory: str = None):
        """
        初始化向量记忆系统

        Args:
            persist_directory: 向量数据库持久化存储路径
        """
        # 选择使用哪个版本的记忆系统
        if FULL_MEMORY_AVAILABLE:
            print("[记忆系统] 使用完整版向量记忆系统")
            self._init_full_memory(persist_directory)
        else:
            print("[记忆系统] 使用简化版向量记忆系统")
            self._init_simple_memory(persist_directory)

    def _init_full_memory(self, persist_directory: str):
        """初始化完整版向量记忆系统"""
        # 设置默认持久化路径
        if persist_directory is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            persist_directory = os.path.join(base_dir, "vector_memory_db")

        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # 初始化向量数据库客户端，禁用telemetry
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 标记使用的是完整版
        self._is_full_version = True

        # 初始化编码器
        print("[记忆系统] 初始化编码器...")
        if DASHSCOPE_AVAILABLE:
            print("[记忆系统] 使用Dashscope调用魔搭embedding服务 (text-embedding-v4)")
            self.encoder_type = "dashscope"
        else:
            print("[记忆系统] Dashscope不可用, 将使用简单编码")
            self.encoder_type = "simple"

        # 检查是否需要重新创建集合
        try:
            # 尝试获取现有集合
            existing_collection = self.client.get_collection(name="agent_memory")
            # 检查现有集合的维度
            # 注意：chromadb的API不直接提供获取集合维度的方法
            # 我们通过尝试添加一个样本向量来检查
            test_embedding = self.encode("test")
            test_dim = len(test_embedding)
            
            # 尝试添加一个测试向量
            try:
                existing_collection.add(
                    embeddings=[test_embedding],
                    documents=["test"],
                    ids=["test_id"]
                )
                # 删除测试向量
                existing_collection.delete(ids=["test_id"])
                # 维度匹配，使用现有集合
                self.collection = existing_collection
                print(f"[记忆系统] 使用现有集合，维度: {test_dim}")
            except Exception as e:
                # 维度不匹配，删除并重新创建集合
                print(f"[记忆系统] 集合维度不匹配，重新创建: {e}")
                self.client.delete_collection("agent_memory")
                self.collection = self.client.create_collection(
                    name="agent_memory",
                    metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
                )
                print(f"[记忆系统] 重新创建集合，维度: {test_dim}")
        except Exception as e:
            # 集合不存在，创建新集合
            print(f"[记忆系统] 集合不存在，创建新集合: {e}")
            self.collection = self.client.create_collection(
                name="agent_memory",
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )

        # 统计信息
        self.total_memories = self.collection.count()
        print(f"[记忆系统] 已加载 {self.total_memories} 条记忆")

    def _init_simple_memory(self, persist_directory: str):
        """初始化简化版向量记忆系统"""
        # 使用简化版记忆系统
        self.simple_memory = SimpleVectorMemory(persist_directory)
        # 标记使用的是简化版
        self._is_full_version = False

    def encode(self, text: str) -> List[float]:
        """
        将文本编码为向量

        Args:
            text: 要编码的文本

        Returns:
            向量列表
        """
        if self._is_full_version:
            if DASHSCOPE_AVAILABLE:
                # 使用dashscope调用魔搭的embedding服务
                resp = dashscope.TextEmbedding.call(
                    model="text-embedding-v4",
                    input=text
                )
                if resp.status_code == HTTPStatus.OK:
                    embedding = resp.output['embeddings'][0]['embedding']
                    return embedding
                else:
                    raise Exception(f"[记忆系统] Dashscope服务调用失败: {resp.message}")
            else:
                raise Exception("[记忆系统] Dashscope不可用，无法编码文本")
        else:
            return self.simple_memory._simple_encode(text)
    
    def _simple_encode(self, text: str) -> List[float]:
        """
        简单编码方法，当dashscope不可用时使用

        Args:
            text: 要编码的文本

        Returns:
            向量列表
        """
        # 基本特征
        features = []
        
        # 文本长度
        features.append(min(len(text) / 1000, 1.0))
        
        # 字符频率特征（前26个字母 + 数字 + 空格）
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789 '
        for char in chars:
            freq = text.lower().count(char) / max(len(text), 1)
            features.append(freq)
        
        # 标点符号频率
        punctuation = ',.!?:;()[]{}'  # 中文标点符号
        punctuation_freq = sum(text.count(p) for p in punctuation) / max(len(text), 1)
        features.append(punctuation_freq)
        
        # 中文特征（简单判断）
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        chinese_ratio = chinese_chars / max(len(text), 1)
        features.append(chinese_ratio)
        
        # 中文关键词特征
        keywords = ['猫', '狗', '沙发', '地板', '代码', '用户', '睡觉', '跳', '写', '正在']
        for keyword in keywords:
            freq = text.count(keyword) / max(len(text), 1)
            features.append(freq)
        
        # 文本复杂度特征
        unique_chars = len(set(text))
        complexity = unique_chars / max(len(text), 1)
        features.append(complexity)
        
        return features

    def save_memory(
        self,
        vlm_analysis: str,
        llm_commentary: str,
        metadata: Dict = None,
        timestamp: float = None
    ) -> str:
        """
        保存一条记忆

        Args:
            vlm_analysis: VLM分析结果
            llm_commentary: LLM吐槽
            metadata: 额外的元数据
            timestamp: 时间戳

        Returns:
            记忆ID
        """
        if self._is_full_version:
            return self._save_memory_full(vlm_analysis, llm_commentary, metadata, timestamp)
        else:
            return self.simple_memory.save_memory(vlm_analysis, llm_commentary, metadata, timestamp)

    def _save_memory_full(
        self,
        vlm_analysis: str,
        llm_commentary: str,
        metadata: Dict = None,
        timestamp: float = None
    ) -> str:
        """
        保存一条记忆 (完整版)
        """
        if timestamp is None:
            timestamp = time.time()

        # 生成唯一ID
        memory_id = f"mem_{timestamp}_{len(vlm_analysis) % 1000}"

        # 编码VLM分析
        embedding = self.encode(vlm_analysis)

        # 准备元数据（创建副本以避免修改原始字典）
        if metadata is None:
            metadata = {}
        else:
            # 过滤metadata，只保留支持的类型（str、int、float、bool）
            filtered_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    filtered_metadata[key] = value
                else:
                    # 不支持的类型转换为字符串
                    filtered_metadata[key] = str(value)
            metadata = filtered_metadata

        metadata.update({
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "type": "monitoring"
        })

        # 保存LLM吐槽到元数据
        commentary_id = f"roast_{timestamp}"
        commentary_embedding = self.encode(llm_commentary)

        metadata_commentary = metadata.copy()
        metadata_commentary.update({
            "type": "commentary",
            "related_analysis_id": memory_id
        })

        # 保存用户输入到数据库（如果存在）
        user_inputs_data = metadata.get("user_inputs", "")
        user_inputs_count = 0
        user_inputs = []
        if user_inputs_data:
            try:
                # 尝试解析为列表
                import ast
                user_inputs = ast.literal_eval(user_inputs_data) if isinstance(user_inputs_data, str) else user_inputs_data
                if isinstance(user_inputs, list):
                    user_inputs_count = len(user_inputs)
            except Exception as e:
                print(f"[记忆系统] 解析用户输入失败: {e}")

        # 保存VLM分析历史到数据库（如果存在）
        vlm_analyses_data = metadata.get("vlm_analyses", "")
        vlm_analyses_count = 0
        vlm_analyses = []
        if vlm_analyses_data:
            try:
                # 尝试解析为列表
                import ast
                vlm_analyses = ast.literal_eval(vlm_analyses_data) if isinstance(vlm_analyses_data, str) else vlm_analyses_data
                if isinstance(vlm_analyses, list):
                    # 过滤掉当前分析
                    vlm_analyses = [item for item in vlm_analyses if item != vlm_analysis]
                    vlm_analyses_count = len(vlm_analyses)
            except Exception as e:
                print(f"[记忆系统] 解析VLM分析失败: {e}")

        try:
            # 存储到数据库
            self.collection.add(
                embeddings=[embedding],
                documents=[vlm_analysis],
                metadatas=[metadata],
                ids=[memory_id]
            )

            # 保存LLM吐槽
            self.collection.add(
                embeddings=[commentary_embedding],
                documents=[llm_commentary],
                metadatas=[metadata_commentary],
                ids=[commentary_id]
            )

            # 保存用户输入
            if user_inputs:
                for idx, user_input in enumerate(user_inputs):
                    user_input_id = f"user_{timestamp}_{idx}"
                    user_input_embedding = self.encode(user_input)

                    metadata_user = metadata.copy()
                    metadata_user.update({
                        "type": "user_input",
                        "related_analysis_id": memory_id
                    })

                    self.collection.add(
                        embeddings=[user_input_embedding],
                        documents=[user_input],
                        metadatas=[metadata_user],
                        ids=[user_input_id]
                    )

            # 保存VLM分析历史
            if vlm_analyses:
                for idx, vlm_analysis_item in enumerate(vlm_analyses):
                    vlm_analysis_id = f"vlm_{timestamp}_{idx}"
                    vlm_analysis_embedding = self.encode(vlm_analysis_item)

                    metadata_vlm = metadata.copy()
                    metadata_vlm.update({
                        "type": "vlm_analysis",
                        "related_analysis_id": memory_id
                    })

                    self.collection.add(
                        embeddings=[vlm_analysis_embedding],
                        documents=[vlm_analysis_item],
                        metadatas=[metadata_vlm],
                        ids=[vlm_analysis_id]
                    )

            self.total_memories += 2 + user_inputs_count + vlm_analyses_count
            print(f"[记忆系统] 保存记忆: {memory_id}, 吐槽: {commentary_id}, 用户输入: {user_inputs_count}条, VLM分析历史: {vlm_analyses_count}条")
            print(f"[记忆系统] VLM分析: {vlm_analysis[:50]}...")
            print(f"[记忆系统] LLM吐槽: {llm_commentary[:50]}...")
            if user_inputs_count > 0:
                print(f"[记忆系统] 用户输入字符串: {user_inputs_data[:100]}...")
            if vlm_analyses_count > 0:
                print(f"[记忆系统] VLM分析历史字符串: {vlm_analyses_data[:100]}...")

            return memory_id

        except Exception as e:
            print(f"[记忆系统] 保存失败: {e}")
            # 检查是否是集合不存在的错误
            if "does not exist" in str(e):
                # 重新初始化集合
                if self._reinitialize_collection():
                    print("[记忆系统] 集合已重新初始化，再次尝试保存")
                    # 再次尝试保存
                    try:
                        # 存储到数据库
                        self.collection.add(
                            embeddings=[embedding],
                            documents=[vlm_analysis],
                            metadatas=[metadata],
                            ids=[memory_id]
                        )

                        # 保存LLM吐槽
                        self.collection.add(
                            embeddings=[commentary_embedding],
                            documents=[llm_commentary],
                            metadatas=[metadata_commentary],
                            ids=[commentary_id]
                        )

                        # 保存用户输入
                        if user_inputs:
                            for idx, user_input in enumerate(user_inputs):
                                user_input_id = f"user_{timestamp}_{idx}"
                                user_input_embedding = self.encode(user_input)

                                metadata_user = metadata.copy()
                                metadata_user.update({
                                    "type": "user_input",
                                    "related_analysis_id": memory_id
                                })

                                self.collection.add(
                                    embeddings=[user_input_embedding],
                                    documents=[user_input],
                                    metadatas=[metadata_user],
                                    ids=[user_input_id]
                                )

                        # 保存VLM分析历史
                        if vlm_analyses:
                            for idx, vlm_analysis_item in enumerate(vlm_analyses):
                                vlm_analysis_id = f"vlm_{timestamp}_{idx}"
                                vlm_analysis_embedding = self.encode(vlm_analysis_item)

                                metadata_vlm = metadata.copy()
                                metadata_vlm.update({
                                    "type": "vlm_analysis",
                                    "related_analysis_id": memory_id
                                })

                                self.collection.add(
                                    embeddings=[vlm_analysis_embedding],
                                    documents=[vlm_analysis_item],
                                    metadatas=[metadata_vlm],
                                    ids=[vlm_analysis_id]
                                )

                        self.total_memories += 2 + user_inputs_count + vlm_analyses_count
                        print(f"[记忆系统] 重新保存成功: {memory_id}")
                        return memory_id
                    except Exception as e2:
                        print(f"[记忆系统] 重新保存失败: {e2}")
            return None

    def retrieve_memory(
        self,
        query_text: str,
        top_k: int = 3,
        memory_type: str = None
    ) -> List[Dict]:
        """
        检索相关的记忆

        Args:
            query_text: 查询文本
            top_k: 返回前k条最相关的记忆
            memory_type: 记忆类型过滤

        Returns:
            相关记忆列表
        """
        if self._is_full_version:
            return self._retrieve_memory_full(query_text, top_k, memory_type)
        else:
            return self.simple_memory.retrieve_memory(query_text, top_k, memory_type)

    def _reinitialize_collection(self):
        """
        重新初始化集合
        """
        try:
            print("[记忆系统] 重新初始化集合...")
            # 尝试删除旧集合（如果存在）
            try:
                self.client.delete_collection("agent_memory")
            except:
                pass
            
            # 创建新集合
            self.collection = self.client.create_collection(
                name="agent_memory",
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
            self.total_memories = 0
            print("[记忆系统] 集合重新初始化成功")
            return True
        except Exception as e:
            print(f"[记忆系统] 集合重新初始化失败: {e}")
            return False

    def _retrieve_memory_full(
        self,
        query_text: str,
        top_k: int = 3,
        memory_type: str = None
    ) -> List[Dict]:
        """
        检索相关的记忆 (完整版)
        """
        if self.total_memories == 0:
            return []

        # 编码查询文本
        query_embedding = self.encode(query_text)

        # 构建查询条件
        where_filter = None
        if memory_type:
            where_filter = {"type": memory_type}

        try:
            # 查询数据库（多查询几条，过滤后再返回）
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2,  # 查询更多的结果，过滤后再返回
                where=where_filter
            )

            # 解析结果
            memories = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i] if 'distances' in results else 0.0
                    # 过滤掉距离过大的记忆（相似度太低）
                    # 余弦距离范围 [0, 2]，0表示完全相似，2表示完全不相似
                    # 设置阈值为 0.8，超过这个距离的记忆会被过滤
                    if distance > 0.8:
                        continue

                    memory = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': distance
                    }
                    memories.append(memory)

                    # 只返回 top_k 条
                    if len(memories) >= top_k:
                        break

            return memories

        except Exception as e:
            print(f"[记忆系统] 检索失败: {e}")
            # 检查是否是集合不存在的错误
            if "does not exist" in str(e):
                # 重新初始化集合
                if self._reinitialize_collection():
                    print("[记忆系统] 集合已重新初始化，再次尝试检索")
                    # 再次尝试检索
                    try:
                        results = self.collection.query(
                            query_embeddings=[query_embedding],
                            n_results=top_k * 2,
                            where=where_filter
                        )
                        memories = []
                        if results['ids'] and results['ids'][0]:
                            for i in range(len(results['ids'][0])):
                                distance = results['distances'][0][i] if 'distances' in results else 0.0
                                if distance > 0.8:
                                    continue
                                memory = {
                                    'id': results['ids'][0][i],
                                    'document': results['documents'][0][i],
                                    'metadata': results['metadatas'][0][i],
                                    'distance': distance
                                }
                                memories.append(memory)
                                if len(memories) >= top_k:
                                    break
                        return memories
                    except Exception as e2:
                        print(f"[记忆系统] 重新检索失败: {e2}")
            return []

    def get_recent_memories(self, limit: int = 5) -> List[Dict]:
        """
        获取最近的记忆

        Args:
            limit: 返回数量

        Returns:
            最近记忆列表
        """
        if self._is_full_version:
            return self._get_recent_memories_full(limit)
        else:
            return self.simple_memory.get_recent_memories(limit)

    def get_all_memories(self) -> List[Dict]:
        """
        获取所有记忆

        Returns:
            所有记忆列表
        """
        if self._is_full_version:
            return self._get_recent_memories_full(limit=1000)  # 设置一个较大的limit值
        else:
            return self.simple_memory.get_recent_memories(limit=1000)

    def _get_recent_memories_full(self, limit: int = 5) -> List[Dict]:
        """
        获取最近的记忆 (完整版)
        """
        if self.total_memories == 0:
            return []

        try:
            # 获取所有记忆
            results = self.collection.get(
                limit=limit
            )

            # 按时间戳排序并分类
            memories = []
            main_memories = {}  # 按主记忆ID分组

            if results['ids']:
                for i in range(len(results['ids'])):
                    memory_type = results['metadatas'][i].get('type', 'unknown')
                    memory_id = results['ids'][i]
                    timestamp = results['metadatas'][i].get('timestamp', 0)

                    memory = {
                        'id': memory_id,
                        'document': results['documents'][i],
                        'metadata': results['metadatas'][i],
                        'type': memory_type
                    }

                    # 按主记忆ID分组（将用户输入和VLM分析历史关联到主记忆）
                    related_id = results['metadatas'][i].get('related_analysis_id', '')
                    if memory_type == 'monitoring':
                        # 主记忆
                        main_memories[memory_id] = {
                            'main': memory,
                            'user_inputs': [],
                            'vlm_analyses': []
                        }
                    elif memory_type == 'user_input' and related_id in main_memories:
                        # 用户输入
                        main_memories[related_id]['user_inputs'].append(memory)
                    elif memory_type == 'vlm_analysis' and related_id in main_memories:
                        # VLM分析历史
                        main_memories[related_id]['vlm_analyses'].append(memory)

                # 将分组后的记忆转换为列表
                for memory_group in main_memories.values():
                    memories.append(memory_group)

                # 按timestamp降序排序
                memories.sort(key=lambda x: x['main']['metadata'].get('timestamp', 0), reverse=True)

                return memories[:limit]

            return []

        except Exception as e:
            print(f"[记忆系统] 获取最近记忆失败: {e}")
            # 检查是否是集合不存在的错误
            if "does not exist" in str(e):
                # 重新初始化集合
                if self._reinitialize_collection():
                    print("[记忆系统] 集合已重新初始化，返回空记忆列表")
            return []

    def format_memories_for_context(self, memories: List[Dict], max_count: int = 3) -> str:
        """
        将记忆格式化为上下文字符串

        Args:
            memories: 记忆列表
            max_count: 最多包含几条记忆

        Returns:
            格式化后的上下文字符串
        """
        if self._is_full_version:
            if not memories:
                return "暂无相关记忆"

            context_parts = []
            for i, memory in enumerate(memories[:max_count]):
                memory_time = memory['metadata'].get('datetime', '未知时间')
                memory_text = memory['document']
                memory_type = memory['metadata'].get('type', 'unknown')

                context_parts.append(
                    f"{i+1}. [{memory_time}] ({memory_type}) {memory_text}"
                )

            return "\n".join(context_parts)
        else:
            return self.simple_memory.format_memories_for_context(memories, max_count)

    def clear_all(self):
        """清空所有记忆"""
        if self._is_full_version:
            try:
                self.client.delete_collection("agent_memory")
                self.collection = self.client.create_collection(
                    name="agent_memory",
                    metadata={"hnsw:space": "cosine"}
                )
                self.total_memories = 0
                print("[记忆系统] 已清空所有记忆")
            except Exception as e:
                print(f"[记忆系统] 清空记忆失败: {e}")
        else:
            self.simple_memory.clear_all()

    def get_stats(self) -> Dict:
        """
        获取记忆系统统计信息
        """
        if self._is_full_version:
            model_name = "text-embedding-v4 (dashscope)" if DASHSCOPE_AVAILABLE else "simple-encoder"
            return {
                "total_memories": self.total_memories,
                "persist_directory": self.persist_directory,
                "model": model_name,
                "version": "full"
            }
        else:
            stats = self.simple_memory.get_stats()
            stats["version"] = "simple"
            return stats


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("测试向量记忆系统")
    print("=" * 60)

    # 创建记忆系统
    memory = VectorMemory()

    # 测试保存记忆
    print("\n1. 测试保存记忆...")
    memory.save_memory(
        vlm_analysis="一只猫在沙发上睡觉",
        llm_commentary="这只猫看起来很嚣张",
        metadata={"category": "cat"}
    )

    time.sleep(0.5)

    memory.save_memory(
        vlm_analysis="猫从沙发上跳到地板",
        llm_commentary="哟，嚣张猫下地视察民情了？",
        metadata={"category": "cat"}
    )

    # 测试检索记忆
    print("\n2. 测试检索记忆...")
    results = memory.retrieve_memory("猫在地上", top_k=2)
    print(f"检索到 {len(results)} 条相关记忆:")
    for result in results:
        print(f"  - {result['document'][:50]}... (相似度: {1 - result['distance']:.2f})")

    # 测试格式化上下文
    print("\n3. 测试格式化上下文...")
    context = memory.format_memories_for_context(results)
    print("上下文字符串:")
    print(context)

    # 测试统计信息
    print("\n4. 记忆系统统计:")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
