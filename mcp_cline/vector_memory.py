"""
向量记忆系统 - 基于向量数据库的长期记忆存储和检索
支持存储VLM分析、LLM吐槽等信息,并提供语义检索功能
"""
import os
import time
from datetime import datetime
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorMemory:
    """向量记忆系统 - 使用向量数据库存储和检索记忆"""

    def __init__(self, persist_directory: str = None):
        """
        初始化向量记忆系统

        Args:
            persist_directory: 向量数据库持久化存储路径
        """
        # 设置默认持久化路径
        if persist_directory is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            persist_directory = os.path.join(base_dir, "vector_memory_db")

        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # 初始化向量数据库客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 获取或创建记忆集合
        self.collection = self.client.get_or_create_collection(
            name="agent_memory",
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )

        # 初始化编码器 (使用轻量级中文模型)
        print("[记忆系统] 初始化编码器...")
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print(f"[记忆系统] 编码器初始化完成, 模型: paraphrase-multilingual-MiniLM-L12-v2")

        # 统计信息
        self.total_memories = self.collection.count()
        print(f"[记忆系统] 已加载 {self.total_memories} 条记忆")

    def encode(self, text: str) -> List[float]:
        """
        将文本编码为向量

        Args:
            text: 要编码的文本

        Returns:
            向量列表
        """
        return self.encoder.encode(text, convert_to_numpy=False).tolist()

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
            vlm_analysis: VLM分析结果 (用于检索)
            llm_commentary: LLM吐槽 (用于展示)
            metadata: 额外的元数据
            timestamp: 时间戳 (默认为当前时间)

        Returns:
            记忆ID
        """
        if timestamp is None:
            timestamp = time.time()

        # 生成唯一ID
        memory_id = f"mem_{timestamp}_{len(vlm_analysis) % 1000}"

        # 编码VLM分析 (使用分析结果作为检索向量,更准确)
        embedding = self.encode(vlm_analysis)

        # 准备元数据
        if metadata is None:
            metadata = {}

        metadata.update({
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "type": "monitoring"
        })

        # 存储到数据库
        self.collection.add(
            embeddings=[embedding],
            documents=[vlm_analysis],  # 存储VLM分析作为文档
            metadatas=[metadata],
            ids=[memory_id]
        )

        # 保存LLM吐槽到元数据 (单独存储,方便检索)
        memory_info = {
            "id": memory_id,
            "vlm_analysis": vlm_analysis,
            "llm_commentary": llm_commentary,
            "metadata": metadata
        }

        # 将吐槽信息也存入数据库 (使用吐槽文本)
        commentary_id = f"roast_{timestamp}"
        commentary_embedding = self.encode(llm_commentary)

        metadata_commentary = metadata.copy()
        metadata_commentary.update({
            "type": "commentary",
            "related_analysis_id": memory_id
        })

        self.collection.add(
            embeddings=[commentary_embedding],
            documents=[llm_commentary],
            metadatas=[metadata_commentary],
            ids=[commentary_id]
        )

        self.total_memories += 2
        print(f"[记忆系统] 保存记忆: {memory_id}, 吐槽: {commentary_id}")
        print(f"[记忆系统] VLM分析: {vlm_analysis[:50]}...")
        print(f"[记忆系统] LLM吐槽: {llm_commentary[:50]}...")

        return memory_id

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
            memory_type: 记忆类型过滤 (None/VLM_analysis/commentary)

        Returns:
            相关记忆列表,每条记忆包含id、document、metadata等
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
            # 查询数据库
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter
            )

            # 解析结果
            memories = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    memory = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else 0.0
                    }
                    memories.append(memory)

            return memories

        except Exception as e:
            print(f"[记忆系统] 检索失败: {e}")
            return []

    def get_recent_memories(self, limit: int = 5) -> List[Dict]:
        """
        获取最近的记忆 (按时间)

        Args:
            limit: 返回数量

        Returns:
            最近记忆列表
        """
        if self.total_memories == 0:
            return []

        try:
            # 获取所有记忆
            results = self.collection.get(
                limit=limit
            )

            # 按时间戳排序
            memories = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    memory = {
                        'id': results['ids'][i],
                        'document': results['documents'][i],
                        'metadata': results['metadatas'][i]
                    }
                    memories.append(memory)

                # 按timestamp降序排序
                memories.sort(key=lambda x: x['metadata'].get('timestamp', 0), reverse=True)

                return memories[:limit]

            return []

        except Exception as e:
            print(f"[记忆系统] 获取最近记忆失败: {e}")
            return []

    def format_memories_for_context(self, memories: List[Dict], max_count: int = 3) -> str:
        """
        将记忆格式化为上下文字符串,用于LLM

        Args:
            memories: 记忆列表
            max_count: 最多包含几条记忆

        Returns:
            格式化后的上下文字符串
        """
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

    def clear_all(self):
        """清空所有记忆"""
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

    def get_stats(self) -> Dict:
        """获取记忆系统统计信息"""
        return {
            "total_memories": self.total_memories,
            "persist_directory": self.persist_directory,
            "model": "paraphrase-multilingual-MiniLM-L12-v2"
        }


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
