# ==================== 1. 最顶部环境配置 ====================
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"  # 强制离线模式

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from src.config import (
    EMBEDDING_MODEL_PATH,
    RERANK_MODEL_PATH,
    RERANK_MAX_LENGTH,
)


# ==================== 本地模型路径（从 config 读取，支持环境变量覆盖） ====================
class EmbeddingManager:
    def __init__(self):
        self.EMBEDDING_MODEL_PATH = EMBEDDING_MODEL_PATH
        self.RERANK_MODEL_PATH = RERANK_MODEL_PATH

        self.embeddings = None
        self.reranker = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # fp16 仅在 GPU 上启用：half 精度推理约 2x 加速、显存减半，对排序精度几乎无影响。
        # CPU 上多数算子不支持 fp16，保持 fp32。
        self.use_fp16 = self.device == 'cuda'

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """获取向量模型"""
        if self.embeddings is None:
            print(f"🚀 正在加载本地 Embedding 模型...")
            print(f"   路径: {self.EMBEDDING_MODEL_PATH}")
            print(f"   设备: {self.device} (fp16={self.use_fp16})")

            model_kwargs = {
                'device': self.device,
                'local_files_only': True,
            }
            if self.use_fp16:
                # 透传给底层 transformers.from_pretrained，启用半精度权重
                model_kwargs['model_kwargs'] = {'torch_dtype': torch.float16}

            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.EMBEDDING_MODEL_PATH,
                model_kwargs=model_kwargs,
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ Embedding 模型加载成功！")
        return self.embeddings

    def get_reranker(self) -> CrossEncoder:
        """获取重排模型"""
        if self.reranker is None:
            print(f"🚀 正在加载本地 Reranker 模型...")
            print(f"   路径: {self.RERANK_MODEL_PATH}")
            print(f"   设备: {self.device} (fp16={self.use_fp16})")

            reranker_kwargs = {
                'model_name_or_path': self.RERANK_MODEL_PATH,
                'device': self.device,
                # 病史可达 12000 字符，但模型有效上下文仅 512 token，显式截断避免无谓 tokenize 开销
                'max_length': RERANK_MAX_LENGTH,
            }
            if self.use_fp16:
                reranker_kwargs['model_kwargs'] = {'torch_dtype': torch.float16}

            self.reranker = CrossEncoder(**reranker_kwargs)
            print("✅ Reranker 模型加载成功！")
        return self.reranker


# ==================== 单例导出（供其他模块调用） ====================
manager = EmbeddingManager()


def get_embedding_model() -> HuggingFaceEmbeddings:
    return manager.get_embeddings()


def get_reranker_model() -> CrossEncoder:
    return manager.get_reranker()


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("\n" + "=" * 30 + " 本地模型加载测试 " + "=" * 30)

    # 测试 Embedding
    print("\n[1/2] 测试 Embedding 模型...")
    try:
        emb = get_embedding_model()
        test_embedding = emb.embed_query("这是一个测试句子")
        print(f"✅ Embedding 测试成功！向量维度: {len(test_embedding)}")
    except Exception as e:
        print(f"❌ Embedding 测试失败: {e}")

    # 测试 Reranker
    print("\n[2/2] 测试 Reranker 模型...")
    try:
        reranker = get_reranker_model()
        scores = reranker.predict([("查询文本", "文档文本测试")])
        print(f"✅ Reranker 测试成功！得分示例: {scores}")
    except Exception as e:
        print(f"❌ Reranker 测试失败: {e}")

    print("\n" + "=" * 30 + " 测试结束 " + "=" * 30)