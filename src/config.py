import os
from dotenv import load_dotenv

load_dotenv()

# ====================== 项目配置 ======================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# ====================== 数据路径 ======================

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw"
)

KNOWLEDGE_FILE = os.path.join(
    DATA_DIR,
    "知识库2.0.xlsx"
)

# ====================== Milvus ======================

MILVUS_HOST = "10.200.10.152"

MILVUS_PORT = "19530"

MILVUS_URI = (
    f"http://{MILVUS_HOST}:{MILVUS_PORT}"
)

MILVUS_COLLECTION_NAME = (
    "tcm_knowledge_base"
)

MILVUS_DIMENSION = 1024

# ====================== 检索参数 ======================

# 召回池：全部送进 reranker 重排
TOP_K_RECALL = 150

# 最终返回条数
TOP_K_FINAL = 100

# ====================== 模型配置 ======================

BGE_EMBEDDING_MODEL = (
    "BAAI/bge-large-zh-v1.5"
)

BGE_RERANKER_MODEL = (
    "BAAI/bge-reranker-large"
)

# ---- 本地模型路径：默认指向项目内 BGE_models，可用环境变量覆盖（GPU 服务器部署时改环境变量即可）----

_DEFAULT_EMBEDDING_PATH = os.path.join(
    PROJECT_ROOT,
    "BGE_models",
    "BAAI--bge-large-zh-v1.5"
)

_DEFAULT_RERANK_PATH = os.path.join(
    PROJECT_ROOT,
    "BGE_models",
    "BAAI--bge-reranker-large"
)

EMBEDDING_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL_PATH",
    _DEFAULT_EMBEDDING_PATH
)

RERANK_MODEL_PATH = os.getenv(
    "RERANK_MODEL_PATH",
    _DEFAULT_RERANK_PATH
)

# ---- 推理参数 ----

# reranker 有效上下文长度（病史虽长，但模型只吃 512 token，显式截断避免浪费）
RERANK_MAX_LENGTH = 512

# reranker 批大小：候选成批喂给 GPU，吃满吞吐
RERANK_BATCH_SIZE = 64

print("✅ 配置加载完成")