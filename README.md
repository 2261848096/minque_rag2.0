输入：患者自然语言主诉
知识库：中医病案 + ICD映射
核心：BGE向量检索 + BM25混合召回 + Reranker重排序
输出：中医疾病、中医证型、西医疾病
结合舌象/脉象信息增强诊断

TCM Medical RAG System
基于大语言模型（LLM）与检索增强生成（RAG, Retrieval-Augmented Generation）的中医智能辅助诊断系统。
本项目针对中医临床诊疗场景设计，通过结合患者自然语言描述、结构化病史信息、中医知识库、向量检索、关键词检索以及重排序模型，实现从患者症状描述到：
- 中医疾病（中医病名）
- 中医证型
- 西医疾病
的智能匹配与辅助诊断。

项目简介
传统医疗知识具有：
- 专业术语复杂
- 病症描述方式多样
- 同一症状对应多个疾病
- 医生经验依赖程度高
等特点。

本项目通过 RAG 架构，将大规模中医临床知识库与深度学习模型结合，使 AI 能够理解患者自然语言描述，并从知识库中检索最相关病例，实现更加可靠、可解释的辅助诊断。
系统整体流程：
患者主诉
|
↓
文本预处理
|
↓
症状特征提取
|
↓
Hybrid Retrieval
(BGE Embedding + BM25)
|
↓
Reranker重排序
|
↓
Top-K知识匹配
|
↓
诊断结果生成
|
↓
中医疾病 / 中医证型 / 西医疾病

核心功能
1. 自然语言症状理解
支持患者非结构化描述：
例如：
最近经常感觉乏力，
晚上睡不好，
口干，
舌红少苔

系统能够自动提取：
- 主症
- 临床表现
- 舌象
- 脉象
等诊断相关信息。

2. 医疗知识库检索
系统构建中医知识库：
数据格式：
|字段|说明|
|病史信息|患者症状描述|
|病史要点|结构化症状|
|中医疾病_ICD|中医疾病编码|
|中医证型_ICD|证型编码|
|西医疾病|西医疾病名称|

当前知识库规模：
30000+ 条医疗知识数据

RAG技术架构
1. Embedding向量模型
采用：
BAAI/bge-large-zh-v1.5
作用：
将文本转换为高维语义向量。
配置：
Embedding Dimension:
1024
实现：
文本
↓
BGE Encoder
↓
1024维向量
↓
向量数据库

2. Hybrid Retrieval 混合检索
为了提高医疗场景召回能力，采用：
Dense Retrieval
+
Sparse Retrieval
Dense Retrieval
使用：
BGE Embedding
捕获：
- 同义症状
- 语义相似表达
例如：
胃脘胀满
≈
胃部感觉堵胀
Sparse Retrieval
使用：
BM25
增强：
- 专业医学关键词匹配
- 疾病名称匹配

最终融合：
Hybrid Score
=
0.5 × BGE Score
+
0.5 × BM25 Score

Reranker排序模型

初始召回：

TOP_K_RECALL = 10
使用：

BAAI/bge-reranker-large

重新计算：
Query
+
Candidate Document
↓
相关性评分
↓
TOP_K_FINAL = 5
提高最终诊断准确率。

检索权重设计

针对中医诊断特点：

当前特征权重：

|特征|权重|
|主症|35%|
|临床表现|45%|
|舌象|10%|
|脉象|10%|

原因：

中医辨证主要依据：

症状表现
+
四诊信息

其中：

- 主症决定疾病方向
- 临床表现提高细粒度匹配
- 舌象、脉象作为辅助证据

技术栈

后端
Python 3.10
FastAPI
Uvicorn

深度学习

PyTorch
Transformers
Sentence-Transformers

NLP模型
BGE-large-zh-v1.5
BGE-reranker-large

检索
FAISS
BM25
Hybrid Retrieval

数据处理
Pandas
OpenPyXL
NumPy

环境安装
创建环境：
bash
conda create -n tcm_rag python=3.10
conda activate tcm_rag

安装依赖：
bash
pip install -r requirements.txt

使用方式
1. 构建知识库
bash
python build_database.py

流程：
Excel知识库
↓
文本切分
↓
Embedding
↓
向量数据库

2. 启动服务
bash
python api/fastapi_server.py
启动：
http://localhost:8000
3. API调用
请求：
json
{
    "symptom":
    "头晕乏力，失眠，舌红少苔"
}
返回：
json
{
    "中医疾病":
        "xxx",

    "中医证型":
        "xxx",

    "西医疾病":
        "xxx",

    "similar_cases":[]
}
当前优化方向

1. 医疗知识增强
计划加入：
《中医诊断学》
《方剂学》
临床病案数据库
提升知识覆盖。

2. 多模态融合
结合舌象AI：
文本RAG
+
舌象视觉模型
+
脉象信息
↓
多模态中医辅助诊断

视觉模型：
YOLO-Seg
+
EfficientNet
+
Multi-task Classification
实现：
* 舌色识别
* 舌苔识别
* 裂纹检测
* 齿痕检测

项目目标

构建一个：

> 面向中医临床场景的多模态智能辅助诊断系统

最终实现：
患者输入症状
↓
AI理解病情
↓
检索医学知识
↓
结合舌象
↓
输出辅助诊断结果
↓
辅助医生决策

声明
本项目用于：
* 医学人工智能研究
* 中医知识智能化探索
* 辅助诊断技术研究

输出结果不能替代专业医生诊断。
Author
TCM AI Research Project
这份 README 的定位更接近科研项目 + 企业技术项目，不是简单代码仓库介绍。

结合你现在两个方向：
1. TCM-RAG文本诊断系统
2. 舌象多任务视觉模型

实际上已经形成一个完整路线：

         患者输入
            |
    -----------------
    |               |
 症状文本          舌象图片
    |               |
  RAG系统       Vision模型
    |               |
    --------融合--------
            |
      中医辅助诊断
