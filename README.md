# MINQUE RAG 2.0

基于大语言模型增强检索（RAG, Retrieval-Augmented Generation）的医疗病史精准检索系统。

RAG 2.0 是在原有 RAG 项目基础上的业务优化版本，整体技术架构保持不变，主要针对以下几点进行优化：

- 调整 Excel 数据结构
- 优化查询逻辑
- 提升基于患者病史信息的精准匹配能力

## 1. 项目简介

传统医疗数据检索通常依赖关键词匹配，存在以下问题：

- 患者描述与医学术语存在差异
- 同义表达无法有效匹配
- 长文本病史难以精准检索

本项目基于 RAG 技术，将患者病史信息转换为高维语义向量，并存储至 Milvus 向量数据库。用户输入新的病史描述后，系统执行以下流程：

1. 病史文本向量化
2. Milvus 向量相似度检索
3. BGE-Reranker 结果重排序
4. 返回最匹配病例信息

## 2. 核心技术栈

- 中文 Embedding 模型：BAAI/bge-large-zh-v1.5
- 重排序模型：BAAI/bge-reranker-large
- 向量数据库：Milvus

系统通过将病史信息进行向量化，实现基于语义理解的病例精准查询，同时保留病例 ID 和医生信息用于业务关联。

## 3. 系统架构

```text
用户输入病史信息
        |
        v
BAAI/bge-large-zh-v1.5
        |
        v
      文本向量化
        |
        v
        Milvus
        |
        v
   向量相似度召回
        |
        v
BAAI/bge-reranker-large
        |
        v
    返回精准匹配结果
   （id + 医生信息）
```

## 4. 数据格式设计

RAG 2.0 对知识库 Excel 文件进行了重新设计。

Excel 文件包含三个字段：

| 字段 | 说明 | 是否向量化 |
| --- | --- | --- |
| id | 病例唯一编号 | 否 |
| 病史信息 | 患者历史病情描述 | 是 |
| 医生 | 病例所属医生 | 否 |

示例：

| id | 病史信息 | 医生 |
| --- | --- | --- |
| 001 | 患者咳嗽两月，夜间明显，伴咽干口燥 | 张医生 |
| 002 | 患者胃脘胀满，食欲下降，大便稀溏 | 李医生 |

### 4.1 字段说明

#### id
病例唯一标识。

作用：

- 保留原始数据索引
- 与业务系统病例数据关联
- 查询结果返回使用

#### 病史信息
核心检索字段。

处理流程：

```text
病史信息
   |
   v
BGE-large-zh-v1.5
   |
   v
1024 维向量
   |
   v
写入 Milvus
```

#### 医生
结构化字段。

作用：

- 保存病例来源医生
- 查询结果返回医生信息

## 5. Milvus 数据结构

在 Milvus 中，存储结构如下：

```json
{
  "id": "001",
  "history": "患者咳嗽两月，夜间明显，伴咽干口燥",
  "doctor": "张医生",
  "vector": "..."
}
```

字段说明：

- id：病例编号
- history：原始病史文本
- doctor：医生信息
- vector：病史文本 Embedding 向量

## 6. 技术栈说明

### 6.1 Embedding 模型

模型：BAAI/bge-large-zh-v1.5

作用：

- 中文语义理解
- 病史文本特征提取
- 生成向量表示

输入：

```text
患者近期胃胀，饭后明显，伴乏力
```

输出：

```text
[0.123, 0.456, ...]
```

表示为 1024 维向量。

### 6.2 Reranker 模型

模型：BAAI/bge-reranker-large

作用：

对 Milvus 初步召回结果进行二次排序。

流程：

```text
Milvus Top-K 结果
        |
        v
    BGE-Reranker
        |
        v
    最终排序结果
```

相比单纯向量搜索，提高复杂医疗文本匹配准确率。

### 6.3 向量数据库

模型：Milvus

作用：

- 保存病例 Embedding 向量
- 支持高效 ANN 搜索
- 支持大规模病例扩展

## 7. 数据入库流程

```text
Excel 文件
   |
   v
读取病史信息
   |
   v
BGE Embedding 模型
   |
   v
生成文本向量
   |
   v
写入 Milvus
```

## 8. 查询流程

### Step 1：用户输入病史

例如：

```text
患者胃部胀满，饭后明显，不想进食
```

### Step 2：文本向量化

```text
Query 文本
   |
   v
BGE-large-zh-v1.5
   |
   v
Query Vector
```

### Step 3：Milvus 向量检索

根据向量相似度召回候选病例。

返回示例：

```json
[
  {
    "id": "002",
    "history": "患者胃脘胀满..",
    "doctor": "李医生"
  }
]
```

### Step 4：Reranker 重新排序

对召回结果进一步计算相关性。

最终返回：

```json
{
  "id": "002",
  "doctor": "李医生",
  "score": 0.93
}
```

## 9. 项目结构

```text
minque_rag2.0
├── src
│   ├── models
│   │   ├── embedding_manager.py
│   │   └── reranker_manager.py
│   ├── vectorstore
│   │   └��─ milvus_manager.py
│   ├── retriever
│   │   └── retriever.py
│   ├── service
│   │   └── diagnosis_service.py
│   └── config.py
├── data
│   └── knowledge.xlsx
├── main.py
├── README.md
└── requirements.txt
```

## 10. RAG 1.0 与 RAG 2.0 对比

| 项目 | RAG 1.0 | RAG 2.0 |
| --- | --- | --- |
| 数据来源 | 医疗知识库 | 医生病例数据 |
| Excel 结构 | 多字段医学标签 | id + 病史信息 + 医生 |
| 向量化字段 | 多字段组合 | 仅病史信息 |
| 查询目标 | 医学知识匹配 | 病例精准匹配 |
| Embedding 模型 | BGE-large-zh-v1.5 | BGE-large-zh-v1.5 |
| Reranker | BGE-reranker-large | BGE-reranker-large |
| 向量数据库 | Milvus | Milvus |

## 11. 环境依赖

Python >= 3.10

依赖库：

- torch
- transformers
- sentence-transformers
- pymilvus
- fastapi
- uvicorn
- pandas
- openpyxl

安装方式：

```bash
pip install -r requirements.txt
```

## 12. 服务启动

启动命令：

```bash
python main.py
```

### 接口

#### POST /diagnose

请求示例：

```json
{
  "history": "患者咳嗽，伴咽干"
}
```

响应示例：

```json
{
  "id": "001",
  "doctor": "张医生",
  "score": 0.92
}
```

## 13. 后续优化方向

- 增加 Hybrid Retrieval（BM25 + Vector）
- 增加医学实体识别
- 增加医生画像分析
- 支持多轮病史查询
- 接入大语言模型生成诊疗辅助结果
- 支持更大规模病例库扩展

## 14. 总结

MINQUE RAG 2.0 通过将病史文本转化为语义向量，并结合 Milvus 的高效向量检索与 BGE-Reranker 的重排序能力，实现了对医疗病史的精准匹配。该方案适用于基于真实病例数据的语义检索场景，能够有效提高复杂病史表达与医生病例库之间的匹配精度。
