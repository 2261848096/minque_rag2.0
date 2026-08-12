import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT))

from src.loader.disease_loader import (
    DiseaseLoader
)

from src.models.embedding_manager import (
    get_embedding_model
)

from src.vectorstore.milvus_manager import (
    MilvusManager
)

BATCH_SIZE = 100


def build_index():

    print("=" * 60)
    print("开始构建医疗知识库")
    print("=" * 60)

    # 读取数据
    df = DiseaseLoader.load()

    # Embedding模型
    embedding_model = (
        get_embedding_model()
    )

    # Milvus
    milvus = MilvusManager()

    # 删除旧库
    milvus.drop_collection()

    # 创建新库
    milvus.create_collection()

    total = len(df)

    records = []

    print("\n开始向量化...")

    for idx, row in df.iterrows():

        try:

            history = str(
                row["病史信息"]
            ).strip()

            vector = (
                embedding_model
                .embed_query(history)
            )

            record = {

                "id":
                int(row["id"]),

                "history":
                history,

                "doctor":
                    str(
                        row["医生"]
                    ).strip(),

                "vector":
                vector
            }

            records.append(record)

            if len(records) >= BATCH_SIZE:

                milvus.insert(records)

                print(
                    f"已完成 {idx+1}/{total}"
                )

                records.clear()

        except Exception as e:

            print(
                f"第{idx+1}条数据异常：{e}"
            )

    # 插入剩余数据
    if records:

        milvus.insert(records)

    print("\n知识库构建完成")

    print(
        milvus.count()
    )


if __name__ == "__main__":

    build_index()