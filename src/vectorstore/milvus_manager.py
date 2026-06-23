from pymilvus import (
    MilvusClient,
    DataType
)

from src.config import (
    MILVUS_URI,
    MILVUS_COLLECTION_NAME,
    MILVUS_DIMENSION
)


class MilvusManager:

    def __init__(self):
        self.client = MilvusClient(
            uri=MILVUS_URI
        )

        print("✅ Milvus连接成功")

    def create_collection(self):

        if self.client.has_collection(
                collection_name=MILVUS_COLLECTION_NAME):

            print("✅ Collection已存在")
            return

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_fields=False
        )

        # 主键
        schema.add_field(
            field_name="id",
            datatype=DataType.INT64,
            is_primary=True
        )

        # 病史信息
        schema.add_field(
            field_name="history",
            datatype=DataType.VARCHAR,
            max_length=12000
        )

        # 病史要点
        schema.add_field(
            field_name="history_points",
            datatype=DataType.VARCHAR,
            max_length=500
        )

        # 向量字段
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=MILVUS_DIMENSION
        )

        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="AUTOINDEX"
        )

        self.client.create_collection(
            collection_name=MILVUS_COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

        print("✅ Collection创建成功")

    def insert(self, data):

        self.client.insert(
            collection_name=MILVUS_COLLECTION_NAME,
            data=data
        )

        print(f"✅ 已插入 {len(data)} 条数据")

    def drop_collection(self):

        if self.client.has_collection(
                collection_name=MILVUS_COLLECTION_NAME):

            self.client.drop_collection(
                collection_name=MILVUS_COLLECTION_NAME
            )

            print("🗑️ 旧Collection已删除")

    def count(self):

        try:
            stats = self.client.get_collection_stats(
                collection_name=MILVUS_COLLECTION_NAME
            )

            return stats

        except Exception as e:

            print(f"统计失败: {e}")

            return None