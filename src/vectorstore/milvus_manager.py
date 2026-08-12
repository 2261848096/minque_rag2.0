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

        print(
            "✅ Milvus Lite连接成功"
        )


    def create_collection(self):

        if self.client.has_collection(
            collection_name=MILVUS_COLLECTION_NAME
        ):

            print(
                "✅ Collection已存在"
            )
            return


        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_fields=False
        )


        # 主键ID
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


        # 医生
        schema.add_field(
            field_name="doctor",
            datatype=DataType.VARCHAR,
            max_length=200
        )


        # BGE向量
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=MILVUS_DIMENSION
        )


        # 创建索引
        index_params = (
            self.client
            .prepare_index_params()
        )


        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="HNSW",
            params={
                "M": 32,
                "efConstruction": 200
            }
        )


        self.client.create_collection(
            collection_name=MILVUS_COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )


        print(
            "✅ Collection创建成功"
        )


    def drop_collection(self):

        if self.client.has_collection(
            collection_name=MILVUS_COLLECTION_NAME
        ):

            self.client.drop_collection(
                collection_name=MILVUS_COLLECTION_NAME
            )

            print(
                "🗑️ Collection已删除"
            )


        else:

            print(
                "ℹ️ Collection不存在，无需删除"
            )


    def insert(self, data):

        self.client.insert(
            collection_name=MILVUS_COLLECTION_NAME,
            data=data
        )

        print(
            f"✅ 已插入 {len(data)} 条数据"
        )


    def count(self):

        try:

            result = self.client.query(
                collection_name=MILVUS_COLLECTION_NAME,
                output_fields=["count(*)"]
            )

            return result


        except Exception as e:

            print(
                f"统计失败: {e}"
            )

            return None