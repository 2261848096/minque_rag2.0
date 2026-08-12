from src.models.embedding_manager import get_embedding_model
from src.vectorstore.milvus_manager import MilvusManager
from src.config import TOP_K_RECALL, MILVUS_COLLECTION_NAME


class Retriever:

    def __init__(self):

        self.embedding_model = get_embedding_model()

        self.milvus = MilvusManager()


    def search(
        self,
        query: str,
        doctor: str = None,
        top_k: int = TOP_K_RECALL
    ):

        # 病史信息向量化
        query_vector = (
            self.embedding_model
            .embed_query(query)
        )


        search_params = {

            "collection_name":
                MILVUS_COLLECTION_NAME,

            "data":[query_vector],

            "limit":top_k,

            "output_fields":[
                "id",
                "history",
                "doctor"
            ]
        }


        # ==========================
        # 医生过滤
        # ==========================

        if doctor:

            search_params["filter"] = (
                f'doctor == "{doctor}"'
            )


        results = self.milvus.client.search(
            **search_params
        )


        return results[0]