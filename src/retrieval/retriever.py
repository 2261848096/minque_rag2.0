from src.models.embedding_manager import (
    get_embedding_model
)

from src.vectorstore.milvus_manager import (
    MilvusManager
)

from src.config import (
    TOP_K_RECALL,
    MILVUS_COLLECTION_NAME
)


class Retriever:

    def __init__(self):

        self.embedding_model = (
            get_embedding_model()
        )

        self.milvus = MilvusManager()

    def search(
            self,
            query: str,
            top_k: int = TOP_K_RECALL
    ):

        query_vector = (
            self.embedding_model
            .embed_query(query)
        )

        results = self.milvus.client.search(
            collection_name=
            MILVUS_COLLECTION_NAME,

            data=[query_vector],

            limit=top_k,

            output_fields=[
                "id",
                "history"
            ]
        )

        return results[0]