from src.models.embedding_manager import (
    get_reranker_model
)

from src.config import (
    RERANK_BATCH_SIZE
)


class MedicalReranker:

    def __init__(self):

        self.model = (
            get_reranker_model()
        )

    def rerank(
            self,
            query,
            candidates
    ):

        pairs = []

        for item in candidates:

            history = (
                item["entity"]["history"]
            )

            pairs.append(
                [query, history]
            )

        scores = (
            self.model.predict(
                pairs,
                # 50 条候选成批喂给 GPU，吃满吞吐，远快于小批多次
                batch_size=RERANK_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        )

        rerank_results = []

        for item, score in zip(
                candidates,
                scores
        ):
            rerank_results.append(
                {
                    "score": float(score),
                    "data": item
                }
            )

        rerank_results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return rerank_results

    def pick_best(
            self,
            query,
            candidates
    ):
        """从一组纯文本候选里，用 reranker 打分挑出最匹配的一个。

        返回 (最佳候选文本, 分数)；候选为空时返回 (None, None)。
        """

        if not candidates:
            return None, None

        pairs = [
            [query, str(c)]
            for c in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=RERANK_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        best_idx = int(scores.argmax())

        return (
            candidates[best_idx],
            float(scores[best_idx])
        )