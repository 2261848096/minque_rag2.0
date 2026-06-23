from src.retrieval.retriever import (
    Retriever
)

from src.retrieval.reranker import (
    MedicalReranker
)

from src.config import (
    TOP_K_FINAL
)


class DiagnosisService:

    def __init__(self):

        self.retriever = Retriever()

        self.reranker = (
            MedicalReranker()
        )

        # 启动预热：GPU 首次推理要做 cuDNN/kernel 初始化，
        # 不预热则线上第一条真实请求会异常慢（3-5s）。
        # 这里把这部分代价提前到启动期。
        self._warmup()

    def _warmup(self):
        try:
            dummy = "预热测试：患者头痛发热，咳嗽"

            # 预热 embedding（触发向量化前向）
            self.retriever.embedding_model.embed_query(dummy)

            # 预热 reranker（触发 CrossEncoder 前向 / GPU kernel 编译）
            self.reranker.model.predict(
                [[dummy, dummy]],
                show_progress_bar=False,
            )

            print("✅ 模型预热完成")

        except Exception as e:
            print(f"⚠️ 预热跳过: {e}")

    def diagnose(
            self,
            complaint: str
    ):

        # Top10召回
        recall_results = (
            self.retriever.search(
                complaint
            )
        )

        # 重排序
        rerank_results = (
            self.reranker.rerank(
                complaint,
                recall_results
            )
        )

        final_results = []

        for item in rerank_results[
                    :TOP_K_FINAL]:

            entity = item[
                "data"
            ]["entity"]

            final_results.append(
                {
                    "id":
                    entity["id"],

                    "score":
                    round(
                        item["score"],
                        4
                    )
                }
            )

        return final_results

    @staticmethod
    def _build_query(
            patient_info: str,
            history: str
    ) -> str:
        """把患者信息 + 病史信息拼成一段查询文本。"""

        parts = [
            p.strip()
            for p in (patient_info, history)
            if p and p.strip()
        ]

        return "\n".join(parts)

    def select_diagnosis(
            self,
            patient_info: str,
            history: str,
            western_candidates: list,
            tcm_disease_candidates: list,
            tcm_syndrome_candidates: list,
    ):
        """根据患者信息 + 病史信息，从三类候选诊断里各自重排挑出最可能的一个。

        candidates 均为纯文本字符串列表（如 ["颈椎病", "肩周炎"]）。
        """

        query = self._build_query(
            patient_info,
            history
        )

        def pick(candidates):
            best, score = (
                self.reranker.pick_best(
                    query,
                    candidates
                )
            )

            if best is None:
                return None

            return {
                "诊断": best,
                "分数": round(score, 4),
            }

        return {
            "西医疾病诊断":
                pick(western_candidates),

            "中医疾病诊断":
                pick(tcm_disease_candidates),

            "中医证型诊断":
                pick(tcm_syndrome_candidates),
        }