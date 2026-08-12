from src.retrieval.retriever import Retriever
from src.retrieval.reranker import MedicalReranker
from src.config import TOP_K_FINAL


class DiagnosisService:

    def __init__(self):

        self.retriever = Retriever()

        self.reranker = MedicalReranker()

        self._warmup()


    def _warmup(self):

        try:

            dummy = "预热测试：患者头痛发热，咳嗽"

            self.retriever.embedding_model.embed_query(
                dummy
            )

            self.reranker.model.predict(
                [[dummy, dummy]],
                show_progress_bar=False
            )

            print("✅ 模型预热完成")


        except Exception as e:

            print(f"⚠️ 预热跳过: {e}")



    # ==========================
    # RAG病例检索
    # ==========================

    def diagnose(
        self,
        history: str,
        doctor: str = None
    ):

        """
        输入:
            history:
                病史信息

            doctor:
                医生

        流程:
            医生metadata过滤
            +
            病史embedding检索
        """


        recall_results = self.retriever.search(
            query=history,
            doctor=doctor
        )


        rerank_results = self.reranker.rerank(
            history,
            recall_results
        )


        final_results = []


        for item in rerank_results[:TOP_K_FINAL]:

            entity = item["data"]["entity"]


            final_results.append(
                {
                    "id": entity["id"],

                    "score": round(
                        item["score"],
                        4
                    )
                }
            )


        return final_results



    # ==========================
    # 候选诊断重排
    # ==========================

    @staticmethod
    def _build_query(
        patient_info: str,
        history: str
    ) -> str:

        parts = [
            p.strip()
            for p in (
                patient_info,
                history
            )
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

        """
        根据患者信息+病史
        从候选诊断中选择最高匹配结果
        """


        query = self._build_query(
            patient_info,
            history
        )


        def pick(candidates):

            best, score = self.reranker.pick_best(
                query,
                candidates
            )


            if best is None:
                return None


            return {
                "诊断": best,
                "分数": round(score,4)
            }



        return {

            "西医疾病诊断":
                pick(western_candidates),

            "中医疾病诊断":
                pick(tcm_disease_candidates),

            "中医证型诊断":
                pick(tcm_syndrome_candidates)

        }