import pandas as pd

from src.config import KNOWLEDGE_FILE


class DiseaseLoader:

    @staticmethod
    def load():

        df = pd.read_excel(KNOWLEDGE_FILE)

        df = df.fillna("")

        # 去掉病史信息为空的数据
        df = df[
            df["病史信息"].astype(str).str.strip() != ""
        ]

        print(
            f"✅ 知识库加载完成，共 {len(df)} 条数据"
        )

        return df