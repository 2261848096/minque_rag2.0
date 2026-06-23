from pydantic import BaseModel


class DiagnosisRequest(
    BaseModel
):
    complaint: str