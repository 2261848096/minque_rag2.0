from pydantic import BaseModel, Field


class DiagnosisRequest(BaseModel):
    history: str = Field(..., alias="病史信息")
    doctor: str = Field(..., alias="医生")

    class Config:
        populate_by_name = True


class SelectDiagnosisRequest(BaseModel):
    patient_info: str = Field(..., alias="患者信息")
    history: str = Field("", alias="病史信息")

    western_candidates: list = Field(
        default_factory=list,
        alias="西医疾病诊断候选"
    )

    tcm_disease_candidates: list = Field(
        default_factory=list,
        alias="中医疾病诊断候选"
    )

    tcm_syndrome_candidates: list = Field(
        default_factory=list,
        alias="中医证型诊断候选"
    )

    class Config:
        populate_by_name = True