import sys
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import logging
import os

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.service.diagnosis_service import DiagnosisService

service = DiagnosisService()

app = FastAPI(
    title="医疗RAG诊断服务",
    description="输入病史信息和医生，进行病例向量检索",
    version="2.0.0"
)


@app.post("/diagnose")
async def diagnose(request: Request):
    try:
        data = await request.json()

        history = data.get("context")
        doctor = data.get("医生")

        if not history or not isinstance(history, str):
            raise HTTPException(400, "缺少有效的病史信息字段")

        if not doctor or not isinstance(doctor, str):
            raise HTTPException(400, "缺少有效的医生字段")

        results = service.diagnose(
            history=history,
            doctor=doctor
        )

        response = {
            "success": True,
            "count": len(results),
            "results": results
        }

        return JSONResponse(response)

    except HTTPException:
        raise

    except Exception as e:
        logging.error(
            f"RAG诊断接口异常: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            500,
            f"服务器内部错误: {str(e)}"
        )


@app.post("/select_diagnosis")
async def select_diagnosis(request: Request):
    """根据患者信息 + 病史信息，从三类候选诊断里各自重排挑出最可能的一个。

    请求体（字段均为中文 key）：
      {
        "患者信息": "一生气就胃疼，反酸打嗝",
        "病史信息": "既往慢性胃炎史",          # 可选
        "西医疾病诊断候选": ["慢性胃炎", "胃溃疡", "功能性消化不良"],
        "中医疾病诊断候选": ["胃脘痛", "嘈杂", "痞满"],
        "中医证型诊断候选": ["肝胃不和证", "脾胃虚寒证", "胃热证"]
      }
    """
    try:
        data = await request.json()

        patient_info = data.get("患者信息")
        history = data.get("病史信息", "")

        western = data.get("西医疾病诊断候选", [])
        tcm_disease = data.get("中医疾病诊断候选", [])
        tcm_syndrome = data.get("中医证型诊断候选", [])

        if not patient_info or not isinstance(patient_info, str):
            raise HTTPException(status_code=400, detail="缺少有效的 患者信息 字段（字符串）")

        if not (western or tcm_disease or tcm_syndrome):
            raise HTTPException(status_code=400, detail="三类候选诊断列表不能全部为空")

        result = service.select_diagnosis(
            patient_info=patient_info,
            history=history,
            western_candidates=western,
            tcm_disease_candidates=tcm_disease,
            tcm_syndrome_candidates=tcm_syndrome,
        )

        return JSONResponse({
            "success": True,
            "results": result
        })

    except HTTPException:
        raise

    except Exception as e:
        logging.error(f"诊断筛选接口异常: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")


# ====================== 启动配置 ======================
if __name__ == "__main__":
    # 配置日志（只输出到文件，不输出到控制台）
    log_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/diagnose.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(log_formatter)

    for logger_name in ["uvicorn.error", "uvicorn.access", "uvicorn"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    # 检测运行环境
    is_docker = os.environ.get("RUNNING_IN_DOCKER") == "1"
    is_debugger = "pydevd" in sys.modules

    server_config = {
        "host": "0.0.0.0",
        "port": 8000,
        "log_config": None,  # 使用我们自定义的日志
    }

    if is_debugger or is_docker:
        server_config.update(reload=False, workers=1)
    else:
        server_config["reload"] = True

    uvicorn.run("run:app", **server_config)
