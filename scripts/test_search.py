import sys
import time
from pathlib import Path

ROOT = (
    Path(__file__)
    .resolve()
    .parent.parent
)

sys.path.append(
    str(ROOT)
)

import torch

from src.service.diagnosis_service import (
    DiagnosisService
)

print(
    f"设备: {'cuda' if torch.cuda.is_available() else 'cpu'} | "
    f"torch={torch.__version__}"
)

service = DiagnosisService()

query = """
患者:杨贵华，女，53。因“反复颈部疼痛半年，
加重5+天”由门诊于2017-01-30以“颈椎病”收入住院。
■■病例特点:■■1.患者中年女性，病程长,发病缓.
■■2.现病史半年前患者因长期伏案工作出现颈部疼痛不适，并伴有左侧肩背部放射性疼痛，自觉左侧肩部乏力，时伴有头昏，
活动后加重，患者未予重视，未予特殊处理。5+天前，患者无明显诱因出现颈部疼痛明显，颈椎活动受限，伴头昏，
左肩背部放射性疼痛加重，起床或躺下时明显，今为求进一步治疗来我院就诊，门诊以“颈椎病”收入我科。入院症见:患者神清，
左侧颈项及肩部疼痛，伴头昏，活动后加重。纳可，眠差，二便调。■■3.中医望闻、切、诊:神清，形体适中，左侧颈项部及肩部疼痛，伴头昏，
纳可，二便调，舌淡，苔薄白，脉弱。
"""

# 跑多次取稳定耗时（warmup 已在 service 初始化时完成，首次也是热的）
RUNS = 3

results = None

print("\n" + "=" * 50)
print("耗时测试")
print("=" * 50)

for run in range(1, RUNS + 1):

    t0 = time.perf_counter()

    results = service.diagnose(query)

    elapsed = time.perf_counter() - t0

    print(
        f"第 {run} 次: {elapsed * 1000:.1f} ms | "
        f"返回 {len(results)} 条"
    )

print("\n" + "=" * 50)
print("结果（Top10 预览）")
print("=" * 50)

for i, item in enumerate(
        results[:10],
        start=1
):

    print(
        f"Top{i}"
    )

    print(item)

    print("-" * 50)