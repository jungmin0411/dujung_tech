"""Real-time weld inspection API for the Dashboard.jsx camera capture flow.

YOLO는 용접부 위치(박스) 탐지 용도로만 쓰고, YOLO 자체의 good/bad 분류는 무시한다.
양품/불량 판정과 불량 세부유형 분류는 전부 defect_cnn.classify()가 전담한다.
"""
import base64
import io
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

from defect_cnn import load_model as load_defect_cnn, classify as cnn_classify

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "weld_yolov10x_box010_e12_deploy"
WEIGHTS = PACKAGE_ROOT / "model" / "weld_yolov10x_box010_e12.pt"
DEFECT_CNN_WEIGHTS = Path(__file__).resolve().parent / "defect_cnn_weights.pt"
CONF = 0.60
IOU = 0.70
IMGSZ = 640

app = FastAPI(title="Weld Inspection AI Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dujungtech123.jim0411.workers.dev"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(str(WEIGHTS))
# 양품/불량 판정 + 결함 세부유형 분류용 CNN — 남우현님이 학습 완료한 실제 가중치(custom_cnn_8class_best.pt).
defect_cnn = load_defect_cnn(str(DEFECT_CNN_WEIGHTS))


class PredictRequest(BaseModel):
    image: str  # "data:image/jpeg;base64,...."


class Box(BaseModel):
    cls: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


class PredictResponse(BaseModel):
    judgment: Optional[str]  # CNN이 판정 (good/bad), YOLO 박스가 하나도 없으면 None
    boxes: List[Box]
    processing_time_ms: int
    defect_type: Optional[str] = None  # judgment가 "bad"일 때만 CNN이 채움


def decode_data_uri(data_uri: str) -> Image.Image:
    _, b64data = data_uri.split(",", 1)
    raw = base64.b64decode(b64data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    img = decode_data_uri(req.image)
    t0 = time.time()
    results = model.predict(
        source=np.array(img), imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False,
    )
    elapsed_ms = int((time.time() - t0) * 1000)

    boxes: List[Box] = []
    r = results[0]
    for b in r.boxes:
        cls_idx = int(b.cls[0])
        x1, y1, x2, y2 = (float(v) for v in b.xyxyn[0])
        boxes.append(Box(
            cls=model.names[cls_idx], conf=float(b.conf[0]),
            x1=x1, y1=y1, x2=x2, y2=y2,
        ))

    # YOLO는 이제 용접부가 화면에 있는지(=박스 존재 여부)와 위치만 보고, good/bad
    # 판정 자체는 사용하지 않는다. 박스가 하나라도 있으면 CNN이 같은 원본 프레임을
    # 그대로 받아 양품/불량 + 세부유형을 전담 판정한다 (팀원 지침: YOLO 박스로
    # 재crop하지 않고 원본 이미지를 병렬로 넣는다).
    if boxes:
        judgment, defect_type = cnn_classify(defect_cnn, img)
        for b in boxes:
            b.cls = judgment
    else:
        judgment = None
        defect_type = None

    return PredictResponse(judgment=judgment, boxes=boxes, processing_time_ms=elapsed_ms, defect_type=defect_type)


@app.get("/health")
def health():
    return {"status": "ok", "classes": model.names}
