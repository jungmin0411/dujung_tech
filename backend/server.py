"""Real-time weld good/bad inference API for the Dashboard.jsx camera capture flow.

Loads weld_yolov10x_box010_e12.pt once at startup and exposes /predict,
which the frontend calls with a captured camera frame (data URI) and gets
back real detection boxes + a good/bad judgment.
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

from defect_cnn import load_model as load_defect_cnn, predict_defect_type

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
# 결함 세부유형(균열/기공 등) 분류용 CNN — 남우현님이 학습 완료한 실제 가중치(custom_cnn_8class_best.pt).
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
    judgment: Optional[str]
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

    if any(b.cls == "bad" for b in boxes):
        judgment = "bad"
    elif any(b.cls == "good" for b in boxes):
        judgment = "good"
    else:
        judgment = None

    defect_type = None
    if judgment == "bad":
        # 팀원 지침: YOLO와 CNN은 같은 입력 이미지를 병렬로 받는 구조 — YOLO 박스로
        # 재crop하지 않고 원본 프레임을 그대로 CNN에 넣는다.
        defect_type = predict_defect_type(defect_cnn, img)

    return PredictResponse(judgment=judgment, boxes=boxes, processing_time_ms=elapsed_ms, defect_type=defect_type)


@app.get("/health")
def health():
    return {"status": "ok", "classes": model.names}
