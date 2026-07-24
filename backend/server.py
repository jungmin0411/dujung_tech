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
from sqlalchemy import delete, insert, select
from ultralytics import YOLO

from db import engine, init_db, inspections

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "weld_yolov10x_box010_e12_deploy"
WEIGHTS = PACKAGE_ROOT / "model" / "weld_yolov10x_box010_e12.pt"
CONF = 0.60
IOU = 0.70
IMGSZ = 640

app = FastAPI(title="Weld Inspection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(str(WEIGHTS))
init_db()


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

    return PredictResponse(judgment=judgment, boxes=boxes, processing_time_ms=elapsed_ms)


@app.get("/health")
def health():
    return {"status": "ok", "classes": model.names}


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class InspectionIn(BaseModel):
    id: str
    session_id: str
    part: str
    ai_judgment: str
    defect_type: Optional[str] = None
    confidence: float
    processing_time_ms: int
    timestamp: str  # 프론트가 만든 로컬 시각 문자열 그대로 저장 (서버에서 타임존 재계산 안 함)
    bbox: BBox
    image_data_uri: str


def _row_to_inspection(row) -> dict:
    return {
        "id": row.id,
        "session_id": row.session_id,
        "part": row.part,
        "ai_judgment": row.ai_judgment,
        "defect_type": row.defect_type,
        "confidence": row.confidence,
        "processing_time_ms": row.processing_time_ms,
        "timestamp": row.timestamp,
        "bbox": {"x": row.bbox_x, "y": row.bbox_y, "w": row.bbox_w, "h": row.bbox_h},
        "image_data_uri": row.image_data_uri,
    }


@app.post("/inspections", response_model=InspectionIn)
def create_inspection(item: InspectionIn) -> InspectionIn:
    with engine.begin() as conn:
        conn.execute(insert(inspections).values(
            id=item.id, session_id=item.session_id, part=item.part,
            ai_judgment=item.ai_judgment, defect_type=item.defect_type,
            confidence=item.confidence, processing_time_ms=item.processing_time_ms,
            timestamp=item.timestamp,
            bbox_x=item.bbox.x, bbox_y=item.bbox.y, bbox_w=item.bbox.w, bbox_h=item.bbox.h,
            image_data_uri=item.image_data_uri,
        ))
    return item


@app.get("/inspections")
def list_inspections(start_date: str, end_date: str) -> List[dict]:
    end_bound = f"{end_date}T23:59:59"
    with engine.connect() as conn:
        rows = conn.execute(
            select(inspections)
            .where(inspections.c.timestamp >= start_date, inspections.c.timestamp <= end_bound)
            .order_by(inspections.c.timestamp.desc())
        ).fetchall()
    return [_row_to_inspection(r) for r in rows]


@app.delete("/inspections/{inspection_id}")
def delete_inspection(inspection_id: str) -> dict:
    with engine.begin() as conn:
        conn.execute(delete(inspections).where(inspections.c.id == inspection_id))
    return {"status": "deleted", "id": inspection_id}
