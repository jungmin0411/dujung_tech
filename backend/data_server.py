"""Inspection record storage API — saves/reads/deletes results in Postgres.

Separate from server.py (AI 추론 전용) on purpose: this server doesn't need a
GPU or the YOLO model at all, so it can run anywhere (local, cloud) while the
AI server runs wherever the model/GPU actually lives (VDI 등).
"""
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import delete, insert, select

from db import engine, init_db, inspections

app = FastAPI(title="Weld Inspection Data API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dujungtech123.jim0411.workers.dev"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()


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


@app.get("/health")
def health():
    return {"status": "ok"}


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
