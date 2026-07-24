"""PostgreSQL persistence for inspection records (검사 상세/대시보드/리포트가 여기서 읽어감)."""
import os

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, Text, create_engine

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://localhost/dujungtech")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
metadata = MetaData()

# timestamp은 프론트가 만든 로컬 시각 문자열("YYYY-MM-DDTHH:MM:SS")을 그대로 저장한다 —
# 서버에서 타임존을 다시 계산하지 않고, 화면에 보여줄 시각 그대로를 신뢰한다.
inspections = Table(
    "inspections", metadata,
    Column("id", String, primary_key=True),
    Column("session_id", String, nullable=False, index=True),
    Column("part", String, nullable=False),
    Column("ai_judgment", String, nullable=False),
    Column("defect_type", String, nullable=True),
    Column("confidence", Float, nullable=False),
    Column("processing_time_ms", Integer, nullable=False),
    Column("timestamp", String, nullable=False, index=True),
    Column("bbox_x", Float, nullable=False),
    Column("bbox_y", Float, nullable=False),
    Column("bbox_w", Float, nullable=False),
    Column("bbox_h", Float, nullable=False),
    Column("image_data_uri", Text, nullable=False),
)


def init_db():
    metadata.create_all(engine)
