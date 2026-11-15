"""Log predictions to a database using SQLAlchemy.

Default: a local SQLite file (predictions.db), so nothing else needs to be installed.
To use PostgreSQL, set the DATABASE_URL environment variable, e.g.
    DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/hdp
"""
import os
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.data import ROOT_DIR

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT_DIR / 'predictions.db'}")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(64), nullable=False)
    inputs: Mapped[dict] = mapped_column(JSON, nullable=False)


Base.metadata.create_all(engine)


def save_prediction(inputs, probability, risk_level):
    with SessionLocal() as session:
        session.add(Prediction(inputs=inputs, probability=probability, risk_level=risk_level))
        session.commit()


def load_history(limit=500):
    """Return the most recent predictions as a list of dicts, newest first."""
    with SessionLocal() as session:
        rows = session.scalars(
            select(Prediction).order_by(Prediction.created_at.desc(), Prediction.id.desc()).limit(limit)
        ).all()
        return [
            {
                "saved_at_utc": row.created_at.strftime("%Y-%m-%d %H:%M"),
                "risk_level": row.risk_level,
                "probability": round(row.probability, 3),
                **row.inputs,
            }
            for row in rows
        ]
