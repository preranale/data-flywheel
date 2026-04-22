# ============================================================
# services/feedback_api/main.py
# ============================================================
# The Feedback API — captures user signals after predictions.
#
# Endpoints:
#   GET  /health              — is the service alive?
#   POST /feedback/rating     — user rates a movie (1-5 stars)
#   POST /feedback/click      — user clicks a movie
#   POST /feedback/skip       — user skips a recommendation
#   GET  /feedback/stats      — how many events in the stream?
#
# Every event goes to Redis Stream → feature_pipeline reads it
# → processed into training data → triggers retraining.
#
# Design decision: We use Pydantic models for ALL request bodies.
# WHY: Pydantic validates types automatically. If someone sends
# rating=6 or user_id="abc", FastAPI rejects it with a clear
# error before our code even runs. No manual validation needed.
# ============================================================

import os
import uuid
import logging
from datetime import datetime
from typing import Optional

import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Redis connection -------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_STREAM = os.getenv("REDIS_STREAM", "feedback_events")

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
)


# ---- Pydantic request models ------------------------------
# These define exactly what shape of data each endpoint accepts.
# FastAPI auto-generates the /docs UI from these definitions.

class RatingFeedback(BaseModel):
    """
    User explicitly rates a movie.
    This is the highest-quality training signal we have.
    """
    user_id: int = Field(..., gt=0, description="User ID (must be positive)")
    movie_id: int = Field(..., gt=0, description="Movie ID (must be positive)")
    rating: float = Field(..., ge=1.0, le=5.0, description="Rating between 1.0 and 5.0")
    # Optional: which model version served this recommendation?
    # Lets us correlate model versions with user satisfaction.
    model_version: Optional[str] = Field(default="unknown")

    @field_validator("rating")
    @classmethod
    def round_rating(cls, v):
        # Round to 1 decimal — avoids 4.123456 in training data
        return round(v, 1)


class ClickFeedback(BaseModel):
    """
    User clicks on a recommended movie.
    Implicit signal — weaker than a rating, but much more common.
    We treat a click as an implicit rating of 3.5.
    """
    user_id: int = Field(..., gt=0)
    movie_id: int = Field(..., gt=0)
    # Position in the recommendation list (1 = top result)
    # WHY track position? Clicking result #1 is less meaningful
    # than clicking result #8 — that's strong positive signal.
    position: Optional[int] = Field(default=1, ge=1, le=20)
    model_version: Optional[str] = Field(default="unknown")


class SkipFeedback(BaseModel):
    """
    User explicitly skips / dismisses a recommendation.
    Negative signal — we'll treat this as an implicit rating of 1.5.
    Skips help the model learn what NOT to recommend.
    """
    user_id: int = Field(..., gt=0)
    movie_id: int = Field(..., gt=0)
    model_version: Optional[str] = Field(default="unknown")


# ---- FastAPI app ------------------------------------------
app = FastAPI(
    title="Data Flywheel — Feedback API",
    description=(
        "Captures user signals (ratings, clicks, skips) and "
        "writes them to the Redis Stream for downstream processing."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Helper -----------------------------------------------

def write_to_stream(event_type: str, payload: dict) -> str:
    """
    Write a feedback event to the Redis Stream.

    WHY Redis Streams over a simple Redis list?
    Streams give us:
    - Persistent log (events aren't deleted after reading)
    - Consumer groups (multiple readers without duplication)
    - Built-in message IDs with timestamps
    - Exactly what Kafka provides, but simpler for our scale

    Returns the Redis message ID on success.
    Raises HTTPException if Redis is unavailable.
    """
    event = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        **{k: str(v) for k, v in payload.items()},
        # Convert all values to strings — Redis Streams
        # only stores string key-value pairs
    }

    try:
        msg_id = redis_client.xadd(REDIS_STREAM, event)
        logger.info(f"Event written to stream: {event_type} | msg_id={msg_id}")
        return msg_id
    except redis.RedisError as e:
        logger.error(f"Redis write failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Feedback stream unavailable. Please retry."
        )


# ---- Endpoints --------------------------------------------

@app.get("/health")
def health():
    """Health check — also verifies Redis connectivity."""
    try:
        redis_client.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "unavailable"

    return {
        "status": "ok",
        "service": "feedback_api",
        "redis": redis_status,
    }


@app.post("/feedback/rating")
def submit_rating(feedback: RatingFeedback):
    """
    User explicitly rates a movie.

    Example request body:
    {
        "user_id": 42,
        "movie_id": 1,
        "rating": 4.5
    }
    """
    msg_id = write_to_stream("explicit_rating", {
        "user_id": feedback.user_id,
        "movie_id": feedback.movie_id,
        "rating": feedback.rating,
        "model_version": feedback.model_version,
    })

    return {
        "accepted": True,
        "event_type": "explicit_rating",
        "msg_id": msg_id,
        "message": f"Rating {feedback.rating} for movie {feedback.movie_id} recorded.",
    }


@app.post("/feedback/click")
def submit_click(feedback: ClickFeedback):
    """
    User clicks a recommended movie.
    We assign it an implicit rating of 3.5 — above neutral,
    below explicitly positive.

    Position matters: clicking rank #5 > clicking rank #1.
    We store position so the feature pipeline can weight it.
    """
    # Implicit rating: base 3.5, bonus if deep in the list
    implicit_rating = 3.5 + (max(0, feedback.position - 1) * 0.1)
    implicit_rating = round(min(implicit_rating, 4.5), 2)

    msg_id = write_to_stream("implicit_click", {
        "user_id": feedback.user_id,
        "movie_id": feedback.movie_id,
        "rating": implicit_rating,
        "position": feedback.position,
        "model_version": feedback.model_version,
    })

    return {
        "accepted": True,
        "event_type": "implicit_click",
        "msg_id": msg_id,
        "implicit_rating": implicit_rating,
    }


@app.post("/feedback/skip")
def submit_skip(feedback: SkipFeedback):
    """
    User skips / dismisses a recommendation.
    Negative signal — implicit rating of 1.5.
    """
    msg_id = write_to_stream("implicit_skip", {
        "user_id": feedback.user_id,
        "movie_id": feedback.movie_id,
        "rating": 1.5,
        "model_version": feedback.model_version,
    })

    return {
        "accepted": True,
        "event_type": "implicit_skip",
        "msg_id": msg_id,
        "implicit_rating": 1.5,
    }


@app.get("/feedback/stats")
def feedback_stats():
    """
    How many feedback events are in the stream?
    Used by the scheduler to decide when to trigger retraining.
    Also useful for monitoring and your demo video.
    """
    try:
        # xlen returns total number of messages in the stream
        total_events = redis_client.xlen(REDIS_STREAM)

        # xinfo_stream gives metadata about the stream
        stream_info = redis_client.xinfo_stream(REDIS_STREAM)

        return {
            "total_events": total_events,
            "first_event_at": stream_info.get("first-entry", [None])[0],
            "last_event_at": stream_info.get("last-entry", [None])[0],
            "stream_name": REDIS_STREAM,
        }
    except redis.ResponseError:
        # Stream doesn't exist yet — no events written
        return {
            "total_events": 0,
            "stream_name": REDIS_STREAM,
            "note": "No events yet — stream will be created on first feedback.",
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
