# ============================================================
# services/inference_api/main.py
# ============================================================
# The Inference API — serves movie recommendations.
#
# Endpoints:
#   GET  /health          — is the service alive?
#   GET  /recommend/{uid} — get recommendations for a user
#   GET  /model/status    — is a model loaded?
#
# Design: This file ONLY handles HTTP concerns — routing,
# request validation, response formatting. Business logic
# lives in model.py. Redis logic lives in its own block.
# ============================================================

import os
import json
import uuid
import logging
from datetime import datetime

import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from model import recommender

# ---- Logging setup ----------------------------------------
# Always configure logging at the top of your entry point.
# Format includes timestamp + level + message for easy debugging.
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
    decode_responses=True,  # Return strings not bytes
)


# ---- App lifecycle ----------------------------------------
# FastAPI's lifespan replaces the old @app.on_event("startup").
# Code before yield runs on startup, after yield on shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: load model into memory
    logger.info("Starting Inference API...")
    recommender.load()
    if recommender.is_ready:
        logger.info("Model loaded and ready")
    else:
        logger.warning("No model found — using fallback recommendations")
    yield
    # SHUTDOWN: nothing to clean up for now
    logger.info("Inference API shutting down")


# ---- FastAPI app ------------------------------------------
app = FastAPI(
    title="Data Flywheel — Inference API",
    description=(
        "Serves movie recommendations and logs predictions "
        "to the feedback loop pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: allows a frontend (React, etc.) to call this API
# In prod you'd restrict origins — for dev we allow all
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Endpoints --------------------------------------------

@app.get("/health")
def health():
    """
    Health check endpoint.
    Docker and load balancers ping this to know if
    the service is alive. Always keep this simple and fast.
    """
    return {"status": "ok", "service": "inference_api"}


@app.get("/model/status")
def model_status():
    """
    Reports whether a trained model is currently loaded.
    Useful for debugging the cold-start state.
    """
    return {
        "model_loaded": recommender.is_ready,
        "model_path": os.getenv("MODEL_PATH", "/data/models"),
    }


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 5):
    """
    Get top-N movie recommendations for a user.

    Args:
        user_id: integer user ID (from the MovieLens dataset)
        n: number of recommendations (default 5, max 20)

    Returns:
        List of recommended movies with predicted ratings.
        Also logs the prediction event to Redis for the
        feedback pipeline to consume.
    """
    if n > 20:
        raise HTTPException(
            status_code=400,
            detail="n cannot exceed 20. Keep responses fast."
        )

    # Get recommendations from the model
    recommendations = recommender.recommend(user_id=user_id, n=n)

    # Log this prediction to Redis Stream
    # WHY: We want to know what we showed the user, so when
    # they click or rate, we can pair action with prediction.
    prediction_event = {
        "event_id": str(uuid.uuid4()),
        "event_type": "prediction_served",
        "user_id": str(user_id),
        "movie_ids": json.dumps([r["movie_id"] for r in recommendations]),
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": "fallback" if not recommender.is_ready else "v1",
    }

    try:
        redis_client.xadd(REDIS_STREAM, prediction_event)
        logger.info(f"Prediction logged for user {user_id}")
    except Exception as e:
        # Never let Redis failure break the API response!
        # Log the error but still return recommendations.
        logger.error(f"Failed to log prediction to Redis: {e}")

    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "model_ready": recommender.is_ready,
        "served_at": datetime.utcnow().isoformat(),
    }


@app.post("/model/reload")
def reload_model():
    """
    Hot-reload the model from disk without restarting the API.
    
    The scheduler calls this after a successful training run
    so users immediately get the new model.
    
    WHY POST not GET: This endpoint changes server state
    (swaps the model). HTTP convention says state-changing
    actions use POST/PUT, not GET.
    """
    logger.info("Model reload requested")
    success = recommender.load()
    return {
        "success": success,
        "model_ready": recommender.is_ready,
        "reloaded_at": datetime.utcnow().isoformat(),
    }
