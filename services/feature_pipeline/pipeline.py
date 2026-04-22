# ============================================================
# services/feature_pipeline/pipeline.py
# ============================================================
# Background worker that transforms raw feedback events into
# clean training data.
#
# Flow:
#   1. On first run: download MovieLens dataset as base data
#   2. Every 20 seconds: read new events from Redis Stream
#   3. Clean + deduplicate events
#   4. Merge with existing training data
#   5. Write updated CSV to /data/processed/
#
# WHY a separate worker instead of doing this in the Feedback API?
# The Feedback API must respond in milliseconds — it can't afford
# to do file I/O, dataset merging, or deduplication on every request.
# Offloading to a worker decouples speed from correctness.
# ============================================================

import os
import time
import logging
import zipfile
import requests
import pandas as pd
import numpy as np
import redis

from datetime import datetime
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Config -----------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_STREAM = os.getenv("REDIS_STREAM", "feedback_events")
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "/data/raw")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "/data/processed")
MODEL_PATH = os.getenv("MODEL_PATH", "/data/models")
POLL_INTERVAL = 20   # seconds between each pipeline run
STREAM_BATCH = 100   # how many Redis events to read per run

# Paths for output files
TRAIN_CSV = os.path.join(PROCESSED_DATA_PATH, "train.csv")
VAL_CSV   = os.path.join(PROCESSED_DATA_PATH, "val.csv")
MOVIES_CSV = os.path.join(PROCESSED_DATA_PATH, "movies.csv")

# Redis Stream consumer tracking
# We use a "last seen" ID so we never re-process old events.
# "0" means "start from the very beginning of the stream"
LAST_ID_FILE = os.path.join(RAW_DATA_PATH, "last_stream_id.txt")

# ---- Redis client -----------------------------------------
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
)


# ===========================================================
# SECTION 1: Dataset bootstrap
# ===========================================================

def download_movielens():
    """
    Download the MovieLens ml-latest-small dataset.

    MovieLens is a free, real-world movie rating dataset
    from the University of Minnesota. It has:
    - 100,000 ratings from 600 users across 9,000 movies
    - Perfect size for a portfolio project (fast to train)
    - Well known — any ML interviewer will recognise it

    We only download once — if files already exist, skip.
    """
    ratings_file = os.path.join(RAW_DATA_PATH, "ratings.csv")
    movies_file  = os.path.join(RAW_DATA_PATH, "movies.csv")

    if os.path.exists(ratings_file) and os.path.exists(movies_file):
        logger.info("MovieLens dataset already exists — skipping download")
        return

    logger.info("Downloading MovieLens ml-latest-small dataset...")
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        zip_path = os.path.join(RAW_DATA_PATH, "movielens.zip")
        with open(zip_path, "wb") as f:
            f.write(response.content)

        # Unzip and move the files we need
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extract("ml-latest-small/ratings.csv", RAW_DATA_PATH)
            z.extract("ml-latest-small/movies.csv",  RAW_DATA_PATH)

        # Move from subfolder to raw root
        os.rename(
            os.path.join(RAW_DATA_PATH, "ml-latest-small", "ratings.csv"),
            ratings_file,
        )
        os.rename(
            os.path.join(RAW_DATA_PATH, "ml-latest-small", "movies.csv"),
            movies_file,
        )

        # Cleanup
        os.remove(zip_path)
        os.rmdir(os.path.join(RAW_DATA_PATH, "ml-latest-small"))

        logger.info(f"Dataset downloaded: {ratings_file}")

    except Exception as e:
        logger.error(f"Failed to download MovieLens: {e}")
        raise


def build_base_dataset():
    """
    Turn raw MovieLens ratings into train/val splits.

    Schema of output CSVs:
        user_id  | movie_id | rating | source
        ---------|----------|--------|--------
        1        | 296      | 4.0    | movielens
        42       | 1        | 3.5    | feedback

    The 'source' column tells us where this row came from.
    WHY: During evaluation we might want to test only on
    feedback-sourced data to see if the model is improving
    on real user behaviour, not just the original dataset.
    """
    ratings_file = os.path.join(RAW_DATA_PATH, "ratings.csv")
    movies_file  = os.path.join(RAW_DATA_PATH, "movies.csv")

    logger.info("Building base dataset from MovieLens ratings...")

    # Load raw data
    ratings = pd.read_csv(ratings_file)
    movies  = pd.read_csv(movies_file)

    # Keep only the columns we need
    # MovieLens also has 'timestamp' — we drop it for simplicity
    ratings = ratings[["userId", "movieId", "rating"]].rename(columns={
        "userId":  "user_id",
        "movieId": "movie_id",
    })
    ratings["source"] = "movielens"

    # Keep only movies that have at least 10 ratings
    # WHY: Rare movies make the model unstable — not enough
    # signal to learn a reliable embedding for them
    movie_counts = ratings["movie_id"].value_counts()
    popular_movies = movie_counts[movie_counts >= 10].index
    ratings = ratings[ratings["movie_id"].isin(popular_movies)]

    logger.info(f"Base dataset: {len(ratings)} ratings, "
                f"{ratings['user_id'].nunique()} users, "
                f"{ratings['movie_id'].nunique()} movies")

    # Train/val split (80/20)
    # stratify=None — random split is fine for this project
    train, val = train_test_split(ratings, test_size=0.2, random_state=42)

    # Save
    train.to_csv(TRAIN_CSV, index=False)
    val.to_csv(VAL_CSV, index=False)

    # Save movies lookup (used by inference_api for titles)
    movies = movies.rename(columns={"movieId": "movie_id"})
    movies = movies[movies["movie_id"].isin(popular_movies)]
    movies.to_csv(MOVIES_CSV, index=False)

    logger.info(f"Saved: {TRAIN_CSV} ({len(train)} rows), "
                f"{VAL_CSV} ({len(val)} rows)")


# ===========================================================
# SECTION 2: Feedback event processing
# ===========================================================

def get_last_stream_id() -> str:
    """
    Read the last processed Redis Stream ID from disk.
    This is our cursor — we only process NEW events each run.

    WHY file-based cursor instead of Redis?
    If the pipeline crashes, we want to resume from where we
    left off — even if Redis restarts. A file survives container
    restarts as long as it's in a mounted volume (/data/raw).
    """
    if os.path.exists(LAST_ID_FILE):
        with open(LAST_ID_FILE, "r") as f:
            return f.read().strip()
    return "0"  # Start from beginning if no cursor exists


def save_last_stream_id(msg_id: str):
    """Persist the cursor so we resume correctly after restarts."""
    with open(LAST_ID_FILE, "w") as f:
        f.write(msg_id)


def read_new_events() -> list[dict]:
    """
    Read new events from Redis Stream since our last cursor.

    XREAD COUNT N STREAMS stream_name last_id
    - COUNT N: read at most N messages per call
    - last_id: only return messages NEWER than this ID
    """
    last_id = get_last_stream_id()

    try:
        results = redis_client.xread(
            {REDIS_STREAM: last_id},
            count=STREAM_BATCH,
        )
    except redis.ResponseError:
        # Stream doesn't exist yet
        return []

    if not results:
        return []

    events = []
    latest_id = last_id

    for stream_name, messages in results:
        for msg_id, fields in messages:
            events.append({**fields, "_redis_id": msg_id})
            latest_id = msg_id

    # Save cursor so next run skips these events
    if events:
        save_last_stream_id(latest_id)
        logger.info(f"Read {len(events)} new events from stream")

    return events


def events_to_dataframe(events: list[dict]) -> pd.DataFrame:
    """
    Convert raw Redis event dicts into a clean DataFrame.

    Redis stores everything as strings — we need to cast
    types back to int/float for training.
    """
    if not events:
        return pd.DataFrame()

    rows = []
    for event in events:
        event_type = event.get("event_type", "")

        # Only process feedback events, skip prediction_served logs
        if event_type not in ("explicit_rating", "implicit_click", "implicit_skip"):
            continue

        try:
            rows.append({
                "user_id":  int(event["user_id"]),
                "movie_id": int(event["movie_id"]),
                "rating":   float(event["rating"]),
                "source":   event_type,
                "timestamp": event.get("timestamp", ""),
            })
        except (KeyError, ValueError) as e:
            # Malformed event — log and skip
            logger.warning(f"Skipping malformed event: {e} | {event}")
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info(f"Parsed {len(df)} valid feedback events")
    return df


def clean_feedback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and deduplicate feedback events.

    Rules:
    1. Drop duplicate (user_id, movie_id) pairs — keep the latest
       WHY: If a user clicks then rates the same movie, we want
       only the explicit rating (higher quality signal)
    2. Remove ratings outside [1.0, 5.0]
    3. Keep only users and movies that exist in our base dataset
       WHY: Unknown IDs can't be used by the matrix factorisation model
    """
    if df.empty:
        return df

    # Sort by timestamp so "keep last" keeps the most recent
    df = df.sort_values("timestamp")

    # Priority: explicit_rating > implicit_click > implicit_skip
    # Achieved by sorting source values in that priority order
    priority = {"explicit_rating": 0, "implicit_click": 1, "implicit_skip": 2}
    df["priority"] = df["source"].map(priority).fillna(99)
    df = df.sort_values(["user_id", "movie_id", "priority"])

    # Keep only the highest-priority event per (user, movie) pair
    df = df.drop_duplicates(subset=["user_id", "movie_id"], keep="first")
    df = df.drop(columns=["priority", "timestamp"])

    # Validate rating range
    df = df[df["rating"].between(1.0, 5.0)]

    # Filter to known users and movies
    if os.path.exists(TRAIN_CSV):
        train = pd.read_csv(TRAIN_CSV)
        known_users  = set(train["user_id"].unique())
        known_movies = set(train["movie_id"].unique())

        before = len(df)
        df = df[
            df["user_id"].isin(known_users) &
            df["movie_id"].isin(known_movies)
        ]
        dropped = before - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} events with unknown user/movie IDs")

    logger.info(f"Clean feedback: {len(df)} rows after deduplication")
    return df


def merge_into_training_data(new_feedback: pd.DataFrame):
    """
    Append cleaned feedback events to the training CSV.

    Strategy: feedback events AUGMENT, not replace, the base data.
    WHY: The base MovieLens data is high quality and large.
    New feedback is small and noisy initially. Replacing would
    make the model worse before it gets better (cold start problem).

    As feedback volume grows, its influence naturally increases
    because it becomes a larger fraction of total training data.
    """
    if new_feedback.empty:
        logger.info("No new feedback to merge")
        return

    if not os.path.exists(TRAIN_CSV):
        logger.warning("No training CSV found — run base dataset build first")
        return

    train = pd.read_csv(TRAIN_CSV)
    original_size = len(train)

    # Append new feedback
    train = pd.concat([train, new_feedback], ignore_index=True)

    # Deduplicate again at the full dataset level
    # If feedback contradicts base data, feedback wins (keep last)
    train = train.drop_duplicates(subset=["user_id", "movie_id"], keep="last")

    train.to_csv(TRAIN_CSV, index=False)
    added = len(train) - original_size
    logger.info(
        f"Training data updated: {original_size} → {len(train)} rows "
        f"(+{added} new, {original_size - len(train) + added} replaced)"
    )


# ===========================================================
# SECTION 3: Main loop
# ===========================================================

def run_pipeline_once():
    """
    Single pipeline run:
    1. Read new Redis events
    2. Parse → clean → merge into training data
    """
    logger.info("--- Pipeline run starting ---")

    events = read_new_events()
    if not events:
        logger.info("No new events — nothing to process")
        return

    df = events_to_dataframe(events)
    df = clean_feedback(df)
    merge_into_training_data(df)

    logger.info("--- Pipeline run complete ---")


def main():
    """
    Entry point — runs forever with a sleep between each cycle.

    Startup sequence:
    1. Wait for Redis to be ready
    2. Download MovieLens dataset (first run only)
    3. Build base train/val split (first run only)
    4. Loop: process new feedback events every POLL_INTERVAL seconds
    """
    # Wait for Redis
    logger.info("Waiting for Redis...")
    while True:
        try:
            redis_client.ping()
            logger.info("Redis connected")
            break
        except Exception:
            logger.info("Redis not ready — retrying in 3s...")
            time.sleep(3)

    # Bootstrap dataset on first run
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)

    download_movielens()
    build_base_dataset()

    # Also copy movies.csv to model path for inference_api
    movies_src = MOVIES_CSV
    movies_dst = os.path.join(MODEL_PATH, "movies.pkl")
    if os.path.exists(movies_src) and not os.path.exists(movies_dst):
        import pickle
        movies_df = pd.read_csv(movies_src)
        with open(movies_dst, "wb") as f:
            pickle.dump(movies_df, f)
        logger.info(f"Movies index saved to {movies_dst}")

    logger.info(
        f"Pipeline ready. Polling every {POLL_INTERVAL}s "
        f"for new feedback events..."
    )

    # Main processing loop
    while True:
        try:
            run_pipeline_once()
        except Exception as e:
            # Never let one bad run crash the whole worker
            logger.error(f"Pipeline run failed: {e}", exc_info=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
