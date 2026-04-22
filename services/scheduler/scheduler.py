# ============================================================
# services/scheduler/scheduler.py
# ============================================================
# Watchdog that monitors feedback volume and triggers retraining.
#
# Logic:
#   Every CHECK_INTERVAL seconds:
#     1. Count total events in Redis Stream
#     2. Compare to last known count at previous training run
#     3. If delta >= RETRAIN_THRESHOLD → trigger trainer
#     4. Record new baseline count
#
# WHY delta not total count?
# If we used total count, we'd only ever train once
# (after 50 events, count stays above 50 forever).
# Delta means "50 NEW events since last training run" —
# this is what actually measures new information arriving.
# ============================================================

import os
import time
import logging
import subprocess
import requests
import redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Config -----------------------------------------------
REDIS_HOST        = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT        = int(os.getenv("REDIS_PORT", 6379))
REDIS_STREAM      = os.getenv("REDIS_STREAM", "feedback_events")
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "50"))
CHECK_INTERVAL    = 30   # seconds between checks
BASELINE_KEY      = "scheduler:last_train_event_count"


# ---- Redis client -----------------------------------------
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
)


def get_event_count() -> int:
    """Total events in the Redis Stream."""
    try:
        return redis_client.xlen(REDIS_STREAM)
    except redis.ResponseError:
        return 0   # Stream doesn't exist yet


def get_baseline_count() -> int:
    """
    Event count at the time of the last training run.
    Stored in Redis so it survives scheduler restarts.
    """
    val = redis_client.get(BASELINE_KEY)
    return int(val) if val else 0


def set_baseline_count(count: int):
    """Update the baseline after a training run fires."""
    redis_client.set(BASELINE_KEY, count)


def trigger_training():
    """
    Trigger a training run.

    We use subprocess to call docker compose run trainer.
    WHY: The trainer is defined as a Docker Compose service
    with profiles: [training] — it doesn't start automatically
    with docker compose up. We launch it on demand here.

    The /var/run/docker.sock volume mount in docker-compose.yml
    gives this container access to the HOST Docker daemon,
    so it can spin up sibling containers.
    """
    logger.info("Triggering training run via docker compose...")

    try:
        result = subprocess.run(
            ["docker", "compose", "run", "--rm", "trainer"],
            capture_output=True,
            text=True,
            timeout=600,   # 10 min max for training
        )

        if result.returncode == 0:
            logger.info("Training run completed successfully")
            logger.info(result.stdout[-500:] if result.stdout else "")
        else:
            logger.error(f"Training run failed (exit {result.returncode})")
            logger.error(result.stderr[-500:] if result.stderr else "")

    except subprocess.TimeoutExpired:
        logger.error("Training run timed out after 10 minutes")
    except FileNotFoundError:
        # Docker CLI not found — fallback to HTTP trigger
        logger.warning(
            "Docker CLI not available — trying HTTP trigger instead"
        )
        trigger_training_via_http()
    except Exception as e:
        logger.error(f"Failed to trigger training: {e}", exc_info=True)


def trigger_training_via_http():
    """
    Fallback: call trainer directly via HTTP if Docker CLI unavailable.
    Only works if trainer is running as a long-lived service.
    """
    try:
        requests.post("http://trainer:8002/train", timeout=5)
        logger.info("Training triggered via HTTP fallback")
    except Exception as e:
        logger.error(f"HTTP trigger also failed: {e}")


def main():
    """
    Main scheduler loop.

    Startup: wait for Redis, then start watching.
    Every CHECK_INTERVAL seconds: check delta, maybe train.
    """
    logger.info("Scheduler starting — waiting for Redis...")
    while True:
        try:
            redis_client.ping()
            logger.info("Redis connected")
            break
        except Exception:
            logger.info("Redis not ready — retrying in 3s...")
            time.sleep(3)

    logger.info(
        f"Scheduler ready. Checking every {CHECK_INTERVAL}s. "
        f"Retrain threshold: {RETRAIN_THRESHOLD} new events."
    )

    while True:
        try:
            total_events  = get_event_count()
            baseline      = get_baseline_count()
            new_events    = total_events - baseline

            logger.info(
                f"Event check — total: {total_events} | "
                f"baseline: {baseline} | new: {new_events} | "
                f"threshold: {RETRAIN_THRESHOLD}"
            )

            if new_events >= RETRAIN_THRESHOLD:
                logger.info(
                    f"Threshold reached ({new_events} >= {RETRAIN_THRESHOLD}) "
                    f"— triggering retraining"
                )
                set_baseline_count(total_events)
                trigger_training()
            else:
                remaining = RETRAIN_THRESHOLD - new_events
                logger.info(
                    f"Not enough new events yet. "
                    f"Need {remaining} more to trigger retraining."
                )

        except Exception as e:
            logger.error(f"Scheduler check failed: {e}", exc_info=True)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
