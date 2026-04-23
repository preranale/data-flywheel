import os, time, logging, requests
import redis

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REDIS_HOST        = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT        = int(os.getenv("REDIS_PORT", 6379))
REDIS_STREAM      = os.getenv("REDIS_STREAM", "feedback_events")
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "50"))
TRAINER_URL       = os.getenv("TRAINER_URL", "http://trainer:8002")
CHECK_INTERVAL    = 30
BASELINE_KEY      = "scheduler:last_train_event_count"

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def get_event_count():
    try:
        return redis_client.xlen(REDIS_STREAM)
    except:
        return 0


def get_baseline():
    val = redis_client.get(BASELINE_KEY)
    return int(val) if val else 0


def set_baseline(count):
    redis_client.set(BASELINE_KEY, count)


def trigger_training():
    """
    Write a trigger flag to Redis.
    The trainer service polls this flag and runs when it sees it.
    WHY: Avoids Docker-in-Docker complexity on Mac.
    Same pattern used in production — event-driven triggers.
    """
    logger.info("Writing retrain trigger to Redis...")
    redis_client.set("scheduler:retrain_trigger", "1")
    redis_client.set("scheduler:retrain_requested_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    logger.info("Retrain trigger set — trainer will pick it up")


def main():
    logger.info("Scheduler starting — waiting for Redis...")
    while True:
        try:
            redis_client.ping()
            logger.info("Redis connected")
            break
        except:
            time.sleep(3)

    logger.info(f"Scheduler ready. Checking every {CHECK_INTERVAL}s. Threshold: {RETRAIN_THRESHOLD}")

    while True:
        try:
            total    = get_event_count()
            baseline = get_baseline()
            new      = total - baseline

            logger.info(f"Event check — total: {total} | baseline: {baseline} | new: {new} | threshold: {RETRAIN_THRESHOLD}")

            if new >= RETRAIN_THRESHOLD:
                logger.info(f"Threshold reached ({new} >= {RETRAIN_THRESHOLD}) — triggering retraining")
                set_baseline(total)
                trigger_training()
            else:
                logger.info(f"Not enough new events yet. Need {RETRAIN_THRESHOLD - new} more.")

        except Exception as e:
            logger.error(f"Scheduler error: {e}")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
