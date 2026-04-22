# Data Flywheel Pipeline

An end-to-end ML data flywheel — a self-improving movie recommendation
system that automatically retrains as users interact with it.

Built as a portfolio project demonstrating ML engineering skills:
FastAPI, Redis Streams, SVD matrix factorisation, MLflow, and Docker.

---

## What is a Data Flywheel?

> Better model → better predictions → more user trust →
> more feedback → better training data → better model → repeat

The system improves itself continuously without manual intervention.
Every user interaction makes the next recommendation slightly better.

---

## Architecture
User Request
│
├── GET /recommend/{user_id} ──→ Inference API ──→ Redis Stream
│                                      │               (logs predictions)
│                                 loads model
│
└── POST /feedback/{type} ──→ Feedback API ──→ Redis Stream
(logs signals)
│
Feature Pipeline
(background worker)
• reads events every 20s
• cleans + deduplicates
• merges into train CSV
│
Scheduler
(background worker)
• checks every 30s
• 50 new events? → fire
│
Trainer (one-shot job)
• SVD + Ridge Regression
• eval gate (RMSE < 1.0)
• logs to MLflow
• saves model to disk
│
POST /model/reload
│
Inference API
hot-swaps new model
│
(loop repeats)

---

## Services

| Service           | Port | Role                                      |
|-------------------|------|-------------------------------------------|
| `inference_api`   | 8000 | Serves movie recommendations              |
| `feedback_api`    | 8001 | Captures user signals (clicks, ratings)   |
| `feature_pipeline`| —    | ETL worker — Redis events → clean CSV     |
| `scheduler`       | —    | Watches event count, triggers retraining  |
| `trainer`         | —    | One-shot training job + MLflow logging    |
| `redis`           | 6379 | Message queue (event stream)              |
| `mlflow`          | 5000 | Experiment tracking + model registry      |

---

## Tech Stack

| Layer              | Tool                    | Why                                        |
|--------------------|-------------------------|--------------------------------------------|
| API framework      | FastAPI                 | Async, auto-docs, Pydantic validation      |
| Message queue      | Redis Streams           | Lightweight Kafka alternative              |
| ML algorithm       | SVD + Ridge Regression  | Fast CPU training, interpretable           |
| Experiment tracking| MLflow                  | Industry standard, clean UI                |
| Containerisation   | Docker Compose          | One command to run everything              |
| Dataset            | MovieLens (100k)        | Real-world ratings, free, well known       |

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/preranale/data-flywheel
cd data-flywheel

# 2. Set up environment
cp .env.example .env

# 3. Start all services (first run takes 3-5 min to build + download dataset)
docker compose up --build

# 4. Open the UIs
open http://localhost:8000/docs   # Inference API auto-docs
open http://localhost:8001/docs   # Feedback API auto-docs
open http://localhost:5000        # MLflow experiment tracker
```

---

## Try the Flywheel

```bash
# Get recommendations (fallback mode until first model trains)
curl http://localhost:8000/recommend/1

# Check model status
curl http://localhost:8000/model/status

# Submit an explicit rating
curl -X POST http://localhost:8001/feedback/rating \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "movie_id": 1, "rating": 4.5}'

# Submit an implicit click
curl -X POST http://localhost:8001/feedback/click \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "movie_id": 2, "position": 3}'

# Submit a skip (negative signal)
curl -X POST http://localhost:8001/feedback/skip \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "movie_id": 3}'

# Check how many events are queued
curl http://localhost:8001/feedback/stats

# Manually trigger training (without waiting for threshold)
docker compose run --rm trainer

# Recommendations now served by real trained model
curl http://localhost:8000/recommend/42
```

---

## Key Design Decisions

**Why Redis Streams over Kafka?**
Kafka requires Zookeeper, JVM, and significant configuration overhead.
Redis Streams provides the same persistent log and consumer group
semantics at a fraction of the operational complexity — right tool
for the scale of a portfolio project.

**Why SVD + Ridge over a neural network?**
Trains in seconds on CPU, fully interpretable, and competitive with
neural approaches on datasets this size. Matrix factorisation is
what Netflix used in production. Simple and effective wins.

**Why a delta-based retrain trigger?**
Using total event count would trigger training exactly once — after
50 events, the count stays above 50 forever. Delta (new events since
last training run) correctly measures the rate of new information
arriving, which is the right signal for retraining decisions.

**Why separate Feedback and Inference APIs?**
The inference API must respond in milliseconds — users are waiting.
The feedback path can be async. Coupling them creates a single point
of failure and adds latency to the critical path.

**Why an eval gate before deploying?**
Without it, a buggy training run could silently deploy a worse model.
RMSE < 1.0 and coverage > 90% must both pass before the new model
replaces the current one in production.

---

## Project Structure
data-flywheel/
├── services/
│   ├── inference_api/       # GET /recommend — serves predictions
│   │   ├── main.py          # FastAPI routes + Redis logging
│   │   ├── model.py         # Model loader + recommender class
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── feedback_api/        # POST /feedback — captures user signals
│   │   ├── main.py          # Rating, click, skip endpoints
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── feature_pipeline/    # ETL background worker
│   │   ├── pipeline.py      # Download → clean → merge → CSV
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── trainer/             # One-shot training job
│   │   ├── train.py         # SVD + Ridge + MLflow logging
│   │   ├── eval.py          # RMSE/MAE/coverage + pass/fail gate
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── scheduler/           # Threshold watcher
│       ├── scheduler.py     # Delta counter + training trigger
│       ├── requirements.txt
│       └── Dockerfile
├── data/
│   ├── raw/                 # MovieLens download + stream cursor
│   ├── processed/           # train.csv, val.csv, movies.csv
│   └── models/              # model.pkl, movies.pkl, version.txt
├── mlflow/mlruns/           # MLflow experiment runs (gitignored)
├── docker-compose.yml       # Full system definition
├── .env.example             # Config template
└── README.md

---

## How the Flywheel Closes

1. Feature pipeline downloads **MovieLens** (100k ratings) on first run
2. Trainer trains an initial SVD model on that base data
3. Inference API loads the model and starts serving recommendations
4. Users interact → Feedback API writes events to Redis Stream
5. Feature pipeline reads events, cleans them, appends to train CSV
6. Scheduler detects 50+ new events → fires trainer
7. Trainer retrains on base data + all accumulated feedback
8. If eval passes, model saved and Inference API hot-reloads it
9. Better model → better recommendations → more engagement → repeat

---

## Running Tests

```bash
# Verify all services are healthy
docker compose ps

# Check logs for any service
docker compose logs inference_api
docker compose logs feature_pipeline
docker compose logs scheduler

# Tail live logs
docker compose logs -f
```
