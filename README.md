# Data Flywheel Pipeline

An end-to-end ML data flywheel — a self-improving movie recommendation
system that automatically retrains as users interact with it.

Built as a portfolio project demonstrating ML engineering skills:
FastAPI, Redis Streams, SVD matrix factorisation, MLflow, React, and Docker.

---

## What is a Data Flywheel?

Better model -> better predictions -> more user trust ->
more feedback -> better training data -> better model -> repeat

The system improves itself continuously without manual intervention.
Every user interaction makes the next recommendation slightly better.

---

## Live Demo

Open http://localhost:5173 after running docker compose up --build.

1. Enter a User ID and click Recommend
2. React to movies — click Loved it, Meh, Clicked, or Skip
3. Watch the progress bar fill toward the retrain threshold
4. When threshold hits 50 new events, the model retrains automatically
5. Recommendations update — the flywheel has closed

---

## Architecture

User opens React UI
     |
     |-- GET /recommend/{user_id} --> Inference API --> Redis Stream
     |                                     |              (logs predictions)
     |                                loads model
     |
     |-- POST /feedback/{type} --> Feedback API --> Redis Stream
                                                       (logs signals)
                                                            |
                                                   Feature Pipeline
                                                   (background worker)
                                                   reads events every 20s
                                                   cleans + deduplicates
                                                   merges into train CSV
                                                            |
                                                       Scheduler
                                                   (background worker)
                                                   checks every 30s
                                                   50 new events? fire
                                                            |
                                                   Trainer (always-on)
                                                   watches Redis trigger
                                                   SVD + Ridge Regression
                                                   eval gate RMSE < 1.0
                                                   logs to MLflow
                                                   saves model to disk
                                                            |
                                                   POST /model/reload
                                                            |
                                                   Inference API
                                                   hot-swaps new model
                                                            |
                                                      (loop repeats)

---

## Services

| Service          | Port | Role                                    |
|------------------|------|-----------------------------------------|
| ui               | 5173 | React dashboard — recommendations + feedback |
| inference_api    | 8002 | Serves movie recommendations            |
| feedback_api     | 8001 | Captures user signals (clicks, ratings) |
| feature_pipeline | —    | ETL worker — Redis events to clean CSV  |
| scheduler        | —    | Watches event count, triggers retraining|
| trainer          | —    | Always-on training job + MLflow logging |
| redis            | 6379 | Message queue (event stream)            |
| mlflow           | 5001 | Experiment tracking + model registry   |

---

## Tech Stack

| Layer               | Tool                   | Why                                   |
|---------------------|------------------------|---------------------------------------|
| Frontend            | React + Vite           | Live dashboard, real-time event log   |
| API framework       | FastAPI                | Async, auto-docs, Pydantic validation |
| Message queue       | Redis Streams          | Lightweight Kafka alternative         |
| ML algorithm        | SVD + Ridge Regression | Fast CPU training, interpretable      |
| Experiment tracking | MLflow                 | Industry standard, clean UI           |
| Containerisation    | Docker Compose         | One command to run everything         |
| Dataset             | MovieLens 100k         | Real-world ratings, free, well known  |

---

## Quick Start

1. Clone the repo
   git clone https://github.com/preranale/data-flywheel
   cd data-flywheel

2. Set up environment
   cp .env.example .env

3. Install React UI dependencies
   cd services/ui && npm install && cd ../..

4. Start all backend services
   docker compose up --build

5. Start the React UI (new terminal)
   cd services/ui && npm run dev

6. Open the interfaces
   React UI (demo dashboard) : http://localhost:5173
   Inference API docs         : http://localhost:8002/docs
   Feedback API docs          : http://localhost:8001/docs
   MLflow experiment tracker  : http://localhost:5001

---

## Try the Flywheel via curl

Get recommendations
   curl http://localhost:8002/recommend/1

Check model status
   curl http://localhost:8002/model/status

Submit an explicit rating
   curl -X POST http://localhost:8001/feedback/rating \
     -H "Content-Type: application/json" \
     -d '{"user_id": 1, "movie_id": 1, "rating": 4.5}'

Submit an implicit click
   curl -X POST http://localhost:8001/feedback/click \
     -H "Content-Type: application/json" \
     -d '{"user_id": 1, "movie_id": 2, "position": 3}'

Submit a skip (negative signal)
   curl -X POST http://localhost:8001/feedback/skip \
     -H "Content-Type: application/json" \
     -d '{"user_id": 1, "movie_id": 3}'

Check how many events are queued
   curl http://localhost:8001/feedback/stats

Simulate 50 users to trigger automatic retraining
   for i in $(seq 1 25); do
     curl -s -X POST http://localhost:8001/feedback/rating \
       -H "Content-Type: application/json" \
       -d "{\"user_id\": $i, \"movie_id\": $((i % 10 + 1)), \"rating\": 4.0}" > /dev/null
     curl -s -X POST http://localhost:8001/feedback/click \
       -H "Content-Type: application/json" \
       -d "{\"user_id\": $i, \"movie_id\": $((i % 5 + 1)), \"position\": $i}" > /dev/null
   done

---

## Key Design Decisions

Why Redis Streams over Kafka?
Kafka requires Zookeeper, JVM, and significant configuration overhead.
Redis Streams provides the same persistent log and consumer group
semantics at a fraction of the complexity. Right tool for this scale.

Why SVD + Ridge over a neural network?
Trains in seconds on CPU, fully interpretable, and competitive with
neural approaches on datasets this size. Matrix factorisation is what
Netflix used in production. Simple and effective wins.

Why a delta-based retrain trigger?
Using total event count would trigger training exactly once. Delta
measures new events since last training run — the correct signal for
deciding when retraining adds value.

Why separate Feedback and Inference APIs?
The inference API must respond in milliseconds. The feedback path can
be async. Coupling them creates a single point of failure and adds
latency to the critical path.

Why an eval gate before deploying?
Without it, a buggy training run could silently deploy a worse model.
RMSE < 1.0 and coverage > 90% must both pass before the new model
replaces the current one.

Why a long-running trainer instead of one-shot?
On Mac, Docker-in-Docker socket mounting is unreliable for triggering
sibling containers. The trainer watches a Redis flag set by the
scheduler — same event-driven pattern used in production systems.

---

## How the Flywheel Closes

1.  Feature pipeline downloads MovieLens (100k ratings) on first run
2.  Trainer trains initial SVD model on that base data
3.  Inference API loads model and serves recommendations
4.  User interacts via React UI — Feedback API writes to Redis Stream
5.  Feature pipeline reads events, cleans them, appends to train CSV
6.  Scheduler detects 50+ new events, sets Redis trigger flag
7.  Trainer picks up trigger, retrains on base data + all feedback
8.  Eval gate checks RMSE < 1.0 and coverage > 90%
9.  If passed, model saved and Inference API hot-reloads it
10. Better model serves better recommendations — loop continues

---

## Project Structure

data-flywheel/
├── services/
│   ├── ui/                  # React + Vite dashboard
│   │   ├── src/
│   │   │   ├── App.jsx
│   │   │   └── components/
│   │   │       ├── Stats.jsx
│   │   │       ├── RecommendPanel.jsx
│   │   │       └── FeedbackPanel.jsx
│   │   └── package.json
│   ├── inference_api/       # GET /recommend
│   │   ├── main.py
│   │   ├── model.py
│   │   └── Dockerfile
│   ├── feedback_api/        # POST /feedback
│   │   ├── main.py
│   │   └── Dockerfile
│   ├── feature_pipeline/    # ETL worker
│   │   ├── pipeline.py
│   │   └── Dockerfile
│   ├── trainer/             # Training + MLflow
│   │   ├── train.py
│   │   ├── eval.py
│   │   └── Dockerfile
│   └── scheduler/           # Threshold watcher
│       ├── scheduler.py
│       └── Dockerfile
├── data/
│   ├── raw/                 # MovieLens + stream cursor
│   ├── processed/           # train.csv, val.csv, movies.csv
│   └── models/              # model.pkl, movies.pkl
├── mlflow/mlruns/           # MLflow runs (gitignored)
├── docker-compose.yml
├── .env.example
└── README.md

---

## Monitoring

Check all services are healthy
   docker compose ps

Watch scheduler and trainer live
   docker compose logs -f scheduler trainer

Check feature pipeline processed new events
   docker compose logs feature_pipeline

View all logs
   docker compose logs -f
