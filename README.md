# Data Flywheel Pipeline

An end-to-end ML data flywheel — a self-improving movie recommendation system that automatically retrains as users interact with it.

Built as a portfolio project: FastAPI · Redis Streams · SVD Matrix Factorisation · MLflow · React · Docker

---

## What is a Data Flywheel?

> Better model → better predictions → more user trust → more feedback → better training data → better model → repeat

The system improves itself continuously without manual intervention.

---

## Live Demo

Open `http://localhost:5173` after running `docker compose up --build`

1. Enter a User ID and click **Recommend**
2. React to movies — Loved it / Meh / Clicked / Skip
3. Watch the progress bar fill toward the retrain threshold
4. At 50 new events — model retrains automatically, no human involved
5. Recommendations update — flywheel has closed

---

## Services

| Service            | Port | Role                                 |
| ------------------ | ---- | ------------------------------------ |
| `ui`               | 5173 | React dashboard                      |
| `inference_api`    | 8002 | Serves recommendations               |
| `feedback_api`     | 8001 | Captures clicks, ratings, skips      |
| `feature_pipeline` | —    | ETL: Redis → clean CSV               |
| `scheduler`        | —    | Triggers retraining at threshold     |
| `trainer`          | —    | SVD model + MLflow logging           |
| `redis`            | 6379 | Event stream (message queue)         |
| `mlflow`           | 5001 | Experiment tracking + model registry |

---

## Tech Stack

| Layer    | Tool                   | Why                                   |
| -------- | ---------------------- | ------------------------------------- |
| Frontend | React + Vite           | Live dashboard, real-time event log   |
| API      | FastAPI                | Async, auto-docs, Pydantic validation |
| Queue    | Redis Streams          | Lightweight Kafka alternative         |
| ML       | SVD + Ridge Regression | Fast CPU training, interpretable      |
| Tracking | MLflow                 | Industry standard, beautiful UI       |
| DevOps   | Docker Compose         | One command boots everything          |
| Dataset  | MovieLens 100k         | Real ratings, free, well known        |

---

## Key Design Decisions

**Why Redis Streams over Kafka?**
Kafka needs Zookeeper, JVM, and heavy config. Redis Streams gives the same persistent log semantics at a fraction of the complexity.

**Why SVD + Ridge over a neural network?**
Trains in seconds on CPU. Interpretable. Competitive with neural approaches at this scale. Matrix factorisation is what Netflix used in production.

**Why delta-based retrain trigger?**
Total event count only triggers once. Delta (new events since last run) correctly measures new information arriving.

**Why separate Feedback and Inference APIs?**
Inference must respond in milliseconds. Feedback can be async. Coupling them adds latency to the hot path.

**Why an eval gate?**
Without it, a buggy training run deploys a worse model silently. RMSE < 1.0 and coverage > 90% must pass before going live.

---

## How the Flywheel Closes

1. Feature pipeline downloads MovieLens (100k ratings) on first run
2. Trainer trains initial SVD model on that base data
3. Inference API loads model, starts serving recommendations
4. User interacts in React UI → Feedback API writes to Redis Stream
5. Feature pipeline reads events, cleans them, appends to train CSV
6. Scheduler detects 50+ new events → sets Redis trigger flag
7. Trainer picks up trigger, retrains on base data + all feedback
8. Eval gate: RMSE < 1.0 and coverage > 90% must pass
9. Model saved → Inference API hot-reloads without restart
10. Better recommendations → more engagement → loop continues

---

## Architecture

```text
React UI (port 5173)
    |
    |-- GET /recommend/{id} --> Inference API --> recommendations
    |
    |-- POST /feedback -------> Feedback API --> Redis Stream
                                                      |
                                             Feature Pipeline
                                             reads + cleans events
                                             merges into train.csv
                                                      |
                                               Scheduler
                                             checks every 30s
                                             50 new events? fire
                                                      |
                                               Trainer
                                             SVD + Ridge Regression
                                             RMSE eval gate
                                             logs to MLflow
                                                      |
                                             Inference API
                                             hot-swaps new model
                                                      |
                                              (loop repeats)
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/preranale/data-flywheel
cd data-flywheel

# 2. Environment
cp .env.example .env

# 3. Install UI
cd services/ui && npm install && cd ../..

# 4. Start backend
docker compose up --build

# 5. Start UI (new terminal)
cd services/ui && npm run dev
```

Open these in your browser:

- React UI → http://localhost:5173
- MLflow → http://localhost:5001
- API docs → http://localhost:8002/docs

---

## Project Structure

```text
data-flywheel/
├── services/
│   ├── ui/                  # React + Vite dashboard
│   │   └── src/
│   │       ├── App.jsx
│   │       └── components/
│   │           ├── Stats.jsx
│   │           ├── RecommendPanel.jsx
│   │           └── FeedbackPanel.jsx
│   ├── inference_api/       # Serves recommendations
│   │   ├── main.py
│   │   ├── model.py
│   │   └── Dockerfile
│   ├── feedback_api/        # Captures user signals
│   │   ├── main.py
│   │   └── Dockerfile
│   ├── feature_pipeline/    # ETL worker
│   │   ├── pipeline.py
│   │   └── Dockerfile
│   ├── trainer/             # SVD training + MLflow
│   │   ├── train.py
│   │   ├── eval.py
│   │   └── Dockerfile
│   └── scheduler/           # Threshold watcher
│       ├── scheduler.py
│       └── Dockerfile
├── data/
│   ├── raw/                 # MovieLens dataset
│   ├── processed/           # train.csv, val.csv
│   └── models/              # model.pkl, movies.pkl
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Monitoring

```bash
# All services status
docker compose ps

# Watch scheduler + trainer live
docker compose logs -f scheduler trainer

# All logs
docker compose logs -f
```
