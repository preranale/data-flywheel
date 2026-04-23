import os, time, pickle, logging, requests
import numpy as np, pandas as pd
import mlflow, mlflow.sklearn
import redis
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from eval import compute_metrics, passes_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "/data/processed")
MODEL_PATH          = os.getenv("MODEL_PATH", "/data/models")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "movie-recommender")
INFERENCE_API_URL   = os.getenv("INFERENCE_API_URL", "http://inference_api:8000")
REDIS_HOST          = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT          = int(os.getenv("REDIS_PORT", 6379))

TRAIN_CSV  = os.path.join(PROCESSED_DATA_PATH, "train.csv")
VAL_CSV    = os.path.join(PROCESSED_DATA_PATH, "val.csv")
MOVIES_CSV = os.path.join(PROCESSED_DATA_PATH, "movies.csv")
N_COMPONENTS = 50

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def load_data():
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"No training data at {TRAIN_CSV}")
    train = pd.read_csv(TRAIN_CSV)
    val   = pd.read_csv(VAL_CSV)
    logger.info(f"Loaded {len(train)} train, {len(val)} val rows")
    return train, val


def train_model(train_df):
    user_ids  = sorted(train_df["user_id"].unique())
    movie_ids = sorted(train_df["movie_id"].unique())
    user_idx  = {u: i for i, u in enumerate(user_ids)}
    movie_idx = {m: i for i, m in enumerate(movie_ids)}

    logger.info("Building interaction matrix...")
    matrix = np.zeros((len(user_ids), len(movie_ids)))
    for _, row in train_df.iterrows():
        u = user_idx.get(row["user_id"])
        m = movie_idx.get(row["movie_id"])
        if u is not None and m is not None:
            matrix[u, m] = row["rating"]
    logger.info(f"Matrix: {matrix.shape} | Sparsity: {(matrix==0).mean():.1%}")

    logger.info(f"Running SVD with {N_COMPONENTS} components...")
    svd           = TruncatedSVD(n_components=N_COMPONENTS)
    user_factors  = svd.fit_transform(matrix)
    movie_factors = svd.components_.T

    logger.info("Building training pairs...")
    X_rows, y_vals = [], []
    for _, row in train_df.iterrows():
        u = user_idx.get(int(row["user_id"]))
        m = movie_idx.get(int(row["movie_id"]))
        if u is None or m is None:
            continue
        X_rows.append(np.concatenate([user_factors[u], movie_factors[m]]))
        y_vals.append(row["rating"])

    X, y = np.array(X_rows), np.array(y_vals)
    mean_rating = float(np.mean(y))

    logger.info(f"Training Ridge on {len(X)} pairs...")
    regressor = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    regressor.fit(X, y)
    logger.info("Training complete")

    return {
        "user_idx":      user_idx,
        "movie_idx":     movie_idx,
        "user_factors":  user_factors,
        "movie_factors": movie_factors,
        "regressor":     regressor,
        "mean_rating":   mean_rating,
        "n_components":  N_COMPONENTS,
    }


def make_predict_fn(model_data):
    class _W:
        def predict(self, pairs):
            preds = []
            for pair in pairs:
                u = model_data["user_idx"].get(int(pair[0]))
                m = model_data["movie_idx"].get(int(pair[1]))
                if u is None or m is None:
                    preds.append(model_data["mean_rating"])
                    continue
                feat = np.concatenate([
                    model_data["user_factors"][u],
                    model_data["movie_factors"][m],
                ]).reshape(1, -1)
                preds.append(float(np.clip(model_data["regressor"].predict(feat)[0], 1.0, 5.0)))
            return np.array(preds)
    return _W()


def save_model(model_data, version):
    os.makedirs(MODEL_PATH, exist_ok=True)
    with open(os.path.join(MODEL_PATH, "model.pkl"), "wb") as f:
        pickle.dump(model_data, f)
    with open(os.path.join(MODEL_PATH, "version.txt"), "w") as f:
        f.write(version)
    if not os.path.exists(os.path.join(MODEL_PATH, "movies.pkl")) and os.path.exists(MOVIES_CSV):
        movies = pd.read_csv(MOVIES_CSV)
        with open(os.path.join(MODEL_PATH, "movies.pkl"), "wb") as f:
            pickle.dump(movies, f)
    logger.info(f"Model saved to {MODEL_PATH}")


def reload_inference_api():
    try:
        r = requests.post(f"{INFERENCE_API_URL}/model/reload", timeout=10)
        if r.status_code == 200:
            logger.info("Inference API reloaded new model successfully")
    except Exception as e:
        logger.warning(f"Could not reload inference API: {e}")


def run_training():
    start   = time.time()
    version = f"v{int(start)}"
    logger.info(f"=== Training run starting | version: {version} ===")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("movie-recommender")

    with mlflow.start_run(run_name=version):
        try:
            train_df, val_df = load_data()
        except Exception as e:
            logger.error(f"Data load failed: {e}")
            return

        mlflow.log_params({
            "n_train": len(train_df), "n_val": len(val_df),
            "n_users": train_df["user_id"].nunique(),
            "n_movies": train_df["movie_id"].nunique(),
            "n_components": N_COMPONENTS, "version": version,
        })

        model_data = train_model(train_df)
        wrapper    = make_predict_fn(model_data)
        metrics    = compute_metrics(wrapper, val_df)
        mlflow.log_metrics(metrics)

        if not passes_threshold(metrics):
            mlflow.log_param("status", "failed_eval_gate")
            return

        save_model(model_data, version)
        mlflow.log_param("status", "deployed")
        mlflow.log_metric("training_time_seconds", round(time.time() - start, 2))

        reload_inference_api()
        logger.info(f"=== Training complete in {time.time()-start:.1f}s | RMSE: {metrics['rmse']} ===")


def main():
    """
    Long-running service mode.
    Watches Redis for a retrain trigger flag set by the scheduler.
    WHY: Avoids Docker-in-Docker issues on Mac.
    """
    logger.info("Trainer waiting for Redis...")
    while True:
        try:
            redis_client.ping()
            logger.info("Redis connected — trainer ready, watching for trigger")
            break
        except:
            time.sleep(3)

    # Run once on startup
    run_training()

    # Then watch for triggers
    while True:
        trigger = redis_client.get("scheduler:retrain_trigger")
        if trigger == "1":
            logger.info("Retrain trigger detected!")
            redis_client.delete("scheduler:retrain_trigger")
            run_training()
        time.sleep(10)


if __name__ == "__main__":
    main()
