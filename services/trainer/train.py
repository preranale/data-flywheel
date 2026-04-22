# ============================================================
# services/trainer/train.py
# ============================================================
# One-shot training job. Runs, logs to MLflow, exits.
#
# Algorithm: SVD via TruncatedSVD + Ridge Regression
# WHY not a neural network?
# - Trains in seconds on CPU (great for demos)
# - Interpretable — you can explain it in interviews
# - Competitive with neural approaches on small datasets
# - Matrix factorisation IS what Netflix used in production
#
# What we actually do:
#   1. Load train/val CSVs from /data/processed
#   2. Build a user-movie interaction matrix
#   3. Decompose it with SVD to get latent factors
#   4. Train a Ridge regression on those factors
#   5. Evaluate on val set
#   6. If eval passes, save model + log to MLflow
#   7. Signal the inference API to reload
# ============================================================

import os
import time
import pickle
import logging
import requests
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eval import compute_metrics, passes_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Config -----------------------------------------------
PROCESSED_DATA_PATH  = os.getenv("PROCESSED_DATA_PATH", "/data/processed")
MODEL_PATH           = os.getenv("MODEL_PATH", "/data/models")
MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME           = os.getenv("MODEL_NAME", "movie-recommender")
INFERENCE_API_URL    = os.getenv("INFERENCE_API_URL", "http://inference_api:8000")

TRAIN_CSV  = os.path.join(PROCESSED_DATA_PATH, "train.csv")
VAL_CSV    = os.path.join(PROCESSED_DATA_PATH, "val.csv")
MOVIES_CSV = os.path.join(PROCESSED_DATA_PATH, "movies.csv")

# SVD hyperparameters
N_COMPONENTS = 50   # Number of latent factors
# WHY 50? Rule of thumb for datasets this size.
# Too few = underfitting (can't capture taste diversity)
# Too many = overfitting + slow training
# 50 is the Netflix Prize sweet spot for similar-scale data


# ===========================================================
# SECTION 1: Data loading
# ===========================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and val CSVs.
    Validates they exist and have minimum viable size.
    """
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(
            f"Training data not found at {TRAIN_CSV}. "
            "Has the feature pipeline run yet?"
        )

    train = pd.read_csv(TRAIN_CSV)
    val   = pd.read_csv(VAL_CSV)

    if len(train) < 100:
        raise ValueError(
            f"Training set too small ({len(train)} rows). "
            f"Need at least 100 ratings to train a meaningful model."
        )

    logger.info(
        f"Loaded training data: {len(train)} train rows, "
        f"{val.__len__()} val rows"
    )
    return train, val


# ===========================================================
# SECTION 2: Feature engineering
# ===========================================================

def build_interaction_matrix(
    df: pd.DataFrame,
) -> tuple[np.ndarray, dict, dict]:
    """
    Build a user-movie interaction matrix for SVD.

    Shape: (n_users, n_movies)
    Value: user's rating for that movie, 0 if not rated

    Also returns index mappings so we can go from
    matrix row/col → user_id/movie_id and back.

    WHY a matrix?
    SVD (Singular Value Decomposition) decomposes this matrix
    into two smaller matrices — one for users, one for movies.
    The decomposition captures LATENT FACTORS: abstract concepts
    like "likes action films" or "prefers 90s comedies" that
    explain the rating patterns, even without labelling them.
    """
    user_ids  = sorted(df["user_id"].unique())
    movie_ids = sorted(df["movie_id"].unique())

    user_idx  = {uid: i for i, uid in enumerate(user_ids)}
    movie_idx = {mid: i for i, mid in enumerate(movie_ids)}

    matrix = np.zeros((len(user_ids), len(movie_ids)))

    for _, row in df.iterrows():
        u = user_idx.get(row["user_id"])
        m = movie_idx.get(row["movie_id"])
        if u is not None and m is not None:
            matrix[u, m] = row["rating"]

    logger.info(
        f"Interaction matrix: {matrix.shape} | "
        f"Sparsity: {(matrix == 0).mean():.1%} zeros"
    )

    return matrix, user_idx, movie_idx


def build_training_pairs(
    df: pd.DataFrame,
    user_idx: dict,
    movie_idx: dict,
    user_factors: np.ndarray,
    movie_factors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) training pairs for Ridge regression.

    X: concatenated user + movie latent factor vectors
       Shape: (n_ratings, 2 * N_COMPONENTS)

    y: actual ratings
       Shape: (n_ratings,)

    The idea: if we concatenate a user's taste vector with
    a movie's style vector, Ridge regression learns to predict
    how much that user will like that movie.
    """
    X_rows = []
    y_vals = []

    for _, row in df.iterrows():
        u_idx = user_idx.get(int(row["user_id"]))
        m_idx = movie_idx.get(int(row["movie_id"]))

        if u_idx is None or m_idx is None:
            continue

        # Concatenate user factors + movie factors
        features = np.concatenate([
            user_factors[u_idx],
            movie_factors[m_idx],
        ])
        X_rows.append(features)
        y_vals.append(row["rating"])

    return np.array(X_rows), np.array(y_vals)


# ===========================================================
# SECTION 3: Model wrapper
# ===========================================================

class SVDRecommender:
    """
    Wraps SVD decomposition + Ridge regression into a single
    object that the inference API can call with .predict().

    WHY wrap instead of using sklearn Pipeline directly?
    We need to store the user/movie index maps alongside
    the model so inference works correctly. A Pipeline
    doesn't have a natural place for those mappings.
    Also: the inference API expects .predict([[user_id, movie_id]])
    which needs our custom lookup logic.
    """

    def __init__(self, n_components: int = N_COMPONENTS):
        self.n_components = n_components
        self.svd          = TruncatedSVD(n_components=n_components)
        self.regressor    = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=1.0)),
        ])
        self.user_idx   = {}
        self.movie_idx  = {}
        self.user_factors  = None
        self.movie_factors = None
        self.mean_rating   = 3.5   # fallback for unknown users/movies

    def fit(self, train_df: pd.DataFrame):
        """Train the model on the interaction matrix."""
        logger.info("Building interaction matrix...")
        matrix, self.user_idx, self.movie_idx = build_interaction_matrix(train_df)

        logger.info(f"Running SVD with {self.n_components} components...")
        self.user_factors  = self.svd.fit_transform(matrix)
        self.movie_factors = self.svd.components_.T

        logger.info("Building training pairs for Ridge regression...")
        X, y = build_training_pairs(
            train_df,
            self.user_idx,
            self.movie_idx,
            self.user_factors,
            self.movie_factors,
        )
        self.mean_rating = float(np.mean(y))

        logger.info(f"Training Ridge regression on {len(X)} pairs...")
        self.regressor.fit(X, y)
        logger.info("Training complete")

    def predict(self, pairs: list) -> np.ndarray:
        """
        Predict ratings for a list of [user_id, movie_id] pairs.
        Returns mean_rating for unknown users or movies.
        """
        predictions = []

        for pair in pairs:
            user_id, movie_id = int(pair[0]), int(pair[1])
            u_idx = self.user_idx.get(user_id)
            m_idx = self.movie_idx.get(movie_id)

            if u_idx is None or m_idx is None:
                predictions.append(self.mean_rating)
                continue

            features = np.concatenate([
                self.user_factors[u_idx],
                self.movie_factors[m_idx],
            ]).reshape(1, -1)

            pred = self.regressor.predict(features)[0]
            pred = float(np.clip(pred, 1.0, 5.0))
            predictions.append(pred)

        return np.array(predictions)


# ===========================================================
# SECTION 4: MLflow logging + model saving
# ===========================================================

def save_model(model: SVDRecommender, version: str):
    """
    Save model artifact to disk for the inference API to load.
    Also saves the movies index needed for title lookups.
    """
    os.makedirs(MODEL_PATH, exist_ok=True)

    model_file = os.path.join(MODEL_PATH, "model.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_file}")

    # Save movies index if not already there
    movies_dst = os.path.join(MODEL_PATH, "movies.pkl")
    if not os.path.exists(movies_dst) and os.path.exists(MOVIES_CSV):
        movies_df = pd.read_csv(MOVIES_CSV)
        with open(movies_dst, "wb") as f:
            pickle.dump(movies_df, f)
        logger.info(f"Movies index saved to {movies_dst}")

    # Save version tag
    version_file = os.path.join(MODEL_PATH, "version.txt")
    with open(version_file, "w") as f:
        f.write(version)


def reload_inference_api():
    """
    Tell the inference API to hot-reload the new model.
    Called after successful training + eval.
    """
    try:
        response = requests.post(
            f"{INFERENCE_API_URL}/model/reload",
            timeout=10,
        )
        if response.status_code == 200:
            logger.info("Inference API reloaded new model successfully")
        else:
            logger.warning(f"Inference API reload returned {response.status_code}")
    except Exception as e:
        logger.warning(
            f"Could not reach inference API to reload: {e}. "
            "Model is saved — API will load it on next restart."
        )


# ===========================================================
# SECTION 5: Main training run
# ===========================================================

def main():
    start_time = time.time()
    run_version = f"v{int(start_time)}"

    logger.info(f"=== Training run starting | version: {run_version} ===")

    # Connect to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("movie-recommender")

    with mlflow.start_run(run_name=run_version):

        # 1. Load data
        try:
            train_df, val_df = load_data()
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Data loading failed: {e}")
            mlflow.log_param("status", "failed_data_load")
            return

        # 2. Log data stats to MLflow
        mlflow.log_params({
            "n_train":       len(train_df),
            "n_val":         len(val_df),
            "n_users":       train_df["user_id"].nunique(),
            "n_movies":      train_df["movie_id"].nunique(),
            "n_components":  N_COMPONENTS,
            "model_version": run_version,
        })

        # 3. Train
        model = SVDRecommender(n_components=N_COMPONENTS)
        try:
            model.fit(train_df)
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            mlflow.log_param("status", "failed_training")
            return

        # 4. Evaluate
        metrics = compute_metrics(model, val_df)
        mlflow.log_metrics(metrics)

        # 5. Gate — only deploy if eval passes
        if not passes_threshold(metrics):
            mlflow.log_param("status", "failed_eval_gate")
            logger.warning(
                "Model did not pass eval gate — NOT deployed. "
                "Check MLflow UI for metrics details."
            )
            return

        # 6. Save model
        save_model(model, run_version)
        mlflow.log_param("status", "deployed")
        mlflow.log_param("model_path", MODEL_PATH)

        # 7. Log to MLflow model registry
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        # 8. Tell inference API to load new model
        reload_inference_api()

        elapsed = time.time() - start_time
        mlflow.log_metric("training_time_seconds", round(elapsed, 2))

        logger.info(
            f"=== Training complete in {elapsed:.1f}s | "
            f"RMSE: {metrics['rmse']} | version: {run_version} ==="
        )


if __name__ == "__main__":
    main()
