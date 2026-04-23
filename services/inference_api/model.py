import os, pickle, logging
import numpy as np

logger    = logging.getLogger(__name__)
MODEL_PATH  = os.getenv("MODEL_PATH", "/data/models")
MODEL_FILE  = os.path.join(MODEL_PATH, "model.pkl")
MOVIES_FILE = os.path.join(MODEL_PATH, "movies.pkl")


class MovieRecommender:
    def __init__(self):
        self.model_data = None
        self.movies     = None
        self.is_ready   = False

    def load(self):
        if not os.path.exists(MODEL_FILE):
            logger.warning(f"No model at {MODEL_FILE} — using fallback")
            self.is_ready = False
            return False
        try:
            with open(MODEL_FILE, "rb") as f:
                self.model_data = pickle.load(f)
            with open(MOVIES_FILE, "rb") as f:
                self.movies = pickle.load(f)
            self.is_ready = True
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_ready = False
            return False

    def _predict(self, user_id, movie_id):
        d     = self.model_data
        u_idx = d["user_idx"].get(user_id)
        m_idx = d["movie_idx"].get(movie_id)
        if u_idx is None or m_idx is None:
            return d["mean_rating"]
        feat = np.concatenate([
            d["user_factors"][u_idx],
            d["movie_factors"][m_idx],
        ]).reshape(1, -1)
        pred = d["regressor"].predict(feat)[0]
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user_id: int, n: int = 5) -> list[dict]:
        if not self.is_ready:
            return self._fallback(n)
        try:
            movie_ids = self.movies["movie_id"].tolist()
            scores    = [self._predict(user_id, mid) for mid in movie_ids]
            scores    = np.array(scores)
            top_idx   = np.argsort(scores)[::-1][:n]
            results   = []
            for idx in top_idx:
                mid = movie_ids[idx]
                row = self.movies[self.movies["movie_id"] == mid].iloc[0]
                results.append({
                    "movie_id":         int(mid),
                    "title":            row["title"],
                    "predicted_rating": round(float(scores[idx]), 2),
                })
            return results
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._fallback(n)

    def _fallback(self, n):
        return [
            {"movie_id": 1, "title": "Toy Story (1995)",                   "predicted_rating": 4.5},
            {"movie_id": 2, "title": "Jumanji (1995)",                     "predicted_rating": 4.2},
            {"movie_id": 3, "title": "Grumpier Old Men (1995)",            "predicted_rating": 4.0},
            {"movie_id": 4, "title": "Waiting to Exhale (1995)",           "predicted_rating": 3.9},
            {"movie_id": 5, "title": "Father of the Bride Part II (1995)", "predicted_rating": 3.8},
        ][:n]


recommender = MovieRecommender()
