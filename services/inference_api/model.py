# ============================================================
# services/inference_api/model.py
# ============================================================
# Responsible for loading and running the ML model.
#
# Design decision: we separate model loading from API routing.
# This is called the "Single Responsibility Principle" — each
# file does exactly one thing. main.py handles HTTP, model.py
# handles ML. Easier to test, easier to swap models later.
# ============================================================

import os
import pickle
import numpy as np
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "/data/models")
MODEL_FILE = os.path.join(MODEL_PATH, "model.pkl")
MOVIES_FILE = os.path.join(MODEL_PATH, "movies.pkl")


class MovieRecommender:
    """
    Wraps the trained scikit-learn model.
    Provides a clean .recommend() interface to the API.
    
    Why a class instead of bare functions?
    The model is loaded once at startup and reused across
    all requests. A class holds that state cleanly.
    """

    def __init__(self):
        self.model = None
        self.movies = None
        self.is_ready = False

    def load(self):
        """
        Load model from disk.
        Called once at API startup.
        Returns False if no model exists yet (cold start).
        """
        if not os.path.exists(MODEL_FILE):
            logger.warning(
                f"No model found at {MODEL_FILE}. "
                "API will return fallback recommendations until "
                "the first training run completes."
            )
            self.is_ready = False
            return False

        try:
            with open(MODEL_FILE, "rb") as f:
                self.model = pickle.load(f)
            with open(MOVIES_FILE, "rb") as f:
                self.movies = pickle.load(f)
            self.is_ready = True
            logger.info(f"Model loaded successfully from {MODEL_FILE}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_ready = False
            return False

    def recommend(self, user_id: int, n: int = 5) -> list[dict]:
        """
        Return top-N movie recommendations for a user.
        
        If model isn't ready yet (cold start), return popular
        movies as a fallback. This keeps the API always usable.
        
        Args:
            user_id: the user requesting recommendations
            n: how many recommendations to return
        
        Returns:
            list of dicts with movie_id, title, score
        """
        if not self.is_ready:
            return self._fallback_recommendations(n)

        try:
            # Get all movie IDs
            movie_ids = self.movies["movie_id"].tolist()
            
            # Ask the model to predict ratings for all movies
            # We create (user_id, movie_id) pairs for each movie
            pairs = [[user_id, mid] for mid in movie_ids]
            scores = self.model.predict(pairs)
            
            # Sort by predicted score, take top N
            top_indices = np.argsort(scores)[::-1][:n]
            
            recommendations = []
            for idx in top_indices:
                movie_id = movie_ids[idx]
                movie_row = self.movies[self.movies["movie_id"] == movie_id].iloc[0]
                recommendations.append({
                    "movie_id": int(movie_id),
                    "title": movie_row["title"],
                    "predicted_rating": round(float(scores[idx]), 2),
                })
            return recommendations

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._fallback_recommendations(n)

    def _fallback_recommendations(self, n: int) -> list[dict]:
        """
        Returns hardcoded popular movies when model isn't ready.
        
        WHY: A good API never returns an error when it can return
        something useful. This is called 'graceful degradation'.
        Users get results; logs tell us the model isn't loaded.
        """
        fallback = [
            {"movie_id": 1, "title": "Toy Story (1995)", "predicted_rating": 4.5},
            {"movie_id": 2, "title": "Jumanji (1995)", "predicted_rating": 4.2},
            {"movie_id": 3, "title": "Grumpier Old Men (1995)", "predicted_rating": 4.0},
            {"movie_id": 4, "title": "Waiting to Exhale (1995)", "predicted_rating": 3.9},
            {"movie_id": 5, "title": "Father of the Bride Part II (1995)", "predicted_rating": 3.8},
        ]
        return fallback[:n]


# Single instance shared across all requests
# This is the "singleton pattern" — we load the model once,
# not on every request (that would be extremely slow)
recommender = MovieRecommender()
