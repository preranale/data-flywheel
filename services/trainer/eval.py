# ============================================================
# services/trainer/eval.py
# ============================================================
# Evaluation suite for the trained model.
#
# Metrics we track:
#   RMSE  — Root Mean Squared Error (lower is better)
#          Standard metric for rating prediction tasks.
#          "On average, our predictions are X stars off"
#
#   MAE   — Mean Absolute Error (lower is better)
#          More interpretable than RMSE for non-technical
#          stakeholders. Less sensitive to outliers.
#
#   Coverage — % of val set items we can predict
#          A model that refuses to predict is useless even
#          if its RMSE on predictions looks great.
#
# Pass/fail threshold:
#   RMSE < 1.0 — predictions within 1 star on average
#   Coverage > 90% — model handles most users/movies
#
# WHY have a pass/fail gate?
# Without it, a buggy training run could deploy a broken model.
# The gate is your safety net between training and production.
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Thresholds — model must beat these to be promoted
RMSE_THRESHOLD     = 1.0   # Max acceptable RMSE
COVERAGE_THRESHOLD = 0.90  # Min acceptable coverage


def compute_metrics(model, val_df: pd.DataFrame) -> dict:
    """
    Evaluate model on the validation set.

    Args:
        model: trained scikit-learn model with .predict()
        val_df: DataFrame with columns [user_id, movie_id, rating]

    Returns:
        dict of metric name → value
    """
    if val_df.empty:
        logger.warning("Empty validation set — skipping evaluation")
        return {"rmse": 999.0, "mae": 999.0, "coverage": 0.0}

    pairs = val_df[["user_id", "movie_id"]].values.tolist()
    actual = val_df["rating"].values

    predictions = []
    failed = 0

    for pair in pairs:
        try:
            pred = model.predict([pair])[0]
            # Clip to valid rating range — model can predict outside [1,5]
            pred = float(np.clip(pred, 1.0, 5.0))
            predictions.append(pred)
        except Exception:
            # Model couldn't predict this pair
            # Use mean rating as fallback so we can still compute coverage
            predictions.append(3.0)
            failed += 1

    predictions = np.array(predictions)
    coverage = 1.0 - (failed / len(pairs))

    rmse = float(np.sqrt(np.mean((predictions - actual) ** 2)))
    mae  = float(np.mean(np.abs(predictions - actual)))

    metrics = {
        "rmse":     round(rmse, 4),
        "mae":      round(mae, 4),
        "coverage": round(coverage, 4),
        "n_samples": len(val_df),
        "n_failed":  failed,
    }

    logger.info(
        f"Eval results — RMSE: {rmse:.4f} | MAE: {mae:.4f} | "
        f"Coverage: {coverage:.1%} | Samples: {len(val_df)}"
    )

    return metrics


def passes_threshold(metrics: dict) -> bool:
    """
    Gate: should this model be promoted to production?

    Returns True only if the model meets ALL thresholds.
    If False, the model is logged to MLflow but NOT deployed.
    """
    rmse_ok     = metrics["rmse"] < RMSE_THRESHOLD
    coverage_ok = metrics["coverage"] >= COVERAGE_THRESHOLD

    if not rmse_ok:
        logger.warning(
            f"Model FAILED eval gate: RMSE {metrics['rmse']:.4f} "
            f">= threshold {RMSE_THRESHOLD}"
        )
    if not coverage_ok:
        logger.warning(
            f"Model FAILED eval gate: coverage {metrics['coverage']:.1%} "
            f"< threshold {COVERAGE_THRESHOLD:.1%}"
        )

    passed = rmse_ok and coverage_ok
    if passed:
        logger.info("Model PASSED eval gate — promoting to production")

    return passed
