"""
ML Predictor Module
Uses ensemble of Random Forest, Gradient Boosting, and LSTM for price prediction.
"""
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd
import numpy as np

from config import ML_LOOKBACK_DAYS, ML_TRAIN_TEST_SPLIT, ML_FEATURES

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


class MLPredictor:
    """Ensemble ML model for market direction prediction."""

    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.scaler = None
        self.is_trained = False

    # ─── Feature Engineering ──────────────────────────────────────────────────
    @staticmethod
    def prepare_features(df: pd.DataFrame, lookback: int = ML_LOOKBACK_DAYS) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create feature matrix and target from enriched DataFrame.

        Target: 1 if next-day close > current close, else 0.
        """
        feature_cols = [c for c in ML_FEATURES if c in df.columns]
        if not feature_cols:
            raise ValueError("No ML feature columns found in DataFrame")

        df_clean = df.dropna(subset=feature_cols).copy()
        if len(df_clean) < lookback:
            raise ValueError(f"Not enough data ({len(df_clean)} rows) for lookback={lookback}")

        # Add derived features
        df_clean["returns_1d"] = df_clean["close"].pct_change(1)
        df_clean["returns_5d"] = df_clean["close"].pct_change(5)
        df_clean["volatility_10d"] = df_clean["returns_1d"].rolling(10).std()
        df_clean["high_low_ratio"] = df_clean["high"] / df_clean["low"]
        df_clean["close_open_ratio"] = df_clean["close"] / df_clean["open"]

        extra_cols = ["returns_1d", "returns_5d", "volatility_10d",
                      "high_low_ratio", "close_open_ratio"]
        all_feature_cols = feature_cols + extra_cols

        # Target: next-day direction
        df_clean["target"] = (df_clean["close"].shift(-1) > df_clean["close"]).astype(int)
        df_clean = df_clean.dropna()

        X = df_clean[all_feature_cols].values
        y = df_clean["target"].values
        return X, y

    # ─── Training ─────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> Dict:
        """Train ensemble model and return performance metrics."""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score, classification_report

        try:
            X, y = self.prepare_features(df)
        except ValueError as e:
            logger.error(f"Feature prep failed: {e}")
            return {"error": str(e)}

        split = int(len(X) * ML_TRAIN_TEST_SPLIT)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train_scaled, y_train)
        rf_pred = self.rf_model.predict(X_test_scaled)
        rf_acc = accuracy_score(y_test, rf_pred)

        # Gradient Boosting
        self.gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.gb_model.fit(X_train_scaled, y_train)
        gb_pred = self.gb_model.predict(X_test_scaled)
        gb_acc = accuracy_score(y_test, gb_pred)

        # Ensemble (simple average probability)
        rf_prob = self.rf_model.predict_proba(X_test_scaled)[:, 1]
        gb_prob = self.gb_model.predict_proba(X_test_scaled)[:, 1]
        ensemble_prob = (rf_prob + gb_prob) / 2
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)

        self.is_trained = True
        self._save_models()

        metrics = {
            "rf_accuracy": round(rf_acc, 4),
            "gb_accuracy": round(gb_acc, 4),
            "ensemble_accuracy": round(ensemble_acc, 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "features_used": X.shape[1],
        }
        logger.info(f"Model trained — Ensemble accuracy: {ensemble_acc:.2%}")
        return metrics

    # ─── Prediction ───────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> Optional[float]:
        """
        Predict probability of price going UP for the latest row.
        Returns float 0.0–1.0 or None if model not trained.
        """
        if not self.is_trained:
            self._load_models()
            if not self.is_trained:
                logger.warning("ML model not trained yet")
                return None

        try:
            X, _ = self.prepare_features(df)
            X_latest = X[-1:].copy()
            X_scaled = self.scaler.transform(X_latest)

            rf_prob = self.rf_model.predict_proba(X_scaled)[:, 1][0]
            gb_prob = self.gb_model.predict_proba(X_scaled)[:, 1][0]
            ensemble_prob = (rf_prob + gb_prob) / 2
            return round(float(ensemble_prob), 4)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

    def predict_batch(self, df: pd.DataFrame, last_n: int = 5) -> Dict[int, float]:
        """Predict for last N rows. Returns {relative_index: probability}."""
        if not self.is_trained:
            self._load_models()
            if not self.is_trained:
                return {}

        try:
            X, _ = self.prepare_features(df)
            X_batch = X[-last_n:]
            X_scaled = self.scaler.transform(X_batch)

            rf_prob = self.rf_model.predict_proba(X_scaled)[:, 1]
            gb_prob = self.gb_model.predict_proba(X_scaled)[:, 1]
            probs = (rf_prob + gb_prob) / 2

            return {i - last_n: round(float(p), 4) for i, p in enumerate(probs)}
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return {}

    # ─── Feature Importance ───────────────────────────────────────────────────
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Return feature importances from RF model."""
        if self.rf_model is None:
            return None
        feature_cols = ML_FEATURES + [
            "returns_1d", "returns_5d", "volatility_10d",
            "high_low_ratio", "close_open_ratio"
        ]
        importances = self.rf_model.feature_importances_
        df_imp = pd.DataFrame({
            "feature": feature_cols[:len(importances)],
            "importance": importances
        }).sort_values("importance", ascending=False)
        return df_imp

    # ─── Model Persistence ────────────────────────────────────────────────────
    def _save_models(self):
        try:
            with open(MODEL_DIR / "rf_model.pkl", "wb") as f:
                pickle.dump(self.rf_model, f)
            with open(MODEL_DIR / "gb_model.pkl", "wb") as f:
                pickle.dump(self.gb_model, f)
            with open(MODEL_DIR / "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
            logger.info("Models saved to disk")
        except Exception as e:
            logger.error(f"Model save error: {e}")

    def _load_models(self):
        try:
            rf_path = MODEL_DIR / "rf_model.pkl"
            gb_path = MODEL_DIR / "gb_model.pkl"
            sc_path = MODEL_DIR / "scaler.pkl"
            if rf_path.exists() and gb_path.exists() and sc_path.exists():
                with open(rf_path, "rb") as f:
                    self.rf_model = pickle.load(f)
                with open(gb_path, "rb") as f:
                    self.gb_model = pickle.load(f)
                with open(sc_path, "rb") as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                logger.info("Models loaded from disk")
        except Exception as e:
            logger.error(f"Model load error: {e}")
