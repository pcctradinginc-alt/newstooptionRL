"""
Stufe 6: Adaptive Quasi-ML Scoring (Selbstlern-Kern)
- Unverändert vom Original (keine Bugs gefunden)
- cfg: MIN_BIN_COUNT aus config.yaml
"""
import logging
from modules.config import cfg

log = logging.getLogger(__name__)


class QuasiML:
    def __init__(self, history: dict):
        self.history    = history
        self.feat_stats = history.get("feature_stats", {})
        self.weights    = history.get("model_weights", {
            "impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20,
        })

    def run(self, simulated: list[dict]) -> list[dict]:
        scored = []
        for s in simulated:
            final_score = self._compute_final_score(s)
            scored.append({**s, "final_score": round(final_score, 4)})
            log.info(f"  [{s['ticker']}] FinalScore={final_score:.4f}")
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored

    def _compute_final_score(self, s: dict) -> float:
        features = s.get("features", {})
        total = 0.0
        for feat_name, bin_label in {
            "impact": features.get("bin_impact"),
            "mismatch": features.get("bin_mismatch"),
            "eps_drift": features.get("bin_eps_drift"),
        }.items():
            if bin_label is None:
                continue
            total += self._get_bin_avg_return(feat_name, bin_label) * self.weights.get(feat_name, 0.0)
        if total == 0.0:
            total = self._fallback_score(features)
        return total

    def _get_bin_avg_return(self, feature: str, bin_label: str) -> float:
        try:
            stats = self.feat_stats.get(feature, {}).get(bin_label, {})
            if stats.get("count", 0) < cfg.learning.min_bin_count:
                return self._prior_return(feature, bin_label)
            return stats.get("avg_return", 0.0)
        except Exception:
            return 0.0

    def _prior_return(self, feature: str, bin_label: str) -> float:
        priors = {
            "impact":    {"low": -0.02, "mid": 0.04, "high": 0.12},
            "mismatch":  {"weak": -0.01, "good": 0.05, "strong": 0.15},
            "eps_drift": {"noise": 0.00, "relevant": 0.04, "massive": 0.10},
        }
        return priors.get(feature, {}).get(bin_label, 0.0)

    def _fallback_score(self, features: dict) -> float:
        impact   = features.get("impact", 0) / 10.0
        mismatch = min(features.get("mismatch", 0) / 10.0, 1.0)
        drift    = min(abs(features.get("eps_drift", 0)), 0.2) / 0.2
        return (
            impact   * self.weights.get("impact",    0.35) +
            mismatch * self.weights.get("mismatch",  0.45) +
            drift    * self.weights.get("eps_drift", 0.20)
        )
