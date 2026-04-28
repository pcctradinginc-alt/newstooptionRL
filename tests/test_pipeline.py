"""
Tests für den Adaptive Asymmetry-Scanner
Ausführen: pytest tests/ -v

v8.2: Mismatch-Scorer-Tests aktualisiert:
  - _compute_48h_move wird gemockt (statt price_move_48h als Feld)
  - data_validation.eps_cross_check.deviation_pct (statt eps_drift.drift)
  - sample_analysis Fixture angepasst
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_history():
    return {
        "feature_stats": {
            "impact":    {"low": {"count": 0, "avg_return": 0.0}, "mid": {"count": 0, "avg_return": 0.0}, "high": {"count": 0, "avg_return": 0.0}},
            "mismatch":  {"weak": {"count": 0, "avg_return": 0.0}, "good": {"count": 0, "avg_return": 0.0}, "strong": {"count": 0, "avg_return": 0.0}},
            "eps_drift": {"noise": {"count": 0, "avg_return": 0.0}, "relevant": {"count": 0, "avg_return": 0.0}, "massive": {"count": 0, "avg_return": 0.0}},
        },
        "active_trades": [],
        "closed_trades": [],
        "model_weights": {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20},
    }


@pytest.fixture
def sample_candidate():
    return {
        "ticker": "AAPL",
        "info": {
            "marketCap": 3_000_000_000_000,
            "averageVolume10days": 50_000_000,
            "forwardEps": 7.5,
            "recommendationMean": 1.8,
            "sector": "Technology",
        },
        # v8.2: EPS-Drift jetzt über data_validation (korrekter Pfad)
        "data_validation": {
            "eps_cross_check": {
                "sec_eps": 6.7,
                "yf_eps": 7.5,
                "deviation_pct": 0.1067,
                "consistent": False,
                "confidence": "low",
                "source": "SEC_EDGAR",
                "data_anomaly": False,
            }
        },
        "news": ["Apple announces revolutionary AI chip", "iPhone sales beat estimates by 15%"],
        "prescreen_reason": "Technologischer Durchbruch – neue Produktkategorie",
    }


@pytest.fixture
def sample_analysis(sample_candidate):
    return {
        **sample_candidate,
        # v8.2: price_move_48h entfernt — wird jetzt intern vom Scorer berechnet
        "deep_analysis": {
            "ticker":                  "AAPL",
            "impact":                  8,
            "surprise":                7,
            "mispricing_logic":        "Markt unterschätzt langfristige Margenwirkung des AI-Chips.",
            "catalyst":                "Earnings Q2 2026 – 14. August",
            "time_to_materialization": "2-3 Monate",
            "bear_case":               "Makro-Abschwung könnte Consumer-Spending drücken.",
            "bear_case_severity":      4,
            "direction":               "BULLISH",
        },
    }


# ── Mismatch Scorer ───────────────────────────────────────────────────────────

class TestMismatchScorer:
    def test_high_impact_low_move_gives_high_mismatch(self):
        """Impact=9, 48h-Move=1% bei σ=2% → Z=0.5 → Mismatch=9-2.5=6.5 (strong)"""
        from modules.mismatch_scorer import MismatchScorer
        scorer = MismatchScorer()

        candidate = {
            "ticker": "AAPL",
            "info": {},
            "data_validation": {
                "eps_cross_check": {"deviation_pct": 0.12}
            },
            "news": [],
            "deep_analysis": {"impact": 9, "surprise": 8, "direction": "BULLISH"},
        }

        with patch.object(scorer, "_compute_sigma", return_value=0.02), \
             patch.object(scorer, "_compute_48h_move", return_value=0.01):
            result = scorer._score(candidate)

        assert result is not None
        mismatch = result["features"]["mismatch"]
        assert mismatch > 3, f"Erwartetes hohes Mismatch, aber: {mismatch}"
        assert result["features"]["bin_mismatch"] in ("good", "strong")
        # v8.2: EPS-Drift wird jetzt korrekt gelesen
        assert result["features"]["eps_drift"] == 0.12

    def test_low_impact_high_move_gives_negative_mismatch_filtered(self):
        """Impact=3, 48h-Move=8% bei σ=2% → Z=4.0 → Mismatch=3-20=-17 → gefiltert (None)"""
        from modules.mismatch_scorer import MismatchScorer
        scorer = MismatchScorer()

        candidate = {
            "ticker": "MSFT",
            "info": {},
            "data_validation": {
                "eps_cross_check": {"deviation_pct": 0.01}
            },
            "news": [],
            "deep_analysis": {"impact": 3, "surprise": 2, "direction": "BULLISH"},
        }

        with patch.object(scorer, "_compute_sigma", return_value=0.02), \
             patch.object(scorer, "_compute_48h_move", return_value=0.08):
            result = scorer._score(candidate)

        # v8.2: Negativer Mismatch → H-03 Filter → None
        assert result is None

    def test_moderate_impact_moderate_move(self):
        """Impact=5, 48h-Move=2% bei σ=2% → Z=1.0 → Mismatch=5-5=0 → gefiltert"""
        from modules.mismatch_scorer import MismatchScorer
        scorer = MismatchScorer()

        candidate = {
            "ticker": "GOOG",
            "info": {},
            "data_validation": {
                "eps_cross_check": {"deviation_pct": 0.03}
            },
            "news": [],
            "deep_analysis": {"impact": 5, "surprise": 4, "direction": "BULLISH"},
        }

        with patch.object(scorer, "_compute_sigma", return_value=0.02), \
             patch.object(scorer, "_compute_48h_move", return_value=0.02):
            result = scorer._score(candidate)

        # Mismatch=5-(1.0*5)=0 → ≤0 → gefiltert
        assert result is None

    def test_high_impact_tiny_move_passes(self):
        """Impact=8, 48h-Move=0.2% bei σ=2% → Z=0.1 → Mismatch=8-0.5=7.5 (strong)"""
        from modules.mismatch_scorer import MismatchScorer
        scorer = MismatchScorer()

        candidate = {
            "ticker": "NVDA",
            "info": {},
            "data_validation": {
                "eps_cross_check": {"deviation_pct": 0.15}
            },
            "news": [],
            "deep_analysis": {"impact": 8, "surprise": 7, "direction": "BULLISH"},
        }

        with patch.object(scorer, "_compute_sigma", return_value=0.02), \
             patch.object(scorer, "_compute_48h_move", return_value=0.002):
            result = scorer._score(candidate)

        assert result is not None
        mismatch = result["features"]["mismatch"]
        assert mismatch > 6, f"Erwartetes starkes Mismatch, aber: {mismatch}"
        assert result["features"]["bin_mismatch"] == "strong"
        assert result["features"]["eps_drift"] == 0.15

    def test_zero_sigma_returns_none(self):
        from modules.mismatch_scorer import MismatchScorer
        scorer = MismatchScorer()

        candidate = {
            "ticker": "X",
            "deep_analysis": {"impact": 5},
            "data_validation": {"eps_cross_check": {"deviation_pct": 0.0}},
            "info": {},
            "news": [],
        }

        with patch.object(scorer, "_compute_sigma", return_value=0.0), \
             patch.object(scorer, "_compute_48h_move", return_value=0.0):
            result = scorer._score(candidate)
        assert result is None

    def test_missing_data_validation_defaults_to_zero(self):
        """Wenn data_validation fehlt, soll eps_drift=0.0 sein (nicht crashen)."""
        from modules.mismatch_scorer import MismatchScorer
        scorer = MismatchScorer()

        candidate = {
            "ticker": "TSLA",
            "info": {},
            "news": [],
            "deep_analysis": {"impact": 7, "surprise": 6, "direction": "BULLISH"},
            # Kein data_validation Feld → soll graceful 0.0 liefern
        }

        with patch.object(scorer, "_compute_sigma", return_value=0.02), \
             patch.object(scorer, "_compute_48h_move", return_value=0.005):
            result = scorer._score(candidate)

        assert result is not None
        assert result["features"]["eps_drift"] == 0.0


# ── Quasi-ML ─────────────────────────────────────────────────────────────────

class TestQuasiML:
    def test_fallback_scoring_without_history(self, empty_history):
        from modules.quasi_ml import QuasiML
        qml = QuasiML(history=empty_history)

        signal = {
            "ticker": "NVDA",
            "features": {
                "impact": 9, "surprise": 8,
                "mismatch": 7.0, "z_score": 0.4, "sigma_30d": 0.02, "eps_drift": 0.15,
                "bin_impact": "high", "bin_mismatch": "strong", "bin_eps_drift": "massive",
            },
            "deep_analysis": {"direction": "BULLISH"},
            "simulation": {"hit_rate": 0.82},
        }
        result = qml.run([signal])
        assert len(result) == 1
        assert result[0]["final_score"] > 0

    def test_signals_sorted_by_score(self, empty_history):
        from modules.quasi_ml import QuasiML
        qml = QuasiML(history=empty_history)

        low = {"ticker": "A", "features": {"impact": 2, "mismatch": 1, "eps_drift": 0, "bin_impact": "low", "bin_mismatch": "weak", "bin_eps_drift": "noise"}, "deep_analysis": {}, "simulation": {}}
        high = {"ticker": "B", "features": {"impact": 9, "mismatch": 8, "eps_drift": 0.15, "bin_impact": "high", "bin_mismatch": "strong", "bin_eps_drift": "massive"}, "deep_analysis": {}, "simulation": {}}

        result = qml.run([low, high])
        assert result[0]["ticker"] == "B"


# ── Risk Gates ────────────────────────────────────────────────────────────────

class TestRiskGates:
    def test_vix_below_threshold_passes(self):
        from modules.risk_gates import RiskGates
        gates = RiskGates()
        with patch.object(gates, "_fetch_vix", return_value=20.0):
            assert gates.global_ok() is True

    def test_vix_above_threshold_blocks(self):
        from modules.risk_gates import RiskGates
        gates = RiskGates()
        with patch.object(gates, "_fetch_vix", return_value=40.0):
            assert gates.global_ok() is False


# ── Mirofish Simulation ───────────────────────────────────────────────────────

class TestMirofishSimulation:
    def test_strong_signal_passes_gate(self):
        from modules.mirofish_simulation import MirofishSimulation
        sim = MirofishSimulation()

        candidate = {
            "ticker": "NVDA",
            "features": {"mismatch": 8.0, "impact": 9},
            "deep_analysis": {
                "direction": "BULLISH",
                "time_to_materialization": "2-3 Monate",
            },
        }
        with patch.object(sim, "_get_market_params", return_value=(0.02, 500.0, "Technology")):
            result = sim._simulate(candidate)

        # Mit hohem Mismatch-Drift sollte die Simulation passieren
        assert result is not None
        assert result["simulation"]["hit_rate"] >= 0.70

    def test_zero_price_returns_none(self):
        from modules.mirofish_simulation import MirofishSimulation
        sim = MirofishSimulation()

        candidate = {
            "ticker": "BROKEN",
            "features": {"mismatch": 5.0, "impact": 5},
            "deep_analysis": {"direction": "BULLISH", "time_to_materialization": "2-3 Monate"},
        }
        with patch.object(sim, "_get_market_params", return_value=(0.02, 0.0, "default")):
            result = sim._simulate(candidate)
        assert result is None


# ── Feedback Loop ─────────────────────────────────────────────────────────────

class TestFeedbackLoop:
    def test_bin_update_running_average(self):
        import feedback
        stats = {}
        feedback.update_bin(stats, "impact", "high", 0.20)
        feedback.update_bin(stats, "impact", "high", 0.10)
        assert stats["impact"]["high"]["count"] == 2
        assert abs(stats["impact"]["high"]["avg_return"] - 0.15) < 1e-6

    def test_bin_to_num_mapping(self):
        import feedback
        assert feedback._bin_to_num("impact",   "high")     == 1.0
        assert feedback._bin_to_num("mismatch", "weak")     == 0.0
        assert feedback._bin_to_num("eps_drift","relevant") == 0.5
