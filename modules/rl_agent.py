"""
modules/rl_agent.py – PPO-Agent (Stable-Baselines3) für Options-Scoring

Ersetzt QuasiML komplett in Stufe 6 der Pipeline.

Architektur:
    - Algorithmus: PPO (Proximal Policy Optimization)
      Begründung: Stabil, sample-efficient für kleine Datensätze,
      keine Hyperparameter-Tuning nötig für MVP.
    - Policy: MlpPolicy (2 Hidden Layers à 64 Neurons)
      Begründung: Observation-Space hat nur 9 Dimensionen,
      tiefere Netze würden auf ~50 Trades overfitten.
    - Modell wird als PPO_options_agent.zip gespeichert.
    - Fallback: Wenn kein trainiertes Modell vorhanden →
      QuasiML-Logik als Fallback (Backward-Kompatibilität).

GitHub-Actions-Tauglichkeit:
    - Training läuft nur in feedback.py (offline, nach Trade-Close)
    - Inference in pipeline.py ist CPU-only, <10ms pro Signal
    - Modell-Datei wird in outputs/models/ gecacht und per Git committed
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from modules.rl_environment import (
    OptionsRLEnv,
    features_to_obs,
    ACTION_SKIP,
    ACTION_NORMAL,
    ACTION_BOOST,
    build_env_from_history,
)

log = logging.getLogger(__name__)

MODEL_PATH = Path("outputs/models/ppo_options_agent.zip")


# ── Training ──────────────────────────────────────────────────────────────────

def train_agent(
    history: dict,
    total_timesteps: int = 10_000,
    force_retrain: bool = False,
) -> bool:
    """
    Trainiert den PPO-Agenten auf closed_trades aus history.json.

    Args:
        history:         history.json als Dict
        total_timesteps: Trainings-Steps (10k reichen für <100 Trades)
        force_retrain:   Wenn True, überschreibt existierendes Modell

    Returns:
        True wenn Training erfolgreich, False wenn zu wenig Daten
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
    except ImportError as e:
        log.error(f"stable-baselines3 nicht installiert: {e}")
        return False

    closed = history.get("closed_trades", [])
    if len(closed) < 5:
        log.info(
            f"Nur {len(closed)} closed_trades → Training übersprungen "
            f"(Minimum: 5)."
        )
        return False

    env = build_env_from_history(history)
    if env is None:
        return False

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Existierendes Modell weiter-trainieren (Continual Learning)
    if MODEL_PATH.exists() and not force_retrain:
        try:
            model = PPO.load(str(MODEL_PATH), env=env)
            log.info(
                f"Existierendes Modell geladen, weiter-trainiere "
                f"({total_timesteps} steps)..."
            )
        except Exception as e:
            log.warning(f"Modell-Laden fehlgeschlagen ({e}) → Neu-Training")
            model = _create_new_model(env)
    else:
        model = _create_new_model(env)

    log.info(f"PPO-Training startet ({total_timesteps} timesteps)...")
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model.save(str(MODEL_PATH))
    log.info(f"Modell gespeichert: {MODEL_PATH}")
    return True


def _create_new_model(env: OptionsRLEnv):
    """Erstellt neues PPO-Modell mit bewährten Hyperparametern."""
    from stable_baselines3 import PPO

    return PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=min(64, len(env.trade_data)),   # Kleinere Batches für kleine Datasets
        batch_size=min(32, len(env.trade_data)),
        n_epochs=10,
        gamma=0.95,            # Etwas niedrigerer Discount (kürzere Episoden)
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # Leichte Exploration-Encouragement
        policy_kwargs={
            "net_arch": [64, 64],   # 2 Hidden Layers reichen für 9 Features
        },
        verbose=0,
    )


# ── Inference ─────────────────────────────────────────────────────────────────

class RLScorer:
    """
    Drop-in-Ersatz für QuasiML.
    Nutzt den trainierten PPO-Agenten für final_score-Berechnung.

    Fallback: Wenn kein Modell vorhanden, wird die QuasiML-Logik genutzt.
    """

    def __init__(self, history: dict):
        self.history = history
        self._model  = None
        self._load_model()

    def _load_model(self) -> None:
        """Lazy-Load des trainierten PPO-Modells."""
        if not MODEL_PATH.exists():
            log.info(
                "Kein trainiertes RL-Modell gefunden → QuasiML-Fallback aktiv. "
                "Führe feedback.py aus um das erste Modell zu trainieren."
            )
            return

        try:
            from stable_baselines3 import PPO
            self._model = PPO.load(str(MODEL_PATH))
            log.info(f"RL-Modell geladen: {MODEL_PATH}")
        except Exception as e:
            log.warning(f"RL-Modell-Laden fehlgeschlagen: {e} → QuasiML-Fallback")

    def run(self, simulated: list[dict]) -> list[dict]:
        """
        Stufe 6: Berechnet final_score für jeden Signal-Dict.

        Wenn RL-Modell vorhanden: PPO-Inference
        Sonst: QuasiML-Fallback
        """
        if self._model is None:
            return self._quasi_ml_fallback(simulated)

        scored = []
        for s in simulated:
            obs = features_to_obs(
                features      = s.get("features", {}),
                simulation    = s.get("simulation", {}),
                deep_analysis = s.get("deep_analysis", {}),
            )

            # PPO-Inference: deterministisch (kein Exploration-Noise)
            action, _ = self._model.predict(obs, deterministic=True)
            action    = int(action)

            # raw_score: gewichteter Score aus Observation-Vektor
            # (interpretierbar, ähnlich wie QuasiML-Score)
            raw_score = self._compute_raw_score(s)

            if action == ACTION_SKIP:
                final_score = 0.0
                action_str  = "SKIP"
            elif action == ACTION_NORMAL:
                final_score = raw_score
                action_str  = "NORMAL"
            else:  # BOOST
                final_score = raw_score * 1.5
                action_str  = "BOOST"

            log.info(
                f"  [{s['ticker']}] RL-Action={action_str} "
                f"raw={raw_score:.4f} final={final_score:.4f}"
            )

            # SKIP-Signale werden komplett gefiltert (kein Trade-Vorschlag)
            if action == ACTION_SKIP:
                log.info(f"  [{s['ticker']}] → SKIP: nicht weitergeleitet.")
                continue

            scored.append({**s, "final_score": round(final_score, 4)})

        # Absteigend nach final_score sortieren
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored

    def _compute_raw_score(self, s: dict) -> float:
        """
        Interpretierbare Score-Berechnung aus Observation-Vektor.
        Gleichgewichtete Summe der normierten Features.
        Ähnlich wie QuasiML, aber ohne historische Bins.
        """
        features   = s.get("features", {})
        simulation = s.get("simulation", {})
        da         = s.get("deep_analysis", {})

        obs = features_to_obs(features, simulation, da)

        # Gewichtungen entsprechen den QuasiML-Gewichten aus config.yaml
        weights = np.array([
            0.15,   # impact
            0.10,   # surprise
            0.20,   # mismatch  (stärkstes Signal)
            0.05,   # z_score
            0.10,   # eps_drift
            0.15,   # hit_rate
            0.05,   # iv_rank
            0.10,   # sentiment (neu durch FinBERT)
            0.10,   # bear_severity (invertiert)
        ], dtype=np.float32)

        return float(np.dot(obs, weights))

    def _quasi_ml_fallback(self, simulated: list[dict]) -> list[dict]:
        """
        Fallback auf QuasiML-Logik wenn kein RL-Modell vorhanden.
        Vollständige Backward-Kompatibilität.
        """
        log.info("QuasiML-Fallback aktiv (kein RL-Modell).")
        from modules.quasi_ml import QuasiML
        qml = QuasiML(history=self.history)
        return qml.run(simulated)
