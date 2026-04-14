"""
modules/mirofish_simulation.py v8.0

Änderungen:
  - days_to_expiry Parameter: Simulation läuft exakt bis zum Options-Verfall
  - Adaptive Pfad-Anzahl: 10.000 Pfade für > 90 Tage (stabil für LEAPS)
  - Time-Value-Efficiency Score: ROI/Tag für fairen Laufzeit-Vergleich
  - EMA-200 Hard-Gate (aus v7)
  - MACD-Multiplikator (aus v7)
  - Kopplung mit options_designer: dte wird übergeben, nicht aus config
"""

import logging
import time
from typing import Optional

import numpy as np
import yfinance as yf

from modules.config import cfg

log = logging.getLogger(__name__)

# Pfad-Anzahl nach Laufzeit
N_PATHS_SHORT  = 5_000    # < 90 Tage
N_PATHS_LONG   = 10_000   # >= 90 Tage (LEAPS brauchen mehr für Stabilität)

NARRATIVE_DECAY = {
    "4-8 Wochen":  0.015,
    "2-3 Monate":  0.008,
    "6 Monate":    0.004,
}

SECTOR_VOL_MULT = {
    "Technology": 1.3, "Healthcare": 0.9, "Energy": 1.2,
    "Financial Services": 1.1, "Consumer Cyclical": 1.0,
    "Consumer Defensive": 0.8, "Real Estate": 0.9,
    "Utilities": 0.7, "Industrials": 1.0, "Basic Materials": 1.1,
    "Communication Services": 1.2, "default": 1.0,
}


class MirofishSimulation:

    def __init__(self):
        seed      = int(time.time_ns() % (2**32))
        self._rng = np.random.default_rng(seed=seed)

    def run(self, scored: list[dict]) -> list[dict]:
        """
        Standard-Run: Laufzeit aus config (für Rückwärtskompatibilität).
        Für adaptive Laufzeiten: run_for_dte() direkt aufrufen.
        """
        min_impact = getattr(
            getattr(cfg, "pipeline", None), "min_impact_threshold", 4
        )
        passing = []
        for s in scored:
            if s.get("features", {}).get("impact", 0) < min_impact:
                continue

            # EMA-200 Gate
            ema_ok, ema_info = self._check_ema200(s["ticker"], s.get("deep_analysis", {}).get("direction", "BULLISH"))
            s["ema200_check"] = ema_info
            if not ema_ok:
                log.info(f"  [{s['ticker']}] EMA-200-GATE → verworfen")
                continue

            # Standard-DTE aus config
            dte = getattr(getattr(cfg, "pipeline", None), "simulation_days", 90)
            result = self._simulate(s, days_to_expiry=dte)
            if result:
                passing.append(result)

        return passing

    def run_for_dte(self, s: dict, days_to_expiry: int) -> Optional[dict]:
        """
        Expliziter DTE-Parameter vom options_designer.
        Wird für adaptive Laufzeit-Loop aufgerufen.
        """
        return self._simulate(s, days_to_expiry=days_to_expiry)

    # ── Haupt-Simulation ──────────────────────────────────────────────────────

    def _simulate(self, s: dict, days_to_expiry: int) -> Optional[dict]:
        ticker    = s["ticker"]
        features  = s.get("features", {})
        da        = s.get("deep_analysis", {})
        direction = da.get("direction", "BULLISH")
        ttm       = da.get("time_to_materialization", "2-3 Monate")
        mismatch  = features.get("mismatch", 0)

        if mismatch <= 0:
            return None

        sigma, current_price, sector = self._get_market_params(ticker)
        if current_price <= 0:
            return None

        # Volatilitäts-Skalierung: tägliche Vola bereits korrekt via std(daily_returns)
        # Wurzel-T-Regel ist implizit: sigma ist tägliche Std, GBM skaliert über n_days
        # σ_annual = σ_daily × √252 (zur Dokumentation)
        vol_mult   = SECTOR_VOL_MULT.get(sector, 1.0)
        sigma_adj  = sigma * vol_mult

        impact            = features.get("impact", 5)
        decay_rate        = NARRATIVE_DECAY.get(ttm, 0.008)
        impact_multiplier = min(1.0 + (impact / 20.0), 1.3)
        base_alpha        = (mismatch / 100.0) * impact_multiplier

        # MACD-Multiplikator
        macd_data = self._compute_macd(ticker)
        s["macd"] = macd_data
        if macd_data.get("data_available"):
            score = macd_data["momentum_score"]
            mult  = 1.0 + (score * 0.20 if direction == "BULLISH" else -score * 0.20)
            base_alpha *= max(0.70, min(1.30, mult))

        if direction == "BEARISH":
            base_alpha = -base_alpha

        target_move  = getattr(getattr(cfg, "options", None), "target_move_pct", 0.08)
        target_price = (
            current_price * (1 + target_move)
            if direction == "BULLISH"
            else current_price * (1 - target_move)
        )

        # Adaptive Pfad-Anzahl: mehr Pfade für längere Laufzeiten
        n_paths   = N_PATHS_LONG if days_to_expiry >= 90 else N_PATHS_SHORT
        n_days    = days_to_expiry
        threshold = getattr(
            getattr(cfg, "pipeline", None), "confidence_gate", 0.60
        )

        log.debug(
            f"  [{ticker}] Simulation: {n_paths} Pfade × {n_days} Tage "
            f"(alpha={base_alpha:.4f})"
        )

        # Monte-Carlo GBM
        paths_hit = 0
        for _ in range(n_paths):
            price = current_price
            hit   = False
            for day in range(n_days):
                alpha_today  = base_alpha * np.exp(-decay_rate * day)
                daily_return = alpha_today + sigma_adj * self._rng.standard_normal()
                price       *= (1 + daily_return)
                if direction == "BULLISH" and price >= target_price:
                    hit = True; break
                if direction == "BEARISH" and price <= target_price:
                    hit = True; break
            if hit:
                paths_hit += 1

        hit_rate = paths_hit / n_paths

        # Statistische Stabilität prüfen (Standardfehler des Schätzers)
        stderr = np.sqrt(hit_rate * (1 - hit_rate) / n_paths)
        log.info(
            f"  [{ticker}] Simulation ({n_days}d, {n_paths} Pfade): "
            f"Hit={hit_rate:.1%} ±{stderr:.1%} "
            f"({'PASS' if hit_rate >= threshold else 'FAIL'})"
        )

        if hit_rate < threshold:
            return None

        return {
            **s,
            "simulation": {
                "hit_rate":       round(hit_rate, 4),
                "stderr":         round(stderr, 4),
                "n_paths":        n_paths,
                "n_days":         n_days,
                "days_to_expiry": days_to_expiry,
                "target_price":   round(target_price, 2),
                "current_price":  round(current_price, 2),
                "sigma_adj":      round(sigma_adj, 4),
                "sector":         sector,
                "ema200":         s.get("ema200_check", {}),
                "macd":           macd_data,
            },
        }

    # ── EMA-200 Gate ──────────────────────────────────────────────────────────

    def _check_ema200(self, ticker: str, direction: str) -> tuple[bool, dict]:
        tolerance = -0.02
        try:
            hist = yf.Ticker(ticker).history(period="1y")
            if hist.empty or len(hist) < 50:
                return True, {"data_available": False}
            closes  = hist["Close"]
            ema200  = float(closes.ewm(span=200, adjust=False).mean().iloc[-1])
            current = float(closes.iloc[-1])
            pct     = (current - ema200) / ema200
            info    = {"current": round(current, 2), "ema200": round(ema200, 2),
                       "pct_vs_ema": round(pct, 4), "data_available": True}
            passes  = pct >= tolerance if direction == "BULLISH" else pct <= -tolerance
            return passes, info
        except Exception as e:
            return True, {"data_available": False, "error": str(e)}

    # ── MACD ──────────────────────────────────────────────────────────────────

    def _compute_macd(self, ticker: str) -> dict:
        try:
            hist = yf.Ticker(ticker).history(period="1y")
            if hist.empty or len(hist) < 35:
                return {"data_available": False, "momentum_score": 0.0}
            closes  = hist["Close"]
            macd    = closes.ewm(span=12, adjust=False).mean() - closes.ewm(span=26, adjust=False).mean()
            signal  = macd.ewm(span=9, adjust=False).mean()
            histo   = macd - signal
            h_std   = float(histo.rolling(30).std().iloc[-1])
            score   = float(np.clip(histo.iloc[-1] / h_std, -1.0, 1.0)) if h_std > 0 else 0.0
            return {
                "macd": round(float(macd.iloc[-1]), 4),
                "signal": round(float(signal.iloc[-1]), 4),
                "histogram": round(float(histo.iloc[-1]), 4),
                "momentum_score": round(score, 4),
                "data_available": True,
            }
        except Exception:
            return {"data_available": False, "momentum_score": 0.0}

    # ── Marktdaten ────────────────────────────────────────────────────────────

    def _get_market_params(self, ticker: str) -> tuple[float, float, str]:
        try:
            t    = yf.Ticker(ticker)
            info = t.info
            hist = t.history(period="35d")
            current = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
            sector  = info.get("sector", "default")
            sigma   = float(np.std(hist["Close"].pct_change().dropna())) if len(hist) >= 10 else 0.02
            return sigma, current, sector
        except Exception:
            return 0.02, 0.0, "default"


# ── Time-Value-Efficiency Score ───────────────────────────────────────────────

def compute_time_value_efficiency(roi_net: float, dte: int) -> dict:
    """
    Time-Value-Efficiency = ROI / Tag.
    Ermöglicht fairen Vergleich zwischen Laufzeiten.

    Beispiel:
        15% auf 30d  = 0.50%/Tag  → sehr effizient
        18% auf 180d = 0.10%/Tag  → weniger effizient
        aber annualisiert: 30d=60% p.a. vs. 180d=36% p.a.

    Beide Metriken werden ausgegeben damit das System
    und der Nutzer informiert entscheiden können.
    """
    if dte <= 0:
        return {"roi_per_day": 0.0, "annualized_roi": 0.0, "dte": dte}

    roi_per_day    = roi_net / dte
    annualized_roi = (1 + roi_net) ** (365 / dte) - 1

    efficiency_label = (
        "high"   if roi_per_day >= 0.005  else   # >= 0.5%/Tag
        "medium" if roi_per_day >= 0.002  else   # >= 0.2%/Tag
        "low"
    )

    return {
        "roi_per_day":      round(roi_per_day, 6),
        "roi_per_day_pct":  round(roi_per_day * 100, 4),
        "annualized_roi":   round(annualized_roi, 4),
        "efficiency_label": efficiency_label,
        "dte":              dte,
    }
