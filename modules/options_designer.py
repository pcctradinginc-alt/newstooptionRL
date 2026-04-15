"""
modules/options_designer.py v7.0

Adaptive Laufzeit-Loop:
    Wenn primäre Laufzeit (30d) ROI-Gate nicht besteht →
    automatisch Mid-Term (90-120d) und Long-Term (180d+) prüfen.

    Wichtig: Annualisierter ROI wird geloggt damit Laufzeit-Vergleich
    fair ist. 15% auf 180d = 10% p.a. vs. 15% auf 30d = 60% p.a.

IV-Rank Gate bleibt aktiv für alle Laufzeiten:
    Bei IV-Rank > 70%: Spread statt Long Call (noch wichtiger bei LEAPS).
"""

from __future__ import annotations
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import requests
import yfinance as yf

from modules.config       import cfg
from modules.macro_context import get_macro_regime_multiplier

log = logging.getLogger(__name__)

TRADIER_BASE     = "https://api.tradier.com/v1"
IV_RANK_GATE     = 70.0

# Laufzeit-Stufen für den adaptiven Loop
DTE_TIERS = [
    {"label": "Short-Term",  "dte_min": 21,  "dte_max": 45,  "min_roi": 0.15},
    {"label": "Mid-Term",    "dte_min": 75,  "dte_max": 135, "min_roi": 0.12},
    {"label": "Long-Term",   "dte_min": 150, "dte_max": 270, "min_roi": 0.10},
]

SECTOR_ETF = {
    "Technology": "XLK", "Healthcare": "XLV", "Biotechnology": "XBI",
    "Financial Services": "XLF", "Financials": "XLF", "Energy": "XLE",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
    "Industrials": "XLI", "Basic Materials": "XLB",
    "Real Estate": "XLRE", "Utilities": "XLU",
    "Communication Services": "XLC", "default": "SPY",
}

RELATIVE_STRENGTH_MIN = -0.03


class OptionsDesigner:

    def __init__(self, gates):
        self.gates = gates

    def run(self, signals: list[dict]) -> list[dict]:
        proposals = []
        for s in signals:
            ticker = s.get("ticker", "")

            if not self._bear_case_ok(s):
                continue

            if not self._sector_momentum_ok(s):
                log.info(f"  [{ticker}] SECTOR-GATE → verworfen")
                continue

            proposal = self._design_with_adaptive_dte(s)
            if proposal:
                proposals.append(proposal)

        return proposals

    # ── Adaptiver Laufzeit-Loop ───────────────────────────────────────────────

    def _design_with_adaptive_dte(self, s: dict) -> Optional[dict]:
        """
        Kern des Fix: Prüft Short → Mid → Long-Term Laufzeiten.
        Wählt die erste Laufzeit die ROI-Gate besteht.
        """
        ticker    = s["ticker"]
        direction = s.get("deep_analysis", {}).get("direction", "BULLISH")
        sim       = s.get("simulation", {})
        current   = sim.get("current_price", 0)
        iv_rank   = self._get_iv_rank(ticker)

        if current <= 0:
            return None

        if self.gates.has_upcoming_earnings(ticker):
            log.info(f"  [{ticker}] EARNINGS-GATE → blockiert")
            return None

        results_per_tier = []

        for tier in DTE_TIERS:
            label   = tier["label"]
            min_roi = tier["min_roi"]

            strategy = self._select_strategy(ticker, direction, iv_rank)
            option   = self._find_option_for_dte(
                ticker, strategy, current,
                tier["dte_min"], tier["dte_max"]
            )

            if not option:
                log.info(f"  [{ticker}] {label}: kein Kontrakt verfügbar")
                continue

            roi = self._compute_roi(option, sim, iv_rank, tier)

            annualized_roi = (
                (1 + roi["roi_net"]) ** (365 / max(option["dte"], 1)) - 1
            )

            log.info(
                f"  [{ticker}] {label} ({int(option['dte'] or 0)}d): "
                f"ROI={roi['roi_net']:.1%} "
                f"(ann.={annualized_roi:.1%}) "
                f"{'✅ PASS' if roi['passes_roi_gate'] else '❌ FAIL'}"
            )

            results_per_tier.append({
                "tier":           label,
                "dte":            option["dte"],
                "option":         option,
                "roi":            roi,
                "annualized_roi": round(annualized_roi, 4),
                "strategy":       strategy,
            })

            if roi["passes_roi_gate"]:
                # Erster Tier der besteht → nehmen
                tier_idx = DTE_TIERS.index(tier)
                if tier_idx > 0:
                    # War nicht der primäre Tier → explizit loggen
                    prev_label = DTE_TIERS[0]["label"]
                    prev_roi   = results_per_tier[0]["roi"]["roi_net"] if results_per_tier else None
                    if prev_roi is not None:
                        log.info(
                            f"  [{ticker}] ⚡ LAUFZEIT-RETTUNG: "
                            f"{prev_label} ROI={prev_roi:.1%} (FAIL) → "
                            f"{label} ROI={roi['roi_net']:.1%} (PASS) — "
                            f"Trade akzeptiert mit {option['dte']}d Laufzeit"
                        )

                return {
                    "ticker":            ticker,
                    "strategy":          strategy,
                    "iv_rank":           iv_rank,
                    "iv_gate_applied":   iv_rank >= IV_RANK_GATE,
                    "direction":         direction,
                    "option":            option,
                    "roi_analysis":      roi,
                    "dte_tier":          label,
                    "annualized_roi":    round(annualized_roi, 4),
                    "all_tiers_tried":   results_per_tier,
                    "features":          s.get("features", {}),
                    "simulation":        s.get("simulation", {}),
                    "deep_analysis":     s.get("deep_analysis", {}),
                    "sector_momentum":   s.get("sector_momentum", {}),
                    "final_score":       s.get("final_score", 0),
                }

        # Alle 3 Tiers gescheitert
        tried = ", ".join(
            f"{r['tier']}={r['roi']['roi_net']:.1%}" for r in results_per_tier
        )
        log.info(
            f"  [{ticker}] Alle Laufzeiten unter ROI-Gate: {tried} → verworfen"
        )
        return None

    # ── Strategie-Wahl (IV-Gate) ──────────────────────────────────────────────

    def _select_strategy(self, ticker: str, direction: str, iv_rank: float) -> str:
        if iv_rank >= IV_RANK_GATE:
            s = "BULL_CALL_SPREAD" if direction == "BULLISH" else "BEAR_PUT_SPREAD"
            log.info(
                f"  [{ticker}] IV={iv_rank:.0f}% ≥ {IV_RANK_GATE:.0f}% "
                f"→ {s} (Vega-Schutz)"
            )
        else:
            s = "LONG_CALL" if direction == "BULLISH" else "LONG_PUT"
        return s

    # ── ROI-Berechnung ────────────────────────────────────────────────────────

    def _compute_roi(
        self, option: dict, sim: dict, iv_rank: float, tier: dict
    ) -> dict:
        import math

        bid     = option.get("bid", 0) or 0
        ask     = option.get("ask", 0) or 0
        strike  = option.get("strike", 0) or 0
        iv      = option.get("implied_vol", 0.30) or 0.30
        dte     = option.get("dte", 120) or 120
        current = sim.get("current_price", 0) or 0
        target  = sim.get("target_price", 0) or 0
        min_roi = tier["min_roi"]

        if ask <= 0 or current <= 0:
            return {"roi_net": 0.0, "passes_roi_gate": False,
                    "roi_gross": 0.0, "spread_pct": 0.0,
                    "vega_loss": 0.0, "min_roi_threshold": min_roi}

        spread_pct = (ask - bid) / ask if ask > 0 else 0.0
        T          = dte / 365.0

        # Black-Scholes Delta + Vega
        try:
            d1    = (math.log(current / strike) + 0.5 * iv**2 * T) / (iv * math.sqrt(T))
            delta = (1.0 + math.erf(d1 / math.sqrt(2.0))) / 2.0
            vega  = current * math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi) * math.sqrt(T)
        except Exception:
            delta, vega = 0.5, 0.0

        leverage      = current / ask if ask > 0 else 1.0
        expected_move = (target - current) / current if target > current else 0.0
        roi_delta     = expected_move * delta * leverage

        # IV-Crush nach Event (höher bei kurzen Laufzeiten)
        if dte <= 45:
            iv_drop = 0.25 if iv_rank >= 70 else (0.12 if iv_rank >= 50 else 0.05)
        elif dte <= 135:
            iv_drop = 0.15 if iv_rank >= 70 else (0.07 if iv_rank >= 50 else 0.02)
        else:
            # LEAPS: IV-Crush nach Event verpufft — weniger Einfluss
            iv_drop = 0.08 if iv_rank >= 70 else (0.03 if iv_rank >= 50 else 0.01)

        vega_loss = max(vega * iv * iv_drop * leverage, 0.0)
        roi_net   = roi_delta - (spread_pct * 2) - vega_loss
        passes    = roi_net >= min_roi

        if not passes:
            log.debug(
                f"    ROI: delta={roi_delta:.2%} spread={spread_pct:.2%} "
                f"vega={vega_loss:.2%} net={roi_net:.2%} < {min_roi:.0%}"
            )

        return {
            "roi_gross":         round(roi_delta, 4),
            "roi_net":           round(roi_net, 4),
            "spread_pct":        round(spread_pct, 4),
            "vega_loss":         round(vega_loss, 4),
            "delta":             round(delta, 4),
            "iv_drop_assumed":   iv_drop,
            "passes_roi_gate":   passes,
            "min_roi_threshold": min_roi,
            "dte":               dte,
        }

    # ── Kontrakt-Suche für spezifischen DTE-Bereich ──────────────────────────

    def _find_option_for_dte(
        self, ticker: str, strategy: str, current: float,
        dte_min: int, dte_max: int,
    ) -> Optional[dict]:
        try:
            t     = yf.Ticker(ticker)
            dates = [
                d for d in (t.options or [])
                if dte_min <= self._days_to(d) <= dte_max
            ]
            if not dates:
                return None

            best_expiry = dates[0]
            chain       = t.option_chain(best_expiry)
            is_call     = "CALL" in strategy or "BULL" in strategy
            opts        = chain.calls if is_call else chain.puts

            filtered = opts[
                (opts["strike"] >= current * 1.00) &
                (opts["strike"] <= current * 1.12) &
                (opts["openInterest"] >= cfg.risk.min_open_interest)
            ].copy()

            if filtered.empty:
                return None

            filtered["spread_ratio"] = (
                (filtered["ask"] - filtered["bid"]) /
                filtered["ask"].clip(lower=0.01)
            )
            filtered = filtered[
                filtered["spread_ratio"] <= cfg.risk.max_bid_ask_ratio
            ]
            if filtered.empty:
                return None

            best = filtered.sort_values("openInterest", ascending=False).iloc[0]
            dte  = self._days_to(best_expiry)

            result = {
                "expiry":       best_expiry,
                "strike":       float(best["strike"]),
                "bid":          float(best["bid"]),
                "ask":          float(best["ask"]),
                "open_interest": int(best["openInterest"]),
                "implied_vol":  float(best.get("impliedVolatility", 0.30)),
                "spread_ratio": round(float(best["spread_ratio"]), 4),
                "dte":          int(dte),  # Explizit int — verhindert ValueError
            }

            if "SPREAD" in strategy:
                spread_leg = self._find_spread_leg(opts, best["strike"])
                result["spread_leg"] = spread_leg
                if spread_leg:
                    result["net_debit"] = round(
                        result["ask"] - spread_leg.get("bid", 0), 2
                    )

            return result

        except Exception as e:
            log.debug(f"Kontrakt-Suche [{ticker}] {dte_min}-{dte_max}d: {e}")
            return None

    def _find_spread_leg(self, opts, long_strike: float) -> Optional[dict]:
        candidates = opts[
            (opts["strike"] >= long_strike * 1.05) &
            (opts["strike"] <= long_strike * 1.20)
        ]
        if candidates.empty:
            return None
        best = candidates.iloc[
            (candidates["strike"] - long_strike * 1.10).abs().argsort()
        ].iloc[0]
        return {"strike": float(best["strike"]),
                "bid": float(best["bid"]), "ask": float(best["ask"])}

    # ── Sektor-Momentum ───────────────────────────────────────────────────────

    def _sector_momentum_ok(self, s: dict) -> bool:
        import yfinance as yf
        ticker    = s.get("ticker", "")
        sector    = s.get("info", {}).get("sector", "default")
        direction = s.get("deep_analysis", {}).get("direction", "BULLISH")
        etf       = SECTOR_ETF.get(sector, SECTOR_ETF["default"])
        try:
            sh  = yf.Ticker(ticker).history(period="35d")
            eh  = yf.Ticker(etf).history(period="35d")
            if sh.empty or eh.empty or len(sh) < 5:
                return True
            sr  = float((sh["Close"].iloc[-1] - sh["Close"].iloc[0]) / sh["Close"].iloc[0])
            er  = float((eh["Close"].iloc[-1] - eh["Close"].iloc[0]) / eh["Close"].iloc[0])
            rs  = sr - er
            s["sector_momentum"] = {"etf": etf, "rel_strength": round(rs, 4)}
            return rs >= RELATIVE_STRENGTH_MIN if direction == "BULLISH" else rs <= -RELATIVE_STRENGTH_MIN
        except Exception:
            return True

    # ── Bear-Case Gate ────────────────────────────────────────────────────────

    def _bear_case_ok(self, s: dict) -> bool:
        sev = s.get("deep_analysis", {}).get("bear_case_severity", 0)
        thr = cfg.risk.max_bear_case_severity
        if sev >= thr:
            log.info(f"  [{s['ticker']}] BEAR-CASE={sev} ≥ {thr} → blockiert")
            return False
        return True

    # ── IV-Rank ───────────────────────────────────────────────────────────────

    def _get_iv_rank(self, ticker: str) -> float:
        import math, numpy as np
        try:
            t     = yf.Ticker(ticker)
            dates = t.options
            if not dates:
                return 30.0
            chain      = t.option_chain(dates[0])
            calls      = chain.calls
            if calls.empty or "impliedVolatility" not in calls.columns:
                return 30.0
            iv_current = float(calls["impliedVolatility"].median())
            hist       = t.history(period="1y")
            rets       = hist["Close"].pct_change().dropna()
            roll       = rets.rolling(30).std().dropna() * (252 ** 0.5)
            iv_lo      = float(roll.quantile(0.10))
            iv_hi      = float(roll.quantile(0.90))
            if iv_hi <= iv_lo:
                return 50.0
            rank = ((iv_current - iv_lo) / (iv_hi - iv_lo)) * 100
            return round(max(0.0, min(100.0, rank)), 2)
        except Exception:
            return 30.0

    def _days_to(self, expiry_str: str) -> int:
        try:
            delta = datetime.strptime(expiry_str, "%Y-%m-%d") - datetime.utcnow()
            return max(0, int(delta.days))  # Immer int, nie komplex
        except Exception:
            return 0
