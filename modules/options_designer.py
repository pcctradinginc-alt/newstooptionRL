"""
modules/options_designer.py v8.0

Änderungen v8.0:
    - Tradier Live-API als primäre Datenquelle für Option Chains + Expirations
      Endpoints: /v1/markets/options/expirations
                 /v1/markets/options/chains?greeks=true
    - yfinance bleibt vollständiger Fallback (Option Chain + Term Structure)
    - Tradier liefert echte Real-Time Bid/Ask/IV/Greeks (kein 15-min-Delay)
    - TRADIER_API_KEY via os.environ (bereits als GitHub Secret hinterlegt)
    - RV-Percentile (IV-Rank Basis) weiterhin via yfinance Kurshistorie
      → Tradier hat kein Äquivalent für tägliche OHLCV-Kurshistorie
    - _sector_momentum_ok() weiterhin via yfinance (Kurshistorie)
    - Neue Methoden: _tradier_expirations(), _tradier_chain(), _tradier_headers()

Änderungen v7.5:
    - IV-Rank Kalibrierung: Term-Structure-Gewicht 50%→20%, Formel entschärft
      Vorher: IV-Rank=100 bei fast allem (Term-Structure trieb Score hoch)
      Jetzt:  RV-Percentile dominiert (80%), Term-Structure ergänzt (20%, cap 80)

Änderungen v7.3/v7.4:
    1. ROI bei Spreads: net_debit als Kostenbasis statt ask (Long-Leg)
       + Spread-Delta-Adjustment (Long Delta - Short Delta ~0.80x)
    2. DTE-Tiers lückenlos: 14-60 / 61-149 / 150-365
    3. Timezone-Fix: datetime.now(timezone.utc) in _days_to
    4. IV Sanity-Check: iv < 0.05 → Fallback 0.30
    5. yf.Ticker einmal pro Signal (Performance)
    6. RV-Percentile: quantile(0.05/0.95) — crash-robust
    7. dates[:3] statt dates[:6] — weniger HTTP-Requests
    8. _bear_case_ok: Schwelle aus cfg statt hardcoded
"""

from __future__ import annotations
import logging
import math
import os
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

from modules.config        import cfg
from modules.macro_context import get_macro_regime_multiplier

log = logging.getLogger(__name__)

IV_RANK_GATE = 85.0

DTE_TIERS = [
    {"label": "Short-Term", "dte_min": 14,  "dte_max": 60,  "min_roi": 0.15},
    {"label": "Mid-Term",   "dte_min": 61,  "dte_max": 149, "min_roi": 0.12},
    {"label": "Long-Term",  "dte_min": 150, "dte_max": 365, "min_roi": 0.10},
]

SECTOR_ETF = {
    "Technology": "XLK", "Healthcare": "XLV", "Biotechnology": "XBI",
    "Financial Services": "XLF", "Financials": "XLF", "Energy": "XLE",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
    "Industrials": "XLI", "Basic Materials": "XLB",
    "Real Estate": "XLRE", "Utilities": "XLU",
    "Communication Services": "XLC", "default": "SPY",
}

RELATIVE_STRENGTH_MIN = -0.08

TRADIER_BASE   = "https://api.tradier.com/v1"
TRADIER_TIMEOUT = 10   # Sekunden — schneller als yfinance-Scraping


# ── Tradier Hilfsfunktionen ───────────────────────────────────────────────────

def _tradier_headers() -> dict:
    """Authorization-Header für Tradier Live-API."""
    api_key = os.environ.get("TRADIER_API_KEY", "")
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept":        "application/json",
    }


def _tradier_expirations(symbol: str) -> list[str]:
    """
    Verfallsdaten via Tradier Live-API.

    Endpoint: GET /v1/markets/options/expirations?symbol=AAPL&includeAllRoots=true
    Returns:  Sortierte Liste von Datumsstrings ["2026-05-16", "2026-06-20", ...]
    Fallback: Leere Liste → Caller fällt auf yfinance zurück
    """
    try:
        resp = requests.get(
            f"{TRADIER_BASE}/markets/options/expirations",
            params={"symbol": symbol, "includeAllRoots": "true"},
            headers=_tradier_headers(),
            timeout=TRADIER_TIMEOUT,
        )
        resp.raise_for_status()
        data  = resp.json()
        dates = data.get("expirations", {}).get("date", []) or []
        # Tradier liefert manchmal einen einzelnen String statt Liste
        if isinstance(dates, str):
            dates = [dates]
        return sorted(dates)
    except Exception as e:
        log.debug(f"Tradier Expirations [{symbol}]: {e}")
        return []


def _tradier_chain(symbol: str, expiration: str) -> list[dict]:
    """
    Option Chain via Tradier Live-API (mit Greeks).

    Endpoint: GET /v1/markets/options/chains
              ?symbol=AAPL&expiration=2026-10-16&greeks=true
    Returns:  Liste von Option-Dicts mit Feldern:
              strike, bid, ask, open_interest, option_type,
              greeks.delta, greeks.mid_iv
    Fallback: Leere Liste → Caller fällt auf yfinance zurück
    """
    try:
        resp = requests.get(
            f"{TRADIER_BASE}/markets/options/chains",
            params={
                "symbol":     symbol,
                "expiration": expiration,
                "greeks":     "true",
            },
            headers=_tradier_headers(),
            timeout=TRADIER_TIMEOUT,
        )
        resp.raise_for_status()
        data    = resp.json()
        options = data.get("options", {}).get("option", []) or []
        # Einzelner Kontrakt kommt manchmal als Dict statt Liste
        if isinstance(options, dict):
            options = [options]
        return options
    except Exception as e:
        log.debug(f"Tradier Chain [{symbol} {expiration}]: {e}")
        return []


def _tradier_chain_to_df(options: list[dict], option_type: str) -> pd.DataFrame:
    """
    Konvertiert Tradier-Option-Liste in yfinance-kompatibles DataFrame.

    Spalten-Mapping:
        Tradier             → yfinance-kompatibel
        open_interest       → openInterest
        greeks.mid_iv       → impliedVolatility
        greeks.delta        → delta  (Bonus: direkt verfügbar, kein BS nötig)

    option_type: "call" oder "put"
    """
    rows = []
    for o in options:
        if o.get("option_type") != option_type:
            continue
        greeks = o.get("greeks") or {}

        # mid_iv bevorzugt; Fallback auf smv_vol, dann 0.30
        iv = (
            greeks.get("mid_iv")
            or greeks.get("smv_vol")
            or 0.30
        )
        # Sanity-Check
        if not isinstance(iv, (int, float)) or iv <= 0.01:
            iv = 0.30

        rows.append({
            "strike":          float(o.get("strike", 0)),
            "bid":             float(o.get("bid") or 0),
            "ask":             float(o.get("ask") or 0),
            "openInterest":    int(o.get("open_interest") or 0),
            "impliedVolatility": float(iv),
            "delta":           float(greeks.get("delta") or 0),
            "volume":          int(o.get("volume") or 0),
            # original Tradier-Felder für Debugging
            "_option_symbol":  o.get("symbol", ""),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


# ── Haupt-Klasse ──────────────────────────────────────────────────────────────

class OptionsDesigner:

    def __init__(self, gates):
        self.gates = gates
        # Einmalig prüfen ob Tradier-Key vorhanden
        self._use_tradier = bool(os.environ.get("TRADIER_API_KEY", "").strip())
        if self._use_tradier:
            log.info("OptionsDesigner: Tradier Live-API aktiv (Primary)")
        else:
            log.warning("OptionsDesigner: TRADIER_API_KEY fehlt → yfinance Fallback")

    def run(self, signals: list[dict]) -> list[dict]:
        proposals = []
        for s in signals:
            ticker = s.get("ticker", "")
            if not self._bear_case_ok(s):
                continue
            # yf.Ticker weiterhin für _sector_momentum_ok() (Kurshistorie)
            t_obj = yf.Ticker(ticker)
            if not self._sector_momentum_ok(s, t=t_obj):
                log.info(f"  [{ticker}] SECTOR-GATE → verworfen")
                continue
            proposal = self._design_with_adaptive_dte(s, t_obj)
            if proposal:
                proposals.append(proposal)
        return proposals

    def _design_with_adaptive_dte(self, s: dict, t=None) -> Optional[dict]:
        ticker    = s["ticker"]
        direction = s.get("deep_analysis", {}).get("direction", "BULLISH")
        sim       = s.get("simulation", {})
        current   = sim.get("current_price", 0)

        if current <= 0:
            return None

        if self.gates.has_upcoming_earnings(ticker):
            log.info(f"  [{ticker}] EARNINGS-GATE → blockiert")
            return None

        if t is None:
            t = yf.Ticker(ticker)
        iv_rank = self._get_iv_rank(ticker, t)

        results_per_tier = []

        for tier in DTE_TIERS:
            label = tier["label"]

            strategy = self._select_strategy(ticker, direction, iv_rank)
            option   = self._find_option_for_dte(
                ticker, strategy, current, tier["dte_min"], tier["dte_max"], t
            )

            if not option:
                log.info(f"  [{ticker}] {label}: kein Kontrakt verfügbar")
                if "SPREAD" in strategy:
                    fallback = "LONG_CALL" if "BULL" in strategy else "LONG_PUT"
                    option   = self._find_option_for_dte(
                        ticker, fallback, current, tier["dte_min"], tier["dte_max"], t
                    )
                    if option:
                        strategy = fallback
                        log.info(f"  [{ticker}] {label}: Fallback → {fallback}")
                if not option:
                    continue

            roi = self._compute_roi(option, sim, iv_rank, tier, strategy)

            try:
                dte_safe       = max(int(option.get("dte") or 1), 1)
                roi_net_safe   = float(roi["roi_net"].real if isinstance(roi["roi_net"], complex) else roi["roi_net"])
                annualized_roi = float((1 + roi_net_safe) ** (365 / dte_safe) - 1)
                annualized_roi = min(annualized_roi, 9.99)  # Cap: >999% p.a. = nicht aussagekräftig
            except Exception:
                annualized_roi = 0.0

            log.info(
                f"  [{ticker}] {label} ({int(option['dte'] or 0)}d): "
                f"ROI={roi['roi_net']:.1%} "
                f"(ann.={annualized_roi:.1%}) "
                f"{'✅ PASS' if roi['passes_roi_gate'] else '❌ FAIL'}"
            )

            results_per_tier.append({
                "tier": label, "dte": option["dte"],
                "option": option, "roi": roi,
                "annualized_roi": round(annualized_roi, 4),
                "strategy": strategy,
            })

            if label == "Short-Term" and roi.get("vega_loss", 0) > 0.35:
                log.info(
                    f"  [{ticker}] {label}: Vega-Loss={roi['vega_loss']:.0%} > 35% "
                    f"→ zu hohes IV-Crush-Risiko, versuche längere Laufzeit"
                )
                continue

            if roi["passes_roi_gate"]:
                tier_idx = DTE_TIERS.index(tier)
                if tier_idx > 0:
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
                    "ticker":          ticker,
                    "strategy":        strategy,
                    "iv_rank":         iv_rank,
                    "iv_gate_applied": iv_rank >= IV_RANK_GATE,
                    "direction":       direction,
                    "option":          option,
                    "roi_analysis":    roi,
                    "dte_tier":        label,
                    "annualized_roi":  round(annualized_roi, 4),
                    "all_tiers_tried": results_per_tier,
                    "features":        s.get("features", {}),
                    "simulation":      s.get("simulation", {}),
                    "deep_analysis":   s.get("deep_analysis", {}),
                    "sector_momentum": s.get("sector_momentum", {}),
                    "final_score":     s.get("final_score", 0),
                }

        tried = ", ".join(
            f"{r['tier']}={r['roi']['roi_net']:.1%}" for r in results_per_tier
        )
        log.info(f"  [{ticker}] Alle Laufzeiten unter ROI-Gate: {tried} → verworfen")
        return None

    def _select_strategy(self, ticker: str, direction: str, iv_rank: float) -> str:
        if iv_rank >= IV_RANK_GATE:
            s = "BULL_CALL_SPREAD" if direction == "BULLISH" else "BEAR_PUT_SPREAD"
            log.info(f"  [{ticker}] IV={iv_rank:.0f}% ≥ {IV_RANK_GATE:.0f}% → {s} (Vega-Schutz)")
        else:
            s = "LONG_CALL" if direction == "BULLISH" else "LONG_PUT"
            log.info(f"  [{ticker}] IV={iv_rank:.0f}% < {IV_RANK_GATE:.0f}% → {s} (Optionen günstig)")
        return s

    def _compute_roi(
        self, option: dict, sim: dict, iv_rank: float, tier: dict, strategy: str = ""
    ) -> dict:
        bid     = option.get("bid", 0) or 0
        ask     = option.get("ask", 0) or 0
        strike  = option.get("strike", 0) or 0
        iv      = option.get("implied_vol", 0.30) or 0.30
        dte     = int(option.get("dte", 120) or 120)
        current = sim.get("current_price", 0) or 0
        target  = sim.get("target_price", 0) or 0
        min_roi = tier["min_roi"]

        is_spread = "SPREAD" in strategy
        cost = option.get("net_debit", ask) if is_spread else ask

        if cost <= 0 or current <= 0:
            return {"roi_net": 0.0, "passes_roi_gate": False,
                    "roi_gross": 0.0, "spread_pct": 0.0,
                    "vega_loss": 0.0, "min_roi_threshold": min_roi}

        if iv < 0.05 or iv > 3.0:
            iv = 0.30

        spread_pct = (ask - bid) / ask if ask > 0 else 0.0
        T          = dte / 365.0

        # Tradier liefert echtes Delta direkt — nutze es wenn vorhanden
        tradier_delta = option.get("delta")
        if tradier_delta and 0.01 <= abs(float(tradier_delta)) <= 0.99:
            delta = float(tradier_delta)
            vega  = 0.0   # Vega via BS als Fallback unten
            try:
                d1   = (math.log(current / strike) + 0.5 * iv**2 * T) / (iv * math.sqrt(T))
                vega = current * math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi) * math.sqrt(T)
            except Exception:
                pass
        else:
            try:
                d1    = (math.log(current / strike) + 0.5 * iv**2 * T) / (iv * math.sqrt(T))
                delta = (1.0 + math.erf(d1 / math.sqrt(2.0))) / 2.0
                vega  = current * math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi) * math.sqrt(T)
            except Exception:
                delta, vega = 0.5, 0.0

        if is_spread:
            delta *= 0.80

        leverage      = current / cost if cost > 0 else 1.0
        expected_move = (target - current) / current if target > current else 0.0
        roi_delta     = expected_move * delta * leverage

        if dte <= 60:
            iv_drop = 0.25 if iv_rank >= 70 else (0.12 if iv_rank >= 50 else 0.05)
        elif dte <= 149:
            iv_drop = 0.15 if iv_rank >= 70 else (0.07 if iv_rank >= 50 else 0.02)
        else:
            iv_drop = 0.08 if iv_rank >= 70 else (0.03 if iv_rank >= 50 else 0.01)

        vega_loss = min((vega * iv * iv_drop) / cost, 0.50) if cost > 0 else 0.0
        roi_net   = roi_delta - (spread_pct * 2) - vega_loss
        passes    = roi_net >= min_roi

        def _safe_float(v):
            if isinstance(v, complex): return float(v.real)
            try: return float(v)
            except: return 0.0

        return {
            "roi_gross":         round(_safe_float(roi_delta), 4),
            "roi_net":           round(_safe_float(roi_net), 4),
            "spread_pct":        round(_safe_float(spread_pct), 4),
            "vega_loss":         round(_safe_float(vega_loss), 4),
            "delta":             round(_safe_float(delta), 4),
            "iv_drop_assumed":   _safe_float(iv_drop),
            "cost_basis":        round(float(cost), 4),
            "is_spread":         is_spread,
            "passes_roi_gate":   passes,
            "min_roi_threshold": min_roi,
            "dte":               int(dte),
        }

    # ── Option Chain Abruf: Tradier Primary, yfinance Fallback ───────────────

    def _find_option_for_dte(
        self, ticker: str, strategy: str, current: float,
        dte_min: int, dte_max: int,
        t: Optional[object] = None,
    ) -> Optional[dict]:
        """
        Sucht den besten Kontrakt im DTE-Fenster.
        Reihenfolge: Tradier Live-API → yfinance Fallback
        """
        # ── Versuch 1: Tradier ────────────────────────────────────────────────
        if self._use_tradier:
            result = self._find_option_tradier(ticker, strategy, current, dte_min, dte_max)
            if result is not None:
                return result
            log.debug(f"  [{ticker}] Tradier Chain leer → yfinance Fallback")

        # ── Versuch 2: yfinance Fallback ──────────────────────────────────────
        return self._find_option_yfinance(ticker, strategy, current, dte_min, dte_max, t)

    def _find_option_tradier(
        self, ticker: str, strategy: str, current: float,
        dte_min: int, dte_max: int,
    ) -> Optional[dict]:
        """Option-Kontrakt via Tradier Live-API."""
        try:
            # 1. Verfallsdaten holen
            all_dates = _tradier_expirations(ticker)
            if not all_dates:
                return None

            dates = [d for d in all_dates if dte_min <= self._days_to(d) <= dte_max]
            if not dates:
                return None

            best_expiry = dates[0]
            is_call     = "CALL" in strategy or "BULL" in strategy
            option_type = "call" if is_call else "put"

            # 2. Option Chain holen
            raw = _tradier_chain(ticker, best_expiry)
            if not raw:
                return None

            # 3. In yfinance-kompatibles DataFrame konvertieren
            opts = _tradier_chain_to_df(raw, option_type)
            if opts.empty:
                return None

            # 4. Strike-Filter (identisch zu yfinance-Pfad)
            days_mid = (dte_min + dte_max) / 2
            if days_mid <= 60:
                otm_max, itm_max = 1.03, 0.97
            elif days_mid <= 149:
                otm_max, itm_max = 1.08, 0.96
            else:
                otm_max, itm_max = 1.12, 0.95

            min_oi = max(
                getattr(getattr(cfg, "risk", None), "min_open_interest", 100), 50
            )
            filtered = opts[
                (opts["strike"] >= current * itm_max) &
                (opts["strike"] <= current * otm_max) &
                (opts["openInterest"] >= min_oi)
            ].copy()

            if filtered.empty:
                return None

            # 5. Bid-Ask-Ratio-Gate
            filtered["spread_ratio"] = (
                (filtered["ask"] - filtered["bid"]) /
                filtered["ask"].clip(lower=0.01)
            )
            filtered = filtered[filtered["spread_ratio"] <= cfg.risk.max_bid_ask_ratio]
            if filtered.empty:
                return None

            best = filtered.sort_values("openInterest", ascending=False).iloc[0]
            dte  = self._days_to(best_expiry)

            result = {
                "expiry":        best_expiry,
                "strike":        float(best["strike"]),
                "bid":           float(best["bid"]),
                "ask":           float(best["ask"]),
                "open_interest": int(best["openInterest"]),
                "implied_vol":   float(best["impliedVolatility"]),
                "spread_ratio":  round(float(best["spread_ratio"]), 4),
                "dte":           int(dte),
                "delta":         float(best.get("delta", 0)),   # Bonus: echtes Delta
                "data_source":   "tradier",
            }

            log.info(
                f"  [{ticker}] Tradier Chain: expiry={best_expiry} "
                f"strike={result['strike']:.1f} IV={result['implied_vol']:.1%} "
                f"delta={result['delta']:.2f} OI={result['open_interest']}"
            )

            # 6. Spread-Leg (bei Spread-Strategien)
            if "SPREAD" in strategy:
                spread_leg = self._find_spread_leg(opts, best["strike"])
                result["spread_leg"] = spread_leg
                if spread_leg:
                    result["net_debit"] = round(
                        result["ask"] - spread_leg.get("bid", 0), 2
                    )
                    if spread_leg.get("bid", 0) <= 0:
                        log.debug(f"  [{ticker}] Short-Leg hat keine Liquidität → kein Spread")
                        result.pop("spread_leg", None)
                        result.pop("net_debit", None)

            return result

        except Exception as e:
            log.debug(f"Tradier _find_option [{ticker}] {dte_min}-{dte_max}d: {e}")
            return None

    def _find_option_yfinance(
        self, ticker: str, strategy: str, current: float,
        dte_min: int, dte_max: int,
        t: Optional[object] = None,
    ) -> Optional[dict]:
        """Option-Kontrakt via yfinance (Fallback). Unveränderte v7.5-Logik."""
        try:
            if t is None:
                t = yf.Ticker(ticker)
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

            days_mid = (dte_min + dte_max) / 2
            if days_mid <= 60:
                otm_max, itm_max = 1.03, 0.97
            elif days_mid <= 149:
                otm_max, itm_max = 1.08, 0.96
            else:
                otm_max, itm_max = 1.12, 0.95

            filtered = opts[
                (opts["strike"] >= current * itm_max) &
                (opts["strike"] <= current * otm_max) &
                (opts["openInterest"] >= max(
                    getattr(getattr(cfg, "risk", None), "min_open_interest", 100), 50
                ))
            ].copy()

            if filtered.empty:
                return None

            filtered["spread_ratio"] = (
                (filtered["ask"] - filtered["bid"]) /
                filtered["ask"].clip(lower=0.01)
            )
            filtered = filtered[filtered["spread_ratio"] <= cfg.risk.max_bid_ask_ratio]
            if filtered.empty:
                return None

            best = filtered.sort_values("openInterest", ascending=False).iloc[0]
            dte  = self._days_to(best_expiry)

            result = {
                "expiry":        best_expiry,
                "strike":        float(best["strike"]),
                "bid":           float(best["bid"]),
                "ask":           float(best["ask"]),
                "open_interest": int(best["openInterest"]),
                "implied_vol":   float(best.get("impliedVolatility", 0.30)),
                "spread_ratio":  round(float(best["spread_ratio"]), 4),
                "dte":           int(dte),
                "data_source":   "yfinance",
            }

            if "SPREAD" in strategy:
                spread_leg = self._find_spread_leg(opts, best["strike"])
                result["spread_leg"] = spread_leg
                if spread_leg:
                    result["net_debit"] = round(
                        result["ask"] - spread_leg.get("bid", 0), 2
                    )
                    if spread_leg.get("bid", 0) <= 0:
                        log.debug(f"  [{ticker}] Short-Leg hat keine Liquidität → kein Spread")
                        result.pop("spread_leg", None)
                        result.pop("net_debit", None)

            return result

        except Exception as e:
            log.debug(f"yfinance _find_option [{ticker}] {dte_min}-{dte_max}d: {e}")
            return None

    def _find_spread_leg(self, opts: pd.DataFrame, long_strike: float) -> Optional[dict]:
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

    # ── IV-Rank: RV via yfinance, Term Structure Tradier → yfinance Fallback ──

    def _get_iv_rank(self, ticker: str, t: Optional[object] = None) -> float:
        """
        IV-Rank Proxy: RV-Percentile (yfinance) + Term Structure Slope (Tradier/yfinance).

        v8.0:
            Term Structure: Tradier Primary → yfinance Fallback
            RV-Percentile:  Weiterhin yfinance (Kurshistorie, kein Tradier-Äquivalent)

        v7.5 FIX: Gewichtung 80/20 statt 50/50, Term-Structure entschärft.
            combined = rv_score * 0.8 + term_score * 0.2
            term_score = (slope + 0.05) * 100, cap bei 80

        Beispiel-Werte:
            Ruhiger Markt (RV=30, Term=20):  30*0.8 + 20*0.2 = 28 → LONG_CALL
            Leicht erhöht (RV=55, Term=40):  55*0.8 + 40*0.2 = 52 → LONG_CALL
            Stark erhöht (RV=85, Term=70):   85*0.8 + 70*0.2 = 82 → LONG_CALL (knapp)
            Extrem (RV=95, Term=80):          95*0.8 + 80*0.2 = 92 → SPREAD
        """
        try:
            if t is None:
                t = yf.Ticker(ticker)

            # ── RV-Percentile (80% Gewicht) — weiterhin yfinance ─────────────
            rv_score = 50.0
            info     = t.info
            current  = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)

            hist = t.history(period="1y")
            if not hist.empty and len(hist) >= 60:
                rets    = hist["Close"].pct_change().dropna()
                roll_rv = rets.rolling(21).std().dropna() * (252 ** 0.5)
                if len(roll_rv) >= 20:
                    rv_current = float(roll_rv.iloc[-1])
                    rv_min     = float(roll_rv.quantile(0.05))
                    rv_max     = float(roll_rv.quantile(0.95))
                    if rv_max > rv_min:
                        rv_score = ((rv_current - rv_min) / (rv_max - rv_min)) * 100
                        rv_score = max(0.0, min(100.0, rv_score))

            if current <= 0:
                return round(rv_score, 1)

            # ── Term Structure Slope (20% Gewicht) — Tradier → yfinance ──────
            term_score = 20.0   # Default
            iv_pts     = self._get_term_structure_iv(ticker, current, t)

            if len(iv_pts) >= 2:
                iv_pts.sort()
                iv_short = iv_pts[0][1]
                iv_long  = iv_pts[-1][1]
                if iv_long > 0:
                    slope      = (iv_short / iv_long) - 1.0
                    term_score = max(0.0, min(80.0, (slope + 0.05) * 100))

            combined = round(rv_score * 0.80 + term_score * 0.20, 1)
            log.info(
                f"  [{ticker}] IV-Rank: rv={rv_score:.0f} term={term_score:.0f} "
                f"→ combined={combined:.0f} "
                f"({'SPREAD' if combined >= IV_RANK_GATE else 'LONG'})"
            )
            return combined

        except Exception:
            return 50.0

    def _get_term_structure_iv(
        self, ticker: str, current: float, t=None
    ) -> list[tuple[int, float]]:
        """
        Sammelt (DTE, ATM-IV)-Paare für Term-Structure-Berechnung.
        Tradier Primary → yfinance Fallback.
        Gibt max. 3 Datenpunkte zurück.
        """
        # ── Tradier ───────────────────────────────────────────────────────────
        if self._use_tradier:
            iv_pts = self._term_structure_tradier(ticker, current)
            if len(iv_pts) >= 2:
                return iv_pts
            log.debug(f"  [{ticker}] Term-Structure Tradier unvollständig → yfinance")

        # ── yfinance Fallback ─────────────────────────────────────────────────
        return self._term_structure_yfinance(ticker, current, t)

    def _term_structure_tradier(
        self, ticker: str, current: float
    ) -> list[tuple[int, float]]:
        """Term Structure IV-Punkte via Tradier."""
        iv_pts = []
        try:
            dates = _tradier_expirations(ticker)
            for d in dates[:3]:
                dte = self._days_to(d)
                if dte < 7:
                    continue
                raw = _tradier_chain(ticker, d)
                if not raw:
                    continue

                # ATM Calls (±7% vom aktuellen Kurs)
                atm_ivs = []
                for o in raw:
                    if o.get("option_type") != "call":
                        continue
                    strike = float(o.get("strike", 0))
                    if not (current * 0.93 <= strike <= current * 1.07):
                        continue
                    greeks = o.get("greeks") or {}
                    iv = greeks.get("mid_iv") or greeks.get("smv_vol")
                    if iv and isinstance(iv, (int, float)) and iv > 0.05:
                        atm_ivs.append(float(iv))

                if atm_ivs:
                    median_iv = float(np.median(atm_ivs))
                    iv_pts.append((dte, median_iv))

        except Exception as e:
            log.debug(f"  [{ticker}] Term-Structure Tradier Fehler: {e}")

        return iv_pts

    def _term_structure_yfinance(
        self, ticker: str, current: float, t=None
    ) -> list[tuple[int, float]]:
        """Term Structure IV-Punkte via yfinance (Fallback, unveränderte v7.5-Logik)."""
        iv_pts = []
        try:
            if t is None:
                t = yf.Ticker(ticker)
            dates = t.options or []
            for d in dates[:3]:
                try:
                    dte = self._days_to(d)
                    if dte < 7:
                        continue
                    ch  = t.option_chain(d)
                    atm = ch.calls[
                        (ch.calls["strike"] >= current * 0.93) &
                        (ch.calls["strike"] <= current * 1.07) &
                        (ch.calls["impliedVolatility"] > 0.05)
                    ]
                    if not atm.empty:
                        iv_pts.append((dte, float(atm["impliedVolatility"].median())))
                except Exception:
                    continue
        except Exception as e:
            log.debug(f"  [{ticker}] Term-Structure yfinance Fehler: {e}")

        return iv_pts

    # ── Sektor-Momentum (bleibt yfinance — Kurshistorie) ─────────────────────

    def _sector_momentum_ok(self, s: dict, t=None) -> bool:
        ticker    = s.get("ticker", "")
        sector    = s.get("info", {}).get("sector", "default")
        direction = s.get("deep_analysis", {}).get("direction", "BULLISH")
        etf       = SECTOR_ETF.get(sector, SECTOR_ETF["default"])
        try:
            ticker_obj = t if t is not None else yf.Ticker(ticker)
            sh = ticker_obj.history(period="35d")
            eh = yf.Ticker(etf).history(period="35d")
            if sh.empty or eh.empty or len(sh) < 5:
                return True
            sr = float((sh["Close"].iloc[-1] - sh["Close"].iloc[0]) / sh["Close"].iloc[0])
            er = float((eh["Close"].iloc[-1] - eh["Close"].iloc[0]) / eh["Close"].iloc[0])
            rs = sr - er
            s["sector_momentum"] = {"etf": etf, "rel_strength": round(rs, 4)}
            return rs >= RELATIVE_STRENGTH_MIN if direction == "BULLISH" else rs <= -RELATIVE_STRENGTH_MIN
        except Exception:
            return True

    def _bear_case_ok(self, s: dict) -> bool:
        sev = s.get("deep_analysis", {}).get("bear_case_severity", 0)
        thr = getattr(getattr(cfg, "risk", None), "max_bear_case_severity", 8)
        if sev >= thr:
            log.info(f"  [{s['ticker']}] BEAR-CASE={sev} ≥ {thr} → blockiert")
            return False
        return True

    def _days_to(self, expiry_str: str) -> int:
        try:
            expiry = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            now    = datetime.now(timezone.utc)
            return max(0, (expiry - now).days)
        except Exception:
            return 0
