"""
Stufe 7: Options-Design & Strategie

Fix #2: Bear-Case-Gate war `severity > 7` → Severity=7 passierte.
        Jetzt: `severity >= max_bear_case_severity` → Severity=7 wird blockiert.
        config.yaml: max_bear_case_severity: 7 bedeutet "ab 7 blockieren".
"""

import logging
import os
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional

from modules.config import cfg

log = logging.getLogger(__name__)

TRADIER_BASE = "https://api.tradier.com/v1"


class OptionsDesigner:

    def __init__(self, gates):
        self.gates = gates

    def run(self, signals: list[dict]) -> list[dict]:
        proposals = []
        for s in signals:
            if not self._bear_case_ok(s):
                log.info(f"  [{s['ticker']}] Bear-Case Audit FAILED – übersprungen.")
                continue
            proposal = self._design_option(s)
            if proposal:
                proposals.append(proposal)
        return proposals

    # ── Bear-Case Audit (FIX #2) ──────────────────────────────────────────────

    def _bear_case_ok(self, s: dict) -> bool:
        severity  = s.get("deep_analysis", {}).get("bear_case_severity", 0)
        threshold = cfg.risk.max_bear_case_severity

        # FIX #2: >= statt > → bei Severity=7 und threshold=7 wird blockiert
        if severity >= threshold:
            log.info(
                f"  [{s['ticker']}] Bear-Case-Severity={severity} "
                f">= {threshold} → blockiert."
            )
            return False
        return True

    # ── Options-Design ────────────────────────────────────────────────────────

    def _design_option(self, s: dict) -> Optional[dict]:
        ticker    = s["ticker"]
        direction = s.get("deep_analysis", {}).get("direction", "BULLISH")
        sim       = s.get("simulation", {})
        current   = sim.get("current_price", 0)

        if current <= 0:
            return None

        if self.gates.has_upcoming_earnings(ticker):
            log.info(
                f"  [{ticker}] Earnings < "
                f"{cfg.risk.earnings_buffer_days} Tage → blockiert."
            )
            return None

        iv_rank = self._get_iv_rank(ticker)
        log.info(f"  [{ticker}] IV-Rank={iv_rank:.1f}")

        if iv_rank < 50:
            strategy = "LONG_CALL" if direction == "BULLISH" else "LONG_PUT"
        else:
            strategy = (
                "BULL_CALL_SPREAD" if direction == "BULLISH"
                else "BEAR_PUT_SPREAD"
            )

        option = self._find_best_option(ticker, strategy, current)
        if not option:
            log.warning(f"  [{ticker}] Kein geeigneter Options-Kontrakt.")
            return None

        return {
            "ticker":        ticker,
            "strategy":      strategy,
            "iv_rank":       iv_rank,
            "direction":     direction,
            "option":        option,
            "features":      s.get("features", {}),
            "simulation":    s.get("simulation", {}),
            "deep_analysis": s.get("deep_analysis", {}),
            "final_score":   s.get("final_score", 0),
        }

    # ── IV-Rank ───────────────────────────────────────────────────────────────

    def _get_iv_rank(self, ticker: str) -> float:
        tradier_key = os.getenv("TRADIER_API_KEY", "")
        if not tradier_key:
            return self._estimate_iv_rank_from_yfinance(ticker)
        try:
            headers = {
                "Authorization": f"Bearer {tradier_key}",
                "Accept": "application/json",
            }
            exp_resp = requests.get(
                f"{TRADIER_BASE}/markets/options/expirations",
                params={"symbol": ticker, "includeAllRoots": "true"},
                headers=headers, timeout=10,
            )
            expirations = (
                exp_resp.json().get("expirations", {}).get("date", []) or []
            )
            if not expirations:
                return self._estimate_iv_rank_from_yfinance(ticker)

            target_expiry = None
            for exp in expirations:
                dte = self._days_to(exp)
                if cfg.options.dte_min <= dte <= cfg.options.dte_max:
                    target_expiry = exp
                    break

            if not target_expiry:
                return self._estimate_iv_rank_from_yfinance(ticker)

            quotes_resp = requests.get(
                f"{TRADIER_BASE}/markets/options/chains",
                params={"symbol": ticker, "expiration": target_expiry, "greeks": "true"},
                headers=headers, timeout=10,
            )
            options = quotes_resp.json().get("options", {}).get("option", []) or []
            if not options:
                return self._estimate_iv_rank_from_yfinance(ticker)

            calls = [o for o in options if o.get("option_type") == "call"]
            if not calls:
                return self._estimate_iv_rank_from_yfinance(ticker)

            calls_sorted = sorted(
                calls,
                key=lambda o: abs(o.get("strike", 0) -
                                  (o.get("underlying_price") or o.get("strike", 0)))
            )
            atm_ivs = [
                float(c["greeks"]["mid_iv"])
                for c in calls_sorted[:3]
                if c.get("greeks") and c["greeks"].get("mid_iv")
            ]
            if not atm_ivs:
                return self._estimate_iv_rank_from_yfinance(ticker)

            iv_current = sum(atm_ivs) / len(atm_ivs)
            iv_low, iv_high = self._estimate_iv_range(ticker, iv_current)
            if iv_high <= iv_low:
                return 50.0
            iv_rank = ((iv_current - iv_low) / (iv_high - iv_low)) * 100
            return round(max(0.0, min(100.0, iv_rank)), 2)

        except Exception as e:
            log.debug(f"Tradier IV-Rank Fehler für {ticker}: {e}")
            return self._estimate_iv_rank_from_yfinance(ticker)

    def _estimate_iv_range(self, ticker: str, current_iv: float) -> tuple[float, float]:
        try:
            hist = yf.Ticker(ticker).history(period="1y")
            if len(hist) < 50:
                return current_iv * 0.6, current_iv * 1.4
            daily_returns = hist["Close"].pct_change().dropna()
            rolling_vol   = daily_returns.rolling(30).std().dropna() * (252 ** 0.5)
            return float(rolling_vol.quantile(0.10)), float(rolling_vol.quantile(0.90))
        except Exception:
            return current_iv * 0.6, current_iv * 1.4

    def _estimate_iv_rank_from_yfinance(self, ticker: str) -> float:
        try:
            t = yf.Ticker(ticker)
            dates = t.options
            if not dates:
                return 30.0
            target_date = None
            for d in dates:
                if cfg.options.dte_min <= self._days_to(d) <= cfg.options.dte_max:
                    target_date = d
                    break
            if not target_date:
                target_date = dates[0]
            chain = t.option_chain(target_date)
            calls = chain.calls
            if calls.empty or "impliedVolatility" not in calls.columns:
                return 30.0
            current_iv      = float(calls["impliedVolatility"].median())
            iv_low, iv_high = self._estimate_iv_range(ticker, current_iv)
            if iv_high <= iv_low:
                return 30.0
            iv_rank = ((current_iv - iv_low) / (iv_high - iv_low)) * 100
            return round(max(0.0, min(100.0, iv_rank)), 2)
        except Exception:
            return 30.0

    # ── Kontrakt-Suche ────────────────────────────────────────────────────────

    def _find_best_option(
        self, ticker: str, strategy: str, current_price: float
    ) -> Optional[dict]:
        try:
            t            = yf.Ticker(ticker)
            expiry_dates = [
                d for d in t.options
                if cfg.options.dte_min <= self._days_to(d) <= cfg.options.dte_max
            ]
            if not expiry_dates:
                return None

            best_expiry = expiry_dates[0]
            chain       = t.option_chain(best_expiry)
            options     = (
                chain.calls if "CALL" in strategy or "BULL" in strategy
                else chain.puts
            )

            filtered = options[
                (options["strike"] >= current_price * 1.00) &
                (options["strike"] <= current_price * 1.12) &
                (options["openInterest"] >= cfg.risk.min_open_interest)
            ].copy()

            if filtered.empty:
                return None

            filtered["spread_ratio"] = (
                (filtered["ask"] - filtered["bid"]) / filtered["ask"]
            )
            filtered = filtered[
                filtered["spread_ratio"] <= cfg.risk.max_bid_ask_ratio
            ]
            if filtered.empty:
                return None

            best   = filtered.sort_values("openInterest", ascending=False).iloc[0]
            result = {
                "expiry":        best_expiry,
                "strike":        float(best["strike"]),
                "bid":           float(best["bid"]),
                "ask":           float(best["ask"]),
                "last":          float(best.get("lastPrice", 0)),
                "open_interest": int(best["openInterest"]),
                "implied_vol":   float(best.get("impliedVolatility", 0)),
                "spread_ratio":  round(float(best["spread_ratio"]), 4),
                "dte":           self._days_to(best_expiry),
            }

            if strategy == "BULL_CALL_SPREAD":
                spread_leg = self._find_spread_leg(options, best["strike"], current_price)
                result["spread_leg"] = spread_leg

            return result

        except Exception as e:
            log.debug(f"Kontrakt-Suche Fehler für {ticker}: {e}")
            return None

    def _find_spread_leg(
        self, options, long_strike: float, current_price: float
    ) -> Optional[dict]:
        short_target = long_strike * 1.10
        candidates   = options[
            (options["strike"] >= long_strike * 1.05) &
            (options["strike"] <= long_strike * 1.20) &
            (options["openInterest"] >= cfg.risk.min_open_interest)
        ]
        if candidates.empty:
            return None
        best = candidates.iloc[
            (candidates["strike"] - short_target).abs().argsort()
        ].iloc[0]
        return {
            "strike": float(best["strike"]),
            "bid":    float(best["bid"]),
            "ask":    float(best["ask"]),
        }

    def _days_to(self, expiry_str: str) -> int:
        try:
            d = datetime.strptime(expiry_str, "%Y-%m-%d")
            return (d - datetime.utcnow()).days
        except Exception:
            return 0
