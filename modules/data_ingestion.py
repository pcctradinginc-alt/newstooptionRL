"""
Stufe 1: Daten-Ingestion & Hard-Filter

Fixes:
  #1: Ticker ≤ 2 Zeichen (ON, V, A, C, F...) werden im RSS-Feed NICHT mehr
      per Regex gematcht – zu viele False-Positives ("ON sale", "depends ON").
      Für Kurzticker gilt: nur NewsAPI mit exaktem Quoted-Query.
  #4: EPS-Drift-Berechnung fällt auf yfinance trailingEps zurück wenn
      kein stored_eps in history.json vorhanden (neue Ticker).
  #5: Min-Impact-Schwelle aus config.yaml (min_impact_threshold).
"""

import os
import re
import logging
import feedparser
import requests
import yfinance as yf
from typing import Any

from modules.config import cfg
from modules.universe import get_universe

log = logging.getLogger(__name__)

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
]

# FIX #1: Kurzticker (≤ 2 Zeichen) NICHT per RSS matchen
# Begründung: "ON", "V", "A", "C", "F" sind zu häufige englische Wörter/
# Buchstaben und produzieren massenhaft False-Positives im RSS-Matching.
# Für diese Ticker gilt: ausschließlich NewsAPI (exakter Quoted-Query).
SHORT_TICKER_MIN_LEN = 3   # Ticker kürzer als 3 Zeichen → kein RSS-Matching

_TICKER_PATTERNS: dict[str, re.Pattern] = {}


def _get_ticker_pattern(ticker: str) -> re.Pattern:
    """Word-Boundary-Regex für Ticker-Matching im RSS-Feed."""
    if ticker not in _TICKER_PATTERNS:
        escaped = re.escape(ticker)
        _TICKER_PATTERNS[ticker] = re.compile(
            rf"\b{escaped}\b", re.IGNORECASE
        )
    return _TICKER_PATTERNS[ticker]


def _is_rss_safe(ticker: str) -> bool:
    """
    FIX #1: Prüft ob ein Ticker sicher per RSS gematcht werden kann.
    Kurzticker (≤ 2 Zeichen) sind NICHT RSS-sicher.
    Zusätzlich: bekannte problematische Ticker explizit ausschließen.
    """
    if len(ticker) < SHORT_TICKER_MIN_LEN:
        return False
    # Explizite Ausschlussliste für häufige englische Wörter
    UNSAFE = {"ON", "IT", "OR", "ARE", "BE", "TO", "DO", "GO", "SO", "RE",
               "AI", "GE", "AM", "PM", "IS", "AS", "AT", "BY", "IN", "OF"}
    if ticker.upper() in UNSAFE:
        return False
    return True


class DataIngestion:

    def __init__(self, history: dict):
        self.history      = history
        self.news_api_key = os.getenv("NEWS_API_KEY", "")

    def run(self) -> list[dict]:
        universe = get_universe()
        log.info(
            f"Universum '{cfg.filters.universe}': "
            f"{len(universe)} Ticker geladen."
        )

        news_by_ticker = self._fetch_news(universe)
        candidates     = []

        for ticker in universe:
            info = self._get_ticker_info(ticker)
            if info is None:
                continue
            if not self._passes_hard_filter(info):
                continue

            eps_drift = self._compute_eps_drift(ticker, info)
            news      = news_by_ticker.get(ticker, [])
            if not news:
                continue

            candidates.append({
                "ticker":    ticker,
                "info":      info,
                "eps_drift": eps_drift,
                "news":      news,
            })

        return candidates

    # ── News-Fetching ─────────────────────────────────────────────────────────

    def _fetch_news(self, universe: list[str]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {t: [] for t in universe}

        # NewsAPI: funktioniert für ALLE Ticker (exakter Quoted-Query)
        if self.news_api_key:
            for ticker in universe:
                try:
                    url = (
                        "https://newsapi.org/v2/everything"
                        f"?q=%22{ticker}%22"
                        f"&language=en&pageSize=5"
                        f"&apiKey={self.news_api_key}"
                    )
                    resp     = requests.get(url, timeout=10)
                    articles = resp.json().get("articles", [])
                    result[ticker] += [
                        a["title"] for a in articles if a.get("title")
                    ]
                except Exception as e:
                    log.debug(f"NewsAPI Fehler für {ticker}: {e}")

        # RSS: NUR für RSS-sichere Ticker (FIX #1)
        rss_universe = [t for t in universe if _is_rss_safe(t)]
        log.debug(
            f"RSS-Matching: {len(rss_universe)}/{len(universe)} Ticker "
            f"(Kurzticker ausgeschlossen: "
            f"{len(universe)-len(rss_universe)})"
        )

        for feed_url in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    title = entry.get("title", "")
                    for ticker in rss_universe:
                        if _get_ticker_pattern(ticker).search(title):
                            result[ticker].append(title)
            except Exception as e:
                log.debug(f"RSS Fehler ({feed_url}): {e}")

        return result

    # ── Ticker-Info ───────────────────────────────────────────────────────────

    def _get_ticker_info(self, ticker: str) -> dict | None:
        try:
            t    = yf.Ticker(ticker)
            info = t.info
            if not info or "marketCap" not in info:
                return None
            return info
        except Exception as e:
            log.debug(f"yfinance Fehler für {ticker}: {e}")
            return None

    # ── Hard-Filter ───────────────────────────────────────────────────────────

    def _passes_hard_filter(self, info: dict) -> bool:
        market_cap = info.get("marketCap", 0) or 0
        avg_volume = info.get("averageVolume10days", 0) or 0
        if market_cap < cfg.filters.min_market_cap:
            return False
        if avg_volume < cfg.filters.min_avg_volume:
            return False
        return True

    # ── EPS-Drift (FIX #4: trailingEps-Fallback) ─────────────────────────────

    def _compute_eps_drift(self, ticker: str, info: dict) -> dict[str, Any]:
        current_eps = info.get("forwardEps") or 0.0
        rec_mean    = info.get("recommendationMean") or 0.0

        stored = self._get_stored_eps(ticker)

        # FIX #4: Wenn kein stored_eps vorhanden (neuer Ticker) →
        # trailingEps als Baseline nutzen statt 0.0-Drift auszugeben.
        # Das macht den EPS-Drift für neue Ticker sofort bedeutsam.
        if stored is None or stored == 0:
            trailing_eps = info.get("trailingEps") or 0.0
            if trailing_eps != 0 and current_eps != 0:
                stored = trailing_eps
                log.debug(
                    f"  [{ticker}] EPS-Drift: kein stored_eps → "
                    f"nutze trailingEps={trailing_eps:.2f} als Baseline"
                )

        if stored and stored != 0 and current_eps != 0:
            drift = (current_eps - stored) / abs(stored)
        else:
            drift = 0.0

        abs_drift = abs(drift)
        if abs_drift > cfg.eps_drift.massive_threshold:
            weight = "massive"
        elif abs_drift > cfg.eps_drift.relevant_threshold:
            weight = "relevant"
        else:
            weight = "noise"

        return {
            "current_eps":  current_eps,
            "stored_eps":   stored,
            "drift":        round(drift, 4),
            "drift_weight": weight,
            "rec_mean":     rec_mean,
        }

    def _get_stored_eps(self, ticker: str) -> float | None:
        for trade in self.history.get("active_trades", []):
            if trade.get("ticker") == ticker:
                return trade.get("features", {}).get("eps", None)
        return None

    @staticmethod
    def compute_48h_move(ticker: str) -> float:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) < 3:
                return 0.0
            close = hist["Close"]
            return float((close.iloc[-1] - close.iloc[-3]) / close.iloc[-3])
        except Exception:
            return 0.0
