"""
modules/data_ingestion.py v6.0

Hard-Filter Optimierung:
  - Market Cap Gate:    > 2 Mrd. USD  (war: vermutlich > 10 Mrd.)
  - Liquidity Gate:     Avg Volume > 1 Mio. Stück/Tag
  - Dollar-Volume Gate: Preis × Volumen > 10 Mio. USD/Tag
  - Relative Volume:    RV > 0.8 (war: > 1.5 / nur "High")
  - Logging:            Zeigt genau welcher Filter wie viele Ticker eliminiert

Ziel: 30-60 Kandidaten für Prescreening (statt bisher 4-7).
"""

from __future__ import annotations
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import requests
import yfinance as yf

from modules.config   import cfg
from modules.universe import get_universe

log = logging.getLogger(__name__)

# ── Hard-Filter Schwellenwerte ────────────────────────────────────────────────
MIN_MARKET_CAP_USD    = 2_000_000_000   # > 2 Mrd. USD
MIN_AVG_VOLUME        = 1_000_000       # > 1 Mio. Stück/Tag (30d Durchschnitt)
MIN_DOLLAR_VOLUME_USD = 10_000_000      # > 10 Mio. USD/Tag (Preis × Volumen)
MIN_RELATIVE_VOLUME   = 0.8             # > 80% des normalen Volumens
MAX_INTRADAY_MOVE     = 0.07            # < 7% Intraday-Bewegung (kein Chase)


class DataIngestion:

    def __init__(self, history: dict | None = None):
        self.history       = history or {}
        self.news_api_key  = os.getenv("NEWS_API_KEY", "")

    def run(self) -> list[dict]:
        tickers = get_universe()
        log.info(f"Stufe 1: Hard-Filter auf {len(tickers)} Ticker")

        # Tracking pro Filter-Kriterium
        stats = {
            "total":          len(tickers),
            "no_data":        0,
            "market_cap":     0,
            "avg_volume":     0,
            "dollar_volume":  0,
            "rel_volume":     0,
            "no_news":        0,
            "passed":         0,
        }

        candidates = []
        for ticker in tickers:
            result = self._evaluate_ticker(ticker, stats)
            if result:
                candidates.append(result)

        self._log_filter_stats(stats)
        log.info(f"  → {len(candidates)} Kandidaten nach Hard-Filter")
        return candidates

    # ── Ticker-Evaluation ─────────────────────────────────────────────────────

    def _evaluate_ticker(self, ticker: str, stats: dict) -> Optional[dict]:
        """Prüft einen Ticker gegen alle Hard-Filter."""
        try:
            t    = yf.Ticker(ticker)
            info = t.info

            if not info or not isinstance(info, dict):
                stats["no_data"] += 1
                return None

            # ── Filter 1: Market Cap ──────────────────────────────────────────
            market_cap = info.get("marketCap") or 0
            if market_cap < MIN_MARKET_CAP_USD:
                stats["market_cap"] += 1
                log.debug(
                    f"  [{ticker}] ❌ Market Cap: "
                    f"${market_cap/1e9:.2f}B < ${MIN_MARKET_CAP_USD/1e9:.0f}B"
                )
                return None

            # ── Filter 2: Durchschnittliches Volumen (30d) ────────────────────
            avg_vol = info.get("averageVolume") or info.get("averageVolume10days") or 0
            if avg_vol < MIN_AVG_VOLUME:
                stats["avg_volume"] += 1
                log.debug(
                    f"  [{ticker}] ❌ Avg Volume: "
                    f"{avg_vol/1e6:.2f}M < {MIN_AVG_VOLUME/1e6:.0f}M"
                )
                return None

            # ── Filter 3: Dollar-Volume ───────────────────────────────────────
            current_price = (
                info.get("currentPrice") or
                info.get("regularMarketPrice") or
                info.get("previousClose") or 0
            )
            dollar_volume = current_price * avg_vol
            if dollar_volume < MIN_DOLLAR_VOLUME_USD:
                stats["dollar_volume"] += 1
                log.debug(
                    f"  [{ticker}] ❌ Dollar-Volume: "
                    f"${dollar_volume/1e6:.1f}M < ${MIN_DOLLAR_VOLUME_USD/1e6:.0f}M"
                )
                return None

            # ── Filter 4: Relative Volume (RV) ────────────────────────────────
            volume_today = info.get("volume") or info.get("regularMarketVolume") or 0
            if avg_vol > 0 and volume_today > 0:
                rel_volume = volume_today / avg_vol
            else:
                rel_volume = 0.0

            if rel_volume < MIN_RELATIVE_VOLUME:
                stats["rel_volume"] += 1
                log.debug(
                    f"  [{ticker}] ❌ Rel. Volume: "
                    f"RV={rel_volume:.2f} < {MIN_RELATIVE_VOLUME}"
                )
                return None

            # ── Filter 5: News vorhanden ──────────────────────────────────────
            news = self._fetch_news(ticker, info)
            if not news:
                stats["no_news"] += 1
                log.debug(f"  [{ticker}] ❌ Keine News in letzten 48h")
                return None

            # ── Alle Filter bestanden ─────────────────────────────────────────
            stats["passed"] += 1
            log.info(
                f"  [{ticker}] ✅ "
                f"Cap=${market_cap/1e9:.1f}B "
                f"AvgVol={avg_vol/1e6:.1f}M "
                f"$Vol=${dollar_volume/1e6:.0f}M "
                f"RV={rel_volume:.2f} "
                f"News={len(news)}"
            )

            return {
                "ticker":        ticker,
                "info":          info,
                "news":          news,
                "market_cap":    market_cap,
                "avg_volume":    avg_vol,
                "dollar_volume": dollar_volume,
                "rel_volume":    round(rel_volume, 3),
                "current_price": current_price,
                "features":      {},
            }

        except Exception as e:
            log.debug(f"  [{ticker}] Fehler: {e}")
            stats["no_data"] += 1
            return None

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_filter_stats(self, stats: dict) -> None:
        """Zeigt transparent warum Ticker eliminiert wurden."""
        total   = stats["total"]
        passed  = stats["passed"]
        dropped = total - passed

        log.info("=" * 55)
        log.info(f"HARD-FILTER ERGEBNIS: {passed}/{total} Ticker bestanden")
        log.info("-" * 55)
        log.info(f"  ❌ Kein Data/Fehler:      {stats['no_data']:>4}  "
                 f"({stats['no_data']/total*100:.1f}%)")
        log.info(f"  ❌ Market Cap < 2 Mrd.:   {stats['market_cap']:>4}  "
                 f"({stats['market_cap']/total*100:.1f}%)")
        log.info(f"  ❌ Avg Volume < 1M:        {stats['avg_volume']:>4}  "
                 f"({stats['avg_volume']/total*100:.1f}%)")
        log.info(f"  ❌ Dollar-Vol < $10M:      {stats['dollar_volume']:>4}  "
                 f"({stats['dollar_volume']/total*100:.1f}%)")
        log.info(f"  ❌ Rel. Volume < 0.8:      {stats['rel_volume']:>4}  "
                 f"({stats['rel_volume']/total*100:.1f}%)")
        log.info(f"  ❌ Keine News:             {stats['no_news']:>4}  "
                 f"({stats['no_news']/total*100:.1f}%)")
        log.info(f"  ✅ Bestanden:              {passed:>4}  "
                 f"({passed/total*100:.1f}%)")
        log.info("=" * 55)

        if passed < 10:
            log.warning(
                f"Nur {passed} Kandidaten — wenig Material für Prescreening. "
                f"Ggf. Filter weiter lockern."
            )
        elif passed > 80:
            log.warning(
                f"{passed} Kandidaten — sehr viel für Prescreening. "
                f"Könnte API-Kosten erhöhen."
            )

    # ── News Fetching ─────────────────────────────────────────────────────────

    def _fetch_news(self, ticker: str, info: dict) -> list[str]:
        """
        Versucht News in dieser Reihenfolge:
        1. Finnhub (ticker-spezifisch, beste Qualität)
        2. NewsAPI (Firmenname-Suche)
        3. yfinance news (Fallback)
        """
        # Finnhub
        finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        if finnhub_key:
            news = self._fetch_finnhub_news(ticker, finnhub_key)
            if news:
                return news

        # NewsAPI Fallback
        if self.news_api_key:
            company_name = info.get("longName", ticker).split()[0]
            news = self._fetch_newsapi(ticker, company_name)
            if news:
                return news

        # yfinance Fallback
        return self._fetch_yfinance_news(ticker)

    def _fetch_finnhub_news(self, ticker: str, api_key: str) -> list[str]:
        try:
            since = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")
            today = datetime.utcnow().strftime("%Y-%m-%d")
            resp  = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={"symbol": ticker, "from": since, "to": today, "token": api_key},
                timeout=8,
            )
            resp.raise_for_status()
            articles = resp.json()
            if not isinstance(articles, list):
                return []
            headlines = [a["headline"] for a in articles[:5] if a.get("headline")]
            return headlines
        except Exception:
            return []

    def _fetch_newsapi(self, ticker: str, company_name: str) -> list[str]:
        try:
            since = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")
            resp  = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q":        f'"{ticker}" OR "{company_name}"',
                    "from":     since,
                    "sortBy":   "publishedAt",
                    "pageSize": 5,
                    "apiKey":   self.news_api_key,
                    "language": "en",
                },
                timeout=8,
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            return [a["title"] for a in articles if a.get("title")]
        except Exception:
            return []

    def _fetch_yfinance_news(self, ticker: str) -> list[str]:
        try:
            t     = yf.Ticker(ticker)
            news  = t.news or []
            since = datetime.utcnow() - timedelta(hours=48)
            headlines = []
            for n in news[:5]:
                ts = n.get("providerPublishTime", 0)
                if ts and datetime.fromtimestamp(ts) >= since:
                    title = n.get("title", "")
                    if title:
                        headlines.append(title)
            return headlines
        except Exception:
            return []
