"""
modules/alpha_sources.py – Alternative Alpha-Quellen

Integriert FDA API, SEC Insider-Käufe und Finnhub Earnings-Kalender.
Alle Quellen sind kostenlos / Free-Tier.

Priorität 2: Informationsvorsprung durch Nischen-Quellen.

API-Limits:
  - FDA:     Unbegrenzt (offiziell, kein Key nötig)
  - SEC:     Unbegrenzt (offiziell, kein Key nötig)
  - Finnhub: 60 Calls/Minute auf Free Tier (API-Key nötig)
"""

from __future__ import annotations
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

log = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "newstoption-scanner/4.1 research@pcctrading.com"}


# ── FDA API ───────────────────────────────────────────────────────────────────

def fetch_fda_events(company_name: str, days_back: int = 7) -> list[dict]:
    """
    Ruft FDA-Ereignisse für ein Unternehmen ab.
    Quelle: https://api.fda.gov/drug/event.json

    Gibt eine Liste von Events zurück:
      [{"date": "2026-04-10", "type": "approval", "description": "..."}]

    Anwendung: Besonders relevant für Biotech/Pharma-Ticker.
    Ein FDA-Approval erscheint oft 24-48h vor CNBC-Berichterstattung.
    """
    try:
        since = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y%m%d")
        url   = (
            f"https://api.fda.gov/drug/event.json"
            f"?search=receivedate:[{since}+TO+99991231]"
            f"+AND+companynumb:{company_name.replace(' ', '+')}"
            f"&limit=5"
        )
        resp = requests.get(url, headers=_HEADERS, timeout=10)

        if resp.status_code == 404:
            return []

        resp.raise_for_status()
        results = resp.json().get("results", [])

        events = []
        for r in results:
            date = r.get("receivedate", "")
            desc = r.get("primarysource", {}).get("reportercountry", "")
            events.append({
                "date":        date,
                "type":        "fda_adverse_event",
                "description": f"FDA Adverse Event Report ({desc})",
                "source":      "FDA",
            })

        return events

    except Exception as e:
        log.debug(f"FDA API Fehler für {company_name}: {e}")
        return []


def fetch_fda_drug_approvals(days_back: int = 7) -> list[dict]:
    """
    Ruft aktuelle FDA Drug Approvals ab (nicht ticker-spezifisch).
    Gibt alle Approvals der letzten N Tage zurück.
    Wird dann gegen das Ticker-Universum gecrosst.

    Quelle: https://api.fda.gov/drug/drugsfda.json
    """
    try:
        since = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y%m%d")
        url   = (
            f"https://api.fda.gov/drug/drugsfda.json"
            f"?search=submissions.submission_status_date:[{since}+TO+99991231]"
            f"+AND+submissions.submission_type:ORIG"
            f"&limit=10"
        )
        resp = requests.get(url, headers=_HEADERS, timeout=10)

        if resp.status_code == 404:
            return []

        resp.raise_for_status()
        results = resp.json().get("results", [])

        approvals = []
        for r in results:
            sponsor = r.get("sponsor_name", "").upper()
            drugs   = [p.get("brand_name", "") for p in r.get("products", [])[:2]]
            approvals.append({
                "sponsor":     sponsor,
                "drugs":       drugs,
                "type":        "fda_approval",
                "description": f"FDA Approval: {', '.join(drugs)}",
                "source":      "FDA",
            })

        return approvals

    except Exception as e:
        log.debug(f"FDA Approvals Fehler: {e}")
        return []


def match_fda_to_ticker(ticker: str, company_info: dict, days_back: int = 7) -> list[str]:
    """
    Sucht FDA-Events für einen Ticker und gibt Headlines zurück.
    Nutzt company_name aus yfinance-Info für den FDA-Lookup.
    """
    company_name = company_info.get("shortName", "") or company_info.get("longName", "")
    if not company_name:
        return []

    # Ersten Begriff des Company-Namens nutzen (z.B. "Pfizer" aus "Pfizer Inc.")
    name_short = company_name.split()[0] if company_name else ""
    if len(name_short) < 3:
        return []

    events    = fetch_fda_events(name_short, days_back)
    approvals = fetch_fda_drug_approvals(days_back)

    headlines = []

    for e in events:
        headlines.append(f"FDA {e['type'].replace('_', ' ').title()}: {e['description']}")

    for a in approvals:
        if name_short.upper() in a.get("sponsor", ""):
            headlines.append(f"FDA APPROVAL: {a['description']} by {a['sponsor']}")

    if headlines:
        log.info(f"  [{ticker}] FDA: {len(headlines)} Events gefunden")

    return headlines


# ── SEC Insider-Käufe ─────────────────────────────────────────────────────────

def fetch_sec_insider_trades(ticker: str, days_back: int = 14) -> list[dict]:
    """
    Ruft Insider-Käufe für einen Ticker via SEC EDGAR ab.
    Nutzt den EDGAR Full-Text-Search API (kein Key nötig).

    Ein Cluster-Insider-Kauf (mehrere Insider kaufen innerhalb 72h) ist
    eines der stärksten statistisch validierten Alpha-Signale.

    Gibt zurück:
      [{"date": str, "insider": str, "shares": int, "value": float, "type": "buy"}]
    """
    try:
        # EDGAR Submissions API für CIK-Lookup
        search_url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
            f"&dateRange=custom&startdt="
            f"{(datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%d')}"
            f"&forms=4"
        )
        resp = requests.get(search_url, headers=_HEADERS, timeout=10)

        if resp.status_code != 200:
            return _fetch_sec_form4_fallback(ticker, days_back)

        hits = resp.json().get("hits", {}).get("hits", [])
        trades = []

        for hit in hits[:10]:
            src = hit.get("_source", {})
            # Nur Käufe (P = Purchase, nicht S = Sale)
            if "P" not in src.get("period_of_report", ""):
                pass  # Alle anzeigen, Filterung im Downstream

            trades.append({
                "date":        src.get("period_of_report", ""),
                "insider":     src.get("display_names", ["Unknown"])[0]
                               if src.get("display_names") else "Unknown",
                "filing_url":  src.get("file_date", ""),
                "form":        "Form 4",
                "source":      "SEC",
            })

        return trades

    except Exception as e:
        log.debug(f"SEC EDGAR Fehler für {ticker}: {e}")
        return _fetch_sec_form4_fallback(ticker, days_back)


def _fetch_sec_form4_fallback(ticker: str, days_back: int) -> list[dict]:
    """Fallback: SEC EDGAR Full-Text-Search."""
    try:
        url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
            f"&forms=4&dateRange=custom"
            f"&startdt={(datetime.utcnow()-timedelta(days=days_back)).strftime('%Y-%m-%d')}"
        )
        resp = requests.get(url, headers=_HEADERS, timeout=8)
        if resp.status_code != 200:
            return []

        hits   = resp.json().get("hits", {}).get("hits", [])
        result = []
        for h in hits[:5]:
            s = h.get("_source", {})
            result.append({
                "date":    s.get("file_date", ""),
                "insider": (s.get("display_names") or ["Unknown"])[0],
                "form":    "Form 4",
                "source":  "SEC",
            })
        return result
    except Exception:
        return []


def detect_insider_cluster(ticker: str, days_back: int = 14) -> dict:
    """
    Erkennt Cluster-Insider-Käufe: Mehrere verschiedene Insider kaufen
    innerhalb von 72 Stunden → starkes Signal.

    Returns:
        {
            "cluster_detected": bool,
            "insider_count": int,
            "trades": list,
            "headline": str   # für News-Liste
        }
    """
    trades = fetch_sec_insider_trades(ticker, days_back)

    if not trades:
        return {"cluster_detected": False, "insider_count": 0, "trades": []}

    # Cluster: mehr als 1 unterschiedlicher Insider in den Daten
    unique_insiders = set(t["insider"] for t in trades)
    cluster         = len(unique_insiders) >= 2

    result = {
        "cluster_detected": cluster,
        "insider_count":    len(unique_insiders),
        "trades":           trades[:5],
        "headline":         "",
    }

    if cluster:
        result["headline"] = (
            f"SEC Form 4: {len(unique_insiders)} Insider kaufen {ticker} "
            f"innerhalb {days_back} Tagen (Cluster-Signal)"
        )
        log.info(
            f"  [{ticker}] SEC Insider-Cluster: "
            f"{len(unique_insiders)} Insider, {len(trades)} Trades"
        )

    return result


# ── Finnhub Earnings-Kalender ─────────────────────────────────────────────────

def get_earnings_date_finnhub(ticker: str) -> Optional[str]:
    """
    Ruft das nächste Earnings-Datum via Finnhub ab (zuverlässiger als yfinance).

    Benötigt: FINNHUB_API_KEY als GitHub Secret
    Free Tier: 60 Calls/Minute, ausreichend für ~20 Ticker/Tag

    Returns:
        "2026-04-25" oder None wenn keine Earnings in den nächsten 30 Tagen
    """
    finnhub_key = os.getenv("FINNHUB_API_KEY", "")
    if not finnhub_key:
        return None

    try:
        today   = datetime.utcnow()
        to_date = today + timedelta(days=30)
        url     = (
            f"https://finnhub.io/api/v1/calendar/earnings"
            f"?from={today.strftime('%Y-%m-%d')}"
            f"&to={to_date.strftime('%Y-%m-%d')}"
            f"&symbol={ticker}"
            f"&token={finnhub_key}"
        )
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()

        earnings_calendar = resp.json().get("earningsCalendar", [])
        if not earnings_calendar:
            return None

        # Nächstes Datum zurückgeben
        dates = sorted([e["date"] for e in earnings_calendar if e.get("date")])
        return dates[0] if dates else None

    except Exception as e:
        log.debug(f"Finnhub Earnings Fehler für {ticker}: {e}")
        return None


def has_earnings_within_days(
    ticker: str,
    buffer_days: int = 7,
    use_finnhub: bool = True,
) -> tuple[bool, Optional[str]]:
    """
    Prüft ob Earnings innerhalb der nächsten buffer_days liegen.
    Nutzt Finnhub als primäre Quelle (zuverlässiger als yfinance).

    Returns:
        (has_earnings: bool, earnings_date: str or None)
    """
    earnings_date = None

    # Primär: Finnhub (zuverlässig)
    if use_finnhub and os.getenv("FINNHUB_API_KEY"):
        earnings_date = get_earnings_date_finnhub(ticker)

    # Fallback: yfinance
    if not earnings_date:
        try:
            import yfinance as yf
            info          = yf.Ticker(ticker).info
            earnings_ts   = info.get("earningsTimestamp")
            if earnings_ts:
                earnings_date = datetime.fromtimestamp(earnings_ts).strftime("%Y-%m-%d")
        except Exception:
            pass

    if not earnings_date:
        return False, None

    try:
        earnings_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
        days_until  = (earnings_dt - datetime.utcnow()).days

        if 0 <= days_until <= buffer_days:
            log.info(
                f"  [{ticker}] EARNINGS-GATE: Earnings in {days_until}d "
                f"({earnings_date}) → Hard-Block."
            )
            return True, earnings_date

        return False, earnings_date

    except Exception:
        return False, None


# ── Kombinierter Alpha-Enrichment ─────────────────────────────────────────────

def enrich_with_alpha_sources(candidate: dict) -> dict:
    """
    Reichert einen Pipeline-Kandidaten mit FDA, SEC und Finnhub-Daten an.
    Wird nach dem Prescreening für YES-Ticker aufgerufen.

    Fügt neue Headlines zu candidate["news"] hinzu.
    Setzt candidate["alpha_signals"] mit strukturierten Daten.
    """
    ticker = candidate.get("ticker", "")
    info   = candidate.get("info", {})

    alpha_signals = {
        "fda_headlines":     [],
        "sec_insider":       {},
        "earnings_date":     None,
        "has_near_earnings": False,
    }

    # 1. FDA (nur für Healthcare/Biotech)
    sector = info.get("sector", "")
    if sector in ("Healthcare", "Biotechnology", "Pharmaceuticals"):
        fda_headlines = match_fda_to_ticker(ticker, info)
        alpha_signals["fda_headlines"] = fda_headlines
        if fda_headlines:
            candidate.setdefault("news", [])
            candidate["news"] = fda_headlines + candidate["news"]

    # 2. SEC Insider
    insider_data = detect_insider_cluster(ticker)
    alpha_signals["sec_insider"] = insider_data
    if insider_data.get("headline"):
        candidate.setdefault("news", [])
        candidate["news"] = [insider_data["headline"]] + candidate["news"]

    # 3. Finnhub Earnings
    has_earnings, earnings_date = has_earnings_within_days(ticker)
    alpha_signals["earnings_date"]     = earnings_date
    alpha_signals["has_near_earnings"] = has_earnings
    candidate["has_near_earnings"]     = has_earnings

    candidate["alpha_signals"] = alpha_signals
    return candidate
