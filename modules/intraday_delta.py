"""
modules/intraday_delta.py – Intraday-Delta seit News-Veröffentlichung

Priorität 1: Verhindert "Late-to-the-party"-Trades.

Logik:
  Wenn eine Aktie seit Veröffentlichung der relevanten News bereits X% gestiegen
  ist, ist die Informations-Asymmetrie eingepreist. Der Scanner hätte zu spät
  reagiert und kauft in die Stärke hinein.

Schwellenwert: max_intraday_move aus config.yaml (default: 0.07 = 7%)
  - Unter 7%: Signal bleibt aktiv (Markt hat noch nicht vollständig reagiert)
  - Über 7%: Signal wird verworfen (zu spät, Asymmetrie weg)

Datenquelle: yfinance intraday (1m-Daten des heutigen Tages)
             Kein API-Key nötig.
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

log = logging.getLogger(__name__)

# Standardschwelle: 7% Move seit Tagesbeginn → Signal verwerfen
DEFAULT_MAX_MOVE = 0.07


def get_intraday_move(ticker: str) -> dict:
    """
    Berechnet den heutigen Intraday-Move vom Eröffnungskurs zum aktuellen Kurs.

    Returns:
        {
            "move_pct": float,        # z.B. 0.082 = +8.2% seit Eröffnung
            "open_price": float,
            "current_price": float,
            "data_available": bool
        }
    """
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="1d", interval="5m")

        if hist.empty or len(hist) < 2:
            return _no_data()

        open_price    = float(hist["Open"].iloc[0])
        current_price = float(hist["Close"].iloc[-1])

        if open_price <= 0:
            return _no_data()

        move_pct = (current_price - open_price) / open_price

        log.debug(
            f"  [{ticker}] Intraday: open={open_price:.2f} "
            f"current={current_price:.2f} move={move_pct:+.2%}"
        )

        return {
            "move_pct":      round(move_pct, 4),
            "open_price":    round(open_price, 2),
            "current_price": round(current_price, 2),
            "data_available": True,
        }

    except Exception as e:
        log.debug(f"  [{ticker}] Intraday-Fehler: {e}")
        return _no_data()


def is_already_moved(
    ticker: str,
    direction: str = "BULLISH",
    max_move: float = DEFAULT_MAX_MOVE,
) -> tuple[bool, dict]:
    """
    Prüft ob eine Aktie bereits zu stark in die erwartete Richtung gelaufen ist.

    Args:
        ticker:    Aktien-Ticker
        direction: "BULLISH" oder "BEARISH"
        max_move:  Maximaler erlaubter Move (default: 7%)

    Returns:
        (already_moved: bool, delta_info: dict)
        already_moved=True → Signal verwerfen, zu spät
    """
    delta = get_intraday_move(ticker)

    if not delta["data_available"]:
        # Keine Daten → konservativ: Signal nicht verwerfen
        return False, delta

    move = delta["move_pct"]

    # Bullish: Wenn Aktie schon stark gestiegen → zu spät
    if direction == "BULLISH" and move > max_move:
        log.info(
            f"  [{ticker}] INTRADAY-GATE: move={move:+.2%} > {max_move:.0%} "
            f"(BULLISH) → Signal zu spät, verworfen."
        )
        return True, delta

    # Bearish: Wenn Aktie schon stark gefallen → zu spät
    if direction == "BEARISH" and move < -max_move:
        log.info(
            f"  [{ticker}] INTRADAY-GATE: move={move:+.2%} < -{max_move:.0%} "
            f"(BEARISH) → Signal zu spät, verworfen."
        )
        return True, delta

    log.info(
        f"  [{ticker}] Intraday-Move={move:+.2%} → noch Asymmetrie vorhanden."
    )
    return False, delta


def filter_by_intraday_delta(
    signals: list[dict],
    max_move: float = DEFAULT_MAX_MOVE,
) -> list[dict]:
    """
    Filtert eine Liste von Pipeline-Signalen nach dem Intraday-Delta.
    Wird nach Stufe 4 (Mismatch-Score) und vor Stufe 5 (Simulation) aufgerufen.

    Fügt `intraday_delta` zu jedem Signal-Dict hinzu.
    Entfernt Signale bei denen der Move zu groß ist.
    """
    filtered = []
    for s in signals:
        ticker    = s.get("ticker", "")
        direction = s.get("deep_analysis", {}).get("direction", "BULLISH")

        already_moved, delta_info = is_already_moved(ticker, direction, max_move)

        s["intraday_delta"] = delta_info

        if already_moved:
            continue

        filtered.append(s)

    log.info(
        f"Intraday-Delta-Filter: {len(signals)} → {len(filtered)} Signale "
        f"({len(signals)-len(filtered)} zu spät)"
    )
    return filtered


def _no_data() -> dict:
    return {
        "move_pct":       0.0,
        "open_price":     0.0,
        "current_price":  0.0,
        "data_available": False,
    }
