"""
feedback.py – Adaptive Lern-Loop v5.0

Änderungen v5.0:
    - Tradier Live-API als primäre Datenquelle für Optionspreise (P&L-Tracking)
      Endpoint: /v1/markets/options/chains (Optionspreis via Strike-Filter)
      Endpoint: /v1/markets/quotes        (Aktienkurs Real-Time)
    - get_current_price():        Tradier Primary → yfinance Fallback
    - get_current_option_price(): Tradier Primary → yfinance Fallback
    - compute_outcome():          strategy-Parameter für saubere Call/Put-Erkennung
    - TRADIER_API_KEY via os.environ (bereits als GitHub Secret hinterlegt)
    - Warum wichtig: RL-Agent trainiert auf Outcomes — falsche Preise (yfinance
      ~15min delayed) führen zu fehlerhaften Lern-Signalen für den PPO-Agenten.

Änderungen v4.0:
    - Nach Trade-Close: PPO-Agent wird auf neuem closed_trade nachtrainiert
    - RL-Training: Inkrementelles Update (Continual Learning)
    - Bestehende Fixes M-04, M-05 bleiben erhalten
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import yfinance as yf
from scipy import stats

from modules.config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

HISTORY_PATH       = Path("outputs/history.json")
MIN_TRADE_AGE_DAYS = 7

TRADIER_BASE    = "https://api.tradier.com/v1"
TRADIER_TIMEOUT = 10


# ── Tradier Hilfsfunktionen ───────────────────────────────────────────────────

def _tradier_headers() -> dict:
    """Authorization-Header für Tradier Live-API."""
    api_key = os.environ.get("TRADIER_API_KEY", "")
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept":        "application/json",
    }


def _use_tradier() -> bool:
    """Gibt True zurück wenn TRADIER_API_KEY gesetzt ist."""
    return bool(os.environ.get("TRADIER_API_KEY", "").strip())


# ── History I/O ───────────────────────────────────────────────────────────────

def load_history() -> dict:
    if not HISTORY_PATH.exists():
        log.error("history.json nicht gefunden.")
        sys.exit(1)
    with open(HISTORY_PATH) as f:
        return json.load(f)


def save_history(history: dict) -> None:
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2, default=str)
    log.info("history.json aktualisiert.")


# ── Preis-Abruf: Aktienkurs ───────────────────────────────────────────────────

def get_current_price(ticker: str) -> float:
    """
    Aktueller Aktienkurs: Tradier Primary → yfinance Fallback.

    Tradier /v1/markets/quotes liefert Real-Time-Kurse ohne Delay.
    yfinance als Fallback wenn Tradier nicht erreichbar.
    """
    if _use_tradier():
        price = _tradier_stock_price(ticker)
        if price > 0:
            return price
        log.debug(f"[{ticker}] Tradier Aktienkurs fehlgeschlagen → yfinance")

    # yfinance Fallback
    try:
        info = yf.Ticker(ticker).info
        return float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    except Exception:
        return 0.0


def _tradier_stock_price(ticker: str) -> float:
    """
    Aktienkurs via Tradier /v1/markets/quotes.

    Response-Struktur:
        {"quotes": {"quote": {"last": 150.25, "bid": ..., "ask": ...}}}
    """
    try:
        resp = requests.get(
            f"{TRADIER_BASE}/markets/quotes",
            params={"symbols": ticker, "greeks": "false"},
            headers=_tradier_headers(),
            timeout=TRADIER_TIMEOUT,
        )
        resp.raise_for_status()
        data  = resp.json()
        quote = data.get("quotes", {}).get("quote", {})

        # Mehrere Symbole → Liste; einzelnes Symbol → Dict
        if isinstance(quote, list):
            quote = next((q for q in quote if q.get("symbol") == ticker), {})

        # "last" bevorzugt; Fallback auf Mid aus Bid/Ask
        last = quote.get("last")
        if last and float(last) > 0:
            return float(last)

        bid = float(quote.get("bid") or 0)
        ask = float(quote.get("ask") or 0)
        if bid > 0 and ask > 0:
            return round((bid + ask) / 2, 4)

        return 0.0

    except Exception as e:
        log.debug(f"Tradier Aktienkurs [{ticker}]: {e}")
        return 0.0


# ── Preis-Abruf: Optionspreis ─────────────────────────────────────────────────

def get_current_option_price(
    ticker: str, option: dict, strategy: str = ""
) -> float:
    """
    Aktueller Mid-Price einer Options-Position: Tradier Primary → yfinance Fallback.

    Args:
        ticker:   Ticker-Symbol (z.B. "AAPL")
        option:   Option-Dict aus history.json (strike, expiry, ...)
        strategy: Trade-Strategie (z.B. "LONG_CALL", "BEAR_PUT_SPREAD")
                  → bestimmt ob Call oder Put gesucht wird.
                  Leer → versucht Call zuerst, dann Put.
    """
    if not option:
        return 0.0
    strike = option.get("strike")
    expiry = option.get("expiry")
    if not strike or not expiry:
        return 0.0

    # Option-Type aus Strategie ableiten
    option_type = _option_type_from_strategy(strategy)

    # ── Versuch 1: Tradier ────────────────────────────────────────────────────
    if _use_tradier():
        price = _tradier_option_price(ticker, strike, expiry, option_type)
        if price > 0:
            log.debug(
                f"[{ticker}] Tradier Options-Mid: strike={strike} "
                f"expiry={expiry} → ${price:.2f}"
            )
            return price
        log.debug(f"[{ticker}] Tradier Options-Preis fehlgeschlagen → yfinance")

    # ── Versuch 2: yfinance Fallback ──────────────────────────────────────────
    return _yfinance_option_price(ticker, strike, expiry)


def _option_type_from_strategy(strategy: str) -> str:
    """
    Leitet "call" oder "put" aus der Trade-Strategie ab.

    "LONG_CALL", "BULL_CALL_SPREAD" → "call"
    "LONG_PUT",  "BEAR_PUT_SPREAD"  → "put"
    ""                              → "call" (Standard-Fallback; wird in
                                      _tradier_option_price auch als Put versucht)
    """
    s = strategy.upper()
    if "PUT" in s or "BEAR" in s:
        return "put"
    return "call"  # Default: Call (häufiger Fall)


def _tradier_option_price(
    ticker: str, strike: float, expiry: str, option_type: str
) -> float:
    """
    Options-Mid-Price via Tradier /v1/markets/options/chains.

    Filtert die Chain nach Strike ± 0.01 und option_type.
    Wenn option_type="call" und nichts gefunden → versucht "put" (Fallback
    bei alten Trades ohne Strategy-Info in history.json).
    """
    def _fetch_mid(o_type: str) -> float:
        try:
            resp = requests.get(
                f"{TRADIER_BASE}/markets/options/chains",
                params={
                    "symbol":     ticker,
                    "expiration": expiry,
                    "greeks":     "false",
                },
                headers=_tradier_headers(),
                timeout=TRADIER_TIMEOUT,
            )
            resp.raise_for_status()
            data    = resp.json()
            options = data.get("options", {}).get("option", []) or []

            # Einzelner Kontrakt kommt als Dict
            if isinstance(options, dict):
                options = [options]

            for o in options:
                if o.get("option_type") != o_type:
                    continue
                # Strike-Vergleich mit Float-Toleranz
                if abs(float(o.get("strike", 0)) - float(strike)) > 0.01:
                    continue

                bid = float(o.get("bid") or 0)
                ask = float(o.get("ask") or 0)
                if bid > 0 and ask > 0:
                    return round((bid + ask) / 2, 4)
                # Nur Ask vorhanden
                if ask > 0:
                    return float(ask)

            return 0.0

        except Exception as e:
            log.debug(f"Tradier Options-Chain [{ticker} {expiry}]: {e}")
            return 0.0

    # Primärer Versuch
    price = _fetch_mid(option_type)
    if price > 0:
        return price

    # Fallback: anderer Option-Type (für alte Trades ohne Strategy-Info)
    other_type = "put" if option_type == "call" else "call"
    return _fetch_mid(other_type)


def _yfinance_option_price(
    ticker: str, strike: float, expiry: str
) -> float:
    """Options-Mid-Price via yfinance (Fallback, unveränderte v4.0-Logik)."""
    try:
        t = yf.Ticker(ticker)
        if expiry not in t.options:
            return 0.0
        chain   = t.option_chain(expiry)
        # Versuche Calls zuerst, dann Puts
        for opts in [chain.calls, chain.puts]:
            matches = opts[(opts["strike"] == strike) & (opts["ask"] > 0)]
            if not matches.empty:
                row = matches.iloc[0]
                return float((row["bid"] + row["ask"]) / 2)
        return 0.0
    except Exception as e:
        log.debug(f"yfinance Options-Preis Fehler für {ticker}: {e}")
        return 0.0


# ── Outcome-Berechnung ────────────────────────────────────────────────────────

def compute_outcome(trade: dict, current_stock_price: float) -> float:
    """
    Berechnet Trade-Outcome (Return) für das RL-Training.

    Reihenfolge:
    1. Echter Options-P&L (wenn entry_last bekannt → Tradier/yfinance Preis)
    2. Delta-approximierter Return (Leverage-Schätzung)
    3. Reiner Stock-Return als letzter Fallback
    """
    ticker      = trade["ticker"]
    option      = trade.get("option", {})
    strategy    = trade.get("strategy", "")   # v5.0: für Option-Type-Erkennung
    sim         = trade.get("simulation", {})
    entry_stock = sim.get("current_price", 0)

    stock_return = 0.0
    if entry_stock > 0 and current_stock_price > 0:
        stock_return = (current_stock_price - entry_stock) / entry_stock

    entry_last = option.get("last", 0) if option else 0
    if entry_last > 0:
        # v5.0: strategy wird weitergegeben für saubere Call/Put-Erkennung
        current_option = get_current_option_price(ticker, option, strategy)
        if current_option > 0:
            options_return = (current_option - entry_last) / entry_last
            log.info(
                f"    Options-P&L: entry=${entry_last:.2f} → "
                f"current=${current_option:.2f} = {options_return:+.2%}"
            )
            return options_return

    if entry_last > 0 and entry_stock > 0:
        leverage      = (entry_stock / entry_last) * 0.65
        approx_return = stock_return * leverage
        log.info(f"    Delta-approx: {stock_return:+.2%} × {leverage:.1f} = {approx_return:+.2%}")
        return approx_return

    log.info(f"    Stock-Return Fallback: {stock_return:+.2%}")
    return stock_return


# ── Bin-Updates (Legacy, für Backward-Kompatibilität) ─────────────────────────

def update_bin(stats_dict: dict, feature: str, bin_label: str, outcome: float) -> None:
    bin_data = stats_dict.setdefault(feature, {}).setdefault(
        bin_label, {"count": 0, "avg_return": 0.0}
    )
    old_avg = bin_data["avg_return"]
    old_cnt = bin_data["count"]
    new_cnt = old_cnt + 1
    new_avg = (old_avg * old_cnt + outcome) / new_cnt
    bin_data["count"]      = new_cnt
    bin_data["avg_return"] = round(new_avg, 6)


# ── RL-Training ───────────────────────────────────────────────────────────────

def retrain_rl_agent(history: dict) -> None:
    """
    Trainiert den PPO-Agenten inkrementell auf allen closed_trades.

    Continual Learning: 2.000 Steps pro Feedback-Lauf (~5s auf CPU).
    GitHub-Actions-tauglich: Modell als .zip committed, nächster Run nutzt es.
    """
    try:
        from modules.rl_agent import train_agent
    except ImportError as e:
        log.warning(f"RL-Agent nicht importierbar: {e} → Training übersprungen")
        return

    closed = history.get("closed_trades", [])
    if len(closed) < 5:
        log.info(
            f"Nur {len(closed)} closed_trades → RL-Training übersprungen "
            f"(Minimum: 5)."
        )
        return

    log.info(f"Starte RL-Nachtraining auf {len(closed)} closed_trades...")
    success = train_agent(
        history         = history,
        total_timesteps = 2_000,
        force_retrain   = False,
    )

    if success:
        log.info("RL-Agent erfolgreich nachtrainiert.")
    else:
        log.warning("RL-Nachtraining fehlgeschlagen (nicht kritisch).")


# ── Pearson-Gewichte (Legacy-Support) ────────────────────────────────────────

def compute_pearson_weights(history: dict) -> dict:
    closed = history.get("closed_trades", [])
    if len(closed) < 5:
        return history.get("model_weights", {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20})

    outcomes, impacts, mismatches, drifts = [], [], [], []
    for t in closed:
        outcome = t.get("outcome")
        if outcome is None:
            continue
        feat = t.get("features", {})
        outcomes.append(outcome)
        impacts.append(_bin_to_num("impact",    feat.get("bin_impact",    "mid")))
        mismatches.append(_bin_to_num("mismatch", feat.get("bin_mismatch",  "good")))
        drifts.append(_bin_to_num("eps_drift", feat.get("bin_eps_drift", "noise")))

    if len(outcomes) < 5:
        return history.get("model_weights", {})

    outcomes_arr = np.array(outcomes)
    correlations = {}
    for name, arr in [("impact", np.array(impacts)),
                       ("mismatch", np.array(mismatches)),
                       ("eps_drift", np.array(drifts))]:
        r, _ = stats.pearsonr(arr, outcomes_arr)
        correlations[name] = max(r, 0)

    total = sum(correlations.values()) or 1.0
    old_w = history.get("model_weights", {})
    new_w = {}
    for feat, corr in correlations.items():
        raw_new     = corr / total
        old         = old_w.get(feat, 1/3)
        new_w[feat] = round(old + cfg.learning.learning_rate * (raw_new - old), 4)

    total_w = sum(new_w.values())
    return {k: round(v / total_w, 4) for k, v in new_w.items()}


def _bin_to_num(feature: str, bin_label: str) -> float:
    mapping = {
        "impact":    {"low": 0.0, "mid": 0.5, "high": 1.0},
        "mismatch":  {"weak": 0.0, "good": 0.5, "strong": 1.0},
        "eps_drift": {"noise": 0.0, "relevant": 0.5, "massive": 1.0},
    }
    return mapping.get(feature, {}).get(bin_label, 0.5)


# ── Haupt-Loop ────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Feedback-Loop v5.0 gestartet ===")
    log.info(f"Tradier: {'aktiv' if _use_tradier() else 'KEIN KEY → yfinance Fallback'}")

    history      = load_history()
    today        = datetime.utcnow()
    active       = history.get("active_trades", [])
    still_active = []
    newly_closed = 0

    for trade in active:
        ticker     = trade["ticker"]
        entry_date = datetime.strptime(trade["entry_date"][:10], "%Y-%m-%d")
        age_days   = (today - entry_date).days

        if age_days < MIN_TRADE_AGE_DAYS:
            log.info(f"  [{ticker}] Alter={age_days}d < {MIN_TRADE_AGE_DAYS} → zu jung.")
            still_active.append(trade)
            continue

        current = get_current_price(ticker)
        if current <= 0:
            still_active.append(trade)
            continue

        outcome = compute_outcome(trade, current)
        log.info(f"  [{ticker}] Alter={age_days}d Outcome={outcome:+.2%}")

        # Legacy Bin-Updates (für Backward-Kompatibilität mit QuasiML)
        feat = trade.get("features", {})
        for f_name, bin_key in [("impact",    "bin_impact"),
                                  ("mismatch",  "bin_mismatch"),
                                  ("eps_drift", "bin_eps_drift")]:
            bin_label = feat.get(bin_key)
            if bin_label:
                update_bin(history["feature_stats"], f_name, bin_label, outcome)

        if age_days >= cfg.learning.close_after_days:
            trade["outcome"]      = round(outcome, 4)
            trade["close_date"]   = today.strftime("%Y-%m-%d")
            trade["close_price"]  = current
            history.setdefault("closed_trades", []).append(trade)
            log.info(f"  [{ticker}] Trade abgeschlossen (Return={outcome:+.2%})")
            newly_closed += 1
        else:
            trade["last_price"]     = current
            trade["current_return"] = round(outcome, 4)
            still_active.append(trade)

    history["active_trades"] = still_active
    history["model_weights"] = compute_pearson_weights(history)

    save_history(history)

    if newly_closed > 0:
        log.info(f"{newly_closed} neue closed_trades → starte RL-Nachtraining...")
        retrain_rl_agent(history)
    else:
        log.info("Keine neuen closed_trades → RL-Training übersprungen.")

    log.info("=== Feedback-Loop abgeschlossen ===")


if __name__ == "__main__":
    main()
