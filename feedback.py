"""
feedback.py – Adaptive Lern-Loop

Fixes:
  M-04: history.json zeigte avg_return=0.0 für alle Bins trotz 55+ Trades.
        Ursache: feedback.py lief täglich NACH pipeline.py (scanner.yml),
        aber berechnete stock_return=0.0 wenn option_proxy==current (Trades
        vom selben Tag). Fix: Trades < 7 Tage alt werden übersprungen.
  M-05: `stock_return` (Aktienrendite) war falsches Proxy für Options-P&L.
        Fix: Wenn Options-Preis (last) im Trade gespeichert, wird dessen
        Rendite berechnet. Fallback: Stock-Return mit Delta-Approximation.
  cfg:  LEARNING_RATE und CLOSE_AFTER_DAYS aus config.yaml.
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yfinance as yf
from scipy import stats

from modules.config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

HISTORY_PATH = Path("outputs/history.json")

# FIX M-04: Mindest-Alter bevor ein Trade im Feedback bewertet wird.
# Trades vom selben Tag haben entry_price ≈ current_price → Return ≈ 0.
MIN_TRADE_AGE_DAYS = 7


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


def get_current_price(ticker: str) -> float:
    try:
        info = yf.Ticker(ticker).info
        return float(
            info.get("currentPrice") or info.get("regularMarketPrice") or 0
        )
    except Exception:
        return 0.0


def get_current_option_price(ticker: str, option: dict) -> float:
    """
    FIX M-05: Versucht den aktuellen Options-Preis via yfinance abzurufen.
    Sucht nach dem Kontrakt mit passendem Strike und Expiry.
    Fallback: 0.0 (dann wird Stock-Return verwendet).
    """
    if not option:
        return 0.0

    strike = option.get("strike")
    expiry = option.get("expiry")
    if not strike or not expiry:
        return 0.0

    try:
        t = yf.Ticker(ticker)
        if expiry not in t.options:
            return 0.0

        chain = t.option_chain(expiry)
        calls = chain.calls

        matches = calls[
            (calls["strike"] == strike) &
            (calls["ask"] > 0)
        ]
        if matches.empty:
            return 0.0

        # Mid-Price als fairen Preis
        row = matches.iloc[0]
        return float((row["bid"] + row["ask"]) / 2)

    except Exception as e:
        log.debug(f"Options-Preis Fehler für {ticker}: {e}")
        return 0.0


def compute_outcome(trade: dict, current_stock_price: float) -> float:
    """
    FIX M-05: Berechnet den Outcome korrekt.

    Priorität:
    1. Options-P&L: (current_option_mid - entry_option_last) / entry_option_last
    2. Delta-approximierter Return: stock_return × avg_delta (0.65)
    3. Reiner Stock-Return als letzter Fallback

    Begründung: Option-P&L ist der betriebswirtschaftlich relevante Output.
    Stock-Return als Lern-Signal führt zu falscher Feature-Gewichtung.
    """
    ticker      = trade["ticker"]
    option      = trade.get("option", {})
    sim         = trade.get("simulation", {})
    entry_stock = sim.get("current_price", 0)

    # Stock-Return (immer verfügbar)
    if entry_stock > 0 and current_stock_price > 0:
        stock_return = (current_stock_price - entry_stock) / entry_stock
    else:
        stock_return = 0.0

    # Versuch 1: Echter Options-P&L
    entry_last = option.get("last", 0) if option else 0
    if entry_last > 0:
        current_option = get_current_option_price(ticker, option)
        if current_option > 0:
            options_return = (current_option - entry_last) / entry_last
            log.info(
                f"    Options-P&L: entry=${entry_last:.2f} "
                f"→ current=${current_option:.2f} "
                f"= {options_return:+.2%}"
            )
            return options_return

    # Versuch 2: Delta-approximierter Return
    # Für Long Calls/Puts mit Delta ~0.65:
    # Option-Return ≈ Stock-Return × (Aktienpreis / Optionspreis) × Delta
    # Vereinfacht: stock_return × leverage_factor
    if entry_last > 0 and entry_stock > 0:
        leverage = (entry_stock / entry_last) * 0.65
        approx_return = stock_return * leverage
        log.info(
            f"    Delta-approx Return: {stock_return:+.2%} × "
            f"{leverage:.1f} = {approx_return:+.2%}"
        )
        return approx_return

    # Fallback: Reiner Stock-Return
    log.info(f"    Stock-Return Fallback: {stock_return:+.2%}")
    return stock_return


def update_bin(stats_dict: dict, feature: str, bin_label: str, outcome: float) -> None:
    """Aktualisiert laufenden Durchschnitt für einen Bin."""
    bin_data = stats_dict.setdefault(feature, {}).setdefault(
        bin_label, {"count": 0, "avg_return": 0.0}
    )
    old_avg          = bin_data["avg_return"]
    old_cnt          = bin_data["count"]
    new_cnt          = old_cnt + 1
    new_avg          = (old_avg * old_cnt + outcome) / new_cnt
    bin_data["count"]      = new_cnt
    bin_data["avg_return"] = round(new_avg, 6)


def compute_pearson_weights(history: dict) -> dict:
    """Pearson-Korrelation → Feature-Gewichte."""
    closed = history.get("closed_trades", [])
    if len(closed) < 5:
        log.info("Zu wenig abgeschlossene Trades für Gewichts-Update.")
        return history["model_weights"]

    outcomes   = []
    impacts    = []
    mismatches = []
    drifts     = []

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
        return history["model_weights"]

    outcomes_arr = np.array(outcomes)
    correlations = {}
    for name, arr in [
        ("impact",    np.array(impacts)),
        ("mismatch",  np.array(mismatches)),
        ("eps_drift", np.array(drifts)),
    ]:
        r, _ = stats.pearsonr(arr, outcomes_arr)
        correlations[name] = max(r, 0)

    total = sum(correlations.values()) or 1.0
    old_w = history["model_weights"]
    new_w = {}
    for feat, corr in correlations.items():
        raw_new     = corr / total
        old         = old_w.get(feat, 1/3)
        new_w[feat] = round(
            old + cfg.learning.learning_rate * (raw_new - old), 4
        )

    total_w = sum(new_w.values())
    new_w   = {k: round(v / total_w, 4) for k, v in new_w.items()}

    log.info(f"Neue Gewichte: {new_w} (alt: {old_w})")
    return new_w


def _bin_to_num(feature: str, bin_label: str) -> float:
    mapping = {
        "impact":    {"low": 0.0, "mid": 0.5, "high": 1.0},
        "mismatch":  {"weak": 0.0, "good": 0.5, "strong": 1.0},
        "eps_drift": {"noise": 0.0, "relevant": 0.5, "massive": 1.0},
    }
    return mapping.get(feature, {}).get(bin_label, 0.5)


def main() -> None:
    log.info("=== Feedback-Loop gestartet ===")
    history      = load_history()
    today        = datetime.utcnow()
    active       = history.get("active_trades", [])
    still_active = []

    for trade in active:
        ticker     = trade["ticker"]
        entry_date = datetime.strptime(trade["entry_date"][:10], "%Y-%m-%d")
        age_days   = (today - entry_date).days

        # FIX M-04: Zu junge Trades überspringen
        if age_days < MIN_TRADE_AGE_DAYS:
            log.info(
                f"  [{ticker}] Alter={age_days}d < {MIN_TRADE_AGE_DAYS} "
                f"→ noch zu jung für Feedback."
            )
            still_active.append(trade)
            continue

        current = get_current_price(ticker)
        if current <= 0:
            still_active.append(trade)
            continue

        # FIX M-05: Korrekter Outcome (Options-P&L statt Stock-Return)
        outcome = compute_outcome(trade, current)

        log.info(
            f"  [{ticker}] Alter={age_days}d "
            f"Outcome={outcome:+.2%}"
        )

        # Bin-Updates
        feat = trade.get("features", {})
        for f_name, bin_key in [
            ("impact",    "bin_impact"),
            ("mismatch",  "bin_mismatch"),
            ("eps_drift", "bin_eps_drift"),
        ]:
            bin_label = feat.get(bin_key)
            if bin_label:
                update_bin(
                    history["feature_stats"], f_name, bin_label, outcome
                )

        if age_days >= cfg.learning.close_after_days:
            trade["outcome"]     = round(outcome, 4)
            trade["close_date"]  = today.strftime("%Y-%m-%d")
            trade["close_price"] = current
            history.setdefault("closed_trades", []).append(trade)
            log.info(f"  [{ticker}] Trade abgeschlossen (Return={outcome:+.2%})")
        else:
            trade["last_price"]    = current
            trade["current_return"] = round(outcome, 4)
            still_active.append(trade)

    history["active_trades"]  = still_active
    history["model_weights"]  = compute_pearson_weights(history)

    save_history(history)
    log.info("=== Feedback-Loop abgeschlossen ===")


if __name__ == "__main__":
    main()
