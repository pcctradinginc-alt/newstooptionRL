"""
Adaptive Asymmetry-Scanner v5.0
NEU: Tägliche Status-Email wird IMMER gesendet.
Korrektur: Sentiment-Drift in die Ingestion-Schleife integriert.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from modules.data_ingestion      import DataIngestion
from modules.prescreener          import Prescreener
from modules.deep_analysis        import DeepAnalysis
from modules.mismatch_scorer      import MismatchScorer
from modules.mirofish_simulation import MirofishSimulation
from modules.rl_agent             import RLScorer
from modules.options_designer     import OptionsDesigner
from modules.reporter             import Reporter
from modules.risk_gates           import RiskGates
from modules.email_reporter       import send_email, send_status_email
from modules.sentiment_tracker    import enrich_with_sentiment_drift
from modules.finbert_sentiment    import score_candidate  # Wichtig: Import ergänzt
from modules.reddit_signals       import enrich_candidate
from modules.intraday_delta       import filter_by_intraday_delta
from modules.alpha_sources        import enrich_with_alpha_sources
from modules.data_validator       import validate_candidate_data, compute_option_roi
from modules.premium_signals      import enrich_top_candidates
from modules.config               import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

HISTORY_PATH = Path("outputs/history.json")
REPORTS_DIR  = Path("outputs/daily_reports")


def load_history() -> dict:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return {
        "feature_stats": {
            "impact":    {"low": {"count": 0, "avg_return": 0.0}, "mid": {"count": 0, "avg_return": 0.0}, "high": {"count": 0, "avg_return": 0.0}},
            "mismatch":  {"weak": {"count": 0, "avg_return": 0.0}, "good": {"count": 0, "avg_return": 0.0}, "strong": {"count": 0, "avg_return": 0.0}},
            "eps_drift": {"noise": {"count": 0, "avg_return": 0.0}, "relevant": {"count": 0, "avg_return": 0.0}, "massive": {"count": 0, "avg_return": 0.0}},
        },
        "active_trades": [], "closed_trades": [],
        "model_weights": {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20},
    }


def save_history(history: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2, default=str)
    log.info("history.json gespeichert.")


def _inject_vix(proposals, vix):
    if vix is None:
        return proposals
    for p in proposals:
        if "simulation" in p:
            p["simulation"]["vix"] = round(vix, 2)
    return proposals


def _dedup_trades(existing, new_proposals, today):
    existing_keys = {(t["ticker"], t.get("entry_date", "")) for t in existing}
    new_trades = []
    for proposal in new_proposals:
        key = (proposal["ticker"], today)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        new_trades.append({
            "ticker": proposal["ticker"], "entry_date": today,
            "features": proposal.get("features", {}),
            "strategy": proposal.get("strategy", ""),
            "option": proposal.get("option"),
            "simulation": proposal.get("simulation"),
            "deep_analysis": proposal.get("deep_analysis"),
            "roi_analysis": proposal.get("roi_analysis"),
            "outcome": None,
        })
    return new_trades


def main() -> None:
    log.info("=== Adaptive Asymmetry-Scanner v5.0 gestartet ===")
    today   = datetime.utcnow().strftime("%Y-%m-%d")
    history = load_history()

    # Pipeline-Stats für tägliche Status-Email
    stats = {
        "vix": None, "candidates": 0, "prescreened": 0,
        "analyzed": 0, "mismatch_ok": 0, "intraday_ok": 0,
        "simulated": 0, "rl_scored": 0, "roi_ok": 0,
        "trades": 0, "stop_reason": "",
    }

    def send_daily_email(proposals=None):
        """Sendet immer eine Email — Trade oder Status."""
        try:
            if proposals:
                send_email(proposals, today)
            else:
                send_status_email(stats, today)
        except Exception as e:
            log.error(f"Email-Fehler: {e}")

    # ── STUFE 0: Risk-Gates ──────────────────────────────────────────────────
    gates = RiskGates()
    if not gates.global_ok():
        log.warning("Globales Risk-Gate ausgelöst. Abbruch.")
        stats["stop_reason"] = f"VIX-Gate ausgelöst (VIX={gates.last_vix:.1f} > Schwelle)"
        stats["vix"] = gates.last_vix
        send_daily_email()
        return

    stats["vix"] = gates.last_vix

    # ── STUFE 1: Daten-Ingestion ─────────────────────────────────────────────
    log.info("Stufe 1: Daten-Ingestion")
    ingestion  = DataIngestion(history=history)
    candidates = ingestion.run()
    stats["candidates"] = len(candidates)
    log.info(f"  → {len(candidates)} Kandidaten nach Hard-Filter")
    if not candidates:
        stats["stop_reason"] = "Keine Ticker mit relevanten News heute."
        send_daily_email()
        return

    # ── STUFE 1b: FinBERT + Reddit + Sentiment Drift ─────────────────────────
    log.info("Stufe 1b: Sentiment + Reddit-Enrichment")
    enriched = []
    for c in candidates:
        # Reddit Signale
        try: 
            c = enrich_candidate(c)
        except Exception: 
            pass
        
        # Sentiment Drift
        try:
            c = enrich_with_sentiment_drift(c, history)
        except Exception as e:
            log.debug(f"Sentiment Drift Fehler für {c.get('ticker')}: {e}")

        # FinBERT Sentiment
        try:
            sentiment = score_candidate(c)
            c.setdefault("features", {}).update(sentiment)
        except Exception:
            c.setdefault("features", {}).update({
                "sentiment_score": 0.0, 
                "sentiment_label": "neutral", 
                "sentiment_confidence": 0.0
            })
        enriched.append(c)
    candidates = enriched

    # ── STUFE 2: Prescreening ────────────────────────────────────────────────
    log.info("Stufe 2: Prescreening (Claude Haiku)")
    shortlist = Prescreener().run(candidates)
    stats["prescreened"] = len(shortlist)
    log.info(f"  → {len(shortlist)} Ticker nach Prescreening")
    if not shortlist:
        stats["stop_reason"] = f"Alle {len(candidates)} Kandidaten im Prescreening als 'kein strukturelles Signal' bewertet."
        send_daily_email()
        return

    # ── STUFE 1c+1d: Alpha-Sources + Validierung ─────────────────────────────
    log.info("Stufe 1c: Alpha-Sources (FDA, SEC Insider)")
    shortlist = [enrich_with_alpha_sources(c) for c in shortlist]
    log.info("Stufe 1d: EPS Cross-Check (Alpha Vantage)")
    shortlist = [validate_candidate_data(c) for c in shortlist]

    # ── STUFE 3: Deep Analysis ───────────────────────────────────────────────
    log.info("Stufe 3: Deep Analysis (Claude Sonnet)")
    analyses = DeepAnalysis().run(shortlist)
    stats["analyzed"] = len(analyses)
    log.info(f"  → {len(analyses)} Analysen abgeschlossen")

    # ── STUFE 4: Mismatch-Score ──────────────────────────────────────────────
    log.info("Stufe 4: Mismatch-Score")
    scored = MismatchScorer().run(analyses)
    stats["mismatch_ok"] = len(scored)
    log.info(f"  → {len(scored)} Ticker nach Mismatch-Filter")

    # ── STUFE 4b: Intraday-Delta ─────────────────────────────────────────────
    log.info("Stufe 4b: Intraday-Delta-Filter")
    max_move = getattr(getattr(cfg, "pipeline", None), "max_intraday_move", 0.07)
    scored   = filter_by_intraday_delta(scored, max_move=max_move)
    stats["intraday_ok"] = len(scored)
    log.info(f"  → {len(scored)} Ticker nach Intraday-Filter")

    # ── STUFE 5: MiroFish Simulation ─────────────────────────────────────────
    log.info("Stufe 5: MiroFish Monte-Carlo-Simulation")
    simulated = MirofishSimulation().run(scored)
    stats["simulated"] = len(simulated)
    log.info(f"  → {len(simulated)} Ticker nach Simulation")
    if not simulated:
        stats["stop_reason"] = (
            f"Kein Ticker hat die Confidence-Gate-Schwelle "
            f"({cfg.pipeline.confidence_gate:.0%}) in der Monte-Carlo-Simulation erreicht."
        )
        save_history(history)
        send_daily_email()
        return

    # ── STUFE 6: RL-Scoring ──────────────────────────────────────────────────
    log.info("Stufe 6: RL-Agent Final-Scoring")
    final_signals = RLScorer(history=history).run(simulated)
    stats["rl_scored"] = len(final_signals)
    log.info(f"  → {len(final_signals)} Signale nach RL-Filter")
    if not final_signals:
        stats["stop_reason"] = "RL-Agent hat alle Signale als SKIP klassifiziert (zu wenig Confidence)."
        save_history(history)
        send_daily_email()
        return

    # ── STUFE 6b: Premium-Signals für Top-2 ─────────────────────────────────
    log.info("Stufe 6b: FLASH Alpha + Eulerpool (Top-2)")
    final_signals = enrich_top_candidates(final_signals, top_n=2)

    # ── STUFE 7: Options-Design + ROI-Gate ───────────────────────────────────
    log.info("Stufe 7: Options-Design + ROI-Gate")
    trade_proposals = OptionsDesigner(gates=gates).run(final_signals)

    min_roi     = getattr(getattr(cfg, "options", None), "min_roi_after_spread", 0.15)
    roi_filtered = []
    for p in trade_proposals:
        option = p.get("option", {})
        if option:
            roi = compute_option_roi(option, p.get("simulation", {}))
            p["roi_analysis"] = roi
            if not roi["passes_roi_gate"]:
                log.info(f"  [{p['ticker']}] ROI-GATE: {roi['roi_net']:.1%} < {min_roi:.0%} → verworfen.")
                continue
        roi_filtered.append(p)

    trade_proposals = roi_filtered
    stats["roi_ok"] = len(trade_proposals)
    stats["trades"] = len(trade_proposals)

    if not trade_proposals:
        stats["stop_reason"] = f"Alle Options-Kontrakte scheitern am ROI-Gate (Mindest-ROI nach Spread: {min_roi:.0%})."

    trade_proposals = _inject_vix(trade_proposals, gates.last_vix)

    # ── REPORT & HISTORY ─────────────────────────────────────────────────────
    reporter = Reporter(reports_dir=REPORTS_DIR)
    reporter.save(today=today, proposals=trade_proposals, history=history)

    new_trades = _dedup_trades(history["active_trades"], trade_proposals, today)
    history["active_trades"].extend(new_trades)
    log.info(f"  {len(new_trades)} neue Trade(s) in history geschrieben.")
    save_history(history)

    # ── EMAIL — IMMER SENDEN ─────────────────────────────────────────────────
    send_daily_email(trade_proposals if trade_proposals else None)

    log.info(f"=== Pipeline beendet. {len(trade_proposals)} Trade-Vorschläge. ===")


if __name__ == "__main__":
    main()
