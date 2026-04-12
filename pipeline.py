"""
Adaptive Asymmetry-Scanner v5.0
Hauptpipeline – täglich ausgeführt via GitHub Actions (14:30 MEZ)

Neue Module gegenüber v4.0:
  - Stufe 1c: Alpha-Sources (FDA, SEC Insider) für YES-Ticker
  - Stufe 1d: Daten-Validierung (Alpha Vantage EPS Cross-Check)
  - Stufe 4b: Intraday-Delta-Filter (Move seit News)
  - Stufe 6b: Premium-Signals (FLASH Alpha + Eulerpool) für Top-2
  - Stufe 7:  Bid-Ask ROI Gate in OptionsDesigner
  - Earnings: Finnhub statt yfinance als primäre Quelle
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from modules.data_ingestion    import DataIngestion
from modules.prescreener       import Prescreener
from modules.deep_analysis     import DeepAnalysis
from modules.mismatch_scorer   import MismatchScorer
from modules.mirofish_simulation import MirofishSimulation
from modules.rl_agent          import RLScorer
from modules.options_designer  import OptionsDesigner
from modules.reporter          import Reporter
from modules.risk_gates        import RiskGates
from modules.email_reporter    import send_email
from modules.finbert_sentiment import score_candidate
from modules.reddit_signals    import enrich_candidate

# NEU v5.0
from modules.intraday_delta  import filter_by_intraday_delta
from modules.alpha_sources   import enrich_with_alpha_sources
from modules.data_validator  import validate_candidate_data, compute_option_roi
from modules.premium_signals import enrich_top_candidates
from modules.config          import cfg

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
            "impact":    {"low": {"count": 0, "avg_return": 0.0},
                          "mid": {"count": 0, "avg_return": 0.0},
                          "high": {"count": 0, "avg_return": 0.0}},
            "mismatch":  {"weak": {"count": 0, "avg_return": 0.0},
                          "good": {"count": 0, "avg_return": 0.0},
                          "strong": {"count": 0, "avg_return": 0.0}},
            "eps_drift": {"noise": {"count": 0, "avg_return": 0.0},
                          "relevant": {"count": 0, "avg_return": 0.0},
                          "massive": {"count": 0, "avg_return": 0.0}},
        },
        "active_trades": [],
        "closed_trades": [],
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
    new_trades    = []
    for proposal in new_proposals:
        key = (proposal["ticker"], today)
        if key in existing_keys:
            log.info(f"  [{proposal['ticker']}] Duplikat → übersprungen.")
            continue
        existing_keys.add(key)
        new_trades.append({
            "ticker":        proposal["ticker"],
            "entry_date":    today,
            "features":      proposal.get("features", {}),
            "strategy":      proposal.get("strategy", ""),
            "option":        proposal.get("option"),
            "simulation":    proposal.get("simulation"),
            "deep_analysis": proposal.get("deep_analysis"),
            "flash_alpha":   proposal.get("flash_alpha"),
            "eulerpool":     proposal.get("eulerpool"),
            "intraday_delta": proposal.get("intraday_delta"),
            "roi_analysis":  proposal.get("roi_analysis"),
            "outcome":       None,
        })
    return new_trades


def main() -> None:
    log.info("=== Adaptive Asymmetry-Scanner v5.0 gestartet ===")
    today   = datetime.utcnow().strftime("%Y-%m-%d")
    history = load_history()

    # ── STUFE 0: Risk-Gates ──────────────────────────────────────────────────
    gates = RiskGates()
    if not gates.global_ok():
        log.warning("Globales Risk-Gate ausgelöst. Abbruch.")
        return

    # ── STUFE 1: Daten-Ingestion ─────────────────────────────────────────────
    log.info("Stufe 1: Daten-Ingestion")
    ingestion  = DataIngestion(history=history)
    candidates = ingestion.run()
    log.info(f"  → {len(candidates)} Kandidaten nach Hard-Filter")
    if not candidates:
        return

    # ── STUFE 1b: FinBERT + Reddit ───────────────────────────────────────────
    log.info("Stufe 1b: FinBERT-Sentiment + Reddit-Enrichment")
    enriched = []
    for c in candidates:
        try:
            c = enrich_candidate(c)
        except Exception as e:
            log.debug(f"Reddit Fehler ({c.get('ticker')}): {e}")
        try:
            sentiment = score_candidate(c)
            c.setdefault("features", {}).update(sentiment)
        except Exception as e:
            log.debug(f"FinBERT Fehler ({c.get('ticker')}): {e}")
            c.setdefault("features", {}).update({
                "sentiment_score": 0.0, "sentiment_label": "neutral",
                "sentiment_confidence": 0.0,
            })
        enriched.append(c)
    candidates = enriched
    log.info(f"  → {len(candidates)} Kandidaten nach Enrichment")

    # ── STUFE 2: Prescreening ────────────────────────────────────────────────
    log.info("Stufe 2: Prescreening (Claude Haiku)")
    prescreener = Prescreener()
    shortlist   = prescreener.run(candidates)
    log.info(f"  → {len(shortlist)} Ticker nach Prescreening")
    if not shortlist:
        return

    # ── NEU STUFE 1c: Alpha-Sources für YES-Ticker ───────────────────────────
    log.info("Stufe 1c: Alpha-Sources (FDA, SEC Insider, Finnhub)")
    shortlist = [enrich_with_alpha_sources(c) for c in shortlist]

    # ── NEU STUFE 1d: Daten-Validierung (EPS Cross-Check) ───────────────────
    log.info("Stufe 1d: EPS Cross-Check (Alpha Vantage)")
    shortlist = [validate_candidate_data(c) for c in shortlist]

    # ── STUFE 3: Deep Analysis ───────────────────────────────────────────────
    log.info("Stufe 3: Deep Analysis (Claude Sonnet)")
    analyzer = DeepAnalysis()
    analyses = analyzer.run(shortlist)
    log.info(f"  → {len(analyses)} Analysen abgeschlossen")

    # ── STUFE 4: Mismatch-Score ──────────────────────────────────────────────
    log.info("Stufe 4: Mismatch-Score")
    scorer = MismatchScorer()
    scored = scorer.run(analyses)
    log.info(f"  → {len(scored)} Ticker nach Mismatch-Filter")

    # ── NEU STUFE 4b: Intraday-Delta-Filter ─────────────────────────────────
    log.info("Stufe 4b: Intraday-Delta-Filter")
    max_move = getattr(getattr(cfg, 'pipeline', None), 'max_intraday_move', 0.07)
    scored   = filter_by_intraday_delta(scored, max_move=max_move)
    log.info(f"  → {len(scored)} Ticker nach Intraday-Filter")

    # ── STUFE 5: MiroFish Simulation ─────────────────────────────────────────
    log.info("Stufe 5: MiroFish Monte-Carlo-Simulation")
    simulator = MirofishSimulation()
    simulated = simulator.run(scored)
    log.info(f"  → {len(simulated)} Ticker nach Simulation")
    if not simulated:
        log.info("Keine Signale. Pipeline beendet.")
        reporter = Reporter(reports_dir=REPORTS_DIR)
        reporter.save(today=today, proposals=[], history=history)
        save_history(history)
        return

    # ── STUFE 6: RL-Scoring ──────────────────────────────────────────────────
    log.info("Stufe 6: RL-Agent Final-Scoring")
    rl_scorer     = RLScorer(history=history)
    final_signals = rl_scorer.run(simulated)
    log.info(f"  → {len(final_signals)} Signale nach RL-Filter")

    if not final_signals:
        log.info("RL-Agent hat alle Signale gefiltert. Pipeline beendet.")
        reporter = Reporter(reports_dir=REPORTS_DIR)
        reporter.save(today=today, proposals=[], history=history)
        save_history(history)
        return

    # ── NEU STUFE 6b: Premium-Signals für Top-2 ─────────────────────────────
    log.info("Stufe 6b: FLASH Alpha + Eulerpool (Top-2)")
    final_signals = enrich_top_candidates(final_signals, top_n=2)

    # ── STUFE 7: Options-Design ──────────────────────────────────────────────
    log.info("Stufe 7: Options-Design + ROI-Gate")
    designer        = OptionsDesigner(gates=gates)
    trade_proposals = designer.run(final_signals)

    # NEU: Bid-Ask ROI Gate
    roi_filtered = []
    min_roi      = float(getattr(getattr(cfg, 'options', None), 'min_roi_after_spread', 0.15))
    for proposal in trade_proposals:
        option     = proposal.get("option", {})
        simulation = proposal.get("simulation", {})
        if option:
            roi = compute_option_roi(option, simulation)
            proposal["roi_analysis"] = roi
            if not roi["passes_roi_gate"]:
                log.info(
                    f"  [{proposal['ticker']}] ROI-GATE: "
                    f"net_roi={roi['roi_net']:.1%} < {min_roi:.0%} → verworfen."
                )
                continue
        roi_filtered.append(proposal)

    trade_proposals = roi_filtered
    log.info(f"  → {len(trade_proposals)} Trade-Vorschläge nach ROI-Gate")

    trade_proposals = _inject_vix(trade_proposals, gates.last_vix)

    # ── REPORT & HISTORY ─────────────────────────────────────────────────────
    reporter = Reporter(reports_dir=REPORTS_DIR)
    reporter.save(today=today, proposals=trade_proposals, history=history)

    new_trades = _dedup_trades(history["active_trades"], trade_proposals, today)
    history["active_trades"].extend(new_trades)
    log.info(f"  {len(new_trades)} neue Trade(s) in history geschrieben.")

    save_history(history)

    try:
        send_email(trade_proposals, today)
    except Exception as e:
        log.error(f"Email-Fehler (nicht kritisch): {e}")

    log.info(
        f"=== Pipeline beendet. "
        f"{len(trade_proposals)} Trade-Vorschläge generiert. ==="
    )


if __name__ == "__main__":
    main()
