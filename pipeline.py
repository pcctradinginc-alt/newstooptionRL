"""
pipeline.py v8.2 – Optimierte Reihenfolge + Korrelations-Check

v8.2 Änderungen:
  - NEU Stufe 3b: Pre-Deep-Analysis MC Gate (1k Pfade, sigma-only)
    Filtert hoffnungslose Kandidaten VOR dem teuren Sonnet-Call.
    Spart ~30-50% der Deep-Analysis-Kosten.
  - NEU Stufe 10b: Korrelations-Check
    Verhindert Sektor-Konzentration (z.B. 3× Halbleiter am selben Tag).
    Entfernt das schwächere Signal bei Korrelation > 0.75.

Reihenfolge v8.2:
  1.  Hard-Filter
  1b. FinBERT + Sentiment-Drift
  2.  Prescreening (Haiku)
  2b. Alpha Sources + Data Validation
  3.  ROI Pre-Check (Fail Fast)
  3b. Pre-MC Gate (1k Pfade, kein Signal-Alpha — sigma-only)  ← NEU
  4.  Deep Analysis (Sonnet + Red Team) — jetzt auf weniger Kandidaten
  5.  Mismatch-Score
  6.  Quick MC (3k, 30d, mit Mismatch-Alpha)
  7.  Intraday-Delta
  8.  Final MC (10k, adaptive DTE)
  9.  RL-Scoring
 10.  Options Design + ROI-Gate
 10b. Korrelations-Check                                       ← NEU
"""

import json
import math
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yfinance as yf

from modules.data_ingestion      import DataIngestion
from modules.prescreener         import Prescreener
from modules.deep_analysis       import DeepAnalysis
from modules.mismatch_scorer     import MismatchScorer
from modules.mirofish_simulation import MirofishSimulation, compute_time_value_efficiency
from modules.trade_scorer        import rank_proposals
from modules.rl_agent            import RLScorer
from modules.options_designer    import OptionsDesigner
from modules.reporter            import Reporter
from modules.risk_gates          import RiskGates
from modules.email_reporter      import send_status_email
from modules.finbert_sentiment   import score_candidate
from modules.intraday_delta      import filter_by_intraday_delta, get_intraday_move
from modules.alpha_sources       import enrich_with_alpha_sources
from modules.data_validator      import validate_candidate_data, compute_option_roi
from modules.premium_signals     import enrich_top_candidates
from modules.sentiment_tracker   import enrich_with_sentiment_drift
from modules.macro_context       import get_macro_context
from modules.config              import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

HISTORY_PATH = Path("outputs/history.json")
REPORTS_DIR  = Path("outputs/daily_reports")

QUICK_MC_PATHS    = 3_000
QUICK_MC_DAYS     = 30
FINAL_MC_PATHS    = 10_000

# v8.2: Pre-MC Gate Einstellungen
PRE_MC_PATHS      = 1_000
PRE_MC_DAYS       = 30
PRE_MC_THRESHOLD  = 0.25   # Sehr niedrig — filtert nur hoffnungslose Kandidaten


def get_mc_threshold(vix) -> float:
    """VIX-abhängige MC-Schwelle. Robust gegen None/negative Werte."""
    try:
        vix = float(vix)
    except (TypeError, ValueError):
        return 0.45  # Neutral-Default
    if vix < 0:
        return 0.45
    if vix < 20:
        return 0.48  # Ruhiger Markt → mehr Sicherheit nötig
    elif vix > 30:
        return 0.42  # Hohe Vola → Pfadweite sorgt selbst für Signal
    else:
        return 0.45


# ── Reject Tracking ──────────────────────────────────────────────────────────

reject_stats: dict = {}

def reject(reason: str, ticker: str | None = None) -> None:
    if reason not in reject_stats:
        reject_stats[reason] = {"count": 0, "tickers": []}
    reject_stats[reason]["count"] += 1
    if ticker:
        reject_stats[reason]["tickers"].append(ticker)
        log.info(f"  [{ticker}] REJECT → {reason}")
    else:
        log.info(f"  REJECT → {reason}")


# ── Validation Layer ──────────────────────────────────────────────────────────

def validate_strict(c: dict):
    """Ingress Validator: nach Enrichment, vor jeglicher Logik."""
    if not isinstance(c, dict):
        return None
    ticker = c.get("ticker")
    if not isinstance(ticker, str) or not ticker.strip():
        return None
    features = c.get("features")
    if not isinstance(features, dict):
        c["features"] = {}
        features = c["features"]
    for key in ["sentiment_score", "mismatch"]:
        val = features.get(key)
        if val is None:
            features[key] = 0.0
        elif not isinstance(val, (int, float)):
            return None
    return c


def validate_for_simulation(c: dict):
    """Pre-MC Validator: mismatch muss valide und im Bereich sein."""
    if not isinstance(c, dict):
        return None
    features = c.get("features", {})
    mismatch = features.get("mismatch")
    if not isinstance(mismatch, (int, float)):
        return None
    if abs(mismatch) > 10:
        return None
    return c


def validate_mc_result(result: dict):
    """MC Output Validator: gibt hit_rate als float zurück oder None."""
    if not result or "simulation" not in result:
        return None
    hit_rate = result["simulation"].get("hit_rate")
    if not isinstance(hit_rate, (int, float)):
        return None
    if not (0.0 <= float(hit_rate) <= 1.0):
        return None
    return float(hit_rate)


# ── v8.2: Korrelations-Check ────────────────────────────────────────────────

def filter_correlated_proposals(
    proposals: list[dict],
    max_corr: float = 0.75,
) -> list[dict]:
    """
    Entfernt das schwächere Signal wenn zwei Proposals > max_corr korreliert sind.

    Problem: Am 15. April wurden AVGO, AMD und AMAT gleichzeitig empfohlen —
    alles Halbleiter. Das ist keine Diversifikation, sondern ein
    konzentriertes Sektor-Bet.

    Methode: 30-Tage Return-Korrelation via yfinance (kostenlos).
    Bei Korrelation > 0.75 wird der Trade mit dem niedrigeren trade_score entfernt.
    """
    if len(proposals) <= 1:
        return proposals

    tickers = [p["ticker"] for p in proposals]

    try:
        data = yf.download(tickers, period="35d", progress=False, auto_adjust=True)
        if data.empty:
            return proposals

        close = data["Close"]
        if hasattr(close, "columns") and len(close.columns) < 2:
            return proposals

        returns     = close.pct_change().dropna()
        corr_matrix = returns.corr()

        to_remove = set()
        for i in range(len(tickers)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(tickers)):
                if j in to_remove:
                    continue
                try:
                    corr = float(corr_matrix.loc[tickers[i], tickers[j]])
                except (KeyError, TypeError):
                    continue

                if corr > max_corr:
                    score_i = proposals[i].get("trade_score", {}).get("total", 0)
                    score_j = proposals[j].get("trade_score", {}).get("total", 0)
                    weaker  = j if score_i >= score_j else i
                    to_remove.add(weaker)
                    log.info(
                        f"  KORRELATION: {tickers[i]}↔{tickers[j]} = {corr:.2f} "
                        f"→ {tickers[weaker]} entfernt (Score {proposals[weaker].get('trade_score',{}).get('total',0)})"
                    )

        if to_remove:
            log.info(f"  Korrelations-Check: {len(to_remove)} redundante(r) Trade(s) entfernt")

        return [p for idx, p in enumerate(proposals) if idx not in to_remove]

    except Exception as e:
        log.debug(f"Korrelations-Check Fehler: {e} → alle behalten")
        return proposals


def load_history() -> dict:
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH) as f:
                data = json.load(f)
            # Minimale Schema-Validierung
            if not isinstance(data, dict):
                raise ValueError("history.json ist kein dict")
            return data
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"history.json beschädigt ({e}) → Reset auf Default")
    return {
        "feature_stats": {}, "active_trades": [], "closed_trades": [],
        "model_weights": {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20},
        "sentiment_history": {},
    }


def save_history(history: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2, default=str)


def main() -> None:
    log.info("=== Adaptive Asymmetry-Scanner v8.2 gestartet ===")
    today   = datetime.utcnow().strftime("%Y-%m-%d")
    history = load_history()

    # Reset reject_stats für diesen Run
    reject_stats.clear()

    stats = {
        "vix": None, "candidates": 0, "prescreened": 0,
        "pre_mc": 0, "roi_precheck": 0, "analyzed": 0, "mismatch_ok": 0,
        "quick_mc": 0, "intraday_ok": 0, "final_mc": 0,
        "rl_scored": 0, "roi_ok": 0, "trades": 0, "stop_reason": "",
    }

    # trade_proposals wird am Ende gesetzt — Referenz via Liste
    _proposals_ref = []

    def send_email():
        try:
            proposals = _proposals_ref[0] if len(_proposals_ref) > 0 else []
            if proposals:
                from modules.email_reporter import send_email as _send_trade_email
                _send_trade_email(proposals, today)
            else:
                send_status_email(stats, today)
        except Exception as e:
            log.error(f"Email-Fehler: {e}")

    # ── STUFE 0: Risk Gates ──────────────────────────────────────────────────
    gates = RiskGates()
    if not gates.global_ok():
        stats["stop_reason"] = f"VIX-Gate (VIX={gates.last_vix:.1f})" if gates.last_vix is not None else "VIX-Gate (VIX nicht abrufbar)"
        stats["vix"] = gates.last_vix
        send_email(); return
    stats["vix"] = gates.last_vix

    macro = get_macro_context()
    log.info(f"Makro: {macro.get('macro_regime')} | YC={macro.get('yield_curve_desc','n/a')}")

    # ── STUFE 1: Hard-Filter ─────────────────────────────────────────────────
    log.info("Stufe 1: Hard-Filter (Cap>2B, Vol>1M, RV>0.6)")
    candidates = DataIngestion(history=history).run()
    stats["candidates"] = len(candidates)
    if not candidates:
        stats["stop_reason"] = "Keine Kandidaten nach Hard-Filter."
        send_email(); return

    # ── STUFE 1b: FinBERT + Sentiment-Drift ─────────────────────────────────
    log.info("Stufe 1b: FinBERT + Sentiment-Drift")
    enriched = []
    for c in candidates:
        try:
            sentiment = score_candidate(c)
            c.setdefault("features", {}).update(sentiment)
        except Exception:
            c.setdefault("features", {}).update({"sentiment_score": 0.0})
        c = enrich_with_sentiment_drift(c, history)
        enriched.append(c)
    candidates = enriched

    # ── STUFE 2: Prescreening (Haiku) ────────────────────────────────────────
    log.info("Stufe 2: Prescreening (Claude Haiku)")
    shortlist = Prescreener().run(candidates)
    stats["prescreened"] = len(shortlist)
    log.info(f"  → {len(shortlist)} nach Prescreening")
    if not shortlist:
        stats["stop_reason"] = f"Alle {len(candidates)} im Prescreening als kein Signal bewertet."
        send_email(); return

    enriched_with_alpha = []
    for c in shortlist:
        c = enrich_with_alpha_sources(c)
        # Earnings-Gate: Wenn Earnings in < 7 Tagen → aus Pipeline entfernen
        if c.get("has_near_earnings"):
            earnings_date = c.get("alpha_signals", {}).get("earnings_date", "?")
            log.info(f"  [{c['ticker']}] EARNINGS-GATE: Earnings in 7d ({earnings_date}) → Hard-Block.")
            continue
        enriched_with_alpha.append(c)
    shortlist = enriched_with_alpha
    shortlist = [validate_candidate_data(c) for c in shortlist]

    # ── Ingress Validation Gate ───────────────────────────────────────────────
    shortlist_raw = shortlist[:]
    shortlist = []
    for c in shortlist_raw:
        valid = validate_strict(c)
        if valid is None:
            reject("ingress_invalid_data", c.get("ticker") if isinstance(c, dict) else None)
        else:
            shortlist.append(valid)

    # ── STUFE 3: ROI Pre-Check (Fail Fast) ───────────────────────────────────
    # Schema-Check: ticker muss valide String sein
    valid_shortlist = []
    for c in shortlist:
        if not isinstance(c.get("ticker"), str) or not c.get("ticker", "").strip():
            reject("no_ticker")
            continue
        valid_shortlist.append(c)
    shortlist = valid_shortlist

    log.info("Stufe 3: ROI Pre-Check (Fail Fast)")
    roi_viable = []
    designer_pre = OptionsDesigner(gates=gates)
    for c in shortlist:
        ticker = c["ticker"]
        try:
            _, current, _ = MirofishSimulation()._get_market_params(ticker)
            if current <= 0:
                roi_viable.append(c); continue
            iv_rank  = designer_pre._get_iv_rank(ticker)
            strategy = designer_pre._select_strategy(ticker, "BULLISH", iv_rank)
            option   = designer_pre._find_option_for_dte(ticker, strategy, current, 21, 45)
            if option:
                sim_fake = {"current_price": current, "target_price": current * 1.08, "iv_rank": iv_rank}
                roi      = designer_pre._compute_roi(
                    option, sim_fake, iv_rank,
                    {"label": "Short-Term", "dte_min": 21, "dte_max": 45, "min_roi": 0.15}
                )
                if roi["roi_net"] < -0.30:
                    log.info(f"  [{ticker}] ROI-PRECHECK: {roi['roi_net']:.1%} → hoffnungslos")
                    continue
            roi_viable.append(c)
            log.info(f"  [{ticker}] ROI-PRECHECK: viable")
        except Exception as e:
            reject("roi_error", c.get("ticker"))
            continue

    stats["roi_precheck"] = len(roi_viable)
    log.info(f"  → {len(roi_viable)} nach ROI Pre-Check")
    if not roi_viable:
        stats["stop_reason"] = "Alle Optionsketten hoffnungslos (ROI Pre-Check)."
        send_email(); return

    # ── STUFE 3b: Pre-MC Gate (sigma-only, vor Deep Analysis) ────────────────
    # v8.2 NEU: Leichtgewichtiger MC-Check OHNE Signal-Alpha.
    # Prüft: "Hat diese Aktie genug Volatilität, damit sich Options lohnen?"
    # Spart Sonnet-Calls für Low-Vol-Aktien die selbst mit Signal nie das
    # Target erreichen würden.
    log.info(f"Stufe 3b: Pre-MC Gate ({PRE_MC_PATHS} Pfade, {PRE_MC_DAYS}d, sigma-only)")
    pre_mc_sim = MirofishSimulation()
    pre_mc_viable = []
    for c in roi_viable:
        ticker = c["ticker"]
        # Minimales deep_analysis-Dict: default Impact/Surprise
        # → MirofishSimulation nutzt impact=5, surprise=5 als Default
        c_temp = {**c, "deep_analysis": {"impact": 5, "surprise": 5,
                  "time_to_materialization": "2-3 Monate"}}
        result = pre_mc_sim.run_for_dte(c_temp, days_to_expiry=PRE_MC_DAYS)
        if result and result.get("simulation", {}).get("hit_rate", 0) >= PRE_MC_THRESHOLD:
            pre_mc_viable.append(c)
            log.info(
                f"  [{ticker}] Pre-MC: {result['simulation']['hit_rate']:.1%} "
                f"≥ {PRE_MC_THRESHOLD:.0%} ✅"
            )
        else:
            hr = result["simulation"]["hit_rate"] if result and "simulation" in result else 0
            reject("pre_mc_sigma_too_low", ticker)
            log.info(
                f"  [{ticker}] Pre-MC: {hr:.1%} < {PRE_MC_THRESHOLD:.0%} "
                f"→ zu wenig Vola für Options"
            )

    stats["pre_mc"] = len(pre_mc_viable)
    log.info(f"  → {len(pre_mc_viable)} nach Pre-MC Gate (eingespart: {len(roi_viable)-len(pre_mc_viable)} Sonnet-Calls)")
    if not pre_mc_viable:
        stats["stop_reason"] = "Alle unter Pre-MC-Schwelle (zu wenig Volatilität)."
        save_history(history); send_email(); return

    # ── STUFE 4: Deep Analysis (Sonnet) ──────────────────────────────────────
    log.info("Stufe 4: Deep Analysis (Claude Sonnet + Red Team)")
    analyses = DeepAnalysis().run(pre_mc_viable)
    stats["analyzed"] = len(analyses)
    log.info(f"  → {len(analyses)} nach Deep Analysis")
    if not analyses:
        stats["stop_reason"] = "Alle Signale im Red-Team-Check verworfen."
        send_email(); return

    # ── STUFE 5: Mismatch-Score ───────────────────────────────────────────────
    log.info("Stufe 5: Mismatch-Score")
    scored = MismatchScorer().run(analyses)
    # Post-DeepAnalysis Validation: mismatch muss vorhanden sein
    before_da = len(scored)
    scored = [validate_for_simulation(s) for s in scored]
    scored = [s for s in scored if s is not None]
    for _ in range(before_da - len(scored)):
        reject("post_deep_analysis_invalid")
    stats["mismatch_ok"] = len(scored)
    log.info(f"  → {len(scored)} nach Mismatch-Score")
    if not scored:
        stats["stop_reason"] = "Kein Signal hat Mismatch-Filter bestanden."
        save_history(history); send_email(); return

    # ── STUFE 6: Quick MC (NACH Mismatch — jetzt mit echtem alpha) ───────────
    # ── Pre-Simulation Gate ──────────────────────────────────────────────────
    before_sim = len(scored)
    scored = [validate_for_simulation(s) for s in scored]
    scored = [s for s in scored if s is not None]
    for _ in range(before_sim - len(scored)):
        reject("pre_mc_invalid_mismatch")

    log.info(f"Stufe 6: Quick MC (n={QUICK_MC_PATHS}, {QUICK_MC_DAYS}d) — mit Mismatch-Alpha")
    sim        = MirofishSimulation()
    mc_viable  = []
    for s in scored:
        ticker   = s["ticker"]
        mismatch = s.get("features", {}).get("mismatch", 0)
        log.info(f"  [{ticker}] Mismatch={mismatch:.2f} → Quick MC startet")

        mc_threshold = get_mc_threshold(gates.last_vix or 20.0)
        result   = sim.run_for_dte(s, days_to_expiry=QUICK_MC_DAYS)
        hit_rate = validate_mc_result(result)
        if hit_rate is None:
            if not result or "simulation" not in result:
                reject("mc_no_result", ticker)
            else:
                reject("mc_invalid_hit_rate", ticker)
            continue

        if hit_rate < mc_threshold:
            log.info(f"  [{ticker}] Quick MC: {hit_rate:.1%} < {mc_threshold:.0%} → verworfen")
            reject("mc_below_threshold", ticker)
            continue

        s["quick_mc"] = {"hit_rate": hit_rate, "n_paths": QUICK_MC_PATHS, "n_days": QUICK_MC_DAYS}
        s["features"]["quick_mc_hit_rate"] = hit_rate
        mc_viable.append(s)
        log.info(f"  [{ticker}] Quick MC: {hit_rate:.1%} ✅ PASS")

    stats["quick_mc"] = len(mc_viable)
    log.info(f"  → {len(mc_viable)} nach Quick MC")
    if not mc_viable:
        stats["stop_reason"] = f"Alle unter Quick-MC-Schwelle ({get_mc_threshold(gates.last_vix or 20.0):.0%})."
        save_history(history); send_email(); return

    # ── STUFE 7: Intraday-Delta ───────────────────────────────────────────────
    log.info("Stufe 7: Intraday-Delta-Filter")
    # Dynamischer Intraday-Filter: Limit steigt mit Mismatch-Score
    base_move = getattr(getattr(cfg, "pipeline", None), "max_intraday_move", 0.07)

    before_intraday = {s["ticker"]: s for s in mc_viable}
    mc_viable_dynamic = []
    for s in mc_viable:
        ticker   = s["ticker"]
        mismatch = s.get("features", {}).get("mismatch", 0)

        if mismatch >= 7:
            current_max = max(base_move, 0.12)
        elif mismatch >= 5:
            current_max = max(base_move, 0.09)
        else:
            current_max = base_move

        # Berechne intraday_delta falls noch nicht vorhanden
        if "intraday_delta" not in s:
            s["intraday_delta"] = get_intraday_move(ticker)
        delta_info = s.get("intraday_delta", {})
        move = abs(float(delta_info.get("move_pct", 0) or 0))

        if move <= current_max:
            mc_viable_dynamic.append(s)
            log.info(
                f"  [{ticker}] Intraday-Move={move:+.2%} → "
                f"noch Asymmetrie vorhanden (Limit={current_max:.0%}, Mismatch={mismatch:.1f})."
            )
        else:
            reject("intraday_too_late", ticker)
            log.info(
                f"  [{ticker}] REJECT: Move={move:.1%} > Limit={current_max:.0%} "
                f"(Mismatch={mismatch:.1f})"
            )

    mc_viable = mc_viable_dynamic
    stats["intraday_ok"] = len(mc_viable)
    log.info(
        f"Intraday-Delta-Filter: {len(before_intraday)} → {len(mc_viable)} Signale "
        f"({len(before_intraday) - len(mc_viable)} zu spät)"
    )
    log.info(f"  → {len(mc_viable)} nach Intraday-Filter")
    if not mc_viable:
        stats["stop_reason"] = "Alle durch Intraday-Delta-Filter verworfen (>7% Bewegung)."
        save_history(history); send_email(); return

    # ── STUFE 8: Final MC (10k, adaptive DTE) ────────────────────────────────
    log.info(f"Stufe 8: Final MC (n={FINAL_MC_PATHS}, adaptive DTE)")
    sim_final  = MirofishSimulation()
    final_sims = []
    for s in mc_viable:
        ticker = s["ticker"]
        quick_mc_days = s.get("quick_mc", {}).get("n_days", 30)
        final_dte = 45 if quick_mc_days <= 30 else 120
        result   = sim_final.run_for_dte(s, days_to_expiry=final_dte)
        hit_rate = validate_mc_result(result)
        if hit_rate is None:
            reject("final_mc_invalid", ticker)
            continue
        if hit_rate < 0.01:
            reject("final_mc_zero_prob", ticker)
            continue
        if hit_rate:
            result["simulation"]["n_paths"] = FINAL_MC_PATHS
            final_sims.append(result)
            log.info(f"  [{ticker}] Final MC: {result['simulation']['hit_rate']:.1%} ✅")
        else:
            log.info(f"  [{ticker}] Final MC: FAIL")

    stats["final_mc"] = len(final_sims)
    log.info(f"  → {len(final_sims)} nach Final MC")
    if not final_sims:
        stats["stop_reason"] = "Kein Kandidat besteht Final MC (120d)."
        save_history(history); send_email(); return

    # ── STUFE 9: RL-Scoring ──────────────────────────────────────────────────
    log.info("Stufe 9: RL-Scoring")
    final_signals = RLScorer(history=history).run(final_sims)
    stats["rl_scored"] = len(final_signals)
    log.info(f"  → {len(final_signals)} nach RL-Scoring")
    if not final_signals:
        stats["stop_reason"] = "RL-Agent: alle als SKIP klassifiziert."
        save_history(history); send_email(); return

    final_signals = enrich_top_candidates(final_signals, top_n=2)

    # ── STUFE 10: Options Design + ROI-Gate ──────────────────────────────────
    log.info("Stufe 10: Options Design + adaptiver Laufzeit-Loop")
    designer        = OptionsDesigner(gates=gates)
    try:
        trade_proposals = designer.run(final_signals)
    except Exception as e:
        log.error(f"Options Design Fehler: {e} → Email wird trotzdem gesendet")
        stats["stop_reason"] = f"Options Design Fehler: {type(e).__name__}: {e}"
        save_history(history)
        send_email()
        raise  # Re-raise damit GitHub Actions den Fehler sieht

    for p in trade_proposals:
        roi = p.get("roi_analysis", {})
        dte = p.get("option", {}).get("dte", 90)
        p["time_value_efficiency"] = compute_time_value_efficiency(
            roi.get("roi_net", 0), dte
        )

    # Trade-Score berechnen und ranken
    if trade_proposals:
        trade_proposals = rank_proposals(trade_proposals)
        # AVOID-Trades herausfiltern (Score < 45)
        before = len(trade_proposals)
        trade_proposals = [p for p in trade_proposals
                           if p.get("trade_score", {}).get("total", 0) >= 45]
        if len(trade_proposals) < before:
            log.info(f"  {before - len(trade_proposals)} AVOID-Trade(s) herausgefiltert (Score < 45)")

        # ── STUFE 10b: Korrelations-Check ────────────────────────────────────
        # v8.2 NEU: Verhindert Sektor-Konzentration
        if len(trade_proposals) > 1:
            log.info("Stufe 10b: Korrelations-Check")
            before_corr      = len(trade_proposals)
            trade_proposals   = filter_correlated_proposals(trade_proposals)
            if len(trade_proposals) < before_corr:
                log.info(
                    f"  Korrelations-Check: {before_corr} → {len(trade_proposals)} "
                    f"({before_corr - len(trade_proposals)} korrelierte entfernt)"
                )

        log.info(f"Trade-Ranking:")
        for p in trade_proposals:
            ts = p.get("trade_score", {})
            log.info(
                f"  #{p.get('trade_rank','-')} [{p['ticker']}] "
                f"{ts.get('total',0)}/100 — {ts.get('grade','?')}"
            )

    stats["roi_ok"] = len(trade_proposals)
    stats["trades"] = len(trade_proposals)
    if not trade_proposals:
        stats["stop_reason"] = "Alle Options-Kontrakte scheitern am ROI-Gate."

    # Proposals für Email-Funktion verfügbar machen
    if trade_proposals:
        _proposals_ref.append(trade_proposals)

    # History speichern
    existing = {(t["ticker"], t.get("entry_date","")) for t in history["active_trades"]}
    for p in trade_proposals:
        key = (p["ticker"], today)
        if key not in existing:
            history["active_trades"].append({
                "ticker":        p["ticker"], "entry_date": today,
                "features":      p.get("features", {}),
                "strategy":      p.get("strategy", ""),
                "option":        p.get("option"),
                "simulation":    p.get("simulation"),
                "deep_analysis": p.get("deep_analysis"),
                "tve":           p.get("time_value_efficiency"),
                "outcome":       None,
            })

    Reporter(reports_dir=REPORTS_DIR).save(
        today=today, proposals=trade_proposals, history=history
    )
    save_history(history)
    send_email()
    # Reject Summary
    if reject_stats:
        log.info("=== Reject Summary ===")
        for reason, data in sorted(reject_stats.items(), key=lambda x: -x[1]["count"]):
            tickers = ", ".join(data["tickers"][:5]) if data["tickers"] else "-"
            log.info(f"  {reason}: {data['count']}x → [{tickers}]")
    stats["rejects"] = {k: v["count"] for k, v in reject_stats.items()}

    log.info(f"=== Pipeline v8.2 beendet. {len(trade_proposals)} Trade-Vorschläge. ===")


if __name__ == "__main__":
    main()
