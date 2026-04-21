"""
modules/trade_scorer.py  —  Adaptive Asymmetry-Scanner v8.1

Nachvollziehbarer Trade-Score für die Email.
Jede Komponente ist erklärbar und hat eine klare Logik.

Score-System (0-100 Punkte total):

A. SIGNAL-QUALITÄT (0-40 Punkte)
   √(Impact × Surprise) / 10 × 40
   Geometrisches Mittel → hoch nur wenn BEIDE Dimensionen stark
   Beispiel: Impact=7, Surprise=6 → √42/10 × 40 = 25.9/40

B. OPTIONEN-QUALITÄT (0-30 Punkte)
   - ROI netto:         max 15 Pkt  (nur positiv; Verluste = 0)
   - Bid-Ask Spread:    max 10 Pkt  (<5% = top)
   - Open Interest:     max  5 Pkt  (>1000 = top)

C. RISIKO-ABZÜGE (0 bis -30 Punkte)
   - Bear Case Severity ≥7:   bis -15 Pkt
   - IV-Rank ≥85%:            bis -10 Pkt
   - 48h-Move ≥5%:            bis  -5 Pkt

D. KONTEXT-BONUS (0-30 Punkte)
   - Makro expansiv:           +10  (unknown/FRED-Ausfall → neutral +5)
   - Sektor-Momentum positiv:  +10
   - Haiku+Sonnet Konsistenz:  +5   (Red Team ist Hard-Gate, immer PASSIERT)
   - Richtungskonflikt:         -5  (Strafe wenn dir_conflict=True)

Finale Empfehlung:
   >=75: STRONG BUY | 60-74: BUY | 45-59: WATCH | <45: AVOID
"""

from __future__ import annotations
import math
import logging

log = logging.getLogger(__name__)


def compute_trade_score(proposal: dict) -> dict:
    """
    Berechnet einen nachvollziehbaren Trade-Score.

    Returns:
        {
            "total":             int   (0-100),
            "grade":             str   z.B. "BUY",
            "grade_short":       str   z.B. "BUY",
            "components": {
                "signal_quality":  float (0-40),
                "options_quality": float (0-30),
                "risk_deductions": float (-30 bis 0),
                "context_bonus":   float (0-30),
            },
            "reasoning":           str,
            "best_argument_for":   str,
            "best_argument_against": str,
        }
    """
    da       = proposal.get("deep_analysis", {}) or {}
    option   = proposal.get("option", {}) or {}
    roi_data = proposal.get("roi_analysis", {}) or {}
    features = proposal.get("features", {}) or {}
    red_team = da.get("red_team", {}) or {}
    ticker   = proposal.get("ticker", "?")

    # ── A: SIGNAL-QUALITÄT (0-40) ─────────────────────────────────────────────
    impact   = float(da.get("impact", 0) or 0)
    surprise = float(da.get("surprise", 0) or 0)

    # Geometrisches Mittel: hoch nur wenn BEIDE Dimensionen stark sind
    if impact > 0 and surprise > 0:
        signal_raw = math.sqrt(impact * surprise) / 10.0   # 0-1
    else:
        signal_raw = 0.0
    signal_pts = round(signal_raw * 40, 1)

    # ── B: OPTIONEN-QUALITÄT (0-30) ───────────────────────────────────────────
    roi_net    = float(roi_data.get("roi_net", 0) or 0)
    spread_pct = float(roi_data.get("spread_pct", 1) or 1)
    oi         = int(option.get("open_interest", 0) or 0)

    # ROI: negativer ROI = 0 Punkte (kein Bonus für Verlustgeschäfte)
    roi_capped = min(max(0.0, roi_net), 0.30)           # Verlust=0, cap bei 30%
    roi_pts    = round((roi_capped / 0.30) * 15, 1)     # 0-15

    # Liquidität: Spread-Qualität
    if spread_pct <= 0.05:
        liq_pts = 10.0
    elif spread_pct <= 0.10:
        liq_pts = 7.0
    elif spread_pct <= 0.20:
        liq_pts = 4.0
    else:
        liq_pts = 1.0

    # Open Interest
    if oi >= 1000:
        oi_pts = 5.0
    elif oi >= 500:
        oi_pts = 3.0
    elif oi >= 100:
        oi_pts = 1.5
    else:
        oi_pts = 0.5

    options_pts = min(round(roi_pts + liq_pts + oi_pts, 1), 30.0)

    # ── C: RISIKO-ABZÜGE (0 bis -30) ─────────────────────────────────────────
    bear_sev = float(da.get("bear_case_severity", 0) or 0)
    iv_rank  = float(proposal.get("iv_rank", 50) or 50)
    _raw_move = float(features.get("price_change_48h", 0) or 0)
    # Sicherheits-Check: Wenn Wert > 1.0 → als Integer-Prozent interpretiert → /100
    move_48h = abs(_raw_move / 100.0 if abs(_raw_move) > 1.0 else _raw_move)

    # Bear Case: Severity 0-6 = kein Abzug, 7 = -10, 9-10 = -15
    if bear_sev <= 6:
        bear_deduct = 0.0
    elif bear_sev <= 7:
        bear_deduct = -(bear_sev - 5) * 5.0         # -5 bis -10
    else:
        bear_deduct = -10.0 - (bear_sev - 7) * 2.5  # -10 bis -15

    # IV-Rank: teuer einkaufen kostet Punkte
    # Ausnahme: Bei extremer Surprise (>=8) ist hohe IV durch echtes Asymmetrie-Event
    # gerechtfertigt — Abzug halbieren damit "Home Runs" nicht abgewertet werden
    surprise_high = surprise >= 8.0
    if iv_rank >= 95:
        iv_deduct = -5.0 if surprise_high else -10.0
    elif iv_rank >= 85:
        iv_deduct = -2.5 if surprise_high else -5.0
    elif iv_rank >= 70:
        iv_deduct = -1.0 if surprise_high else -2.0
    else:
        iv_deduct = 0.0

    # 48h-Move: Katalysator bereits eingepreist?
    if move_48h >= 0.10:
        move_deduct = -5.0
    elif move_48h >= 0.05:
        move_deduct = -3.0
    else:
        move_deduct = 0.0

    risk_pts = round(bear_deduct + iv_deduct + move_deduct, 1)

    # ── D: KONTEXT-BONUS (0-30) ───────────────────────────────────────────────
    macro_regime = da.get("macro_regime", "neutral")
    dir_conflict = bool(da.get("direction_conflict", False))
    sector_info  = proposal.get("sector_momentum", {}) or {}
    sector_rs    = float(sector_info.get("rel_strength", 0) or 0)

    # Makro: "unknown" bei FRED-Ausfall = neutral, kein Punktverlust
    if macro_regime == "expansive":
        macro_bonus = 10.0
    elif macro_regime in ("neutral", "unknown"):
        macro_bonus = 5.0
    else:
        macro_bonus = 0.0

    # Sektor-Momentum
    if sector_rs >= 0.03:
        sector_bonus = 10.0
    elif sector_rs >= 0.0:
        sector_bonus = 5.0
    else:
        sector_bonus = 0.0

    # KI-Konsistenz: Red Team ist Hard-Gate → jedes Signal hier hat PASSIERT.
    # Bonus an Haiku+Sonnet-Einigkeit knüpfen statt automatisch vergeben.
    ki_bonus     = 5.0 if not dir_conflict else 0.0
    conflict_pen = 0.0 if not dir_conflict else -5.0  # doppelt bestraft bei Konflikt

    context_pts = min(round(macro_bonus + sector_bonus + ki_bonus + conflict_pen, 1), 30.0)

    # ── TOTAL ─────────────────────────────────────────────────────────────────
    total = max(0, min(100, int(round(signal_pts + options_pts + risk_pts + context_pts, 0))))

    # ── GRADE ─────────────────────────────────────────────────────────────────
    if total >= 75:
        grade = "STRONG BUY"
        emoji = "🟢"
    elif total >= 60:
        grade = "BUY"
        emoji = "🟡"
    elif total >= 45:
        grade = "WATCH"
        emoji = "🟠"
    else:
        grade = "AVOID"
        emoji = "🔴"

    grade_short = grade
    grade_full  = f"{grade} {emoji}"

    # ── REASONING ─────────────────────────────────────────────────────────────
    strengths  = []
    weaknesses = []

    if signal_pts >= 25:
        strengths.append(f"starkes Signal (Impact={impact:.0f}/Surprise={surprise:.0f})")
    elif signal_pts >= 15:
        strengths.append(f"solides Signal (Impact={impact:.0f}/Surprise={surprise:.0f})")
    else:
        weaknesses.append(f"schwaches Signal (Impact={impact:.0f}/Surprise={surprise:.0f})")

    if roi_net >= 0.20:
        strengths.append(f"guter ROI ({roi_net:.0%})")
    elif roi_net < 0:
        weaknesses.append(f"negativer ROI ({roi_net:.0%})")

    if iv_rank >= 85:
        weaknesses.append(f"hohe IV ({iv_rank:.0f}%) — Optionen teuer")
    elif iv_rank <= 40:
        strengths.append(f"günstige IV ({iv_rank:.0f}%)")

    if move_48h >= 0.05:
        weaknesses.append(f"48h-Move +{move_48h:.0%} — Alpha teilw. eingepreist")

    if bear_sev >= 7:
        weaknesses.append(f"hohes Bear-Risiko ({bear_sev:.0f}/10)")

    if sector_rs >= 0.03:
        strengths.append("positive Sektor-Relative-Stärke")
    elif sector_rs < -0.03:
        weaknesses.append("Sektor underperformt Markt")

    if macro_regime == "expansive":
        strengths.append("expansives Makro-Umfeld")

    reasoning_parts = []
    if strengths:
        reasoning_parts.append("Stärken: " + ", ".join(strengths[:3]))
    if weaknesses:
        reasoning_parts.append("Risiken: " + ", ".join(weaknesses[:3]))

    # Für / Gegen aus Deep Analysis (volle Länge für Email)
    best_for     = (da.get("asymmetry_reasoning", "") or "Strukturelles Signal erkannt")[:400]
    best_against = (red_team.get("argument_1", "") or "n/a")[:400]

    log.info(
        f"  [{ticker}] Score: {total}/100 ({grade_short}) | "
        f"Signal={signal_pts} Optionen={options_pts} "
        f"Risiko={risk_pts} Kontext={context_pts}"
    )

    return {
        "total":       total,
        "grade":       grade_full,
        "grade_short": grade_short,
        "components": {
            "signal_quality":  signal_pts,
            "options_quality": options_pts,
            "risk_deductions": risk_pts,
            "context_bonus":   context_pts,
        },
        "reasoning":             " | ".join(reasoning_parts),
        "best_argument_for":     best_for,
        "best_argument_against": best_against,
    }


def rank_proposals(proposals: list[dict]) -> list[dict]:
    """Rankt Trade-Vorschläge nach Score (höchster zuerst), fügt rank + trade_score hinzu."""
    for p in proposals:
        p["trade_score"] = compute_trade_score(p)

    proposals.sort(key=lambda x: x["trade_score"]["total"], reverse=True)

    for i, p in enumerate(proposals):
        p["trade_rank"] = i + 1

    return proposals
