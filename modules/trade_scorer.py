"""
modules/trade_scorer.py

Nachvollziehbarer Trade-Score für die Email.
Jede Komponente ist erklärbar und hat eine klare Logik.

Score-System (0-100 Punkte total):

A. SIGNAL-QUALITÄT (40 Punkte)
   Impact × Surprise / 100 × 40
   Beispiel: Impact=7, Surprise=6 → 7×6/100×40 = 16.8/40

B. OPTIONEN-QUALITÄT (30 Punkte)
   - ROI netto (realistisch): max 15 Punkte
   - Bid-Ask Spread (Liquidität): max 10 Punkte  
   - Open Interest: max 5 Punkte

C. RISIKO-ABZÜGE (0 bis -30 Punkte)
   - Bear Case Severity: -0 bis -15
   - IV-Rank zu hoch (teuer): -0 bis -10
   - Katalysator bereits eingepreist (48h Move): -0 bis -5

D. KONTEXT-BONUS (0-30 Punkte)
   - Makro-Regime expansiv: +10
   - Sektor-Momentum positiv: +10
   - Red Team: PASSIERT ohne Veto: +5
   - Kein Directions-Konflikt: +5

Finale Empfehlung:
   80-100: STRONG BUY  🟢
   65-79:  BUY         🟡
   50-64:  WATCH       🟠
   <50:    AVOID       🔴
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
            "total": int (0-100),
            "grade": "STRONG BUY" | "BUY" | "WATCH" | "AVOID",
            "components": {
                "signal_quality":    int (0-40),
                "options_quality":   int (0-30),
                "risk_deductions":   int (-30 bis 0),
                "context_bonus":     int (0-30),
            },
            "reasoning": str,  # Klare Erklärung
            "best_argument_for":     str,
            "best_argument_against": str,
        }
    """
    da       = proposal.get("deep_analysis", {}) or {}
    option   = proposal.get("option", {}) or {}
    roi_data = proposal.get("roi_analysis", {}) or {}
    sim      = proposal.get("simulation", {}) or {}
    tve      = proposal.get("time_value_efficiency", {}) or {}
    features = proposal.get("features", {}) or {}
    red_team = da.get("red_team", {}) or {}
    ticker   = proposal.get("ticker", "?")

    # ── A: SIGNAL-QUALITÄT (0-40) ─────────────────────────────────────────────
    impact   = float(da.get("impact", 0) or 0)
    surprise = float(da.get("surprise", 0) or 0)
    # Geometrisches Mittel bevorzugt: hoch in BEIDEN Dimensionen
    if impact > 0 and surprise > 0:
        signal_raw = math.sqrt(impact * surprise) / 10.0  # 0-1
    else:
        signal_raw = 0.0
    signal_pts = round(signal_raw * 40, 1)

    # ── B: OPTIONEN-QUALITÄT (0-30) ───────────────────────────────────────────
    roi_net    = float(roi_data.get("roi_net", 0) or 0)
    spread_pct = float(roi_data.get("spread_pct", 1) or 1)
    oi         = int(option.get("open_interest", 0) or 0)
    dte        = int(option.get("dte", 30) or 30)

    # ROI: realistisch 10-30% = gut, >50% = suspekt (cap bei 30%)
    roi_capped    = min(abs(roi_net), 0.30)
    roi_pts       = round((roi_capped / 0.30) * 15, 1)  # 0-15

    # Liquidität: Spread < 5% = sehr gut, > 20% = schlecht
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

    options_pts = round(roi_pts + liq_pts + oi_pts, 1)
    options_pts = min(options_pts, 30.0)

    # ── C: RISIKO-ABZÜGE (0 bis -30) ─────────────────────────────────────────
    bear_sev  = float(da.get("bear_case_severity", 0) or 0)
    iv_rank   = float(proposal.get("iv_rank", 50) or 50)
    move_48h  = abs(float(features.get("price_change_48h", 0) or 0))

    # Bear Case: 0-10 → 0 Abzug, 7 → -10, 9 → -15
    if bear_sev <= 5:
        bear_deduct = 0.0
    elif bear_sev <= 7:
        bear_deduct = -(bear_sev - 5) * 5.0   # -0 bis -10
    else:
        bear_deduct = -10.0 - (bear_sev - 7) * 2.5  # -10 bis -15

    # IV-Rank: >85% = sehr teuer = Abzug
    if iv_rank >= 95:
        iv_deduct = -10.0
    elif iv_rank >= 85:
        iv_deduct = -5.0
    elif iv_rank >= 70:
        iv_deduct = -2.0
    else:
        iv_deduct = 0.0

    # 48h Move > 5% = Katalysator eingepreist
    if move_48h >= 0.10:
        move_deduct = -5.0
    elif move_48h >= 0.05:
        move_deduct = -3.0
    else:
        move_deduct = 0.0

    risk_pts = round(bear_deduct + iv_deduct + move_deduct, 1)

    # ── D: KONTEXT-BONUS (0-30) ───────────────────────────────────────────────
    macro_regime   = da.get("macro_regime", "neutral")
    direction      = da.get("direction", "BULLISH")
    red_verdict    = red_team.get("red_team_verdict", "PASSIERT")
    dir_conflict   = da.get("direction_conflict", False)
    sector_info    = proposal.get("sector_momentum", {}) or {}
    sector_rs      = float(sector_info.get("rel_strength", 0) or 0)

    macro_bonus  = 10.0 if macro_regime == "expansive" else (5.0 if macro_regime == "neutral" else 0.0)
    sector_bonus = 10.0 if sector_rs >= 0.03 else (5.0 if sector_rs >= 0 else 0.0)
    red_bonus    = 5.0 if red_verdict == "PASSIERT" else 0.0
    conflict_pen = 0.0 if not dir_conflict else -5.0

    context_pts = round(macro_bonus + sector_bonus + red_bonus + conflict_pen, 1)
    context_pts = min(context_pts, 30.0)

    # ── TOTAL ─────────────────────────────────────────────────────────────────
    total = round(signal_pts + options_pts + risk_pts + context_pts, 0)
    total = max(0, min(100, int(total)))

    # ── GRADE ─────────────────────────────────────────────────────────────────
    if total >= 75:
        grade = "STRONG BUY 🟢"
        grade_short = "STRONG BUY"
    elif total >= 60:
        grade = "BUY 🟡"
        grade_short = "BUY"
    elif total >= 45:
        grade = "WATCH 🟠"
        grade_short = "WATCH"
    else:
        grade = "AVOID 🔴"
        grade_short = "AVOID"

    # ── REASONING ─────────────────────────────────────────────────────────────
    strengths = []
    weaknesses = []

    if signal_pts >= 25: strengths.append(f"starkes Signal (Impact={impact:.0f}/Surprise={surprise:.0f})")
    elif signal_pts >= 15: strengths.append(f"solides Signal (Impact={impact:.0f}/Surprise={surprise:.0f})")
    else: weaknesses.append(f"schwaches Signal (Impact={impact:.0f}/Surprise={surprise:.0f})")

    if roi_net >= 0.15: strengths.append(f"guter ROI ({roi_net:.0%})")
    elif roi_net < 0: weaknesses.append(f"negativer ROI ({roi_net:.0%})")

    if iv_rank >= 85: weaknesses.append(f"sehr hohe IV ({iv_rank:.0f}%) — Optionen teuer")
    elif iv_rank <= 40: strengths.append(f"günstige IV ({iv_rank:.0f}%)")

    if move_48h >= 0.05: weaknesses.append(f"48h-Move +{move_48h:.0%} — Teil-Alpha eingepreist")

    if bear_sev >= 7: weaknesses.append(f"hohes Bear-Risiko ({bear_sev:.0f}/10)")

    if sector_rs >= 0.03: strengths.append("positive Sektor-Relative-Stärke")
    elif sector_rs < -0.03: weaknesses.append("Sektor underperformt")

    if macro_regime == "expansive": strengths.append("expansives Makro-Umfeld")

    best_for     = da.get("asymmetry_reasoning", "Strukturelles Signal erkannt")[:100]
    best_against = red_team.get("argument_1", "n/a")[:100]

    reasoning_parts = []
    if strengths:
        reasoning_parts.append("Stärken: " + ", ".join(strengths[:3]))
    if weaknesses:
        reasoning_parts.append("Risiken: " + ", ".join(weaknesses[:3]))

    log.info(
        f"  [{ticker}] Trade-Score: {total}/100 ({grade_short}) | "
        f"Signal={signal_pts} Options={options_pts} "
        f"Risk={risk_pts} Context={context_pts}"
    )

    return {
        "total":             total,
        "grade":             grade,
        "grade_short":       grade_short,
        "components": {
            "signal_quality":  signal_pts,
            "options_quality": options_pts,
            "risk_deductions": risk_pts,
            "context_bonus":   context_pts,
        },
        "reasoning":           " | ".join(reasoning_parts),
        "best_argument_for":   best_for,
        "best_argument_against": best_against,
    }


def rank_proposals(proposals: list[dict]) -> list[dict]:
    """Rankt Trade-Vorschläge nach Score, fügt Rang und Score hinzu."""
    scored = []
    for p in proposals:
        score_data = compute_trade_score(p)
        p["trade_score"] = score_data
        scored.append(p)

    # Sortieren nach Score (höchster zuerst)
    scored.sort(key=lambda x: x["trade_score"]["total"], reverse=True)

    for i, p in enumerate(scored):
        p["trade_rank"] = i + 1

    return scored
