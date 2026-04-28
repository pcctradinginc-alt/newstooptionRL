"""
Reporter: Speichert tägliche Trade-Vorschläge als JSON + Markdown
- outputs/daily_reports/YYYY-MM-DD.json
- outputs/daily_reports/YYYY-MM-DD.md (lesbar)

v8.2: Exit-Regeln in jedem Trade-Vorschlag:
    - Take-Profit:  +50% Options-Wert → Position (oder Hälfte) schliessen
    - Stop-Loss:    -40% Options-Wert → Gesamte Position liquidieren
    - Time-Exit:    50% DTE verstrichen und < +20% → schliessen (Theta-Decay)
    - Profit-Dollar: Konkrete $-Beträge basierend auf Entry-Preis
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


def compute_exit_rules(proposal: dict) -> dict:
    """
    Berechnet konkrete Exit-Regeln für einen Trade-Vorschlag.

    Basiert auf:
      - Options-Ask (Entry-Preis)
      - DTE (Days to Expiry)
      - Strategie (Spread vs. Long)

    Returns:
        {
            "take_profit_pct": 50,
            "take_profit_price": 4.50,     # Konkreter Options-Preis
            "stop_loss_pct": -40,
            "stop_loss_price": 1.80,
            "time_exit_dte": 15,           # Konkretes Datum-nahes DTE
            "time_exit_date": "2026-05-01",
            "time_exit_min_profit_pct": 20,
            "entry_cost": 3.00,
        }
    """
    option   = proposal.get("option", {})
    strategy = proposal.get("strategy", "")

    # Entry-Kosten: net_debit für Spreads, ask für Long
    if "SPREAD" in strategy and option.get("net_debit"):
        entry = float(option.get("net_debit", 0))
    else:
        entry = float(option.get("ask", 0))

    if entry <= 0:
        return _empty_exit_rules()

    dte = int(option.get("dte", 0) or 0)

    # Take-Profit: +50% (Hälfte schliessen), +100% (Rest schliessen)
    take_profit_pct  = 50
    take_profit_price = round(entry * 1.50, 2)
    full_profit_price = round(entry * 2.00, 2)

    # Stop-Loss: -40% (gesamte Position schliessen)
    stop_loss_pct   = -40
    stop_loss_price = round(entry * 0.60, 2)

    # Time-Exit: Bei 50% abgelaufener DTE und < +20% → schliessen
    time_exit_dte = max(dte // 2, 1)
    try:
        expiry_str = option.get("expiry", "")
        if expiry_str:
            expiry_dt      = datetime.strptime(expiry_str, "%Y-%m-%d")
            time_exit_date = (expiry_dt - timedelta(days=time_exit_dte)).strftime("%Y-%m-%d")
        else:
            time_exit_date = "N/A"
    except Exception:
        time_exit_date = "N/A"

    return {
        "entry_cost":              entry,
        "take_profit_pct":         take_profit_pct,
        "take_profit_price":       take_profit_price,
        "full_profit_pct":         100,
        "full_profit_price":       full_profit_price,
        "stop_loss_pct":           stop_loss_pct,
        "stop_loss_price":         stop_loss_price,
        "time_exit_dte_remaining": time_exit_dte,
        "time_exit_date":          time_exit_date,
        "time_exit_min_profit_pct": 20,
    }


def _empty_exit_rules() -> dict:
    return {
        "entry_cost": 0, "take_profit_pct": 50, "take_profit_price": 0,
        "full_profit_pct": 100, "full_profit_price": 0,
        "stop_loss_pct": -40, "stop_loss_price": 0,
        "time_exit_dte_remaining": 0, "time_exit_date": "N/A",
        "time_exit_min_profit_pct": 20,
    }


class Reporter:
    def __init__(self, reports_dir: Path):
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def save(self, today: str, proposals: list[dict], history: dict) -> None:
        # v8.2: Exit-Regeln zu jedem Proposal hinzufügen
        for p in proposals:
            if "exit_rules" not in p:
                p["exit_rules"] = compute_exit_rules(p)

        self._save_json(today, proposals)
        self._save_markdown(today, proposals, history)

    def _save_json(self, today: str, proposals: list[dict]) -> None:
        path = self.reports_dir / f"{today}.json"
        with open(path, "w") as f:
            json.dump(
                {"date": today, "proposals": proposals},
                f, indent=2, default=str
            )
        log.info(f"Report gespeichert: {path}")

    def _save_markdown(self, today: str, proposals: list[dict], history: dict) -> None:
        path = self.reports_dir / f"{today}.md"
        lines = [
            f"# Adaptive Asymmetry-Scanner – {today}",
            "",
            f"**Generiert:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Trade-Vorschläge:** {len(proposals)}",
            "",
        ]

        if not proposals:
            lines.append("_Kein Signal heute. Alle Gates haben blockiert._")
        else:
            for i, p in enumerate(proposals, 1):
                da     = p.get("deep_analysis", {})
                sim    = p.get("simulation", {})
                feat   = p.get("features", {})
                option = p.get("option", {})
                ts     = p.get("trade_score", {})

                lines += [
                    f"---",
                    f"## {i}. {p['ticker']} – {p.get('strategy', '')}",
                    "",
                    f"**Richtung:** {p.get('direction', '')}  ",
                    f"**FinalScore:** `{p.get('final_score', 0):.4f}`  ",
                    f"**IV-Rank:** {p.get('iv_rank', 'N/A')}  ",
                    f"**Trade-Score:** {ts.get('total', 'N/A')}/100 — {ts.get('grade', '')}  ",
                    "",
                    "### Asymmetry-Analyse",
                    f"- **Impact:** {feat.get('impact', 'N/A')}/10",
                    f"- **Surprise:** {feat.get('surprise', 'N/A')}/10",
                    f"- **Mismatch-Score:** {feat.get('mismatch', 'N/A')}",
                    f"- **Z-Score (48h):** {feat.get('z_score', 'N/A')}",
                    f"- **48h-Move:** {feat.get('price_move_48h', 'N/A')}",
                    f"- **EPS-Drift:** {feat.get('eps_drift', 'N/A')} ({feat.get('bin_eps_drift', '')})",
                    "",
                    f"**Asymmetry-Reasoning:**  ",
                    f"> {da.get('asymmetry_reasoning', da.get('mispricing_logic', 'N/A'))}",
                    "",
                    f"**Katalysator:** {da.get('catalyst', 'N/A')}  ",
                    f"**Time-to-Materialization:** {da.get('time_to_materialization', 'N/A')}  ",
                    "",
                    "### Bear Case",
                    f"> {da.get('bear_case', 'N/A')}  ",
                    f"**Severity:** {da.get('bear_case_severity', 'N/A')}/10",
                    "",
                    "### Monte-Carlo Simulation",
                    f"- **Hit-Rate:** {sim.get('hit_rate', 0):.1%} ({sim.get('n_paths', 0):,} Pfade)",
                    f"- **Target-Preis:** ${sim.get('target_price', 0):.2f}",
                    f"- **Aktueller Preis:** ${sim.get('current_price', 0):.2f}",
                    f"- **σ:** {sim.get('sigma', sim.get('sigma_adj', 0)):.4f}",
                    f"- **α (Signal-Drift):** {sim.get('alpha', 0):.5f}",
                    "",
                ]

                if option:
                    lines += [
                        "### Options-Vorschlag",
                        f"- **Expiry:** {option.get('expiry', 'N/A')} ({option.get('dte', 'N/A')} DTE)",
                        f"- **Strike:** ${option.get('strike', 0):.2f}",
                        f"- **Bid/Ask:** ${option.get('bid', 0):.2f} / ${option.get('ask', 0):.2f}",
                        f"- **Impl. Vol.:** {option.get('implied_vol', 0):.1%}",
                        f"- **Open Interest:** {option.get('open_interest', 0):,}",
                        f"- **Bid-Ask-Ratio:** {option.get('spread_ratio', 0):.2%}",
                        "",
                    ]
                    if p.get("strategy") == "BULL_CALL_SPREAD" and option.get("spread_leg"):
                        sl = option["spread_leg"]
                        lines += [
                            f"- **Short Strike:** ${sl.get('strike', 0):.2f}  ",
                            f"- **Short Bid/Ask:** ${sl.get('bid', 0):.2f} / ${sl.get('ask', 0):.2f}",
                            f"- **Net Debit:** ${option.get('net_debit', 0):.2f}",
                            "",
                        ]

                # ── v8.2: Exit-Regeln ────────────────────────────────────────
                exit_r = p.get("exit_rules", {})
                if exit_r and exit_r.get("entry_cost", 0) > 0:
                    lines += [
                        "### 🚪 Exit-Regeln",
                        f"- **Entry-Kosten:** ${exit_r['entry_cost']:.2f} pro Kontrakt",
                        f"- **Take-Profit:** bei ${exit_r['take_profit_price']:.2f} "
                        f"(+{exit_r['take_profit_pct']}%) → Hälfte schliessen",
                        f"- **Full-Profit:** bei ${exit_r['full_profit_price']:.2f} "
                        f"(+{exit_r['full_profit_pct']}%) → Rest schliessen",
                        f"- **Stop-Loss:** bei ${exit_r['stop_loss_price']:.2f} "
                        f"({exit_r['stop_loss_pct']}%) → Gesamte Position liquidieren",
                        f"- **Time-Exit:** ab {exit_r['time_exit_date']} "
                        f"({exit_r['time_exit_dte_remaining']} DTE Rest) "
                        f"— schliessen wenn Gewinn < +{exit_r['time_exit_min_profit_pct']}%",
                        "",
                    ]

        # Modell-Gewichte
        weights = history.get("model_weights", {})
        lines += [
            "---",
            "## Modell-Gewichte (aktuell)",
            f"- Impact: `{weights.get('impact', 0.35):.2f}`",
            f"- Mismatch: `{weights.get('mismatch', 0.45):.2f}`",
            f"- EPS-Drift: `{weights.get('eps_drift', 0.20):.2f}`",
            "",
            "_Automatisch generiert durch Adaptive Asymmetry-Scanner v8.2_",
        ]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        log.info(f"Markdown-Report gespeichert: {path}")
