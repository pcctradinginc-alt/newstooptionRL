"""
modules/deep_analysis.py v8.0

Änderung: Devil's Advocate + MC-Kontext im Prompt.
Kein "Asymmetry Council" Overhead — gleicher Effekt, 1/8 der Kosten.

Devil's Advocate Logik:
    Claude muss zuerst aktiv GEGEN den Trade argumentieren.
    Erst dann kommt die finale Bewertung.
    Verhindert Confirmation-Bias (Claude stimmt Prescreening blind zu).

MC-Injection:
    Quick MC Hit-Rate wird im Prompt übergeben.
    Claude prüft ob News-Qualität mit Statistik konsistent ist.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import anthropic
import yfinance as yf

from modules.config        import cfg
from modules.macro_context import get_macro_context

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Du bist ein skeptischer Quant-Analyst mit Fokus auf mittelfristige Optionsstrategien (2-6 Monate).

PFLICHT-ABLAUF — in dieser Reihenfolge, keine Ausnahme:

SCHRITT 1 — RED TEAM (zuerst immer):
Finde die 3 stärksten Argumente GEGEN diesen Trade.
Denke wie ein Short-Seller. Was könnte das Signal zerstören?
Typische Red Flags: Überbewertung, Sektor-Gegenwind, fragliche Datenqualität, 
Makro-Risiko, IV-Crush, Katalysator bereits eingepreist.

SCHRITT 2 — STATISTIK-CHECK:
Ist die MC Hit-Rate realistisch gegeben historischer Volatilität?
Warnung wenn Hit-Rate > 80% (Modell möglicherweise zu optimistisch).

SCHRITT 3 — MAKRO-KONTEXT:
Passt das Signal zum aktuellen Zinsumfeld?
Rezessives Umfeld (invertierte Kurve) → erhöhte Skepsis bei BULLISH-Signalen.

SCHRITT 4 — ERST JETZT: Finale Bewertung.
Im Zweifel BEARISH. Nur eindeutige strukturelle Signale verdienen Impact > 7.

Antworte ausschließlich mit validem JSON."""

ANALYSIS_TEMPLATE = """=== MAKRO-KONTEXT ===
{macro_context}

=== TICKER: {ticker} ===
Aktueller Preis: ${current_price:.2f}
Sektor: {sector}
EPS (yfinance): {forward_eps} | EPS (SEC EDGAR): {sec_eps}
EPS-Abweichung: {eps_deviation}
48h-Preisbewegung: {move_48h:+.1%}

=== QUICK MONTE CARLO (Vorfilter) ===
Hit-Rate: {mc_hit_rate:.1%} ({mc_paths} Pfade, {mc_days}d)
Interpretation: {mc_interpretation}

=== NEWS (letzte 48h) ===
{news_text}

{data_anomaly_warning}

=== DEINE AUFGABE ===
Folge dem Pflicht-Ablauf: Red Team → Statistik → Makro → Finale Bewertung.

Antworte NUR mit diesem JSON:
{{
    "red_team": {{
        "argument_1": "<Stärkstes Argument gegen den Trade>",
        "argument_2": "<Zweitstärkstes Argument>",
        "argument_3": "<Drittstärkstes Argument>",
        "red_team_verdict": "VETO" oder "PASSIERT"
    }},
    "stats_check": {{
        "mc_assessment": "<Ist {mc_hit_rate:.0%} realistisch?>",
        "concern_level": "low" oder "medium" oder "high"
    }},
    "impact": <0-10>,
    "surprise": <0-10>,
    "direction": "BULLISH" oder "BEARISH",
    "bear_case_severity": <0-10>,
    "time_to_materialization": "4-8 Wochen" oder "2-3 Monate" oder "6 Monate",
    "asymmetry_reasoning": "<Max 3 Sätze — warum Markt unterreagiert hat>",
    "catalyst": "<Spezifischer Katalysator>",
    "bear_case": "<Stärkstes Gegenargument>",
    "macro_assessment": "<Bewertung im aktuellen Makro-Umfeld>",
    "data_confidence": "high" oder "medium" oder "low"
}}"""


class DeepAnalysis:

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self._macro = get_macro_context()
        if self._macro["data_available"]:
            log.info(
                f"Makro: {self._macro['macro_regime']} | "
                f"YC={self._macro.get('yield_curve_desc', 'n/a')}"
            )

    def run(self, shortlist: list[dict]) -> list[dict]:
        analyses = []
        for candidate in shortlist:
            analysis = self._analyze(candidate)
            if not analysis:
                continue

            # Red Team Veto: Wenn Red Team "VETO" → Signal verwerfen
            red_team = analysis.get("red_team", {})
            if red_team.get("red_team_verdict") == "VETO":
                log.info(
                    f"  [{candidate['ticker']}] RED TEAM VETO → verworfen. "
                    f"Grund: {red_team.get('argument_1', 'n/a')}"
                )
                continue

            # Stats-Check: Bei hohem MC-Concern → Impact deckeln
            stats = analysis.get("stats_check", {})
            if stats.get("concern_level") == "high" and analysis.get("impact", 0) > 6:
                original = analysis["impact"]
                analysis["impact"] = 6
                log.info(
                    f"  [{candidate['ticker']}] Stats-Concern HIGH: "
                    f"Impact {original} → 6 gedeckelt"
                )

            analyses.append({**candidate, "deep_analysis": analysis})
            log.info(
                f"  [{candidate['ticker']}] "
                f"Impact={analysis['impact']} "
                f"Surprise={analysis['surprise']} "
                f"Direction={analysis['direction']} "
                f"RedTeam={red_team.get('red_team_verdict', '?')}"
            )

        return analyses

    def _analyze(self, candidate: dict) -> Optional[dict]:
        ticker  = candidate.get("ticker", "")
        info    = candidate.get("info", {})
        news    = candidate.get("news", [])

        current_price = float(
            info.get("currentPrice") or
            info.get("regularMarketPrice") or 0
        )
        forward_eps = info.get("forwardEps") or info.get("trailingEps") or 0.0
        sector      = info.get("sector", "Unknown")
        move_48h    = self._get_48h_move(ticker)

        # EPS Cross-Check Info
        eps_check   = candidate.get("data_validation", {}).get("eps_cross_check", {})
        sec_eps     = eps_check.get("sec_eps", "n/a")
        dev_pct     = eps_check.get("deviation_pct")
        eps_deviation = f"{dev_pct:.1%}" if dev_pct is not None else "n/a"

        # Data Anomaly Warnung
        data_anomaly    = candidate.get("data_anomaly", False)
        anomaly_warning = ""
        if data_anomaly:
            anomaly_warning = (
                "⚠️ DATA ANOMALY: EPS-Daten weichen >20% ab. "
                "Red Team sollte Datenqualität als Argument 1 nennen. "
                "data_confidence muss 'low' sein."
            )

        # Quick MC Daten
        qmc         = candidate.get("quick_mc", {})
        mc_hit_rate = qmc.get("hit_rate", 0.0)
        mc_paths    = qmc.get("n_paths", 0)
        mc_days     = qmc.get("n_days", 30)

        if mc_hit_rate == 0:
            mc_interpretation = "Kein Quick MC durchgeführt — keine Statistik verfügbar."
        elif mc_hit_rate > 0.80:
            mc_interpretation = "WARNUNG: >80% Hit-Rate ist ungewöhnlich hoch — Modell möglicherweise zu optimistisch."
        elif mc_hit_rate > 0.60:
            mc_interpretation = "Solide statistische Basis — realistisch für 2-6M Horizont."
        else:
            mc_interpretation = f"Nur {mc_hit_rate:.0%} — knapp über Minimum-Gate, erhöhte Vorsicht."

        # News
        news_text = "\n".join(f"- {h}" for h in news[:8]) if news else "Keine News."

        # Makro
        macro_text = (
            self._macro.get("claude_context", "Makro: nicht verfügbar")
            if self._macro.get("data_available")
            else "Makro-Kontext: FRED nicht erreichbar."
        )

        prompt = ANALYSIS_TEMPLATE.format(
            macro_context    = macro_text,
            ticker           = ticker,
            current_price    = current_price,
            sector           = sector,
            forward_eps      = forward_eps,
            sec_eps          = sec_eps,
            eps_deviation    = eps_deviation,
            move_48h         = move_48h,
            mc_hit_rate      = mc_hit_rate,
            mc_paths         = mc_paths,
            mc_days          = mc_days,
            mc_interpretation= mc_interpretation,
            news_text        = news_text,
            data_anomaly_warning = anomaly_warning,
        )

        try:
            response = self.client.messages.create(
                model      = cfg.models.deep_analysis,
                max_tokens = 1200,
                system     = SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # JSON extrahieren
            if "```" in raw:
                parts = raw.split("```")
                raw   = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
            if not raw.startswith("{"):
                idx = raw.find("{")
                if idx != -1:
                    raw = raw[idx:]

            result = json.loads(raw)
            result["macro_regime"]  = self._macro.get("macro_regime", "unknown")
            result["macro_context"] = {
                "yield_curve": self._macro.get("yield_curve_spread"),
                "regime":      self._macro.get("macro_regime"),
            }
            return result

        except Exception as e:
            log.error(f"  [{ticker}] Deep Analysis Fehler: {e}")
            return None

    def _get_48h_move(self, ticker: str) -> float:
        try:
            hist  = yf.Ticker(ticker).history(period="5d")
            close = hist["Close"]
            if len(close) < 3:
                return 0.0
            return float((close.iloc[-1] - close.iloc[-3]) / close.iloc[-3])
        except Exception:
            return 0.0
