"""
modules/deep_analysis.py v7.0

Fix 3: Makro-Kontext (ISM + Yield Curve) wird in den Claude-Prompt injiziert.
       Claude bekommt damit wirtschaftliches "Wetter" für mittelfristige Bewertung.

Fix: Data-Anomaly Flag aus data_validator.py wird im Prompt kommuniziert.
     Wenn yfinance EPS > 20% von SEC EDGAR abweicht, warnt Claude explizit.
"""

import json
import logging
import os

import anthropic
import yfinance as yf

from modules.config        import cfg
from modules.macro_context import get_macro_context

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Du bist ein erfahrener Quantitativ-Analyst mit Fokus auf mittelfristige Optionsstrategien (2-6 Monate). 

Deine Aufgabe: Analysiere ob eine fundamentale Informations-Asymmetrie existiert — ob der Markt auf eine strukturelle Veränderung unterreagiert hat.

Bewertungskriterien:
- Impact (0-10): Wie stark verändert diese News die fundamentalen Ertragserwartungen?
- Surprise (0-10): Wie unerwartet ist diese Information für den Markt?
- Direction: BULLISH oder BEARISH (aus der Perspektive des Aktienpreises)
- Bear Case Severity (0-10): Wie stark ist das Gegenargument?
- Time to Materialization: Wann wird sich die Asymmetrie einpreisen?

KRITISCH für Mittelfrist-Trades (2-6 Monate):
- Einzelne Schlagzeilen sind KEIN ausreichendes Signal. Nur strukturelle Änderungen zählen.
- Wenn EPS-Daten als "DATA ANOMALY" markiert sind: Sei extra vorsichtig bei EPS-basierten Analysen.
- Berücksichtige den Makro-Kontext explizit in deiner Bewertung.

Antworte ausschließlich mit validem JSON."""

ANALYSIS_TEMPLATE = """{macro_context}

=== TICKER-ANALYSE ===
Ticker: {ticker}
Aktueller Preis: ${current_price}
Forward EPS: {forward_eps} {eps_warning}
EPS-Drift: {eps_drift:.1%}
Sektor: {sector}
48h-Preisbewegung: {move_48h:+.1%}

News (letzte 48h):
{news_text}

{data_anomaly_warning}

Erstelle eine vollständige Analyse im folgenden JSON-Format:
{{
    "impact": <0-10>,
    "surprise": <0-10>,
    "direction": "BULLISH" oder "BEARISH",
    "bear_case_severity": <0-10>,
    "time_to_materialization": "4-8 Wochen" oder "2-3 Monate" oder "6 Monate",
    "asymmetry_reasoning": "<Erklärung der Informations-Asymmetrie, max 3 Sätze>",
    "catalyst": "<Spezifischer Katalysator für Preisbewegung>",
    "bear_case": "<Stärkstes Gegenargument>",
    "macro_assessment": "<Wie bewertest du das Signal im aktuellen Makro-Umfeld?>",
    "data_confidence": "high" oder "medium" oder "low"
}}"""


class DeepAnalysis:

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # Makro-Kontext einmal laden (gecached für alle Ticker des Tages)
        self._macro = get_macro_context()
        if self._macro["data_available"]:
            log.info(
                f"Makro-Kontext: {self._macro['macro_regime']} "
                f"(YC={self._macro.get('yield_curve_desc', 'n/a')})"
            )
        else:
            log.warning("Makro-Kontext nicht verfügbar → FRED offline")

    def run(self, shortlist: list[dict]) -> list[dict]:
        analyses = []
        for candidate in shortlist:
            analysis = self._analyze(candidate)
            if analysis:
                analyses.append({**candidate, "deep_analysis": analysis})
                log.info(
                    f"  [{candidate['ticker']}] "
                    f"Impact={analysis['impact']} "
                    f"Surprise={analysis['surprise']} "
                    f"Direction={analysis['direction']}"
                )
        return analyses

    def _analyze(self, candidate: dict) -> Optional[dict]:
        ticker    = candidate.get("ticker", "")
        info      = candidate.get("info", {})
        news      = candidate.get("news", [])
        eps_drift = candidate.get("eps_drift", {})

        current_price = float(
            info.get("currentPrice") or info.get("regularMarketPrice") or 0
        )
        forward_eps = info.get("forwardEps") or 0.0
        sector      = info.get("sector", "Unknown")
        move_48h    = self._get_48h_move(ticker)

        # News als Text
        news_text = "\n".join(
            f"- {h}" for h in news[:8]
        ) if news else "Keine News verfügbar"

        # EPS-Drift Wert
        drift_val = eps_drift.get("drift", 0.0) if isinstance(eps_drift, dict) else 0.0

        # Data-Anomaly Warnung
        data_anomaly    = candidate.get("data_anomaly", False)
        eps_warning     = ""
        anomaly_warning = ""

        if data_anomaly:
            eps_check    = candidate.get("data_validation", {}).get("eps_cross_check", {})
            sec_eps      = eps_check.get("sec_eps", "n/a")
            deviation    = eps_check.get("deviation_pct", 0)
            eps_warning  = f"⚠️ DATA ANOMALY: SEC EDGAR EPS={sec_eps} (Abweichung {deviation:.0%})"
            anomaly_warning = (
                "⚠️ WICHTIG: Die EPS-Daten zeigen eine starke Inkonsistenz zwischen "
                "yfinance und SEC EDGAR. Bewerte EPS-basierte Argumente mit erhöhter "
                "Vorsicht und setze data_confidence auf 'low'."
            )

        # Makro-Kontext für Prompt
        macro_text = (
            self._macro.get("claude_context", "")
            if self._macro["data_available"]
            else "Makro-Kontext: Nicht verfügbar (FRED offline)"
        )

        prompt = ANALYSIS_TEMPLATE.format(
            macro_context        = macro_text,
            ticker               = ticker,
            current_price        = current_price,
            forward_eps          = forward_eps,
            eps_warning          = eps_warning,
            eps_drift            = drift_val,
            sector               = sector,
            move_48h             = move_48h,
            news_text            = news_text,
            data_anomaly_warning = anomaly_warning,
        )

        try:
            response = self.client.messages.create(
                model      = cfg.models.deep_analysis,
                max_tokens = 1024,
                system     = SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # JSON extrahieren
            if "```" in raw:
                parts = raw.split("```")
                raw   = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            if not raw.startswith("{"):
                idx = raw.find("{")
                if idx != -1:
                    raw = raw[idx:]

            result = json.loads(raw)

            # Makro-Regime in Ergebnis speichern
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
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) < 3:
                return 0.0
            close = hist["Close"]
            return float((close.iloc[-1] - close.iloc[-3]) / close.iloc[-3])
        except Exception:
            return 0.0


# Fix Import für Optional
from typing import Optional
