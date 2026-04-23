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

ANALYSIS_TEMPLATE = """=== ANALYSEDATUM: {analysis_date} (WICHTIG: Alle Jahreszahlen müssen ≥ {analysis_year} sein) ===

=== MAKRO-KONTEXT ===
{macro_context}

=== TICKER: {ticker} ===
Aktueller Preis: ${current_price:.2f}
Sektor: {sector}
Haiku-Prescreening: {prescreen_reason} [Kategorie: {prescreen_category}]
WICHTIG: Wenn deine Bewertung der Direction von der Haiku-Einschätzung abweicht,
erkläre explizit warum im asymmetry_reasoning.
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
        "argument_1": "<Stärkstes Argument gegen den Trade — Min 2 vollständige Sätze, mindestens 200 Zeichen, konkret>",  
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
    "asymmetry_reasoning": "<Genau 3 vollständige Sätze — warum der Markt unterreagiert hat. Maximal 500 Zeichen. Kein Satz darf abgebrochen werden>",  
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
            # Auch: Automatisches VETO bei Narrativ-Mismatch-Schlüsselwörtern
            red_team = analysis.get("red_team", {})
            arg1 = (red_team.get("argument_1", "") or "").lower()
            narrativ_mismatch = any(w in arg1 for w in [
                "narrativ-mismatch", "narrative mismatch", "trifft das geschäftsmodell",
                "falsches narrativ", "datenfehler in der vorselektion",
                "grundlegendes missverständnis", "trifft nicht zu"
            ])
            if narrativ_mismatch and red_team.get("red_team_verdict") != "VETO":
                log.warning(
                    f"  [{candidate['ticker']}] AUTO-VETO: Narrativ-Mismatch erkannt → "
                    f"'{arg1[:60]}'"
                )
                red_team["red_team_verdict"] = "VETO"

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

            # Widerspruchs-Check: Haiku bullish → Sonnet bearish
            haiku_reason  = candidate.get("prescreen_reason", "").lower()
            sonnet_dir    = analysis.get("direction", "")
            haiku_bullish = any(w in haiku_reason for w in
                ["positiv", "erhöht", "wachstum", "deal", "akquisition",
                 "expansion", "gewinn", "stieg", "prognose"])
            if haiku_bullish and sonnet_dir == "BEARISH":
                log.warning(
                    f"  [{candidate['ticker']}] ⚠️ WIDERSPRUCH: "
                    f"Haiku=BULLISH ({haiku_reason[:50]}) "
                    f"aber Sonnet=BEARISH → Impact={analysis['impact']} "
                    f"gedeckelt auf max 6"
                )
                if analysis.get("impact", 0) > 6:
                    analysis["impact"] = 6
                analysis["direction_conflict"] = True
            else:
                analysis["direction_conflict"] = False

            analyses.append({**candidate, "deep_analysis": analysis})
            log.info(
                f"  [{candidate['ticker']}] "
                f"Impact={analysis['impact']} "
                f"Surprise={analysis['surprise']} "
                f"Direction={analysis['direction']} "
                f"RedTeam={red_team.get('red_team_verdict', '?')} "
                f"{'⚠️ KONFLIKT' if analysis.get('direction_conflict') else '✅'}"
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

        prescreen_reason   = candidate.get("prescreen_reason", "n/a")
        prescreen_category = candidate.get("prescreen_category", "n/a")

        from datetime import datetime as _dt
        _today = _dt.now()
        prompt = ANALYSIS_TEMPLATE.format(
            analysis_date    = _today.strftime("%d.%m.%Y"),
            analysis_year    = _today.year,
            macro_context    = macro_text,
            ticker           = ticker,
            current_price    = current_price,
            sector           = sector,
            prescreen_reason   = prescreen_reason,
            prescreen_category = prescreen_category,
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
                max_tokens = 1500,
                system     = SYSTEM_PROMPT.format(current_year=_dt.now().year),
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

            # Robuster JSON-Parser: bei unterminated string → reparieren
            try:
                result = json.loads(raw)
            except json.JSONDecodeError as je:
                # Versuche JSON bis zum letzten vollständigen Feld zu parsen
                log.warning(f"  [{ticker}] JSON teilweise abgeschnitten: {je} → Reparatur-Versuch")
                # Schneide bei letztem vollständigen Wert ab
                last_comma = raw.rfind('",')
                last_brace = raw.rfind('"')
                cutoff = max(last_comma, 0)
                if cutoff > 100:
                    raw_fixed = raw[:cutoff] + '"}'
                    try:
                        result = json.loads(raw_fixed)
                        log.info(f"  [{ticker}] JSON repariert (gekürzt auf {cutoff} Zeichen)")
                    except Exception:
                        # Fallback: Minimal-Response mit niedrigem Impact
                        log.warning(f"  [{ticker}] JSON nicht reparierbar → Fallback-Response")
                        result = {
                            "red_team": {"argument_1": "JSON-Parse-Fehler", "red_team_verdict": "PASSIERT"},
                            "stats_check": {"mc_assessment": "n/a", "concern_level": "medium"},
                            "impact": 3, "surprise": 3, "direction": "BULLISH",
                            "bear_case_severity": 5,
                            "time_to_materialization": "2-3 Monate",
                            "asymmetry_reasoning": "JSON-Parse-Fehler — manuelle Prüfung empfohlen",
                            "catalyst": "n/a", "bear_case": "n/a",
                            "macro_assessment": "n/a", "data_confidence": "low"
                        }
                else:
                    raise
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
