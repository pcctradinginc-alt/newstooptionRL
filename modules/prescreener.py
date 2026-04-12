"""
Stufe 2: Vorselektion – "Der Türsteher"

Fixes:
  M-03: Kein Retry bei API-Fehler → ein transienter Fehler terminierte
        die gesamte Tages-Pipeline. Fix: 3 Versuche mit Exponential Backoff.
  cfg:  Modell-Name aus config.yaml.
"""

import json
import logging
import os
import time
import anthropic

from modules.config import cfg

log = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE = 2  # Sekunden

SYSTEM_PROMPT = """Du bist ein erfahrener Finanzanalyst mit Fokus auf strukturelle Marktveränderungen.
Deine Aufgabe: Unterscheide zwischen temporärem Rauschen und echten strukturellen Änderungen.

Temporäres Rauschen (→ [NO]):
- Aktienrückkäufe ohne strategischen Kontext
- Analysten-Upgrades/-Downgrades ohne fundamentale Begründung
- Quartalsergebnisse im Rahmen der Erwartungen
- Dividendenankündigungen
- CEO-Statements ohne konkrete Ankündigung

Strukturelle Änderungen (→ [YES]):
- Neue Produktkategorien oder Märkte
- Technologische Durchbrüche (neue IP, Patente)
- Management-Turnarounds mit konkretem Plan
- Regulatorische Entscheidungen mit langfristiger Wirkung
- M&A mit strategischer Logik
- Verlust/Gewinn eines Großkunden (>10% Umsatz)
- Fundamentale Geschäftsmodelländerungen

Antworte ausschließlich mit validem JSON."""

USER_TEMPLATE = """Analysiere diese News-Headlines pro Ticker.
Für jeden Ticker: Entscheide ob die News eine strukturelle Änderung darstellt.

Ticker und Headlines:
{ticker_news}

Antworte mit folgendem JSON-Format:
{{
  "results": [
    {{
      "ticker": "AAPL",
      "decision": "[YES]" oder "[NO]",
      "reason": "Kurze Begründung (max 20 Wörter)"
    }}
  ]
}}"""


class Prescreener:

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def run(self, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        ticker_news_str = "\n".join([
            f"[{c['ticker']}]: {' | '.join(c['news'][:5])}"
            for c in candidates
        ])
        prompt = USER_TEMPLATE.format(ticker_news=ticker_news_str)

        # FIX M-03: Retry mit Exponential Backoff
        results = self._call_with_retry(prompt)
        if results is None:
            log.error(
                "Prescreening nach allen Versuchen fehlgeschlagen. "
                "Pipeline wird mit leerer Shortlist fortgesetzt."
            )
            return []

        yes_tickers = {
            r["ticker"]: r["reason"]
            for r in results
            if r.get("decision") == "[YES]"
        }

        shortlist = []
        for c in candidates:
            if c["ticker"] in yes_tickers:
                c["prescreen_reason"] = yes_tickers[c["ticker"]]
                shortlist.append(c)
                log.info(f"  [YES] {c['ticker']}: {yes_tickers[c['ticker']]}")

        return shortlist

    def _call_with_retry(self, prompt: str) -> list | None:
        """
        FIX M-03: Bis zu MAX_RETRIES Versuche mit Exponential Backoff.
        Bei transientem Fehler wird nicht sofort die Pipeline beendet.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.messages.create(
                    model=cfg.models.prescreener,   # cfg statt hardcodiert
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text.strip()

                if "```" in raw:
                    parts = raw.split("```")
                    raw   = parts[1] if len(parts) > 1 else raw
                    if raw.startswith("json"):
                        raw = raw[4:].strip()

                if not raw.startswith("{"):
                    idx = raw.find("{")
                    if idx != -1:
                        raw = raw[idx:]

                parsed  = json.loads(raw)
                results = parsed.get("results", [])
                log.info(
                    f"Prescreening erfolgreich (Versuch {attempt}): "
                    f"{len(results)} Ticker bewertet."
                )
                return results

            except (json.JSONDecodeError, KeyError) as e:
                # Parsing-Fehler: Retry sinnlos wenn Modell falsch antwortet
                log.error(f"Prescreening Parsing-Fehler (Versuch {attempt}): {e}")
                if attempt == MAX_RETRIES:
                    return None
                # Kurze Pause dann nochmal (Modell könnte anders antworten)
                time.sleep(BACKOFF_BASE ** attempt)

            except Exception as e:
                # Netzwerk/API-Fehler: Retry mit Backoff
                wait = BACKOFF_BASE ** attempt
                log.warning(
                    f"Prescreening API-Fehler (Versuch {attempt}/{MAX_RETRIES}): "
                    f"{e} → Warte {wait}s"
                )
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(wait)

        return None
