"""
modules/prescreener.py v7.0 – Verschärfter Haiku-Prefilter

Problem vorher: Haiku liess 17+ Ticker durch → Deep Analysis lief für jeden
                einzeln → teuer (~$0.25 nur für Sonnet) + langsam (4 Min).

Fix: Strengerer System-Prompt + härtere Entscheidungsregeln.
     Ziel: max. 3-5 YES pro Batch (statt 17+).

Verschärfte Regeln:
  - Im Zweifel IMMER [NO]
  - Kein [YES] für Routine-Earnings, Upgrades, Dividenden
  - [YES] nur für eindeutige, strukturelle, unerwartete Änderungen
  - Explizite Kategorie-Abfrage damit Haiku begründen muss
"""

import json
import logging
import os
import time
import anthropic

from modules.config import cfg

log = logging.getLogger(__name__)

MAX_RETRIES   = 3
BACKOFF_BASE  = 2
BATCH_SIZE    = 20
MAX_HEADLINES = 3

SYSTEM_PROMPT = """Du bist ein Options-Analyst mit Fokus auf Informations-Asymmetrie.

KERNFRAGE fuer jeden Ticker: "Enthaelt diese Nachricht eine Information, die der breite
Markt noch nicht vollstaendig verarbeitet hat — und koennte ein Options-Kaeufer davon profitieren?"

[YES] wenn MINDESTENS EINE der folgenden Bedingungen zutrifft:
  a) EINGEBETTETES SIGNAL: Guidance-Erhoehung im Nebensatz, strukturelle Aenderung versteckt
     in Routine-Meldung, Insider-Kauf-Cluster, unerwartete Margenaenderung.
  b) MARKT-UNTERREAKTION: Kurs hat nur halb so stark reagiert wie fundamental gerechtfertigt.
     Typisch bei: komplexen Regulierungsentscheiden, mehrstufigen M&A-Auswirkungen.
  c) KATALYSATOR VORAUS: Bekanntes Event in 10-60 Tagen (Earnings, FDA, Produktlaunch, 
     Regulierungs-Deadline) bei dem die aktuelle News die Ausgangslage veraendert hat.
  d) SEKTOR-SPILLOVER: News bei Unternehmen X beeinflusst Unternehmen Y strukturell,
     Markt hat Y noch nicht repriced.

[NO] fuer: reine Analyst-Upgrades/Downgrades ohne neuen Datenpunkt, Standard-Dividenden,
Ruckkaeufe im Plan, vage CEO-Aussagen, Indexaufnahmen, allgemeine Wirtschaftsnews.

WICHTIG: Du suchst Asymmetrie, keine Gewissheit. Eine 60%-Chance auf +15% reicht fuer [YES].
Ziel: 15-25% YES-Rate. Bei <10 Kandidaten mindestens 1 YES wenn irgendein Signal erkennbar.
Antworte ausschliesslich mit validem JSON."""

USER_TEMPLATE = """Bewerte diese {n} Ticker auf Options-Asymmetrie-Potenzial.

{ticker_news}

Antworte NUR mit diesem JSON:
{{
  "results": [
    {{
      "ticker": "AAPL",
      "decision": "[YES]" oder "[NO]",
      "category": "structural_change|routine_news|analyst_opinion|earnings|catalyst",
      "reason": "Konkreter Grund in max 20 Worten — spezifisch, nicht vage"
    }}
  ]
}}

Richtwert: ~15% YES bei vorhandenen Signalen. Vergib [YES] NUR bei echtem Signal.
An nachrichtenarmen Tagen ist 0% YES korrekt — erzwinge keine Quote."""


class Prescreener:

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def run(self, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        all_yes: dict[str, str] = {}

        batches = [
            candidates[i:i + BATCH_SIZE]
            for i in range(0, len(candidates), BATCH_SIZE)
        ]
        log.info(
            f"Prescreening: {len(candidates)} Kandidaten in "
            f"{len(batches)} Batch(es) à max {BATCH_SIZE}"
        )

        for batch_idx, batch in enumerate(batches, 1):
            log.info(f"  Batch {batch_idx}/{len(batches)}: {len(batch)} Ticker")
            results = self._call_with_retry(batch)

            if results is None:
                log.warning(f"  Batch {batch_idx} fehlgeschlagen → übersprungen")
                continue

            yes_count = 0
            no_count  = 0
            for r in results:
                decision = r.get("decision", "[NO]")
                category = r.get("category", "other")
                ticker   = r.get("ticker", "")

                if decision == "[YES]":
                    # Zusatz-Check: Routine-Kategorien nie durchlassen
                    if category in ("routine_news", "analyst_opinion", "earnings"):
                        log.info(
                            f"  [{ticker}] Override: Kategorie='{category}' "
                            f"→ trotz [YES] auf [NO] gesetzt"
                        )
                        no_count += 1
                        continue
                    # Quick Options-Liquiditäts-Check
                    if not self._has_options_liquidity(ticker):
                        log.info(f"  [{ticker}] OPTIONS-LIQUIDITÄT: zu gering → übersprungen")
                        no_count += 1
                        continue
                    all_yes[ticker] = {
                        "reason":    r.get("reason", ""),
                        "category":  r.get("category", ""),
                    }
                    yes_count += 1
                else:
                    no_count += 1

            log.info(
                f"  Batch {batch_idx}: {yes_count} YES, {no_count} NO "
                f"({yes_count/(yes_count+no_count)*100:.0f}% YES-Rate)"
            )

        shortlist = []
        for c in candidates:
            if c["ticker"] in all_yes:
                prescreen_data = all_yes[c["ticker"]]
                c["prescreen_reason"]   = prescreen_data.get("reason", "")
                c["prescreen_category"] = prescreen_data.get("category", "")
                shortlist.append(c)
                log.info(f"  [YES] {c['ticker']}: {prescreen_data.get('reason','')} [{prescreen_data.get('category','')}]")

        log.info(
            f"Prescreening gesamt: {len(shortlist)}/{len(candidates)} YES "
            f"({len(shortlist)/len(candidates)*100:.0f}% YES-Rate)"
        )
        return shortlist

    def _has_options_liquidity(self, ticker: str) -> bool:
        """Prüft ob genug Optionskontrakte verfügbar sind (schnell, kein API-Call)."""
        try:
            import yfinance as yf
            t     = yf.Ticker(ticker)
            dates = t.options
            if not dates or len(dates) < 2:
                return False
            # Mindestens 2 Verfallsdaten → ausreichend liquide
            return True
        except Exception:
            return True  # Im Zweifel durchlassen

    def _call_with_retry(self, batch: list[dict]) -> list | None:
        ticker_news_str = "\n".join([
            f"[{c['ticker']}]: {' | '.join(c['news'][:MAX_HEADLINES])}"
            for c in batch
        ])
        min_yes = len(batch) // 7   # ~15% Richtwert, kann 0 sein — kein Zwang
        prompt  = USER_TEMPLATE.format(
            n           = len(batch),
            ticker_news = ticker_news_str,
            min_yes     = min_yes,
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.messages.create(
                    model      = cfg.models.prescreener,
                    max_tokens = 4096,
                    system     = SYSTEM_PROMPT,
                    messages   = [{"role": "user", "content": prompt}],
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
                    f"    Batch OK (Versuch {attempt}): "
                    f"{len(results)} Ticker bewertet"
                )
                return results

            except (json.JSONDecodeError, KeyError) as e:
                log.error(f"    Parsing-Fehler (Versuch {attempt}): {e}")
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(BACKOFF_BASE ** attempt)

            except Exception as e:
                wait = BACKOFF_BASE ** attempt
                log.warning(
                    f"    API-Fehler (Versuch {attempt}/{MAX_RETRIES}): "
                    f"{e} → Warte {wait}s"
                )
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(wait)

        return None
