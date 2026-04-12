"""
Risk-Gates: Globale und ticker-spezifische Sicherheitschecks

Fixes:
  C-01: VIX-Fallback war 20.0 (immer grün) → jetzt Exception bei Netzwerkfehler
  H-05: VIX-Wert wird jetzt zurückgegeben damit email_reporter ihn nutzen kann
  cfg:  Alle Schwellenwerte kommen aus config.yaml
"""

import logging
from datetime import datetime, timedelta
import yfinance as yf
from modules.config import cfg

log = logging.getLogger(__name__)


class VIXUnavailableError(Exception):
    """Wird geworfen wenn der VIX nicht abgerufen werden kann."""
    pass


class RiskGates:

    def __init__(self):
        self._last_vix: float | None = None   # H-05: für email_reporter

    def global_ok(self) -> bool:
        """
        Prüft globale Marktbedingungen.

        FIX C-01: Wirft VIXUnavailableError statt 20.0 zurückzugeben.
        Rationale: Der Safety-Gate muss im Fehlerfall sperren, nicht
        durchlassen. Ein Netzwerkfehler ist in Krisenzeiten am
        wahrscheinlichsten – genau dann darf das Gate nicht passiert werden.
        """
        try:
            vix = self._get_vix()
        except VIXUnavailableError as e:
            log.error(
                f"VIX nicht abrufbar: {e} → Pipeline wird abgebrochen "
                f"(Safety-First-Prinzip)."
            )
            return False

        self._last_vix = vix   # H-05: für email_reporter
        log.info(
            f"VIX aktuell: {vix:.2f} "
            f"(Schwelle: {cfg.risk.vix_threshold})"
        )

        if vix > cfg.risk.vix_threshold:
            log.warning(
                f"VIX={vix:.2f} > {cfg.risk.vix_threshold} → Abbruch."
            )
            return False

        return True

    @property
    def last_vix(self) -> float | None:
        """H-05: Gibt den zuletzt gemessenen VIX zurück (für email_reporter)."""
        return self._last_vix

    def has_upcoming_earnings(
        self, ticker: str, days: int | None = None
    ) -> bool:
        """
        Prüft ob Earnings innerhalb der nächsten `days` Tage anstehen.
        days=None → liest Wert aus config.yaml
        """
        days = days if days is not None else cfg.risk.earnings_buffer_days
        try:
            t = yf.Ticker(ticker)
            cal = t.calendar
            if cal is None or cal.empty:
                return False

            if "Earnings Date" in cal.columns:
                earnings_dates = cal["Earnings Date"].dropna()
            elif "Earnings Dates" in cal.columns:
                earnings_dates = cal["Earnings Dates"].dropna()
            else:
                return False

            cutoff = datetime.utcnow() + timedelta(days=days)
            for ed in earnings_dates:
                if isinstance(ed, str):
                    ed = datetime.strptime(ed[:10], "%Y-%m-%d")
                if hasattr(ed, "to_pydatetime"):
                    ed = ed.to_pydatetime().replace(tzinfo=None)
                if datetime.utcnow() <= ed <= cutoff:
                    log.info(
                        f"  [{ticker}] Earnings am {ed.date()} "
                        f"(< {days} Tage)"
                    )
                    return True
        except Exception as e:
            log.debug(f"Earnings-Check Fehler für {ticker}: {e}")
        return False

    def _get_vix(self) -> float:
        """
        FIX C-01: Wirft VIXUnavailableError statt stillen Fallback.
        Drei Versuche mit unterschiedlichen Methoden.
        """
        errors = []

        # Versuch 1: yfinance history
        try:
            hist = yf.Ticker("^VIX").history(period="2d")
            if not hist.empty:
                vix = float(hist["Close"].iloc[-1])
                if vix > 0:
                    return vix
            errors.append("yfinance history: leeres Ergebnis")
        except Exception as e:
            errors.append(f"yfinance history: {e}")

        # Versuch 2: yfinance info
        try:
            info = yf.Ticker("^VIX").info
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            if price and float(price) > 0:
                return float(price)
            errors.append("yfinance info: kein Preis")
        except Exception as e:
            errors.append(f"yfinance info: {e}")

        # Alle Versuche fehlgeschlagen → Exception (FIX C-01)
        raise VIXUnavailableError(
            f"VIX nach 2 Versuchen nicht abrufbar. Details: {'; '.join(errors)}"
        )
