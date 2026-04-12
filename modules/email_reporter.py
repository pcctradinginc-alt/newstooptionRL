"""
modules/email_reporter.py v5.0

NEU: Tägliche Status-Email wird IMMER gesendet — auch wenn kein Trade generiert.
     "No Trade"-Email zeigt transparent warum kein Signal durchkam.

Email-Typen:
  1. TRADE-Email:    Wie bisher — Trade-Vorschlag mit allen Details
  2. NO-TRADE-Email: Täglich, zeigt Pipeline-Status + Filterstatistik

Benötigte Secrets:
  GMAIL_SENDER   → Absender-Adresse
  GMAIL_APP_PW   → Gmail App-Passwort
  NOTIFY_EMAIL   → Empfänger-Adresse
"""

import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger(__name__)


def send_email(proposals: list[dict], today: str) -> None:
    """
    Sendet entweder eine Trade-Email oder eine No-Trade-Status-Email.
    Wird immer aufgerufen — unabhängig von proposals.
    """
    if proposals:
        html = _build_trade_email(proposals, today)
        subject = f"Adaptive Asymmetry-Scanner – Trade Empfehlung – {today}"
    else:
        html = _build_no_trade_email(today)
        subject = f"Adaptive Asymmetry-Scanner – Kein Trade – {today}"

    _send_smtp(subject, html)


def send_status_email(pipeline_stats: dict, today: str) -> None:
    """
    Explizite Status-Email mit Pipeline-Statistiken.
    Wird von pipeline.py am Ende IMMER aufgerufen.

    pipeline_stats:
        {
            "candidates":   int,   # nach Hard-Filter
            "prescreened":  int,   # nach Prescreening
            "analyzed":     int,   # nach Deep Analysis
            "mismatch_ok":  int,   # nach Mismatch-Filter
            "intraday_ok":  int,   # nach Intraday-Delta-Filter
            "simulated":    int,   # nach Monte-Carlo
            "rl_scored":    int,   # nach RL-Filter
            "roi_ok":       int,   # nach ROI-Gate
            "trades":       int,   # finale Trade-Vorschläge
            "vix":          float,
            "stop_reason":  str,   # Warum Pipeline gestoppt (wenn kein Trade)
        }
    """
    trades  = pipeline_stats.get("trades", 0)
    subject = (
        f"Adaptive Asymmetry-Scanner – Trade Empfehlung – {today}"
        if trades > 0
        else f"Adaptive Asymmetry-Scanner – Kein Trade – {today}"
    )
    html = _build_status_email(pipeline_stats, today)
    _send_smtp(subject, html)


def _build_status_email(stats: dict, today: str) -> str:
    """Baut die tägliche Status-Email (Trade oder No-Trade)."""
    vix        = stats.get("vix", "–")
    candidates = stats.get("candidates", 0)
    prescreened = stats.get("prescreened", 0)
    analyzed   = stats.get("analyzed", 0)
    mismatch   = stats.get("mismatch_ok", 0)
    intraday   = stats.get("intraday_ok", analyzed)
    simulated  = stats.get("simulated", 0)
    rl_scored  = stats.get("rl_scored", 0)
    roi_ok     = stats.get("roi_ok", 0)
    trades     = stats.get("trades", 0)
    stop       = stats.get("stop_reason", "")

    # Farbe je nach Ergebnis
    header_color = "#16a34a" if trades > 0 else "#0f172a"
    status_icon  = "🎯" if trades > 0 else "📊"
    status_text  = "Trade Empfehlung" if trades > 0 else "Kein Trade heute"

    # Filter-Funnel als Tabelle
    funnel_rows = [
        ("498 Ticker im Universum", "📋", True),
        (f"{candidates} nach Hard-Filter (Market Cap / Volume)", "🔍", candidates > 0),
        (f"{prescreened} nach Prescreening (Claude Haiku)", "🤖", prescreened > 0),
        (f"{analyzed} nach Deep Analysis (Claude Sonnet)", "🧠", analyzed > 0),
        (f"{mismatch} nach Mismatch-Score", "📐", mismatch > 0),
        (f"{intraday} nach Intraday-Delta-Filter", "⏱️", intraday > 0),
        (f"{simulated} nach Monte-Carlo-Simulation", "🎲", simulated > 0),
        (f"{rl_scored} nach RL-Scoring", "🤖", rl_scored > 0),
        (f"{roi_ok} nach ROI-Gate", "💰", roi_ok > 0),
        (f"{trades} finale Trade-Vorschläge", "✅" if trades > 0 else "❌", trades > 0),
    ]

    funnel_html = ""
    for label, icon, active in funnel_rows:
        bg    = "#f0fdf4" if active else "#fef2f2"
        color = "#16a34a" if active else "#dc2626"
        funnel_html += f"""
        <tr>
          <td style="padding:8px 12px;font-size:13px;color:{color};
                     background:{bg};border-bottom:1px solid #e2e8f0;">
            {icon} {label}
          </td>
        </tr>"""

    stop_section = ""
    if stop and trades == 0:
        stop_section = f"""
        <div style="margin:20px 0;padding:16px;background:#fef3c7;
                    border-left:4px solid #f59e0b;border-radius:4px;">
          <b>Pipeline-Stop:</b> {stop}
        </div>"""

    return f"""
<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;margin:0;padding:0;
  background:#f8fafc;">
<div style="max-width:600px;margin:30px auto;background:#fff;
  border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);">

  <!-- Header -->
  <div style="background:{header_color};padding:28px 32px;">
    <div style="font-size:28px;margin-bottom:4px;">{status_icon}</div>
    <div style="color:#fff;font-size:22px;font-weight:bold;">{status_text}</div>
    <div style="color:rgba(255,255,255,0.7);font-size:14px;margin-top:4px;">
      {today} · VIX {f'{float(vix):.2f}' if vix else '–'} · Adaptive Asymmetry-Scanner v5.0
    </div>
  </div>

  <!-- Pipeline-Funnel -->
  <div style="padding:24px 32px;">
    <h3 style="margin:0 0 16px;color:#0f172a;font-size:16px;">
      Pipeline-Filter heute
    </h3>
    <table style="width:100%;border-collapse:collapse;border-radius:8px;
                  overflow:hidden;border:1px solid #e2e8f0;">
      {funnel_html}
    </table>
    {stop_section}
  </div>

  <!-- Footer -->
  <div style="padding:16px 32px;background:#f8fafc;
              border-top:1px solid #e2e8f0;
              font-size:12px;color:#94a3b8;text-align:center;">
    Adaptive Asymmetry-Scanner v5.0 · GitHub Actions · {datetime.utcnow().strftime('%H:%M UTC')}
  </div>
</div>
</body></html>"""


def _build_trade_email(proposals: list[dict], today: str) -> str:
    """Baut die Trade-Email (bereits vorhanden, hier vereinfacht)."""
    cards = ""
    for i, p in enumerate(proposals, 1):
        ticker     = p.get("ticker", "?")
        strategy   = p.get("strategy", "?")
        direction  = p.get("deep_analysis", {}).get("direction", "?")
        score      = p.get("final_score", 0)
        option     = p.get("option", {}) or {}
        sim        = p.get("simulation", {}) or {}
        da         = p.get("deep_analysis", {}) or {}
        roi        = p.get("roi_analysis", {}) or {}
        flash      = p.get("flash_alpha", {}) or {}
        eulerpool  = p.get("eulerpool", {}) or {}

        strike      = option.get("strike", "–")
        expiry      = option.get("expiry", "–")
        bid         = option.get("bid", "–")
        ask         = option.get("ask", "–")
        oi          = option.get("open_interest", "–")
        dte         = option.get("dte", "–")
        iv_rank     = p.get("iv_rank", "–")
        current     = sim.get("current_price", "–")
        target      = sim.get("target_price", "–")
        hit_rate    = sim.get("hit_rate", 0)
        impact      = da.get("impact", "–")
        surprise    = da.get("surprise", "–")
        bear_sev    = da.get("bear_case_severity", "–")
        roi_net     = roi.get("roi_net", None)
        crush_risk  = eulerpool.get("iv_crush_risk", "–")
        dealer_bias = flash.get("dealer_bullish", None)
        sentiment   = p.get("features", {}).get("sentiment_score", None)

        roi_badge = ""
        if roi_net is not None:
            roi_color = "#16a34a" if roi_net > 0.2 else ("#f59e0b" if roi_net > 0.1 else "#dc2626")
            roi_badge = f'<span style="background:{roi_color};color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;">ROI {roi_net:.1%}</span>'

        cards += f"""
        <div style="border:1px solid #e2e8f0;border-radius:8px;padding:20px;margin-bottom:20px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <span style="font-size:20px;font-weight:bold;color:#0f172a;">{ticker}</span>
            <span style="background:#2563eb;color:#fff;padding:4px 12px;border-radius:20px;font-size:12px;">
              {strategy} · Score {score:.4f}
            </span>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;">
            <div><b>Richtung:</b> {direction}</div>
            <div><b>Aktuell:</b> ${current}</div>
            <div><b>Ziel:</b> ${target}</div>
            <div><b>Hit-Rate:</b> {hit_rate:.1%}</div>
            <div><b>Strike:</b> ${strike}</div>
            <div><b>Expiry:</b> {expiry} ({dte}d)</div>
            <div><b>Bid/Ask:</b> ${bid}/${ask}</div>
            <div><b>OI:</b> {oi}</div>
            <div><b>Impact:</b> {impact}/10</div>
            <div><b>Surprise:</b> {surprise}/10</div>
            <div><b>Bear Severity:</b> {bear_sev}/10</div>
            <div><b>IV-Rank:</b> {iv_rank}%</div>
            {f'<div><b>Sentiment:</b> {sentiment:.2f}</div>' if sentiment is not None else ''}
            {f'<div><b>IV-Crush:</b> {crush_risk}</div>' if crush_risk != "–" else ''}
            {f'<div><b>Dealer-Bias:</b> {dealer_bias:.2f}</div>' if dealer_bias is not None else ''}
          </div>
          {roi_badge}
        </div>"""

    return f"""
<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;background:#f8fafc;margin:0;">
<div style="max-width:600px;margin:30px auto;background:#fff;border-radius:12px;
  box-shadow:0 2px 8px rgba(0,0,0,0.1);overflow:hidden;">
  <div style="background:#16a34a;padding:28px 32px;">
    <div style="font-size:28px;">🎯</div>
    <div style="color:#fff;font-size:22px;font-weight:bold;">
      Adaptive Asymmetry-Scanner – Trade Empfehlung
    </div>
    <div style="color:rgba(255,255,255,0.7);font-size:14px;margin-top:4px;">
      {today} · Adaptive Asymmetry-Scanner v5.0
    </div>
  </div>
  <div style="padding:24px 32px;">{cards}</div>
  <div style="padding:16px 32px;background:#f8fafc;border-top:1px solid #e2e8f0;
    font-size:12px;color:#94a3b8;text-align:center;">
    Kein Finanzberatung. Alle Vorschläge sind algorithmisch generiert.
  </div>
</div>
</body></html>"""


def _build_no_trade_email(today: str) -> str:
    """Fallback No-Trade-Email ohne Pipeline-Stats."""
    return _build_status_email({"trades": 0, "stop_reason": "Kein Signal hat alle Filter passiert."}, today)


def _send_smtp(subject: str, html: str) -> None:
    """Sendet die Email via Gmail SMTP."""
    sender   = os.getenv("GMAIL_SENDER", "")
    password = os.getenv("GMAIL_APP_PW", "")
    receiver = os.getenv("NOTIFY_EMAIL", sender)

    if not sender or not password:
        log.warning("GMAIL_SENDER oder GMAIL_APP_PW nicht gesetzt → Email übersprungen.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = receiver
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.sendmail(sender, receiver, msg.as_string())
        log.info(f"Email gesendet: '{subject}' → {receiver}")
    except Exception as e:
        log.error(f"SMTP-Fehler: {e}")
