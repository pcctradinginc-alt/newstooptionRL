"""
modules/email_reporter.py v8.1

Änderungen v8.1:
    - Textlimit: [:700] statt [:600]
    - Score-Schwelle für Email: >= 50 (WATCH und besser)
"""

import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

log = logging.getLogger(__name__)


def send_status_email(pipeline_stats: dict, today: str) -> None:
    trades  = pipeline_stats.get("trades", 0)
    subject = (
        f"Adaptive Asymmetry-Scanner – Trade Empfehlung – {today}"
        if trades > 0
        else f"Adaptive Asymmetry-Scanner – Kein Trade – {today}"
    )
    html = _build_status_email(pipeline_stats, today)
    _send_smtp(subject, html)


def send_email(proposals: list[dict], today: str) -> None:
    # Nur Trades mit Score >= 50 in der Email
    proposals = [p for p in proposals
                 if p.get("trade_score", {}).get("total", 0) >= 50]
    if proposals:
        html    = _build_trade_email(proposals, today)
        subject = f"Adaptive Asymmetry-Scanner – Trade Empfehlung – {today}"
    else:
        html    = _build_status_email({"trades": 0}, today)
        subject = f"Adaptive Asymmetry-Scanner – Kein Trade – {today}"
    _send_smtp(subject, html)


def _build_status_email(stats: dict, today: str) -> str:
    vix         = stats.get("vix")
    trades      = stats.get("trades", 0)
    stop        = stats.get("stop_reason", "")
    header_col  = "#16a34a" if trades > 0 else "#0f172a"
    status_icon = "🎯" if trades > 0 else "📊"
    status_text = "Trade Empfehlung" if trades > 0 else "Kein Trade heute"

    funnel = [
        ("498 Ticker im Universum",                                                     "📋", True),
        (f"{stats.get('candidates',   0)} nach Hard-Filter (Cap>2B, Vol>1M, RV>0.6)", "🔍", stats.get("candidates",   0) > 0),
        (f"{stats.get('prescreened',  0)} nach Prescreening (Claude Haiku)",           "🤖", stats.get("prescreened",  0) > 0),
        (f"{stats.get('roi_precheck', 0)} nach ROI Pre-Check (Fail Fast)",             "💰", stats.get("roi_precheck", 0) > 0),
        (f"{stats.get('analyzed',     0)} nach Deep Analysis + Red Team (Sonnet)",     "🧠", stats.get("analyzed",     0) > 0),
        (f"{stats.get('mismatch_ok',  0)} nach Mismatch-Score",                        "📐", stats.get("mismatch_ok",  0) > 0),
        (f"{stats.get('quick_mc',     0)} nach Quick Monte Carlo (3k, 30d)",           "🎲", stats.get("quick_mc",     0) > 0),
        (f"{stats.get('intraday_ok',  0)} nach Intraday-Delta-Filter",                 "⏱️", stats.get("intraday_ok",  0) > 0),
        (f"{stats.get('final_mc',     0)} nach Final Monte Carlo (10k, adaptive DTE)", "🎯", stats.get("final_mc",     0) > 0),
        (f"{stats.get('rl_scored',    0)} nach RL-Scoring",                            "🤖", stats.get("rl_scored",    0) > 0),
        (f"{stats.get('roi_ok',       0)} nach Options Design + ROI-Gate",             "✅" if stats.get("roi_ok", 0) > 0 else "❌", stats.get("roi_ok", 0) > 0),
        (f"{trades} finale Trade-Vorschläge",                                          "🏆" if trades > 0 else "❌", trades > 0),
    ]

    rows = ""
    for label, icon, active in funnel:
        bg    = "#f0fdf4" if active else "#fef2f2"
        color = "#16a34a" if active else "#dc2626"
        rows += f"""
        <tr>
          <td style="padding:9px 16px;font-size:13px;color:{color};
                     background:{bg};border-bottom:1px solid #e2e8f0;">
            {icon}&nbsp; {label}
          </td>
        </tr>"""

    stop_html = ""
    if stop and trades == 0:
        stop_html = f"""
        <div style="margin:20px 0;padding:14px 16px;background:#fef3c7;
                    border-left:4px solid #f59e0b;border-radius:4px;
                    font-size:13px;">
          <b>Pipeline-Stop:</b> {stop}
        </div>"""

    # Reject Summary falls vorhanden
    rejects     = stats.get("rejects", {})
    reject_html = ""
    if rejects:
        reject_rows = ""
        for reason, count in sorted(rejects.items(), key=lambda x: -x[1]):
            reject_rows += f"""
            <tr>
              <td style="padding:6px 12px;font-size:12px;color:#475569;
                         border-bottom:1px solid #e2e8f0;">{reason}</td>
              <td style="padding:6px 12px;font-size:12px;color:#dc2626;
                         font-weight:600;border-bottom:1px solid #e2e8f0;
                         text-align:right;">{count}x</td>
            </tr>"""
        reject_html = f"""
        <h3 style="margin:20px 0 10px;color:#0f172a;font-size:14px;font-weight:600;">
          Reject-Gründe
        </h3>
        <table style="width:100%;border-collapse:collapse;border-radius:8px;
                      overflow:hidden;border:1px solid #e2e8f0;">
          {reject_rows}
        </table>"""

    vix_str = f"{float(vix):.2f}" if vix else "–"

    return f"""<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;
  margin:0;padding:0;background:#f8fafc;">
<div style="max-width:620px;margin:30px auto;background:#fff;
  border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);">

  <div style="background:{header_col};padding:28px 32px;">
    <div style="font-size:28px;margin-bottom:6px;">{status_icon}</div>
    <div style="color:#fff;font-size:22px;font-weight:bold;">
      Adaptive Asymmetry-Scanner
    </div>
    <div style="color:rgba(255,255,255,0.85);font-size:16px;margin-top:4px;">
      {status_text}
    </div>
    <div style="color:rgba(255,255,255,0.6);font-size:13px;margin-top:6px;">
      {today} &nbsp;·&nbsp; VIX {vix_str} &nbsp;·&nbsp; v8.1
    </div>
  </div>

  <div style="padding:24px 32px;">
    <h3 style="margin:0 0 14px;color:#0f172a;font-size:15px;font-weight:600;">
      Pipeline-Filter heute
    </h3>
    <table style="width:100%;border-collapse:collapse;border-radius:8px;
                  overflow:hidden;border:1px solid #e2e8f0;">
      {rows}
    </table>
    {stop_html}
    {reject_html}
  </div>

  <div style="padding:14px 32px;background:#f8fafc;
              border-top:1px solid #e2e8f0;
              font-size:11px;color:#94a3b8;text-align:center;">
    Adaptive Asymmetry-Scanner v8.1 &nbsp;·&nbsp;
    GitHub Actions &nbsp;·&nbsp;
    {datetime.utcnow().strftime('%H:%M UTC')}
  </div>
</div></body></html>"""


def _build_trade_email(proposals: list[dict], today: str) -> str:
    cards = ""
    for i, p in enumerate(proposals, 1):
        ticker   = p.get("ticker", "?")
        strategy = p.get("strategy", "?")
        da       = p.get("deep_analysis", {}) or {}
        sim      = p.get("simulation", {}) or {}
        option   = p.get("option", {}) or {}
        roi      = p.get("roi_analysis", {}) or {}
        tve      = p.get("time_value_efficiency", {}) or {}
        red_team = da.get("red_team", {}) or {}

        ts         = p.get("trade_score", {}) or {}
        trade_rank = p.get("trade_rank", i)
        ts_total   = ts.get("total", 0)
        ts_grade   = ts.get("grade", "–")
        ts_comps   = ts.get("components", {}) or {}
        sig_pts    = ts_comps.get("signal_quality", 0)
        opt_pts    = ts_comps.get("options_quality", 0)
        risk_pts_v = ts_comps.get("risk_deductions", 0)
        ctx_pts    = ts_comps.get("context_bonus", 0)
        best_for_v = (ts.get("best_argument_for", "") or "")[:700]
        best_ag_v  = (ts.get("best_argument_against", "") or "")[:700]
        rank_badge = f"#{trade_rank} " if trade_rank else ""
        score_color = (
            "#16a34a" if ts_total >= 75 else
            "#ca8a04" if ts_total >= 60 else
            "#ea580c" if ts_total >= 45 else
            "#dc2626"
        )
        grade_parts = (ts_grade or "–").split(" ")
        trade_grade = " ".join(
            p for p in grade_parts if not any(c in p for c in "🟢🟡🟠🔴")
        ).strip() or "–"

        def _s(v, default="–"):
            if v is None: return default
            return str(v)

        direction = _s(da.get("direction"))
        current   = _s(sim.get("current_price"))
        target    = _s(sim.get("target_price"))
        hit_rate  = float(sim.get("hit_rate") or 0)
        mc_n      = _s(sim.get("n_paths"))
        strike    = _s(option.get("strike"))
        expiry    = _s(option.get("expiry"))
        dte       = _s(option.get("dte"))
        bid       = _s(option.get("bid"))
        ask       = _s(option.get("ask"))
        iv_rank   = _s(p.get("iv_rank"))
        roi_net   = roi.get("roi_net")
        vega_loss = roi.get("vega_loss")
        roi_day   = tve.get("roi_per_day_pct")
        dte_tier  = _s(p.get("dte_tier"))
        rt_verdict = _s(red_team.get("red_team_verdict"))
        rt_arg1    = _s(red_team.get("argument_1"))
        macro      = _s(da.get("macro_assessment"))
        impact     = _s(da.get("impact"))
        surprise   = _s(da.get("surprise"))

        roi_color = (
            "#16a34a" if (roi_net or 0) > 0.15 else
            "#ca8a04" if (roi_net or 0) > 0 else
            "#dc2626"
        )

        ev_html = ""
        if roi.get("delta") is not None and roi_net is not None:
            ev = roi["delta"] * roi_net
            ev_color = "#16a34a" if ev > 0.05 else "#ca8a04"
            ev_html = (
                f'<div><b>Expected Value:</b> '
                f'<span style="color:{ev_color};font-weight:600;">{ev:.1%}</span> '
                f'<span style="font-size:11px;color:#94a3b8;">(ROI×Δ)</span></div>'
            )

        cards += f"""
        <div style="border:1px solid #e2e8f0;border-radius:10px;
                    padding:20px;margin-bottom:24px;">

          <div style="display:flex;justify-content:space-between;
                      align-items:center;margin-bottom:12px;">
            <span style="font-size:22px;font-weight:bold;color:#0f172a;">
              {rank_badge}{ticker}
            </span>
            &nbsp;&nbsp;
            <span style="background:{score_color};color:#fff;padding:4px 14px;
                         border-radius:20px;font-size:13px;font-weight:600;">
              {trade_grade}&nbsp;·&nbsp;{ts_total}/100
            </span>
          </div>

          <div style="background:#f8fafc;border-radius:6px;padding:10px 14px;
                      margin-bottom:14px;font-size:12px;color:#334155;
                      border-left:3px solid {score_color};">
            <b>Score:</b>
            Signal {sig_pts:.0f}/40 &nbsp;|&nbsp;
            Optionen {opt_pts:.0f}/30 &nbsp;|&nbsp;
            Risiko {risk_pts_v:.0f} &nbsp;|&nbsp;
            Kontext {ctx_pts:.0f}/30<br>
            <span style="color:#16a34a;font-weight:600;">✅ Für:</span> {best_for_v}<br>
            <span style="color:#dc2626;font-weight:600;">⚠️ Gegen:</span> {best_ag_v}
          </div>

          <div style="display:grid;grid-template-columns:1fr 1fr;
                      gap:10px;font-size:13px;margin-bottom:14px;">
            <div><b>Strategie:</b> {strategy}</div>
            <div><b>Laufzeit:</b> {dte_tier}</div>
            <div><b>Richtung:</b> {direction}</div>
            <div><b>IV-Rank:</b> {iv_rank}%</div>
            <div><b>Preis:</b> ${current}</div>
            <div><b>Ziel:</b> ${target}</div>
            <div><b>Hit-Rate:</b> {hit_rate:.1%} ({mc_n} Pfade)</div>
            <div><b>Impact / Surprise:</b> {impact} / {surprise}</div>
            <div><b>Strike:</b> ${strike}</div>
            <div><b>Expiry:</b> {expiry} ({dte}d)</div>
            <div><b>Bid / Ask:</b> ${bid} / ${ask}</div>
            {f'<div><b>ROI netto:</b> <span style="color:{roi_color};font-weight:600;">{roi_net:.1%}</span></div>' if roi_net is not None else ''}
            {ev_html}
            {f'<div><b>Vega-Loss:</b> {vega_loss:.1%}</div>' if vega_loss is not None else ''}
            {f'<div><b>ROI/Tag:</b> {roi_day:.3f}%</div>' if roi_day is not None else ''}
          </div>

          <div style="background:#f8fafc;border-radius:6px;
                      padding:12px;font-size:12px;color:#475569;">
            <b>🔴 Red Team:</b> {rt_verdict} &nbsp;·&nbsp; {rt_arg1}<br>
            <b>🌍 Makro:</b> {macro}
          </div>
        </div>"""

    return f"""<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;
  background:#f8fafc;margin:0;padding:0;">
<div style="max-width:640px;margin:30px auto;background:#fff;
  border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.1);overflow:hidden;">
  <div style="background:#16a34a;padding:28px 32px;">
    <div style="font-size:28px;margin-bottom:6px;">🎯</div>
    <div style="color:#fff;font-size:22px;font-weight:bold;">
      Adaptive Asymmetry-Scanner
    </div>
    <div style="color:rgba(255,255,255,0.85);font-size:16px;margin-top:4px;">
      Trade Empfehlung — {len(proposals)} Signal(e)
    </div>
    <div style="color:rgba(255,255,255,0.6);font-size:13px;margin-top:6px;">
      {today} &nbsp;·&nbsp; v8.1
    </div>
  </div>
  <div style="padding:24px 32px;">{cards}</div>
  <div style="padding:14px 32px;background:#f8fafc;
              border-top:1px solid #e2e8f0;
              font-size:11px;color:#94a3b8;text-align:center;">
    Kein Finanzberatung. Algorithmisch generiert. &nbsp;·&nbsp;
    Adaptive Asymmetry-Scanner v8.1
  </div>
</div></body></html>"""


def _send_smtp(subject: str, html: str) -> None:
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
