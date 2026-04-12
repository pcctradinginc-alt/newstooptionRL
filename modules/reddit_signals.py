"""
modules/reddit_signals.py – Reddit als zweite News-Quelle

Inspiriert von: Mattbusel/Reddit-Options-Trader-ROT
Quelle: Reddit r/wallstreetbets, r/stocks, r/investing via PRAW oder
        Pushshift-kompatiblem API (kein API-Key nötig via öffentliches JSON).

Features:
  - Credibility-Scoring: Posts mit hohem Score/Upvotes gewichtet stärker
  - Options-Intent-Erkennung: Sucht nach Calls/Puts-Erwähnungen
  - Früher als NewsAPI: Reddit-Posts erscheinen oft 12–24h vor Mainstream-News

Design:
  - Kein PRAW (kein Reddit API-Key nötig) → nutzt reddit.com/r/sub/.json
  - Fallback: Gibt leere Liste zurück (Reddit-Fehler stoppen Pipeline nicht)
  - Credibility-Score: log(upvotes + 1) × comment_ratio

Output pro Ticker:
    {
        "reddit_posts": [{"title": str, "score": int, "credibility": float}],
        "reddit_sentiment": float,   # -1 bis +1 (Erwähnungs-Sentiment)
        "options_intent": float,     # 0–1 (wie stark werden Options diskutiert)
        "mention_count": int
    }
"""

from __future__ import annotations
import logging
import re
import time
from typing import Optional

import numpy as np
import requests

log = logging.getLogger(__name__)

# Subreddits die gescannt werden (Reihenfolge = Priorität)
_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
]

# Regex für Options-Erwähnungen (Calls, Puts, Strike-Preise)
_OPTIONS_PATTERN = re.compile(
    r"\b(call|calls|put|puts|strike|expiry|dte|leaps?|"
    r"\d+[cp]\s*\$?\d+|\$\d+c|\$\d+p)\b",
    re.IGNORECASE,
)

# Bullish/Bearish-Keywords für Sentiment
_BULLISH_KEYWORDS = re.compile(
    r"\b(bull|bullish|moon|rocket|buy|long|calls?|squeeze|"
    r"breakout|beat|upside|strong|outperform)\b",
    re.IGNORECASE,
)
_BEARISH_KEYWORDS = re.compile(
    r"\b(bear|bearish|short|puts?|crash|dump|sell|downside|"
    r"miss|weak|underperform|resistance)\b",
    re.IGNORECASE,
)

_HEADERS = {
    "User-Agent": "newstoption-scanner/3.5 (research bot, no spam)"
}


def fetch_ticker_mentions(
    ticker: str,
    max_posts: int = 50,
    time_filter: str = "day",   # "day" | "week"
) -> dict:
    """
    Sucht in Reddit-Subreddits nach Erwähnungen eines Tickers.

    Args:
        ticker:      Aktien-Ticker (z.B. "AAPL")
        max_posts:   Maximale Anzahl Posts die geprüft werden
        time_filter: Zeitfenster für Reddit-Suche

    Returns:
        Dict mit reddit_posts, reddit_sentiment, options_intent, mention_count
    """
    all_posts = []

    for subreddit in _SUBREDDITS:
        posts = _fetch_subreddit_posts(subreddit, ticker, max_posts, time_filter)
        all_posts.extend(posts)
        if len(all_posts) >= max_posts:
            break

    if not all_posts:
        return _empty_result()

    # Credibility-Scoring und Sentiment-Berechnung
    scored_posts     = [_score_post(p) for p in all_posts]
    reddit_sentiment = _compute_sentiment(scored_posts)
    options_intent   = _compute_options_intent(scored_posts)

    # Top-5 nach Credibility
    top_posts = sorted(scored_posts, key=lambda x: x["credibility"], reverse=True)[:5]

    log.info(
        f"  [{ticker}] Reddit: {len(all_posts)} Posts gefunden, "
        f"sentiment={reddit_sentiment:.2f}, options_intent={options_intent:.2f}"
    )

    return {
        "reddit_posts":    top_posts,
        "reddit_sentiment": round(reddit_sentiment, 4),
        "options_intent":   round(options_intent, 4),
        "mention_count":    len(all_posts),
    }


def _fetch_subreddit_posts(
    subreddit: str,
    ticker: str,
    max_posts: int,
    time_filter: str,
) -> list[dict]:
    """Ruft Posts via Reddit JSON-API ab (kein API-Key nötig)."""
    try:
        # Reddit öffentliches JSON-Endpoint (kein Auth nötig)
        url = (
            f"https://www.reddit.com/r/{subreddit}/search.json"
            f"?q={ticker}&restrict_sr=1&sort=top&t={time_filter}"
            f"&limit={min(max_posts, 25)}"
        )
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()

        data  = resp.json()
        posts = data.get("data", {}).get("children", [])

        result = []
        for post in posts:
            d = post.get("data", {})
            title = d.get("title", "")

            # Nur Posts die den Ticker tatsächlich erwähnen
            if ticker.upper() not in title.upper() and ticker.lower() not in title.lower():
                # Auch in Selftext suchen (kurze Vorschau)
                selftext = d.get("selftext", "")[:200]
                if ticker.upper() not in selftext.upper():
                    continue

            result.append({
                "title":    title,
                "score":    int(d.get("score", 0)),
                "comments": int(d.get("num_comments", 0)),
                "selftext": d.get("selftext", "")[:500],
                "subreddit": subreddit,
            })

        # Rate-Limit schonen
        time.sleep(0.5)
        return result

    except requests.exceptions.RequestException as e:
        log.debug(f"Reddit-Fetch Fehler (r/{subreddit}, {ticker}): {e}")
        return []
    except Exception as e:
        log.debug(f"Reddit-Parse Fehler: {e}")
        return []


def _score_post(post: dict) -> dict:
    """
    Credibility-Score nach ROT-Prinzip:
    log(upvotes + 1) × (1 + comment_ratio)

    Posts mit vielen Upvotes UND Kommentaren sind glaubwürdiger als
    Posts die nur upgevoted wurden (Engagement-Signal).
    """
    score    = max(post["score"], 0)
    comments = max(post["comments"], 0)

    # Comment-Ratio: Wie aktiv ist die Diskussion?
    comment_ratio = min(comments / (score + 1), 2.0)   # Cap bei 2.0

    credibility = np.log1p(score) * (1.0 + comment_ratio)

    text = (post["title"] + " " + post.get("selftext", "")).lower()

    return {
        **post,
        "credibility":    round(float(credibility), 3),
        "has_options":    bool(_OPTIONS_PATTERN.search(text)),
        "bullish_count":  len(_BULLISH_KEYWORDS.findall(text)),
        "bearish_count":  len(_BEARISH_KEYWORDS.findall(text)),
    }


def _compute_sentiment(scored_posts: list[dict]) -> float:
    """
    Gewichteter Sentiment-Score (Credibility-gewichtet).
    Range: -1.0 (stark bearish) bis +1.0 (stark bullish)
    """
    if not scored_posts:
        return 0.0

    total_weight = sum(p["credibility"] for p in scored_posts)
    if total_weight == 0:
        return 0.0

    weighted_sentiment = 0.0
    for post in scored_posts:
        w        = post["credibility"] / total_weight
        bullish  = post["bullish_count"]
        bearish  = post["bearish_count"]
        total_kw = bullish + bearish + 1   # +1 um Division by Zero zu vermeiden
        post_sentiment = (bullish - bearish) / total_kw
        weighted_sentiment += w * post_sentiment

    return float(np.clip(weighted_sentiment, -1.0, 1.0))


def _compute_options_intent(scored_posts: list[dict]) -> float:
    """
    Anteil der Posts die Options diskutieren (0–1).
    Hohes Options-Intent = frühzeitiges Institutional-Signal.
    """
    if not scored_posts:
        return 0.0

    options_posts = sum(1 for p in scored_posts if p.get("has_options", False))
    return float(options_posts / len(scored_posts))


def _empty_result() -> dict:
    return {
        "reddit_posts":    [],
        "reddit_sentiment": 0.0,
        "options_intent":   0.0,
        "mention_count":    0,
    }


def enrich_candidate(candidate: dict) -> dict:
    """
    Fügt Reddit-Signals zu einem Pipeline-Kandidaten hinzu.
    Modifiziert candidate["features"] in-place.

    Wird von data_ingestion.py aufgerufen.
    """
    ticker = candidate.get("ticker", "")
    reddit = fetch_ticker_mentions(ticker)

    # Reddit-Headlines zu News-Liste hinzufügen (für FinBERT)
    reddit_headlines = [p["title"] for p in reddit["reddit_posts"]]
    if "news" not in candidate:
        candidate["news"] = []
    candidate["news"] = reddit_headlines + candidate["news"]   # Reddit zuerst

    # Reddit-Features in candidate speichern
    candidate["reddit"] = reddit

    return candidate
