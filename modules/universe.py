"""
modules/universe.py – Dynamisches Ticker-Universum

Liest den `universe`-Parameter aus config.yaml und liefert
die entsprechende Ticker-Liste. Unterstützt:
  - "sp500_nasdaq100"  → S&P 500 + Nasdaq 100 (dedupliziert)
  - "sp500"            → nur S&P 500
  - "nasdaq100"        → nur Nasdaq 100

Datenquelle (Priorität):
  1. Wikipedia via pandas.read_html() – täglich aktuell, kein API-Key
  2. Vollständiger statischer Fallback – greift wenn Wikipedia offline
"""

from __future__ import annotations
import logging
from functools import lru_cache

log = logging.getLogger(__name__)

# ── Statischer Fallback (Stand April 2026) ────────────────────────────────────

_SP500_STATIC: list[str] = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB",
    "AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","META","AMZN",
    "AMCR","AEE","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI",
    "ANSS","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ",
    "T","ATO","ADSK","AZO","AVB","AVY","AXON","BKR","BALL","BAC","BK","BBWI",
    "BAX","BDX","BRK-B","BBY","BIO","TECH","BIIB","BLK","BX","BA","BMY",
    "AVGO","BR","BRO","BF-B","BLDR","BG","CDNS","CZR","CPT","CPB","COF","CAH",
    "KMX","CCL","CARR","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNX",
    "CDAY","CF","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS",
    "CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CMA","CAG",
    "COP","ED","STZ","CEG","COO","CPRT","GLW","CTVA","CSGP","COST","CTRA","CCI",
    "CSX","CMI","CVS","DHI","DHR","DRI","DVA","DAY","DECK","DE","DAL","DVN",
    "DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DUK","DD",
    "EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","LLY","EMR","ENPH","ETR",
    "EOG","EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EVRG","ES",
    "EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT","FDX",
    "FIS","FITB","FSLR","FE","FI","FMC","F","FTNT","FTV","FOXA","FOX","BEN",
    "FCX","GRMN","IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC",
    "GILD","GPN","GL","GDDY","GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY",
    "HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM",
    "HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE",
    "IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM","JPM","K","KVUE",
    "KDP","KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LH","LRCX","LW",
    "LVS","LDOS","LEN","LII","LLY","LIN","LYV","LKQ","LMT","L","LOW","LULU",
    "LYB","MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC",
    "MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA",
    "MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI",
    "NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC",
    "NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL",
    "OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PANW","PH","PAYX","PAYC",
    "PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL",
    "PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","QCOM","DGX","RL",
    "RJF","RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP",
    "ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG",
    "SWKS","SJM","SNA","SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE",
    "SYK","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL",
    "TDY","TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV",
    "TRMB","TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI",
    "UNH","UHS","VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI",
    "V","VST","VMC","WRB","GWW","WAB","WBA","WMT","DIS","WBD","WM","WAT",
    "WEC","WFC","WELL","WST","WDC","WY","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS",
]

_NASDAQ100_STATIC: list[str] = [
    "ADBE","ADP","ABNB","GOOGL","GOOG","AMZN","AMD","AEP","AMGN","ADI","ANSS",
    "AAPL","AMAT","APP","ARM","ASML","TEAM","ADSK","AZN","BIDU","BIIB","BKNG",
    "AVGO","CDNS","CDW","CHTR","CTAS","CSCO","CCEP","CTSH","CMCSA","CEG","CPRT",
    "CSGP","COST","CRWD","CSX","DDOG","DXCM","FANG","DLTR","EA","EXC","FAST",
    "FTNT","GEHC","GILD","GFS","HON","IDXX","ILMN","INTC","INTU","ISRG","KDP",
    "KLAC","KHC","LRCX","LIN","MELI","META","MCHP","MU","MSFT","MRNA","MDLZ",
    "MDB","MNST","NFLX","NVDA","NXPI","ORLY","ON","ODFL","PCAR","PANW","PAYX",
    "PYPL","PDD","PEP","QCOM","REGN","ROST","SBUX","SNPS","TTWO","TMUS","TSLA",
    "TXN","TTD","VRSK","VRTX","WBA","WDAY","ZS","ZM",
]


def _fetch_sp500() -> list[str]:
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, attrs={"id": "constituents"})
        tickers = (
            tables[0]["Symbol"]
            .str.replace(".", "-", regex=False)
            .str.strip()
            .tolist()
        )
        log.info(f"S&P 500: {len(tickers)} Ticker von Wikipedia geladen.")
        return tickers
    except Exception as e:
        log.warning(f"S&P 500 Wikipedia-Fehler: {e} → statischer Fallback.")
        return []


def _fetch_nasdaq100() -> list[str]:
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        all_tables = pd.read_html(url)
        for table in all_tables:
            if "Ticker" in table.columns:
                tickers = (
                    table["Ticker"]
                    .str.replace(".", "-", regex=False)
                    .str.strip()
                    .tolist()
                )
                log.info(f"Nasdaq 100: {len(tickers)} Ticker von Wikipedia geladen.")
                return tickers
        log.warning("Nasdaq-100: Keine passende Tabelle → statischer Fallback.")
        return []
    except Exception as e:
        log.warning(f"Nasdaq-100 Wikipedia-Fehler: {e} → statischer Fallback.")
        return []


def _clean(tickers: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for t in tickers:
        if not isinstance(t, str):
            continue
        t = t.strip().upper()
        if not t or len(t) > 6:
            continue
        if not all(c.isalpha() or c == "-" for c in t):
            continue
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


@lru_cache(maxsize=1)
def get_universe(universe: str = "") -> list[str]:
    """
    Gibt die konfigurierte Ticker-Liste zurück.
    Wird pro Programmlauf nur einmal aufgerufen (lru_cache).
    """
    if not universe:
        try:
            from modules.config import cfg
            universe = cfg.filters.universe
        except Exception:
            universe = "sp500_nasdaq100"

    log.info(f"Lade Ticker-Universum: '{universe}'")

    sp500:  list[str] = []
    ndq100: list[str] = []

    if universe in ("sp500", "sp500_nasdaq100"):
        sp500 = _fetch_sp500()
        if not sp500:
            sp500 = list(_SP500_STATIC)

    if universe in ("nasdaq100", "sp500_nasdaq100"):
        ndq100 = _fetch_nasdaq100()
        if not ndq100:
            ndq100 = list(_NASDAQ100_STATIC)

    combined = _clean(sp500 + ndq100)

    log.info(
        f"Universum '{universe}': {len(combined)} Ticker "
        f"(S&P500={len(sp500)}, Nasdaq100={len(ndq100)}, "
        f"nach Deduplizierung={len(combined)})"
    )
    return combined
