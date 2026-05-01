"""
Global Financial Markets Dashboard
Streamlit app that mirrors the layout of the PDF dashboard you uploaded.

Data sources (all free):
  - FRED (fredapi)      -> policy rates, macro indicators, credit-spread indices
  - Stooq (pandas-datareader) -> non-US sovereign yields (Germany / Japan / UK / China 2y & 10y)
  - yfinance            -> equities, individual names, FX, commodities, VIX, US Treasury yields

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st

# ---------- Page config ----------
st.set_page_config(
    page_title="Global Financial Markets Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- API keys ----------
# Reads FRED_API_KEY from Streamlit secrets (when deployed) or env var (locally).
# Wrapped in try/except so a missing secrets.toml doesn't crash the app.
def _load_fred_key() -> str:
    try:
        return st.secrets.get("FRED_API_KEY", os.environ.get("FRED_API_KEY", ""))
    except Exception:
        return os.environ.get("FRED_API_KEY", "")


FRED_API_KEY = _load_fred_key()

# ---------- Caching wrappers ----------
# Cache for 10 minutes so we don't hammer the APIs during a session.
CACHE_TTL = 60 * 10


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fred_series(series_id: str, start: Optional[dt.date] = None) -> pd.Series:
    """Pull a single FRED series. Tries the free public fredgraph CSV endpoint
    first (no auth, very reliable). Falls back to the fredapi package (requires
    key) if the CSV fetch fails.
    """
    # Path 1: public fredgraph CSV (no auth required, most reliable)
    try:
        import io
        import requests

        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        if start is not None:
            url += f"&cosd={start.isoformat()}"
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and r.text.startswith("observation_date"):
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["date", "val"]
            df["date"] = pd.to_datetime(df["date"])
            df["val"] = pd.to_numeric(df["val"], errors="coerce")
            df = df.dropna()
            if not df.empty:
                return pd.Series(
                    df["val"].values, index=df["date"]
                ).sort_index()
    except Exception:
        pass

    # Path 2: fredapi (fallback, requires key)
    if FRED_API_KEY:
        try:
            from fredapi import Fred

            fred = Fred(api_key=FRED_API_KEY)
            s = fred.get_series(series_id, observation_start=start)
            return s.dropna()
        except Exception:
            return pd.Series(dtype=float)

    return pd.Series(dtype=float)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def yf_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Pull a price history from Yahoo Finance."""
    try:
        import yfinance as yf

        df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        return df
    except Exception:
        return pd.DataFrame()


# Shared cached Yahoo helpers (used by WACC, Comps, Asset Comparison, etc.)
# These wrap the slow yfinance endpoints so they only get hit once per
# ticker per cache window.
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def yf_info(ticker: str) -> dict:
    try:
        import yfinance as yf

        return dict(yf.Ticker(ticker).info or {})
    except Exception:
        return {}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def yf_balance_sheet(ticker: str) -> pd.DataFrame:
    try:
        import yfinance as yf

        df = yf.Ticker(ticker).balance_sheet
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def yf_income_stmt(ticker: str) -> pd.DataFrame:
    try:
        import yfinance as yf

        df = yf.Ticker(ticker).income_stmt
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _bs_get(bs: pd.DataFrame, *keys) -> Optional[float]:
    """Return the most-recent value of the first matching balance-sheet row."""
    if bs is None or bs.empty:
        return None
    for k in keys:
        if k in bs.index:
            try:
                v = bs.loc[k].dropna()
                if not v.empty:
                    return float(v.iloc[0])
            except Exception:
                continue
    return None


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def muni_etf_yield_pct() -> Optional[float]:
    """MUB's distribution yield as a percent. Computed from trailing 12-month
    cash dividends divided by the latest close \u2014 avoids the slow
    yfinance `.info` endpoint and only uses already-cached price history.
    """
    try:
        import yfinance as yf

        tk = yf.Ticker("MUB")
        # Dividends call is much faster than .info and rarely hangs.
        divs = tk.dividends
        hist = yf_history("MUB", period="2y")
        if divs is None or divs.empty or hist.empty:
            return None
        if isinstance(divs.index, pd.DatetimeIndex) and divs.index.tz is not None:
            divs.index = divs.index.tz_localize(None)
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
        ttm = float(divs[divs.index >= cutoff].sum())
        last_price = float(hist["Close"].dropna().iloc[-1])
        if last_price <= 0 or ttm <= 0:
            return None
        return (ttm / last_price) * 100
    except Exception:
        return None


WSJ_ENTITLEMENT = "cecc4267a0194af89ca343805a3e57af"


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def wsj_bond_series(ticker: str) -> pd.Series:
    """Pull a 2-year daily history for a benchmark sovereign bond yield from
    the MarketWatch/WSJ public Michelangelo timeseries endpoint.
    ticker examples: 'TMBMKDE-10Y' (DE 10y), 'TMBMKJP-02Y', 'TMBMKGB-10Y',
    'TMBMKIT-10Y', 'TMUBMUSD10Y'.
    """
    import json as _json
    import urllib.parse
    import requests

    if ticker.startswith("TMUBMUSD"):
        key = f"BOND/BX//{ticker}"
    else:
        key = f"BOND/BX//{ticker}"
    payload = {
        "Step": "P1D",
        "TimeFrame": "P2Y",
        "StartDate": None,
        "EndDate": None,
        "EntitlementToken": WSJ_ENTITLEMENT,
        "IncludeMockTicks": False,
        "FilterNullSlots": False,
        "FilterClosedPoints": True,
        "IncludeOriginalTimeStamps": False,
        "IncludeOfficialClose": True,
        "ExcludeNonOfficialPriceTypes": True,
        "Series": [
            {
                "Key": key,
                "Dialect": "Charting",
                "Kind": "Ticker",
                "SeriesId": "s1",
                "DataTypes": ["Last"],
                "Indicators": [],
            }
        ],
    }
    q = urllib.parse.quote(_json.dumps(payload, separators=(",", ":")))
    url = (
        f"https://api-secure.wsj.net/api/michelangelo/timeseries/history?"
        f"json={q}&ckey=cecc4267a0"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Dylan2010.EntitlementToken": WSJ_ENTITLEMENT,
        "Accept": "application/json",
        "Origin": "https://www.marketwatch.com",
        "Referer": "https://www.marketwatch.com/",
    }
    try:
        r = requests.get(url, timeout=20, headers=headers)
        if r.status_code != 200:
            return pd.Series(dtype=float)
        data = r.json()
        series = data.get("Series", [{}])[0]
        points = series.get("DataPoints", [])
        ticks = data.get("TimeInfo", {}).get("Ticks", [])
        if not points or not ticks or len(points) != len(ticks):
            return pd.Series(dtype=float)
        idx = pd.to_datetime(ticks, unit="ms")
        vals = [p[0] if p and p[0] is not None else None for p in points]
        s = pd.Series(vals, index=idx).astype(float).dropna().sort_index()
        return s
    except Exception:
        return pd.Series(dtype=float)


# ---------- Helpers ----------
@dataclass
class Quote:
    last: Optional[float] = None
    chg_1w: Optional[float] = None  # percent for prices, bp for yields
    chg_ytd: Optional[float] = None  # percent for prices, bp for yields
    hi_52w: Optional[float] = None


def latest_and_changes_from_series(s: pd.Series, is_yield: bool) -> Quote:
    """Compute last, 1-week change, and YTD change from a series.
    For yields, changes are returned in basis points (bps).
    For prices, changes are returned in percent.
    """
    if s is None or s.empty:
        return Quote()
    s = s.dropna().sort_index()
    # Strip any timezone info so comparisons with naive Timestamps don't blow up.
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    last = float(s.iloc[-1])

    today = s.index[-1]
    one_week_ago = today - pd.Timedelta(days=7)
    s_week = s[s.index <= one_week_ago]
    week_val = float(s_week.iloc[-1]) if not s_week.empty else None

    year_start = pd.Timestamp(year=today.year, month=1, day=1)
    s_ytd = s[s.index < year_start]
    ytd_val = float(s_ytd.iloc[-1]) if not s_ytd.empty else None

    if is_yield:
        chg_1w = (last - week_val) * 100 if week_val is not None else None
        chg_ytd = (last - ytd_val) * 100 if ytd_val is not None else None
    else:
        chg_1w = (last / week_val - 1) * 100 if week_val else None
        chg_ytd = (last / ytd_val - 1) * 100 if ytd_val else None

    cutoff_52w = today - pd.Timedelta(days=365)
    s_52w = s[s.index >= cutoff_52w]
    hi_52w = float(s_52w.max()) if not s_52w.empty else None

    return Quote(last=last, chg_1w=chg_1w, chg_ytd=chg_ytd, hi_52w=hi_52w)


def quote_from_yf(ticker: str, is_yield: bool = False) -> Quote:
    df = yf_history(ticker, period="2y")
    if df.empty:
        return Quote()
    return latest_and_changes_from_series(df["Close"], is_yield=is_yield)


def quote_from_wsj(ticker: str, is_yield: bool = True) -> Quote:
    s = wsj_bond_series(ticker)
    return latest_and_changes_from_series(s, is_yield=is_yield)


def quote_from_fred(series_id: str, is_yield: bool = True) -> Quote:
    s = fred_series(series_id, start=dt.date(dt.date.today().year - 2, 1, 1))
    return latest_and_changes_from_series(s, is_yield=is_yield)


# ---------- Formatters ----------
def fmt_yield(v: Optional[float]) -> str:
    return f"{v:.2f}%" if v is not None else "—"


def fmt_price(v: Optional[float], decimals: int = 2, prefix: str = "") -> str:
    if v is None:
        return "—"
    return f"{prefix}{v:,.{decimals}f}"


def fmt_bp(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:+.0f}"


def fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:+.2f}%"


def color(v: Optional[float], invert: bool = False) -> str:
    """Return CSS color for positive/negative values."""
    if v is None:
        return "color: #888"
    if invert:
        v = -v
    if v > 0:
        return "color: #1a8a3a; font-weight: 600"  # green
    if v < 0:
        return "color: #c0312c; font-weight: 600"  # red
    return "color: #333"


# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container { padding-top: 4rem; padding-bottom: 1rem; max-width: 1400px; }
    h1.dash-title {
        background-color: #1f3864;
        color: white;
        text-align: center;
        padding: 12px 0;
        margin-top: 0;
        font-size: 22px;
        border-radius: 4px;
    }
    .section-header {
        background-color: #1f3864;
        color: white;
        font-weight: 700;
        padding: 6px 10px;
        margin-top: 10px;
        margin-bottom: 4px;
        font-size: 15px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .sub-header {
        background-color: #2e5597;
        color: white;
        font-weight: 600;
        padding: 4px 10px;
        font-size: 14px;
        margin-top: 4px;
    }
    table.dash { width: 100%; border-collapse: collapse; font-size: 13px; }
    table.dash th {
        text-align: right;
        padding: 5px 8px;
        color: white;
        background-color: #4472c4;
        font-weight: 600;
        border-bottom: 1px solid #1f3864;
    }
    table.dash td { padding: 3px 8px; border-bottom: 1px solid #f0f0f0; }
    table.dash td.label { text-align: left; }
    table.dash td.num { text-align: right; font-variant-numeric: tabular-nums; }
    .footer { color: #666; font-size: 12px; padding: 8px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"<h1 class='dash-title'>Global Financial Markets Dashboard — "
    f"{dt.date.today().strftime('%B %d, %Y')}</h1>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Markets Dashboard",
    "Beta Calculator",
    "WACC / Cost of Capital",
    "Trading Comps",
    "Yield Curves",
    "Return Comparison",
    "Economic Calendar",
])


# ---------- Parallel prefetch ----------
# Every dashboard cell ultimately calls one of three cached functions:
# fred_series, wsj_bond_series, or yf_history. By firing them all
# concurrently up-front, the per-cell render calls become cache hits and
# the page paints in a fraction of the time it would take serially.
def _prefetch_dashboard_data() -> None:
    from concurrent.futures import ThreadPoolExecutor

    today = dt.date.today()
    fred_with_start = dt.date(today.year - 2, 1, 1)

    # FRED IDs called with the 2-year start (yield-like series rendered
    # via quote_from_fred -> latest_and_changes_from_series).
    fred_yields = ["DGS2", "DGS5", "DGS7", "DGS10", "DGS30"]
    # FRED IDs called with no start (policy + credit single-value reads).
    fred_levels = [
        "DFEDTARU", "DFEDTARL", "ECBMRRFR", "IRSTCI01JPM156N",
        "IUDSOIA", "IR3TIB01CNM156N",
        "MORTGAGE30US", "DGS10",
        "BAMLC0A0CMEY", "BAMLC0A0CM",
        "BAMLH0A0HYM2EY", "BAMLH0A0HYM2",
    ]
    wsj_tickers = [
        "TMBMKDE-02Y", "TMBMKDE-05Y", "TMBMKDE-10Y", "TMBMKDE-30Y",
        "TMBMKJP-02Y", "TMBMKJP-05Y", "TMBMKJP-10Y", "TMBMKJP-30Y",
        "TMBMKGB-02Y", "TMBMKGB-05Y", "TMBMKGB-10Y", "TMBMKGB-30Y",
        "TMBMKIT-10Y",
    ]
    yf_tickers = [
        "^GSPC", "^DJI", "^IXIC", "^RUT",
        "^STOXX50E", "^N225", "^FTSE", "000001.SS",
        "^VIX",
        "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
        "DX-Y.NYB", "EURUSD=X", "JPY=X", "GBPUSD=X", "USDCNY=X", "BTC-USD",
        "GC=F", "BZ=F", "HG=F", "ZC=F", "NG=F",
        "MUB",
    ]

    tasks = []
    for sid in fred_yields:
        tasks.append((fred_series, (sid,), {"start": fred_with_start}))
    for sid in fred_levels:
        tasks.append((fred_series, (sid,), {}))
    for tkr in wsj_tickers:
        tasks.append((wsj_bond_series, (tkr,), {}))
    for tkr in yf_tickers:
        tasks.append((yf_history, (tkr,), {"period": "2y"}))
    tasks.append((muni_etf_yield_pct, (), {}))

    # 16 workers is a good balance — Yahoo and FRED both tolerate it,
    # and most of the time is spent waiting on network I/O.
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(fn, *args, **kwargs) for fn, args, kwargs in tasks]
        for f in futures:
            try:
                f.result(timeout=25)
            except Exception:
                # Individual failures fall through; the render path will
                # show "\u2014" for any series that didn't come back.
                pass


with tab1:
    with st.spinner("Loading market data\u2026"):
        _prefetch_dashboard_data()

    if not FRED_API_KEY:
        st.warning(
            "No FRED API key found. Policy rates, macro indicators, and credit indices "
            "will be blank. Add `FRED_API_KEY` to `.streamlit/secrets.toml` (local) or "
            "to your Streamlit Cloud app's secrets (deployed). It's free at "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )


    # =============================================================================
    # RATES + POLICY (top half)
    # =============================================================================
    left, right = st.columns(2)


    def render_yield_block(title: str, rows: list[tuple[str, Quote]]) -> str:
        """Build an HTML table for a yield block (rate label + last + Δ1w + ΔYTD)."""
        html = [f"<div class='sub-header'>{title}</div>"]
        html.append("<table class='dash'><tr><th></th><th>Last</th><th>1 Week Change</th><th>YTD Change</th></tr>")
        for label, q in rows:
            html.append(
                f"<tr>"
                f"<td class='label'>{label}</td>"
                f"<td class='num'>{fmt_yield(q.last)}</td>"
                f"<td class='num' style='{color(q.chg_1w)}'>{fmt_bp(q.chg_1w)}</td>"
                f"<td class='num' style='{color(q.chg_ytd)}'>{fmt_bp(q.chg_ytd)}</td>"
                f"</tr>"
            )
        html.append("</table>")
        return "".join(html)


    # ---------- Left column: Rates ----------
    with left:
        st.markdown("<div class='section-header'>Rates</div>", unsafe_allow_html=True)

        # United States — pull from FRED for accuracy
        us_rows = [
            ("2y", quote_from_fred("DGS2")),
            ("5y", quote_from_fred("DGS5")),
            ("7y", quote_from_fred("DGS7")),
            ("10y", quote_from_fred("DGS10")),
            ("30y", quote_from_fred("DGS30")),
        ]
        st.markdown(render_yield_block("United States", us_rows), unsafe_allow_html=True)

        # Germany — WSJ benchmark Bund yields (daily, real numbers)
        de_rows = [
            ("2y", quote_from_wsj("TMBMKDE-02Y")),
            ("5y", quote_from_wsj("TMBMKDE-05Y")),
            ("10y", quote_from_wsj("TMBMKDE-10Y")),
            ("30y", quote_from_wsj("TMBMKDE-30Y")),
        ]
        st.markdown(render_yield_block("Germany", de_rows), unsafe_allow_html=True)

        # Japan
        jp_rows = [
            ("2y", quote_from_wsj("TMBMKJP-02Y")),
            ("5y", quote_from_wsj("TMBMKJP-05Y")),
            ("10y", quote_from_wsj("TMBMKJP-10Y")),
            ("30y", quote_from_wsj("TMBMKJP-30Y")),
        ]
        st.markdown(render_yield_block("Japan", jp_rows), unsafe_allow_html=True)

        # United Kingdom
        uk_rows = [
            ("2y", quote_from_wsj("TMBMKGB-02Y")),
            ("5y", quote_from_wsj("TMBMKGB-05Y")),
            ("10y", quote_from_wsj("TMBMKGB-10Y")),
            ("30y", quote_from_wsj("TMBMKGB-30Y")),
        ]
        st.markdown(render_yield_block("United Kingdom", uk_rows), unsafe_allow_html=True)


    # ---------- Right column: Policy / Curves / Credit ----------
    with right:
        st.markdown("<div class='section-header'>Policy / Curves / Credit</div>", unsafe_allow_html=True)

        # Policy rates
        policy_pairs = [
            ("Fed funds (upper)", "DFEDTARU"),
            ("Fed funds (lower)", "DFEDTARL"),
            ("ECB MRO", "ECBMRRFR"),
            # IRSTCI01JPM156N = Japan immediate rate (monthly), tracks BOJ policy rate.
            ("BOJ Policy Rate (mo)", "IRSTCI01JPM156N"),
            # IUDSOIA = SONIA, BoE's official policy benchmark since 2018 (daily).
            ("BoE SONIA (≈Bank Rate)", "IUDSOIA"),
            ("CN Loan Prime Rate (1y)", "IR3TIB01CNM156N"),
        ]
        rows_html = ["<div class='sub-header'>Policy</div>"]
        rows_html.append("<table class='dash'><tr><th></th><th>Current</th></tr>")
        for label, sid in policy_pairs:
            s = fred_series(sid)
            last = float(s.iloc[-1]) if not s.empty else None
            rows_html.append(
                f"<tr><td class='label'>{label}</td>"
                f"<td class='num'>{fmt_yield(last)}</td></tr>"
            )
        rows_html.append("</table>")
        st.markdown("".join(rows_html), unsafe_allow_html=True)

        # Curve spreads (current 2s/10s, 5s/30s in bps)
        def spread_bps(short: Quote, long: Quote) -> Optional[float]:
            if short.last is None or long.last is None:
                return None
            return (long.last - short.last) * 100

        us2, us5, _, us10, us30 = (r[1] for r in us_rows)
        de2, de5, de10, de30 = (r[1] for r in de_rows)
        jp2, jp5, jp10, jp30 = (r[1] for r in jp_rows)
        uk2, uk5, uk10, uk30 = (r[1] for r in uk_rows)

        curves = [
            ("United States", spread_bps(us2, us10), spread_bps(us5, us30)),
            ("Germany", spread_bps(de2, de10), spread_bps(de5, de30)),
            ("Japan", spread_bps(jp2, jp10), spread_bps(jp5, jp30)),
            ("United Kingdom", spread_bps(uk2, uk10), spread_bps(uk5, uk30)),
        ]
        html = ["<div class='sub-header'>Curves (Bps)</div>"]
        html.append("<table class='dash'><tr><th></th><th>2s/10s</th><th>5s/30s</th></tr>")
        for label, s210, s530 in curves:
            html.append(
                f"<tr><td class='label'>{label}</td>"
                f"<td class='num'>{fmt_bp(s210)}</td>"
                f"<td class='num'>{fmt_bp(s530)}</td></tr>"
            )
        html.append("</table>")
        st.markdown("".join(html), unsafe_allow_html=True)

        # Credit (yields + option-adjusted spreads in bps)
        # Each entry: (label, yield_fred_id, spread_fred_id_or_None, custom_spread_fn_or_None)
        dgs10_series = fred_series("DGS10")
        dgs10_last = float(dgs10_series.iloc[-1]) if not dgs10_series.empty else None

        def mortgage_spread_bps() -> Optional[float]:
            m = fred_series("MORTGAGE30US")
            if m.empty or dgs10_last is None:
                return None
            return (float(m.iloc[-1]) - dgs10_last) * 100

        def ger_ita_spread_bps() -> Optional[float]:
            ita10 = quote_from_wsj("TMBMKIT-10Y")
            if ita10.last is None or de10.last is None:
                return None
            return (ita10.last - de10.last) * 100

        # Muni: ICE BofA muni indices were discontinued on FRED. We use the
        # iShares National Muni ETF (MUB) trailing-12-month distribution yield
        # as a proxy. Computed from cached dividends + price history so it
        # avoids the slow yfinance `.info` call.
        def muni_yield_and_spread() -> tuple[Optional[float], Optional[float]]:
            yld_pct = muni_etf_yield_pct()
            spread_bp = (
                (yld_pct - dgs10_last) * 100
                if (yld_pct is not None and dgs10_last is not None)
                else None
            )
            return yld_pct, spread_bp

        credit_rows = [
            ("Mortgage 30y (Freddie)", "MORTGAGE30US", None, mortgage_spread_bps),
            ("ICE BofA US Corp Index Yield", "BAMLC0A0CMEY", "BAMLC0A0CM", None),
            ("ICE BofA US HY Index Yield", "BAMLH0A0HYM2EY", "BAMLH0A0HYM2", None),
            ("Muni ETF Yield (MUB SEC)", "__MUNI__", None, None),
            ("GER-ITA 10y spread", None, None, ger_ita_spread_bps),
        ]
        html = ["<div class='sub-header'>Credit</div>"]
        html.append("<table class='dash'><tr><th></th><th>Last</th><th>Spread (Bps)</th></tr>")
        for label, yld_id, oas_id, spread_fn in credit_rows:
            if yld_id == "__MUNI__":
                last, spread_bp = muni_yield_and_spread()
            else:
                s = fred_series(yld_id) if yld_id else pd.Series(dtype=float)
                last = float(s.iloc[-1]) if not s.empty else None
                if spread_fn is not None:
                    spread_bp = spread_fn()
                elif oas_id is not None:
                    oas = fred_series(oas_id)
                    # FRED OAS values are already in percent; multiply by 100 for bps.
                    spread_bp = float(oas.iloc[-1]) * 100 if not oas.empty else None
                else:
                    spread_bp = None
            html.append(
                f"<tr><td class='label'>{label}</td>"
                f"<td class='num'>{fmt_yield(last)}</td>"
                f"<td class='num'>{fmt_bp(spread_bp)}</td></tr>"
            )
        html.append("</table>")
        st.markdown("".join(html), unsafe_allow_html=True)


    # =============================================================================
    # EQUITIES
    # =============================================================================
    st.markdown("<div class='section-header'>Equities</div>", unsafe_allow_html=True)
    eq_left, eq_right = st.columns(2)


    def render_price_block(title: str, rows: list[tuple[str, str, int]], show_52w: bool = False) -> str:
        """rows = list of (display_name, ticker, decimals)."""
        html = [f"<div class='sub-header'>{title}</div>"]
        if show_52w:
            html.append(
                "<table class='dash'><tr><th></th><th>Last</th><th>1 Week Change</th><th>YTD Change</th><th>52-Wk Hi</th></tr>"
            )
        else:
            html.append(
                "<table class='dash'><tr><th></th><th>Last</th><th>1 Week Change</th><th>YTD Change</th></tr>"
            )
        for label, ticker, decimals in rows:
            q = quote_from_yf(ticker, is_yield=False)
            prefix = "$" if show_52w else ""
            cells = (
                f"<td class='label'>{label}</td>"
                f"<td class='num'>{fmt_price(q.last, decimals, prefix)}</td>"
                f"<td class='num' style='{color(q.chg_1w)}'>{fmt_pct(q.chg_1w)}</td>"
                f"<td class='num' style='{color(q.chg_ytd)}'>{fmt_pct(q.chg_ytd)}</td>"
            )
            if show_52w:
                cells += f"<td class='num'>{fmt_price(q.hi_52w, decimals, prefix)}</td>"
            html.append(f"<tr>{cells}</tr>")
        html.append("</table>")
        return "".join(html)


    with eq_left:
        us_indices = [
            ("S&P 500", "^GSPC", 0),
            ("DJIA", "^DJI", 0),
            ("Nasdaq", "^IXIC", 0),
            ("Russell 2000", "^RUT", 0),
        ]
        st.markdown(render_price_block("United States", us_indices), unsafe_allow_html=True)

        global_indices = [
            ("Eurostoxx 50", "^STOXX50E", 0),
            ("Nikkei 225", "^N225", 0),
            ("FTSE 100", "^FTSE", 0),
            ("SHCOMP", "000001.SS", 0),
        ]
        st.markdown(render_price_block("Global", global_indices), unsafe_allow_html=True)

        vol = [("VIX", "^VIX", 2)]
        st.markdown(render_price_block("Volatility", vol), unsafe_allow_html=True)


    with eq_right:
        names = [
            ("AAPL", "AAPL", 2),
            ("NVDA", "NVDA", 2),
            ("MSFT", "MSFT", 2),
            ("GOOGL", "GOOGL", 2),
            ("AMZN", "AMZN", 2),
            ("META", "META", 2),
            ("TSLA", "TSLA", 2),
        ]
        st.markdown(render_price_block("Individual Names", names, show_52w=True), unsafe_allow_html=True)


    # =============================================================================
    # OTHER (Currencies + Commodities)
    # =============================================================================
    st.markdown("<div class='section-header'>Other</div>", unsafe_allow_html=True)
    ot_left, ot_right = st.columns(2)

    with ot_left:
        fx = [
            ("DXY Index", "DX-Y.NYB", 2),
            ("EURUSD", "EURUSD=X", 4),
            ("USDJPY", "JPY=X", 2),
            ("GBPUSD", "GBPUSD=X", 4),
            ("USDCNY", "USDCNY=X", 4),
            ("BTCUSD", "BTC-USD", 0),
        ]
        st.markdown(render_price_block("Currencies", fx), unsafe_allow_html=True)

    with ot_right:
        commodities = [
            ("Gold", "GC=F", 2),
            ("Brent Crude", "BZ=F", 2),
            ("Copper", "HG=F", 4),
            ("Corn", "ZC=F", 2),
            ("HH Natural Gas", "NG=F", 3),
        ]
        st.markdown(render_price_block("Commodities", commodities), unsafe_allow_html=True)

with tab2:
    st.markdown("<h1 class='dash-title' style='font-size:18px;'>Beta Calculator</h1>", unsafe_allow_html=True)
    st.caption(
        "Beta is computed by regressing the stock's periodic returns on the benchmark's. "
        "Formula: \u03b2 = Cov(stock, benchmark) / Var(benchmark). Quarterly betas need a longer window to be meaningful."
    )

    bc1, bc2, bc3, bc4 = st.columns([1.2, 1, 1, 1])
    with bc1:
        beta_ticker = st.text_input("Stock ticker or symbol", value="AAPL", key="beta_ticker").strip().upper()
    with bc2:
        beta_benchmark_label = st.selectbox(
            "Benchmark",
            [
                "S&P 500 (^GSPC)",
                "Nasdaq 100 (^NDX)",
                "Dow Jones (^DJI)",
                "Russell 2000 (^RUT)",
                "MSCI World (URTH)",
                "Custom",
            ],
            index=0,
            key="beta_bench_choice",
        )
    with bc3:
        beta_timeframe = st.selectbox(
            "Timeframe",
            ["1 year", "2 years", "3 years", "5 years", "10 years"],
            index=1,
            key="beta_timeframe",
        )
    with bc4:
        beta_frequency = st.selectbox(
            "Frequency",
            ["Daily", "Weekly", "Monthly", "Quarterly"],
            index=0,
            key="beta_frequency",
        )

    if beta_benchmark_label == "Custom":
        beta_benchmark = st.text_input("Custom benchmark ticker", value="^GSPC", key="beta_bench_custom").strip().upper()
    else:
        beta_benchmark = beta_benchmark_label.split("(")[-1].rstrip(")").strip()

    run_beta = st.button("Calculate beta", type="primary", key="beta_run")

    def _resample_returns(close: pd.Series, freq: str) -> pd.Series:
        s = close.dropna().sort_index()
        if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        if freq == "Daily":
            r = s.pct_change()
        elif freq == "Weekly":
            r = s.resample("W-FRI").last().pct_change()
        elif freq == "Monthly":
            r = s.resample("ME").last().pct_change()
        elif freq == "Quarterly":
            r = s.resample("QE").last().pct_change()
        else:
            r = s.pct_change()
        return r.dropna()

    def _yf_period_for(timeframe: str) -> str:
        return {
            "1 year": "1y",
            "2 years": "2y",
            "3 years": "3y",
            "5 years": "5y",
            "10 years": "10y",
        }.get(timeframe, "2y")

    if run_beta:
        if not beta_ticker or not beta_benchmark:
            st.error("Please enter both a stock ticker and a benchmark.")
        else:
            period = _yf_period_for(beta_timeframe)
            with st.spinner(f"Fetching {period} of price history for {beta_ticker} and {beta_benchmark}\u2026"):
                stock_df = yf_history(beta_ticker, period=period)
                bench_df = yf_history(beta_benchmark, period=period)

            if stock_df.empty:
                st.error(f"No price data found for **{beta_ticker}**. Double-check the ticker.")
            elif bench_df.empty:
                st.error(f"No price data found for benchmark **{beta_benchmark}**.")
            else:
                stock_ret = _resample_returns(stock_df["Close"], beta_frequency)
                bench_ret = _resample_returns(bench_df["Close"], beta_frequency)

                joined = pd.concat([stock_ret, bench_ret], axis=1, join="inner").dropna()
                joined.columns = ["stock", "bench"]

                # Need a reasonable sample size for a meaningful beta
                min_obs = {"Daily": 30, "Weekly": 12, "Monthly": 6, "Quarterly": 4}.get(beta_frequency, 10)
                if len(joined) < min_obs:
                    st.error(
                        f"Only {len(joined)} overlapping {beta_frequency.lower()} returns are available — "
                        f"not enough to compute a meaningful beta. Try a longer timeframe or higher frequency."
                    )
                else:
                    cov = joined["stock"].cov(joined["bench"])
                    var = joined["bench"].var()
                    beta_val = cov / var if var else None
                    corr = joined["stock"].corr(joined["bench"])
                    r_squared = corr ** 2 if corr is not None else None
                    alpha = joined["stock"].mean() - (beta_val or 0) * joined["bench"].mean()

                    # Annualize alpha for display
                    periods_per_year = {"Daily": 252, "Weekly": 52, "Monthly": 12, "Quarterly": 4}.get(beta_frequency, 252)
                    alpha_annual_pct = alpha * periods_per_year * 100 if alpha is not None else None

                    # We deliberately skip the slow yfinance .info company-name lookup —
                    # that endpoint can take 10\u201330s and was the cause of long loads.
                    st.markdown(
                        f"<div class='sub-header'>{beta_ticker} &nbsp;vs&nbsp; {beta_benchmark} &nbsp;|&nbsp; "
                        f"{beta_timeframe} of {beta_frequency.lower()} returns</div>",
                        unsafe_allow_html=True,
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Beta (\u03b2)", f"{beta_val:.3f}" if beta_val is not None else "\u2014")
                    m2.metric("R\u00b2", f"{r_squared:.3f}" if r_squared is not None else "\u2014")
                    m3.metric("Observations", f"{len(joined)}")
                    m4.metric(
                        f"Alpha (annualized)",
                        f"{alpha_annual_pct:+.2f}%" if alpha_annual_pct is not None else "\u2014",
                    )

                    # Scatter of bench vs stock returns + regression line
                    try:
                        import altair as alt

                        plot_df = joined.reset_index().rename(columns={"index": "date"})
                        # Some yfinance frames have a Date or Datetime index
                        date_col = plot_df.columns[0]

                        scatter = (
                            alt.Chart(plot_df)
                            .mark_circle(size=45, opacity=0.55, color="#2e5597")
                            .encode(
                                x=alt.X("bench:Q", title=f"{beta_benchmark} return"),
                                y=alt.Y("stock:Q", title=f"{beta_ticker} return"),
                                tooltip=[
                                    alt.Tooltip(f"{date_col}:T", title="Date"),
                                    alt.Tooltip("stock:Q", format=".2%", title=f"{beta_ticker}"),
                                    alt.Tooltip("bench:Q", format=".2%", title=f"{beta_benchmark}"),
                                ],
                            )
                        )
                        regression = scatter.transform_regression("bench", "stock").mark_line(color="#c0312c", strokeWidth=2)
                        st.altair_chart(scatter + regression, use_container_width=True)
                    except Exception:
                        st.line_chart(joined.cumsum())

                    with st.expander("Show return data"):
                        display = joined.copy()
                        display.columns = [f"{beta_ticker} return", f"{beta_benchmark} return"]
                        display.index.name = "Period end"
                        st.dataframe(display.style.format("{:+.2%}"), use_container_width=True)



# =============================================================================
# Tab 3: WACC / Cost of Capital
# =============================================================================

# --- Damodaran synthetic credit rating tables ----------------------------------
# Source: Aswath Damodaran, NYU Stern, "Ratings, Interest Coverage Ratios and
# Default Spread" (updated annually at pages.stern.nyu.edu/~adamodar/).
# Default spreads below are in percentage points over the risk-free rate and
# reflect Damodaran's January 2025 update for US non-financial firms.
# When Damodaran refreshes these tables, update the numbers here.
#
# ICR thresholds differ for "large" (market cap > $5B) vs. "small/risky" firms.
DAMODARAN_ICR_LARGE = [
    # (min_ICR, max_ICR, rating, default_spread_pct)
    (8.50,  float("inf"), "Aaa/AAA", 0.59),
    (6.50,  8.50,         "Aa2/AA",  0.78),
    (5.50,  6.50,         "A1/A+",   1.03),
    (4.25,  5.50,         "A2/A",    1.14),
    (3.00,  4.25,         "A3/A-",   1.30),
    (2.50,  3.00,         "Baa2/BBB",1.62),
    (2.25,  2.50,         "Ba1/BB+", 2.38),
    (2.00,  2.25,         "Ba2/BB",  2.78),
    (1.75,  2.00,         "B1/B+",   3.57),
    (1.50,  1.75,         "B2/B",    4.37),
    (1.25,  1.50,         "B3/B-",   5.26),
    (0.80,  1.25,         "Caa/CCC", 8.00),
    (0.65,  0.80,         "Ca2/CC", 10.50),
    (0.20,  0.65,         "C2/C",   14.00),
    (float("-inf"), 0.20, "D2/D",   19.50),
]
DAMODARAN_ICR_SMALL = [
    (12.5,  float("inf"), "Aaa/AAA", 0.59),
    (9.50,  12.5,         "Aa2/AA",  0.78),
    (7.50,  9.50,         "A1/A+",   1.03),
    (6.00,  7.50,         "A2/A",    1.14),
    (4.50,  6.00,         "A3/A-",   1.30),
    (4.00,  4.50,         "Baa2/BBB",1.62),
    (3.50,  4.00,         "Ba1/BB+", 2.38),
    (3.00,  3.50,         "Ba2/BB",  2.78),
    (2.50,  3.00,         "B1/B+",   3.57),
    (2.00,  2.50,         "B2/B",    4.37),
    (1.50,  2.00,         "B3/B-",   5.26),
    (1.25,  1.50,         "Caa/CCC", 8.00),
    (0.80,  1.25,         "Ca2/CC", 10.50),
    (0.50,  0.80,         "C2/C",   14.00),
    (float("-inf"), 0.50, "D2/D",   19.50),
]
# Rating -> default spread, used when the user picks a rating directly.
DAMODARAN_RATING_SPREADS = {
    "Aaa/AAA": 0.59, "Aa2/AA": 0.78, "A1/A+": 1.03, "A2/A": 1.14, "A3/A-": 1.30,
    "Baa2/BBB": 1.62, "Ba1/BB+": 2.38, "Ba2/BB": 2.78, "B1/B+": 3.57,
    "B2/B": 4.37, "B3/B-": 5.26, "Caa/CCC": 8.00, "Ca2/CC": 10.50,
    "C2/C": 14.00, "D2/D": 19.50,
}

def _synthetic_rating(icr: float, is_large: bool) -> tuple[str, float]:
    table = DAMODARAN_ICR_LARGE if is_large else DAMODARAN_ICR_SMALL
    for lo, hi, rating, spread in table:
        if lo <= icr < hi:
            return rating, spread
    # Falls through only if icr is NaN
    return "B2/B", 4.37


with tab3:
    st.markdown("<h1 class='dash-title' style='font-size:18px;'>WACC / Cost of Capital</h1>", unsafe_allow_html=True)
    st.caption(
        "WACC = (E/V) \u00d7 Re + (P/V) \u00d7 Rp + (D/V) \u00d7 Rd \u00d7 (1 \u2212 t). "
        "Equity, preferred, and debt use market values where available, with labeled book-value fallbacks. "
        "Cost of debt supports three methodologies: observed YTM, actual credit rating (Damodaran spread), "
        "or a synthetic rating from Damodaran's ICR-to-default-spread table."
    )

    wc1, wc2, wc3 = st.columns([1.4, 1, 1])
    with wc1:
        wacc_ticker = st.text_input("Ticker", value="AAPL", key="wacc_ticker").strip().upper()
    with wc2:
        wacc_window = st.selectbox("Beta window", ["1y", "2y", "3y", "5y"], index=1, key="wacc_window")
    with wc3:
        wacc_freq = st.selectbox("Beta frequency", ["Daily", "Weekly", "Monthly"], index=1, key="wacc_freq")

    # Auto-pull current 10Y as a sensible Rf default
    try:
        _rf_default = float(quote_from_fred("DGS10").last or 4.5)
    except Exception:
        _rf_default = 4.5

    we1, we2, we3, we4 = st.columns(4)
    with we1:
        wacc_rf = st.number_input("Risk-free rate (%)", value=round(_rf_default, 2), step=0.05, key="wacc_rf")
    with we2:
        wacc_erp = st.number_input("Equity risk premium (%)", value=4.60, step=0.10, key="wacc_erp",
                                   help="Damodaran's implied US ERP runs ~4.5\u20135.0%.")
    with we3:
        wacc_tax = st.number_input("Marginal tax rate (%)", value=21.0, step=0.5, key="wacc_tax")
    with we4:
        beta_flavor = st.selectbox(
            "Beta adjustment",
            ["Raw (regression)", "Bloomberg-adjusted"],
            index=1,
            key="wacc_beta_flavor",
            help="Bloomberg-adjusted = 0.67 \u00d7 raw + 0.33 (Blume's mean-reversion formula).",
        )

    st.markdown("<div class='sub-header'>Cost of Debt Methodology</div>", unsafe_allow_html=True)
    method = st.radio(
        "How should we estimate the cost of debt?",
        [
            "Observed YTM (publicly traded debt)",
            "Actual credit rating (Damodaran spread)",
            "Synthetic rating from ICR (EBIT / Interest Expense)",
        ],
        index=2,
        key="wacc_method",
        horizontal=False,
    )

    rating_options = list(DAMODARAN_RATING_SPREADS.keys())

    if method == "Observed YTM (publicly traded debt)":
        ytm_input = st.number_input(
            "Yield-to-maturity on the company's outstanding debt (%)",
            value=round(_rf_default + 1.5, 2), step=0.05, key="wacc_ytm",
            help="Enter the weighted-average YTM across the issuer's outstanding bonds. "
                 "Use the longest liquid benchmark or duration-weighted average.",
        )
    elif method == "Actual credit rating (Damodaran spread)":
        chosen_rating = st.selectbox(
            "Issuer credit rating",
            rating_options,
            index=rating_options.index("Baa2/BBB"),
            key="wacc_rating",
            help="Pick the issuer-level long-term rating from Moody's / S&P / Fitch.",
        )

    st.markdown("<div class='sub-header'>Preferred Stock (optional)</div>", unsafe_allow_html=True)
    has_pref = st.checkbox("Company has preferred stock outstanding", value=False, key="wacc_has_pref")
    if has_pref:
        p1, p2, p3 = st.columns(3)
        with p1:
            pref_div_ps = st.number_input("Preferred dividend per share ($)", value=0.00, step=0.05, key="wacc_pref_div")
        with p2:
            pref_price = st.number_input("Preferred market price ($)", value=0.00, step=0.50, key="wacc_pref_px")
        with p3:
            pref_shares = st.number_input("Preferred shares outstanding (M)", value=0.0, step=0.1, key="wacc_pref_sh")

    run_wacc = st.button("Calculate WACC", type="primary", key="wacc_run")

    if run_wacc:
        if not wacc_ticker:
            st.error("Please enter a ticker.")
        else:
            with st.spinner(f"Pulling fundamentals for {wacc_ticker}\u2026"):
                period_map = {"1y":"1y","2y":"2y","3y":"3y","5y":"5y"}
                period = period_map[wacc_window]
                stock_df = yf_history(wacc_ticker, period=period)
                bench_df = yf_history("^GSPC", period=period)
                info = yf_info(wacc_ticker)
                bs = yf_balance_sheet(wacc_ticker)
                inc = yf_income_stmt(wacc_ticker)

            if stock_df.empty or bench_df.empty:
                st.error(f"Couldn't pull price history for {wacc_ticker}.")
            else:
                # ---- Beta (raw + Bloomberg-adjusted) ----
                def _resample_close(close: pd.Series, freq: str) -> pd.Series:
                    s = close.dropna().sort_index()
                    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
                        s.index = s.index.tz_localize(None)
                    if freq == "Daily":  return s.pct_change().dropna()
                    if freq == "Weekly": return s.resample("W-FRI").last().pct_change().dropna()
                    if freq == "Monthly":return s.resample("ME").last().pct_change().dropna()
                    return s.pct_change().dropna()

                sret = _resample_close(stock_df["Close"], wacc_freq)
                bret = _resample_close(bench_df["Close"], wacc_freq)
                joined = pd.concat([sret, bret], axis=1, join="inner").dropna()
                joined.columns = ["s", "b"]
                raw_beta = joined["s"].cov(joined["b"]) / joined["b"].var() if (len(joined) > 5 and joined["b"].var()) else None
                if raw_beta is not None:
                    bloom_beta = 0.67 * raw_beta + 0.33
                else:
                    bloom_beta = None
                beta_val = bloom_beta if beta_flavor == "Bloomberg-adjusted" else raw_beta

                # ---- Fundamentals ----
                interest_exp = None
                ebit = None
                if not inc.empty:
                    for k in ("Interest Expense", "InterestExpense"):
                        if k in inc.index:
                            try:
                                v = inc.loc[k].dropna()
                                if not v.empty:
                                    interest_exp = abs(float(v.iloc[0]))
                                    break
                            except Exception:
                                pass
                    for k in ("EBIT", "Operating Income", "OperatingIncome"):
                        if k in inc.index:
                            try:
                                v = inc.loc[k].dropna()
                                if not v.empty:
                                    ebit = float(v.iloc[0])
                                    break
                            except Exception:
                                pass

                # Book debt
                book_debt = _bs_get(bs, "Total Debt", "TotalDebt")
                if book_debt is None:
                    ltd = _bs_get(bs, "Long Term Debt", "LongTermDebt") or 0
                    std = _bs_get(bs, "Current Debt", "Short Long Term Debt", "ShortLongTermDebt") or 0
                    book_debt = (ltd + std) if (ltd or std) else None

                # ---- Market value of equity ----
                mkt_cap = info.get("marketCap")
                if not mkt_cap:
                    sh = info.get("sharesOutstanding")
                    px_last = float(stock_df["Close"].dropna().iloc[-1])
                    mkt_cap = sh * px_last if sh else None
                equity_is_market = bool(mkt_cap)
                equity_value = mkt_cap if equity_is_market else (_bs_get(bs, "Total Equity Gross Minority Interest",
                                                                        "Stockholders Equity", "TotalEquityGrossMinorityInterest") or 0)
                equity_label = "Market Value" if equity_is_market else "Book Value (fallback)"

                # ---- Cost of debt (three methods) ----
                icr = None
                if ebit is not None and interest_exp:
                    icr = ebit / interest_exp

                pre_tax_rd = None
                rd_label = ""
                if method == "Observed YTM (publicly traded debt)":
                    pre_tax_rd = float(ytm_input)
                    rd_label = "Observed YTM (user input)"
                elif method == "Actual credit rating (Damodaran spread)":
                    spread = DAMODARAN_RATING_SPREADS[chosen_rating]
                    pre_tax_rd = wacc_rf + spread
                    rd_label = f"Rf + {chosen_rating} spread ({spread:.2f}%)"
                else:  # synthetic
                    is_large = (mkt_cap or 0) >= 5e9
                    if icr is None:
                        rd_label = "Synthetic (fallback spread \u2014 ICR not available)"
                        pre_tax_rd = wacc_rf + 2.00
                        syn_rating = "n/a"
                        syn_spread = 2.00
                    else:
                        syn_rating, syn_spread = _synthetic_rating(icr, is_large)
                        pre_tax_rd = wacc_rf + syn_spread
                        rd_label = (f"Synthetic rating {syn_rating} from ICR = {icr:.2f} "
                                    f"({'large' if is_large else 'small/risky'} firm table; spread {syn_spread:.2f}%)")

                # ---- Market value of debt (Damodaran coupon-bond approx) ----
                # Treats total debt as a bullet bond paying the company's reported
                # interest expense as coupon, discounted at pre-tax cost of debt,
                # with maturity defaulting to 5 years. Matches his Valuation book (ch. 8).
                def _market_value_of_debt(book_debt: float, interest_expense: float,
                                          kd_pct: float, maturity_yrs: float = 5.0) -> float:
                    if not book_debt or kd_pct is None:
                        return book_debt or 0.0
                    r = kd_pct / 100
                    if r == 0:
                        return book_debt
                    coupon = interest_expense if interest_expense else book_debt * r
                    annuity = coupon * (1 - (1 + r) ** (-maturity_yrs)) / r
                    return annuity + book_debt / ((1 + r) ** maturity_yrs)

                debt_is_market = bool(book_debt and interest_exp and pre_tax_rd)
                if debt_is_market:
                    debt_value = _market_value_of_debt(book_debt, interest_exp, pre_tax_rd)
                    debt_label = "Market Value (Damodaran coupon-bond approx, 5y)"
                else:
                    debt_value = book_debt or 0
                    debt_label = "Book Value (fallback)" if book_debt else "None reported"

                # ---- Preferred stock ----
                pref_value = 0.0
                pref_cost_pct = None
                pref_label = ""
                pref_is_market = False
                if has_pref and pref_div_ps and pref_price and pref_price > 0:
                    pref_cost_pct = (pref_div_ps / pref_price) * 100
                    if pref_shares and pref_shares > 0:
                        pref_value = pref_price * pref_shares * 1e6  # shares_in_millions -> shares
                        pref_is_market = True
                        pref_label = "Market Value (price \u00d7 shares)"
                    else:
                        # Fall back to book value of preferred on the balance sheet, if any.
                        pref_book = _bs_get(bs, "Preferred Stock Equity", "Preferred Stock", "PreferredStock")
                        if pref_book:
                            pref_value = float(pref_book)
                            pref_label = "Book Value (fallback)"
                        else:
                            pref_label = "Cost only (no value supplied)"

                # ---- CAPM ----
                re_pct = wacc_rf + (beta_val if beta_val is not None else 1.0) * wacc_erp
                tax = wacc_tax / 100
                rd_at_pct = pre_tax_rd * (1 - tax)

                # ---- Weights ----
                V = (equity_value or 0) + (debt_value or 0) + (pref_value or 0)
                if V <= 0:
                    e_w = 1.0; d_w = 0.0; p_w = 0.0
                else:
                    e_w = (equity_value or 0) / V
                    d_w = (debt_value or 0) / V
                    p_w = (pref_value or 0) / V

                wacc_pct = e_w * re_pct + d_w * rd_at_pct + p_w * (pref_cost_pct or 0)

                # ---- Output ----
                st.markdown(f"<div class='sub-header'>{info.get('longName') or wacc_ticker} \u2014 Weighted Average Cost of Capital</div>", unsafe_allow_html=True)

                # Top metric row depends on whether preferred exists
                if p_w > 0:
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("WACC", f"{wacc_pct:.2f}%")
                    m2.metric("Cost of Equity", f"{re_pct:.2f}%")
                    m3.metric("After-Tax Cost of Debt", f"{rd_at_pct:.2f}%")
                    m4.metric("Cost of Preferred", f"{(pref_cost_pct or 0):.2f}%")
                    m5.metric("Beta", f"{beta_val:.3f}" if beta_val is not None else "\u2014")
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("WACC", f"{wacc_pct:.2f}%")
                    m2.metric("Cost of Equity", f"{re_pct:.2f}%")
                    m3.metric("After-Tax Cost of Debt", f"{rd_at_pct:.2f}%")
                    m4.metric("Beta", f"{beta_val:.3f}" if beta_val is not None else "\u2014")

                # Capital structure table with explicit MV/BV labels
                cap_rows = [
                    {"Component": "Equity", "Basis": equity_label,
                     "Value ($M)": f"{(equity_value or 0)/1e6:,.0f}",
                     "Weight": f"{e_w:.1%}", "Cost (pre-tax)": f"{re_pct:.2f}%",
                     "After-Tax Cost": f"{re_pct:.2f}%"},
                    {"Component": "Debt", "Basis": debt_label,
                     "Value ($M)": f"{(debt_value or 0)/1e6:,.0f}" if debt_value else "\u2014",
                     "Weight": f"{d_w:.1%}", "Cost (pre-tax)": f"{pre_tax_rd:.2f}%",
                     "After-Tax Cost": f"{rd_at_pct:.2f}%"},
                ]
                if has_pref:
                    cap_rows.append({
                        "Component": "Preferred Stock",
                        "Basis": pref_label or "\u2014",
                        "Value ($M)": f"{(pref_value or 0)/1e6:,.0f}" if pref_value else "\u2014",
                        "Weight": f"{p_w:.1%}",
                        "Cost (pre-tax)": f"{(pref_cost_pct or 0):.2f}%" if pref_cost_pct else "\u2014",
                        "After-Tax Cost": f"{(pref_cost_pct or 0):.2f}%" if pref_cost_pct else "\u2014",
                    })
                cap_rows.append({
                    "Component": "Total Capital (V)", "Basis": "",
                    "Value ($M)": f"{V/1e6:,.0f}" if V else "\u2014",
                    "Weight": "100.0%",
                    "Cost (pre-tax)": "", "After-Tax Cost": f"{wacc_pct:.2f}%",
                })
                st.dataframe(pd.DataFrame(cap_rows), use_container_width=True, hide_index=True)

                # Methodology panel \u2014 this is what you asked to be visible on-page.
                st.markdown("<div class='sub-header'>Methodology Notes</div>", unsafe_allow_html=True)
                notes = []
                notes.append(
                    f"- **Equity value:** {equity_label}. "
                    + ("Using Yahoo Finance market capitalization (shares outstanding \u00d7 current price)."
                       if equity_is_market else
                       "Market cap unavailable from Yahoo \u2014 fell back to book value of stockholders' equity.")
                )
                notes.append(
                    f"- **Debt value:** {debt_label}. "
                    + ("Market value estimated using Damodaran's coupon-bond approximation: "
                       "the outstanding book debt is treated as a single bullet bond paying the company's reported interest expense as coupon, "
                       "discounted at the pre-tax cost of debt over an assumed 5-year maturity."
                       if debt_is_market else
                       "Not enough fundamentals to convert to market value, so the book balance-sheet amount is used directly. "
                       "For most investment-grade issuers book and market are close, so the WACC impact is small.")
                )
                if has_pref:
                    notes.append(
                        f"- **Preferred stock:** {pref_label}. "
                        "Cost of preferred = annual preferred dividend per share \u00f7 current preferred price. "
                        "Weight uses preferred market price \u00d7 shares outstanding where provided."
                    )
                notes.append(
                    f"- **Beta:** {beta_flavor}. "
                    + (f"Raw regression beta = {raw_beta:.3f} using {wacc_window} of {wacc_freq.lower()} returns vs. ^GSPC."
                       if raw_beta is not None else
                       "Insufficient overlapping returns \u2014 beta not computable.")
                    + (f" Bloomberg-adjusted beta = 0.67 \u00d7 raw + 0.33 = {bloom_beta:.3f}." if bloom_beta is not None else "")
                )
                notes.append(f"- **Cost of equity:** CAPM \u2014 Re = Rf + \u03b2 \u00d7 ERP = {wacc_rf:.2f}% + "
                             f"{(beta_val or 1.0):.3f} \u00d7 {wacc_erp:.2f}% = {re_pct:.2f}%.")
                notes.append(f"- **Pre-tax cost of debt:** {rd_label}. Pre-tax = {pre_tax_rd:.2f}%, "
                             f"after-tax = {pre_tax_rd:.2f}% \u00d7 (1 \u2212 {wacc_tax:.1f}%) = {rd_at_pct:.2f}%.")
                if method == "Synthetic rating from ICR (EBIT / Interest Expense)":
                    notes.append(
                        "- **Synthetic credit rating:** Damodaran's Interest Coverage Ratio (EBIT / Interest Expense) "
                        "is mapped to a rating and default spread using his published table. "
                        "Large firms (market cap > $5B) use the looser ICR bands; smaller/riskier firms use the tighter bands. "
                        "Damodaran updates this table annually \u2014 the current version is linked below."
                    )
                st.markdown("\n".join(notes))

                # Sensitivity grid \u2014 unchanged conceptually, now uses the new cost of debt
                st.markdown("<div class='sub-header'>Sensitivity \u2014 WACC at varying Beta and ERP</div>", unsafe_allow_html=True)
                betas = [((beta_val or 1.0)) + d for d in (-0.2, -0.1, 0.0, 0.1, 0.2)]
                erps = [wacc_erp + d for d in (-1.0, -0.5, 0.0, 0.5, 1.0)]
                grid = []
                for b in betas:
                    row = []
                    for e in erps:
                        re_i = wacc_rf + b * e
                        wacc_i = e_w * re_i + d_w * rd_at_pct + p_w * (pref_cost_pct or 0)
                        row.append(f"{wacc_i:.2f}%")
                    grid.append(row)
                sens = pd.DataFrame(grid, index=[f"\u03b2 {b:.2f}" for b in betas],
                                    columns=[f"ERP {e:.1f}%" for e in erps])
                st.dataframe(sens, use_container_width=True)

                with st.expander("Data sources and references"):
                    st.markdown(
                        "- Synthetic credit rating table: "
                        "[Damodaran \u2014 Ratings, Interest Coverage Ratios and Default Spread]"
                        "(https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ratings.htm) "
                        "(updated annually; current values encoded reflect the January 2025 update).\n"
                        "- Implied ERP: [Damodaran \u2014 Implied ERP Monthly Updates]"
                        "(https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datacurrent.html).\n"
                        "- Market value of debt methodology: Damodaran, *Investment Valuation*, Ch. 8 (coupon-bond approximation).\n"
                        "- Bloomberg-adjusted beta formula: 0.67 \u00d7 raw beta + 0.33 (Blume 1971)."
                    )


# =============================================================================
# Tab 4: Trading Comps / Multiples Table
# =============================================================================
with tab4:
    st.markdown("<h1 class='dash-title' style='font-size:18px;'>Trading Comps</h1>", unsafe_allow_html=True)
    st.caption(
        "Quick relative-value table. Enter a primary ticker and 1\u20139 peers (comma-separated). "
        "All multiples come from Yahoo Finance fundamentals."
    )

    cm1, cm2 = st.columns([1, 2.5])
    with cm1:
        comp_primary = st.text_input("Primary ticker", value="AAPL", key="comp_primary").strip().upper()
    with cm2:
        comp_peers_raw = st.text_input(
            "Peer tickers (comma-separated)",
            value="MSFT, GOOGL, META, AMZN, NVDA",
            key="comp_peers",
        )

    run_comps = st.button("Build comps table", type="primary", key="comp_run")

    if run_comps:
        peers = [p.strip().upper() for p in comp_peers_raw.split(",") if p.strip()]
        all_tickers = [comp_primary] + [p for p in peers if p != comp_primary]
        if not comp_primary:
            st.error("Please enter a primary ticker.")
        else:
            from concurrent.futures import ThreadPoolExecutor
            with st.spinner(f"Pulling fundamentals for {len(all_tickers)} tickers\u2026"):
                with ThreadPoolExecutor(max_workers=10) as ex:
                    infos = list(ex.map(yf_info, all_tickers))

            rows = []
            for tkr, info in zip(all_tickers, infos):
                if not info:
                    rows.append({"Ticker": tkr, "Name": "(no data)", "Price": None, "Market Cap": None,
                                 "EV": None, "EV/Rev": None, "EV/EBITDA": None,
                                 "P/E (LTM)": None, "P/E (NTM)": None, "P/S": None,
                                 "P/B": None, "Net Debt/EBITDA": None})
                    continue

                mcap = info.get("marketCap")
                ev = info.get("enterpriseValue")
                rev = info.get("totalRevenue")
                ebitda = info.get("ebitda")
                td = info.get("totalDebt") or 0
                cash = info.get("totalCash") or 0
                net_debt = td - cash
                price = info.get("currentPrice") or info.get("regularMarketPrice")

                rows.append({
                    "Ticker": tkr,
                    "Name": (info.get("shortName") or info.get("longName") or "")[:32],
                    "Price": price,
                    "Market Cap ($M)": mcap/1e6 if mcap else None,
                    "EV ($M)": ev/1e6 if ev else None,
                    "EV/Rev": (ev/rev) if (ev and rev) else None,
                    "EV/EBITDA": (ev/ebitda) if (ev and ebitda) else None,
                    "P/E (LTM)": info.get("trailingPE"),
                    "P/E (NTM)": info.get("forwardPE"),
                    "P/S": info.get("priceToSalesTrailing12Months"),
                    "P/B": info.get("priceToBook"),
                    "Net Debt/EBITDA": (net_debt/ebitda) if ebitda else None,
                })

            df = pd.DataFrame(rows)

            # Add median / mean rows on numeric columns
            numeric_cols = [c for c in df.columns if c not in ("Ticker", "Name") and pd.api.types.is_numeric_dtype(df[c])]
            peer_only = df[df["Ticker"] != comp_primary]
            summary_med = {"Ticker": "Peer Median", "Name": ""}
            summary_avg = {"Ticker": "Peer Mean",   "Name": ""}
            for c in numeric_cols:
                summary_med[c] = peer_only[c].median()
                summary_avg[c] = peer_only[c].mean()
            df_full = pd.concat([df, pd.DataFrame([summary_med, summary_avg])], ignore_index=True)

            # Pretty formatting
            fmt = {}
            for c in ("Price",):
                if c in df_full: fmt[c] = "{:,.2f}"
            for c in ("Market Cap ($M)", "EV ($M)"):
                if c in df_full: fmt[c] = "{:,.0f}"
            for c in ("EV/Rev", "EV/EBITDA", "P/E (LTM)", "P/E (NTM)", "P/S", "P/B", "Net Debt/EBITDA"):
                if c in df_full: fmt[c] = "{:,.2f}x" if "x" not in c else "{:,.2f}"
            # Use plain format for ratios; "x" suffix is in the column header convention only
            fmt = {k: "{:,.2f}" for k in fmt} if False else fmt
            try:
                styled = df_full.style.format(fmt, na_rep="\u2014")
                # Highlight primary row
                def _highlight_primary(row):
                    return ["background-color: #e8f0fe; font-weight: 600" if row["Ticker"] == comp_primary else "" for _ in row]
                styled = styled.apply(_highlight_primary, axis=1)
                st.dataframe(styled, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(df_full, use_container_width=True, hide_index=True)


# =============================================================================
# Tab 5: Yield Curve Visualizer
# =============================================================================
with tab5:
    st.markdown("<h1 class='dash-title' style='font-size:18px;'>Yield Curves</h1>", unsafe_allow_html=True)
    st.caption("Sovereign curves built from the same data as the Markets Dashboard tab. No extra calls \u2014 fully cached.")

    yc_country = st.selectbox(
        "Country",
        ["United States", "Germany", "Japan", "United Kingdom", "All"],
        index=0,
        key="yc_country",
    )
    yc_overlay = st.selectbox(
        "Overlay",
        ["Today only", "Today vs. 1 month ago", "Today vs. 1 year ago", "Today, 1m, 1y"],
        index=3,
        key="yc_overlay",
    )

    # Series IDs per tenor and country
    CURVES = {
        "United States": {"2Y":"DGS2", "5Y":"DGS5", "10Y":"DGS10", "30Y":"DGS30"},
        "Germany":       {"2Y":"TMBMKDE-02Y", "5Y":"TMBMKDE-05Y", "10Y":"TMBMKDE-10Y", "30Y":"TMBMKDE-30Y"},
        "Japan":         {"2Y":"TMBMKJP-02Y", "5Y":"TMBMKJP-05Y", "10Y":"TMBMKJP-10Y", "30Y":"TMBMKJP-30Y"},
        "United Kingdom":{"2Y":"TMBMKGB-02Y", "5Y":"TMBMKGB-05Y", "10Y":"TMBMKGB-10Y", "30Y":"TMBMKGB-30Y"},
    }
    TENOR_YEARS = {"2Y": 2, "5Y": 5, "10Y": 10, "30Y": 30}

    def _curve_value_at(country: str, tenor: str, asof: pd.Timestamp) -> Optional[float]:
        sid = CURVES[country][tenor]
        if sid.startswith("TMBMK"):
            s = wsj_bond_series(sid)
        else:
            s = fred_series(sid, start=dt.date(asof.year - 2, 1, 1))
        if s is None or s.empty:
            return None
        if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s = s.dropna()
        if s.empty:
            return None
        # Take the most recent observation at or before asof
        idx = s.index[s.index <= asof]
        if len(idx) == 0:
            return float(s.iloc[0])
        return float(s.loc[idx[-1]])

    today = pd.Timestamp.today().normalize()
    overlays = []
    if yc_overlay in ("Today only", "Today vs. 1 month ago", "Today vs. 1 year ago", "Today, 1m, 1y"):
        overlays.append(("Today", today))
    if yc_overlay in ("Today vs. 1 month ago", "Today, 1m, 1y"):
        overlays.append(("1 month ago", today - pd.DateOffset(months=1)))
    if yc_overlay in ("Today vs. 1 year ago", "Today, 1m, 1y"):
        overlays.append(("1 year ago", today - pd.DateOffset(years=1)))

    countries = list(CURVES.keys()) if yc_country == "All" else [yc_country]

    rows = []
    for c in countries:
        for label, ts in overlays:
            for tenor, _sid in CURVES[c].items():
                v = _curve_value_at(c, tenor, ts)
                rows.append({"Country": c, "Snapshot": label, "Tenor": tenor,
                             "Years": TENOR_YEARS[tenor], "Yield": v})

    plot_df = pd.DataFrame(rows).dropna(subset=["Yield"])
    if plot_df.empty:
        st.info("No curve data available.")
    else:
        try:
            import altair as alt
            color_field = "Country:N" if yc_country == "All" else "Snapshot:N"
            stroke_field = "Snapshot:N" if yc_country == "All" else "Country:N"
            chart = (
                alt.Chart(plot_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Years:Q", title="Maturity (years)", scale=alt.Scale(type="log", domain=[1.5, 35])),
                    y=alt.Y("Yield:Q", title="Yield (%)"),
                    color=alt.Color(color_field, legend=alt.Legend(title=None)),
                    strokeDash=alt.StrokeDash(stroke_field, legend=alt.Legend(title=None)) if yc_country == "All" or len(overlays) > 1 else alt.value([1, 0]),
                    tooltip=["Country", "Snapshot", "Tenor",
                             alt.Tooltip("Yield:Q", format=".2f")],
                )
                .properties(height=420)
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            st.line_chart(plot_df.pivot_table(index="Years", columns=["Country","Snapshot"], values="Yield"))

        with st.expander("Show snapshot data"):
            piv = plot_df.pivot_table(
                index=["Country", "Tenor"], columns="Snapshot", values="Yield"
            ).round(3)
            # Reorder tenors
            piv = piv.reindex(["2Y","5Y","10Y","30Y"], level="Tenor")
            st.dataframe(piv, use_container_width=True)


# =============================================================================
# Tab 6: Multi-Asset Return Comparison
# =============================================================================
with tab6:
    st.markdown("<h1 class='dash-title' style='font-size:18px;'>Return Comparison</h1>", unsafe_allow_html=True)
    st.caption(
        "Enter 2\u201310 tickers. Prices are rebased to 100 at the start. Stats use daily returns."
    )

    rc1, rc2 = st.columns([3, 1])
    with rc1:
        rc_tickers_raw = st.text_input(
            "Tickers (comma-separated)",
            value="^GSPC, ^NDX, AAPL, NVDA, GLD",
            key="rc_tickers",
        )
    with rc2:
        rc_period = st.selectbox(
            "Window",
            ["3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "ytd"],
            index=2,
            key="rc_period",
        )

    run_rc = st.button("Compare", type="primary", key="rc_run")

    if run_rc:
        tickers = [t.strip().upper() for t in rc_tickers_raw.split(",") if t.strip()][:10]
        if len(tickers) < 1:
            st.error("Enter at least one ticker.")
        else:
            from concurrent.futures import ThreadPoolExecutor
            period_for_yf = "10y" if rc_period == "ytd" else rc_period
            with st.spinner(f"Pulling history for {len(tickers)} tickers\u2026"):
                with ThreadPoolExecutor(max_workers=10) as ex:
                    histories = list(ex.map(lambda t: yf_history(t, period=period_for_yf), tickers))

            closes = {}
            for t, h in zip(tickers, histories):
                if h is None or h.empty:
                    continue
                s = h["Close"].dropna().copy()
                if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
                    s.index = s.index.tz_localize(None)
                closes[t] = s

            if not closes:
                st.error("No data returned for any of the tickers.")
            else:
                px = pd.DataFrame(closes).dropna(how="all")
                if rc_period == "ytd":
                    px = px[px.index >= pd.Timestamp(dt.date(dt.date.today().year, 1, 1))]
                px = px.dropna(how="all").ffill()

                # Rebase to 100 at first valid value per column
                rebased = px.apply(lambda c: c / c.dropna().iloc[0] * 100 if c.dropna().size else c)

                try:
                    import altair as alt
                    plot_df = rebased.reset_index().melt("Date" if "Date" in rebased.reset_index().columns else rebased.reset_index().columns[0],
                                                          var_name="Ticker", value_name="Index").dropna()
                    date_col = plot_df.columns[0]
                    chart = (
                        alt.Chart(plot_df)
                        .mark_line()
                        .encode(
                            x=alt.X(f"{date_col}:T", title=None),
                            y=alt.Y("Index:Q", title="Rebased to 100"),
                            color=alt.Color("Ticker:N", legend=alt.Legend(title=None)),
                            tooltip=[alt.Tooltip(f"{date_col}:T", title="Date"),
                                     "Ticker",
                                     alt.Tooltip("Index:Q", format=",.2f")],
                        )
                        .properties(height=420)
                    )
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    st.line_chart(rebased)

                # Stats table
                rets = px.pct_change().dropna(how="all")
                stats = []
                for t in px.columns:
                    s = px[t].dropna()
                    r = rets[t].dropna()
                    if s.empty or r.empty:
                        continue
                    total = s.iloc[-1] / s.iloc[0] - 1
                    days = (s.index[-1] - s.index[0]).days
                    ann_ret = (s.iloc[-1] / s.iloc[0]) ** (365 / max(days, 1)) - 1 if days > 0 else None
                    ann_vol = r.std() * (252 ** 0.5)
                    sharpe = (ann_ret / ann_vol) if (ann_ret is not None and ann_vol) else None
                    # Max drawdown
                    cummax = s.cummax()
                    dd = (s / cummax - 1).min()
                    stats.append({
                        "Ticker": t,
                        "Total Return": f"{total:+.2%}",
                        "Annualized Return": f"{ann_ret:+.2%}" if ann_ret is not None else "\u2014",
                        "Annualized Vol": f"{ann_vol:.2%}",
                        "Max Drawdown": f"{dd:.2%}",
                        "Sharpe (Rf=0)": f"{sharpe:.2f}" if sharpe is not None else "\u2014",
                    })
                st.markdown("<div class='sub-header'>Performance Statistics</div>", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)


# =============================================================================
# Tab 7: Economic Calendar
# =============================================================================
with tab7:
    st.markdown("<h1 class='dash-title' style='font-size:18px;'>Economic Calendar</h1>", unsafe_allow_html=True)
    st.caption(
        "Upcoming Federal Reserve and ECB policy meetings, plus monthly US data releases. "
        "Fed/ECB dates are scheduled; CPI/NFP/GDP/PCE are released on regular monthly cadences."
    )

    @st.cache_data(ttl=60 * 60 * 24, show_spinner=False)  # 24h cache \u2014 these don't move
    def _econ_events(today: dt.date) -> pd.DataFrame:
        events = []

        # FOMC 2026 (publicly announced schedule)
        fomc_2026 = [
            ("Jan 27\u201328, 2026", dt.date(2026, 1, 28)),
            ("Mar 17\u201318, 2026", dt.date(2026, 3, 18)),
            ("Apr 28\u201329, 2026", dt.date(2026, 4, 29)),
            ("Jun 16\u201317, 2026", dt.date(2026, 6, 17)),
            ("Jul 28\u201329, 2026", dt.date(2026, 7, 29)),
            ("Sep 15\u201316, 2026", dt.date(2026, 9, 16)),
            ("Nov 3\u20134, 2026",   dt.date(2026, 11, 4)),
            ("Dec 15\u201316, 2026", dt.date(2026, 12, 16)),
        ]
        for label, d in fomc_2026:
            events.append({"Date": d, "Event": "FOMC Decision", "Region": "US",
                           "Detail": f"Two-day meeting ({label}) \u2014 statement and SEP if applicable"})

        # ECB Governing Council 2026 monetary policy meetings
        ecb_2026 = [
            dt.date(2026, 1, 22), dt.date(2026, 3, 12), dt.date(2026, 4, 30),
            dt.date(2026, 6, 11), dt.date(2026, 7, 23), dt.date(2026, 9, 10),
            dt.date(2026, 10, 29), dt.date(2026, 12, 17),
        ]
        for d in ecb_2026:
            events.append({"Date": d, "Event": "ECB Decision", "Region": "Euro Area",
                           "Detail": "Governing Council monetary policy meeting"})

        # Generate next ~6 monthly US releases from typical BLS/BEA cadences
        # (CPI: ~10\u201315 of next month; NFP: 1st Friday; GDP: ~end of month; PCE: ~end of month)
        def _first_friday(year: int, month: int) -> dt.date:
            d = dt.date(year, month, 1)
            offset = (4 - d.weekday()) % 7  # Mon=0 ... Fri=4
            return d + dt.timedelta(days=offset)

            # CPI typically releases ~12th business day; we use 13th as a stable approximation
        for i in range(0, 6):
            month_anchor = today.replace(day=1)
            # Move forward by i months
            year = month_anchor.year + (month_anchor.month - 1 + i) // 12
            month = (month_anchor.month - 1 + i) % 12 + 1

            cpi_date = dt.date(year, month, 13)
            events.append({"Date": cpi_date, "Event": "US CPI", "Region": "US",
                           "Detail": "Bureau of Labor Statistics release (typical mid-month)"})
            nfp_date = _first_friday(year, month)
            events.append({"Date": nfp_date, "Event": "US Nonfarm Payrolls", "Region": "US",
                           "Detail": "BLS Employment Situation report"})
            # PCE \u2014 typically last Friday of the month
            last_day = (dt.date(year, month, 28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
            offset = (last_day.weekday() - 4) % 7
            pce_date = last_day - dt.timedelta(days=offset)
            events.append({"Date": pce_date, "Event": "US PCE Inflation", "Region": "US",
                           "Detail": "BEA Personal Income and Outlays release"})

        df = pd.DataFrame(events)
        df = df[df["Date"] >= today].sort_values("Date").reset_index(drop=True)
        return df

    cal_df = _econ_events(dt.date.today())

    if cal_df.empty:
        st.info("No upcoming events on file.")
    else:
        ec1, ec2 = st.columns([1, 1])
        with ec1:
            region_filter = st.multiselect("Filter by region", sorted(cal_df["Region"].unique()),
                                           default=sorted(cal_df["Region"].unique()), key="ec_region")
        with ec2:
            event_filter = st.multiselect("Filter by event", sorted(cal_df["Event"].unique()),
                                          default=sorted(cal_df["Event"].unique()), key="ec_event")

        filtered = cal_df[cal_df["Region"].isin(region_filter) & cal_df["Event"].isin(event_filter)].head(40).copy()
        filtered["Date"] = pd.to_datetime(filtered["Date"]).dt.strftime("%a %b %d, %Y")
        st.dataframe(filtered, use_container_width=True, hide_index=True)

        st.caption(
            "FOMC and ECB dates from official 2026 schedules (federalreserve.gov, ecb.europa.eu). "
            "Monthly data release dates are estimated from typical release cadences and can shift by a few days; "
            "always confirm with the issuing agency before trading."
        )




# =============================================================================
# Footer
# =============================================================================
st.markdown(
    f"<div class='footer'>Sources: FRED, WSJ/MarketWatch, Yahoo Finance. "
    f"Data refreshed every 10 minutes (cached). "
    f"Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} local time.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Force refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.caption(
        "Data is cached for 10 minutes. Click the button to bypass the cache "
        "and re-pull every series."
    )
