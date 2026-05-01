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
FRED_API_KEY = st.secrets.get("FRED_API_KEY", os.environ.get("FRED_API_KEY", ""))

# ---------- Caching wrappers ----------
# Cache for 10 minutes so we don't hammer the APIs during a session.
CACHE_TTL = 60 * 10


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fred_series(series_id: str, start: Optional[dt.date] = None) -> pd.Series:
    """Pull a single FRED series. Returns an empty Series on failure."""
    if not FRED_API_KEY:
        return pd.Series(dtype=float)
    try:
        from fredapi import Fred

        fred = Fred(api_key=FRED_API_KEY)
        s = fred.get_series(series_id, observation_start=start)
        return s.dropna()
    except Exception:
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


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def stooq_series(ticker: str) -> pd.Series:
    """Pull daily close from Stooq for sovereign yield tickers like 10deby.b."""
    try:
        import pandas_datareader.data as pdr

        end = dt.date.today()
        start = end - dt.timedelta(days=400)
        df = pdr.DataReader(ticker, "stooq", start, end).sort_index()
        return df["Close"].dropna()
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


def quote_from_stooq(ticker: str, is_yield: bool = True) -> Quote:
    s = stooq_series(ticker)
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
    .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1400px; }
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
        background-color: #cfd8e8;
        font-weight: 700;
        padding: 4px 8px;
        margin-top: 8px;
        margin-bottom: 4px;
        border-bottom: 1px solid #1f3864;
    }
    .sub-header {
        background-color: #e8edf5;
        font-weight: 600;
        padding: 3px 8px;
        font-size: 14px;
    }
    table.dash { width: 100%; border-collapse: collapse; font-size: 13px; }
    table.dash th { text-align: right; padding: 4px 8px; color: #555; font-weight: 600; border-bottom: 1px solid #ccc; }
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
    html.append("<table class='dash'><tr><th></th><th>Last</th><th>Δ1w</th><th>ΔYTD</th></tr>")
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

    # Germany — Stooq tickers
    de_rows = [
        ("2y", quote_from_stooq("2deby.b")),
        ("10y", quote_from_stooq("10deby.b")),
    ]
    st.markdown(render_yield_block("Germany", de_rows), unsafe_allow_html=True)

    # Japan
    jp_rows = [
        ("2y", quote_from_stooq("2jpy.b")),
        ("10y", quote_from_stooq("10jpy.b")),
    ]
    st.markdown(render_yield_block("Japan", jp_rows), unsafe_allow_html=True)

    # United Kingdom
    uk_rows = [
        ("2y", quote_from_stooq("2uky.b")),
        ("10y", quote_from_stooq("10uky.b")),
    ]
    st.markdown(render_yield_block("United Kingdom", uk_rows), unsafe_allow_html=True)

    # China (onshore)
    cn_rows = [
        ("2y", quote_from_stooq("2cny.b")),
        ("10y", quote_from_stooq("10cny.b")),
    ]
    st.markdown(render_yield_block("China (onshore)", cn_rows), unsafe_allow_html=True)


# ---------- Right column: Policy / Curves / Credit ----------
with right:
    st.markdown("<div class='section-header'>Policy / Curves / Credit</div>", unsafe_allow_html=True)

    # Policy rates
    policy_pairs = [
        ("Fed funds (upper)", "DFEDTARU"),
        ("ECB MRO", "ECBMRRFR"),
        ("BOJ Policy Rate", "IRSTCB01JPM156N"),
        ("BoE Bank Rate", "IUDSOIA"),
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

    us2 = us_rows[0][1]
    us5 = us_rows[1][1]
    us10 = us_rows[3][1]
    us30 = us_rows[4][1]
    de2 = de_rows[0][1]
    de10 = de_rows[1][1]
    jp2 = jp_rows[0][1]
    jp10 = jp_rows[1][1]
    uk2 = uk_rows[0][1]
    uk10 = uk_rows[1][1]
    cn2 = cn_rows[0][1]
    cn10 = cn_rows[1][1]

    curves = [
        ("United States", spread_bps(us2, us10), spread_bps(us5, us30)),
        ("Germany", spread_bps(de2, de10), None),
        ("Japan", spread_bps(jp2, jp10), None),
        ("United Kingdom", spread_bps(uk2, uk10), None),
        ("China", spread_bps(cn2, cn10), None),
    ]
    html = ["<div class='sub-header'>Curves (bps)</div>"]
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
    credit_rows = [
        ("Mortgage 30y (Freddie)", "MORTGAGE30US", None),
        ("ICE BofA US Corp Index Yield", "BAMLC0A0CMEY", "BAMLC0A0CM"),
        ("ICE BofA US HY Index Yield", "BAMLH0A0HYM2EY", "BAMLH0A0HYM2"),
        ("ICE BofA Muni Index Yield", "BAMLU0A0MAEY", None),
        ("GER-ITA 10y spread", None, None),  # computed below
    ]
    html = ["<div class='sub-header'>Credit</div>"]
    html.append("<table class='dash'><tr><th></th><th>Last</th><th>Spread (bps)</th></tr>")
    for label, yld_id, oas_id in credit_rows:
        if label.startswith("GER-ITA"):
            ita10 = quote_from_stooq("10ity.b")
            spread = (
                (ita10.last - de10.last) * 100
                if (ita10.last is not None and de10.last is not None)
                else None
            )
            html.append(
                f"<tr><td class='label'>{label}</td>"
                f"<td class='num'>—</td>"
                f"<td class='num'>{fmt_bp(spread)}</td></tr>"
            )
            continue
        s = fred_series(yld_id) if yld_id else pd.Series(dtype=float)
        oas = fred_series(oas_id) if oas_id else pd.Series(dtype=float)
        last = float(s.iloc[-1]) if not s.empty else None
        oas_last = float(oas.iloc[-1]) * 100 if not oas.empty else None  # FRED OAS is in %
        html.append(
            f"<tr><td class='label'>{label}</td>"
            f"<td class='num'>{fmt_yield(last)}</td>"
            f"<td class='num'>{fmt_bp(oas_last)}</td></tr>"
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
            "<table class='dash'><tr><th></th><th>Last</th><th>Δ1w</th><th>ΔYTD</th><th>52-wk Hi</th></tr>"
        )
    else:
        html.append(
            "<table class='dash'><tr><th></th><th>Last</th><th>Δ1w</th><th>ΔYTD</th></tr>"
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
        ("USDCNH", "CNH=X", 4),
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


# =============================================================================
# Footer
# =============================================================================
st.markdown(
    f"<div class='footer'>Sources: FRED, Stooq, Yahoo Finance. "
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
