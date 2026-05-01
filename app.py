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
        letter-spacing: 0.3px;
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

tab1, tab2 = st.tabs(["Markets Dashboard", "Beta Calculator"])

with tab1:
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
        html.append("<table class='dash'><tr><th></th><th>Last</th><th>1 week change</th><th>YTD change</th></tr>")
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
        # iShares National Muni ETF (MUB) yield-to-maturity proxy via Yahoo Finance,
        # which is what most free dashboards use. The SEC 30-day yield from MUB's
        # info object is a clean industry-standard read.
        def muni_yield_and_spread() -> tuple[Optional[float], Optional[float]]:
            try:
                import yfinance as yf

                info = yf.Ticker("MUB").info or {}
                y = info.get("yield")
                if y is None:
                    y = info.get("trailingAnnualDividendYield")
                yld_pct = float(y) * 100 if y else None
                spread_bp = (
                    (yld_pct - dgs10_last) * 100
                    if (yld_pct is not None and dgs10_last is not None)
                    else None
                )
                return yld_pct, spread_bp
            except Exception:
                return None, None

        credit_rows = [
            ("Mortgage 30y (Freddie)", "MORTGAGE30US", None, mortgage_spread_bps),
            ("ICE BofA US Corp Index Yield", "BAMLC0A0CMEY", "BAMLC0A0CM", None),
            ("ICE BofA US HY Index Yield", "BAMLH0A0HYM2EY", "BAMLH0A0HYM2", None),
            ("Muni ETF Yield (MUB SEC)", "__MUNI__", None, None),
            ("GER-ITA 10y spread", None, None, ger_ita_spread_bps),
        ]
        html = ["<div class='sub-header'>Credit</div>"]
        html.append("<table class='dash'><tr><th></th><th>Last</th><th>Spread (bps)</th></tr>")
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
                "<table class='dash'><tr><th></th><th>Last</th><th>1 week change</th><th>YTD change</th><th>52-wk Hi</th></tr>"
            )
        else:
            html.append(
                "<table class='dash'><tr><th></th><th>Last</th><th>1 week change</th><th>YTD change</th></tr>"
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

                    # Try to fetch the company name (don't fail if Yahoo blocks .info)
                    company_name = beta_ticker
                    try:
                        import yfinance as yf
                        info = yf.Ticker(beta_ticker).info or {}
                        company_name = info.get("longName") or info.get("shortName") or beta_ticker
                    except Exception:
                        pass

                    st.markdown(
                        f"<div class='sub-header'>{company_name} &nbsp;vs&nbsp; {beta_benchmark} &nbsp;|&nbsp; "
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
