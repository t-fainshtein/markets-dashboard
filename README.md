# Global Financial Markets Dashboard

A free, live, web-based replica of your PDF dashboard. Built with Streamlit + Python.

## Data sources (all free)

| Section | Source |
|---|---|
| US Treasury yields, policy rates, credit indices, mortgage rate | [FRED](https://fred.stlouisfed.org/) |
| Germany / Japan / UK / China sovereign yields, GER-ITA spread | [Stooq](https://stooq.com/) |
| Equities, individual names, FX, commodities, VIX | [Yahoo Finance](https://finance.yahoo.com/) |

## Quick start

1. Install Python 3.10+
2. `pip install -r requirements.txt`
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste your FRED API key.
4. `streamlit run app.py`

See the message I sent you in chat for the full step-by-step guide, including how to deploy free on Streamlit Community Cloud.
