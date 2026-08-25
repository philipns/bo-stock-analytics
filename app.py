
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import requests
from io import StringIO
from pathlib import Path

st.set_page_config(
    page_title="BO Stock Analytics v8.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: #151515;
    border: 1px solid #343434;
    padding: 12px;
    border-radius: 10px;
}
div[data-testid="stDataFrame"] {border: 1px solid #333; border-radius: 8px;}
.bo-pill {display:inline-block;padding:5px 9px;border-radius:999px;font-weight:700;font-size:.82rem;margin-right:5px;}
.ready{background:#123c27;color:#6ee7a8}
.near{background:#2d3b13;color:#bef264}
.position{background:#15324a;color:#7dd3fc}
.wait{background:#2d2d2d;color:#e5e7eb}
.risk{background:#4a2b12;color:#fdba74}
.avoid{background:#481a1a;color:#fca5a5}
.small-note{font-size:.85rem;color:#9ca3af}
</style>
""", unsafe_allow_html=True)

DEFAULT_AUTO = [
    "ANTM.JK","PTBA.JK","ADRO.JK","INCO.JK","MDKA.JK","ITMG.JK","MEDC.JK","PGAS.JK",
    "BMRI.JK","BBRI.JK","BBCA.JK","BBNI.JK","TLKM.JK","ASII.JK","UNTR.JK","ICBP.JK",
    "INDF.JK","KLBF.JK","CPIN.JK","JPFA.JK"
]


# =========================================================
# IDX STOCK MASTER
# Primary source: public GitHub CSV mirror of IDX issuer list.
# This avoids scraping KSEI/IDX on every Streamlit restart.
# =========================================================
MASTER_URL = "https://raw.githubusercontent.com/wildangunawan/Dataset-Saham-IDX/master/List%20Emiten/all.csv"

@st.cache_data(ttl=86400, show_spinner=False)
def load_idx_stock_master():
    # 1) Local repository snapshot if present.
    local_path = Path("idx_stock_master.csv")
    try:
        if local_path.exists():
            local = pd.read_csv(local_path)
            if {"code","name"}.issubset(local.columns) and len(local) > 100:
                master = local.copy()
                master["code"] = master["code"].astype(str).str.strip().str.upper()
                master["name"] = master["name"].astype(str).str.strip()
                master["ticker"] = master["code"] + ".JK"
                master["label"] = master["code"] + " — " + master["name"]
                return master.sort_values("code").drop_duplicates("code").reset_index(drop=True), "LOCAL CSV"

    except Exception:
        pass

    # 2) Stable public CSV hosted on GitHub.
    try:
        master = pd.read_csv(MASTER_URL)
        if {"code","name"}.issubset(master.columns):
            master["code"] = master["code"].astype(str).str.strip().str.upper()
            master["name"] = master["name"].astype(str).str.strip()
            master = master[master["code"].str.match(r"^[A-Z0-9]{4,5}$", na=False)]
            master["ticker"] = master["code"] + ".JK"
            master["label"] = master["code"] + " — " + master["name"]
            master = master.sort_values("code").drop_duplicates("code").reset_index(drop=True)
            if len(master) > 100:
                return master, "GITHUB CSV"
    except Exception:
        pass

    # 3) Last-resort fallback so app stays usable.
    fallback_codes = sorted(set([x.replace(".JK","") for x in DEFAULT_AUTO]))
    master = pd.DataFrame({
        "code": fallback_codes,
        "name": ["Fallback stock master"] * len(fallback_codes),
        "listingDate": [None] * len(fallback_codes),
        "shares": [None] * len(fallback_codes),
        "listingBoard": [None] * len(fallback_codes),
    })
    master["ticker"] = master["code"] + ".JK"
    master["label"] = master["code"] + " — " + master["name"]
    return master, "FALLBACK"

def master_code_from_label(label):
    return str(label).split(" — ")[0].strip().upper()

if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["ANTM.JK","INCO.JK"]

if "manual_portfolio" not in st.session_state:
    st.session_state.manual_portfolio = []

def normalize_ticker(t):
    t = str(t).strip().upper()
    if not t:
        return None
    if t.startswith("^"):
        return t
    return t if t.endswith(".JK") else t + ".JK"

def rma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def atr(data, length=14):
    prev_close = data["Close"].shift(1)
    tr = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - prev_close).abs(),
        (data["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return rma(tr, length)

def dmi_adx(data, di_len=14, adx_smoothing=14):
    up = data["High"].diff()
    down = -data["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=data.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=data.index)
    prev_close = data["Close"].shift(1)
    tr = pd.concat([
        data["High"] - data["Low"],
        (data["High"] - prev_close).abs(),
        (data["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    tr_rma = rma(tr, di_len)
    plus = 100 * rma(plus_dm, di_len) / tr_rma.replace(0, np.nan)
    minus = 100 * rma(minus_dm, di_len) / tr_rma.replace(0, np.nan)
    dx = 100 * (plus - minus).abs() / (plus + minus).replace(0, np.nan)
    adx = rma(dx, adx_smoothing)
    return plus, minus, adx

def clamp(v, lo, hi):
    if pd.isna(v):
        return lo
    return max(lo, min(hi, v))

def pine_like_pivots(data, left=4, right=4):
    highs = data["High"].to_numpy(float)
    lows = data["Low"].to_numpy(float)
    n = len(data)
    new_ph = np.full(n, np.nan)
    new_pl = np.full(n, np.nan)

    for confirm_i in range(left + right, n):
        pivot_i = confirm_i - right
        ch, cl = highs[pivot_i], lows[pivot_i]
        left_h = highs[pivot_i-left:pivot_i]
        right_h = highs[pivot_i+1:pivot_i+right+1]
        left_l = lows[pivot_i-left:pivot_i]
        right_l = lows[pivot_i+1:pivot_i+right+1]
        if (ch >= np.max(left_h)) and (ch > np.max(right_h)):
            new_ph[confirm_i] = ch
        if (cl <= np.min(left_l)) and (cl < np.min(right_l)):
            new_pl[confirm_i] = cl

    out = data.copy()
    out["newPivotHigh"] = new_ph
    out["newPivotLow"] = new_pl
    out["pivotHigh"] = pd.Series(new_ph, index=out.index).ffill()
    out["pivotLow"] = pd.Series(new_pl, index=out.index).ffill()
    out["pivotHighPrev"] = out["pivotHigh"].shift(1)
    out["pivotLowPrev"] = out["pivotLow"].shift(1)
    out["breakoutBull"] = out["pivotHighPrev"].notna() & (out["Close"] > out["pivotHighPrev"])
    out["breakdownBear"] = out["pivotLowPrev"].notna() & (out["Close"] < out["pivotLowPrev"])
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def download_stock(ticker, data_start):
    try:
        raw = yf.download(ticker, start=data_start, auto_adjust=False, progress=False)
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    needed = ["Open","High","Low","Close","Volume"]
    if not set(needed).issubset(raw.columns):
        return pd.DataFrame()
    df = raw[needed].dropna().copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def standalone_trade_history(bt, backtest_start):
    position = False
    pending_entry = False
    pending_exit = False
    entry_price = None
    entry_date = None
    trades = []

    for i in range(len(bt)):
        row = bt.iloc[i]
        date = bt.index[i]

        if pending_exit and position:
            exit_price = float(row["Open"])
            trades.append({
                "Entry Date": entry_date,
                "Entry Price": entry_price,
                "Exit Date": date,
                "Exit Price": exit_price,
                "Return %": (exit_price / entry_price - 1) * 100,
                "Holding Days": (date-entry_date).days
            })
            position = False
            pending_exit = False
            entry_price = None
            entry_date = None

        if pending_entry and not position:
            entry_price = float(row["Open"])
            entry_date = date
            position = True
            pending_entry = False

        if date >= backtest_start:
            if position:
                if bool(row["breakdownBear"]):
                    pending_exit = True
            else:
                if bool(row["breakoutBull"]):
                    pending_entry = True

    running = None
    if position:
        running = {
            "Entry Date": entry_date,
            "Entry Price": entry_price,
            "Current Close": float(bt["Close"].iloc[-1]),
            "Return %": (float(bt["Close"].iloc[-1]) / entry_price - 1) * 100
        }
    return pd.DataFrame(trades), running

def strategy_equity_from_trades(trades, initial=1.0):
    eq = initial
    vals = []
    for _, r in trades.iterrows():
        eq *= (1 + r["Return %"]/100)
        vals.append((r["Exit Date"], eq))
    if not vals:
        return pd.Series(dtype=float)
    s = pd.Series([v for _,v in vals], index=pd.to_datetime([d for d,_ in vals]))
    return s


def build_bo_daily_equity(bt, trades, running, backtest_start, initial_capital=100_000_000):
    """
    Reconstructs a daily mark-to-market equity curve for a single stock strategy.
    Closed trades are applied using actual entry/exit dates from the BO engine.
    Open trade, if any, is marked to current daily Close.
    """
    period = bt[bt.index >= backtest_start].copy()
    if period.empty:
        return pd.Series(dtype=float)

    equity = float(initial_capital)
    curve = pd.Series(index=period.index, dtype=float)
    position = False
    shares = 0.0
    entry_price = np.nan

    trade_events = {}
    if trades is not None and not trades.empty:
        for _, tr in trades.iterrows():
            trade_events.setdefault(pd.Timestamp(tr["Entry Date"]), []).append(("ENTRY", float(tr["Entry Price"])))
            trade_events.setdefault(pd.Timestamp(tr["Exit Date"]), []).append(("EXIT", float(tr["Exit Price"])))

    if running:
        trade_events.setdefault(pd.Timestamp(running["Entry Date"]), []).append(("ENTRY", float(running["Entry Price"])))

    for dt, row in period.iterrows():
        events = trade_events.get(pd.Timestamp(dt), [])
        for event, price in events:
            if event == "EXIT" and position:
                equity = shares * price
                shares = 0.0
                position = False
                entry_price = np.nan
            elif event == "ENTRY" and not position and price > 0:
                shares = equity / price
                position = True
                entry_price = price

        if position:
            curve.loc[dt] = shares * float(row["Close"])
        else:
            curve.loc[dt] = equity

    return curve.ffill().dropna()

def build_buyhold_curve(price_series, backtest_start, initial_capital=100_000_000):
    s = price_series[price_series.index >= backtest_start].dropna().copy()
    if s.empty:
        return pd.Series(dtype=float)
    return s / float(s.iloc[0]) * float(initial_capital)

def normalize_curve(curve, initial_capital=100_000_000):
    curve = curve.dropna()
    if curve.empty:
        return curve
    return curve / float(curve.iloc[0]) * float(initial_capital)

def fit_score_current(bt, trades, backtest_start):
    total = len(trades)
    wins = trades[trades["Return %"] > 0] if total else pd.DataFrame()
    losses = trades[trades["Return %"] < 0] if total else pd.DataFrame()

    avg_win = wins["Return %"].mean() if len(wins) else np.nan
    avg_loss = abs(losses["Return %"].mean()) if len(losses) else np.nan
    wl = avg_win/avg_loss if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0 else np.nan
    expectancy = trades["Return %"].mean() if total else np.nan
    gp = wins["Return %"].sum() if len(wins) else 0
    gl = abs(losses["Return %"].sum()) if len(losses) else 0
    pf = gp/gl if gl > 0 else (999 if gp > 0 else np.nan)

    eq_series = strategy_equity_from_trades(trades, 1.0)
    strategy_return = (eq_series.iloc[-1]-1)*100 if len(eq_series) else np.nan
    if len(eq_series):
        dd = (eq_series/eq_series.cummax()-1)*100
        max_dd = float(dd.min())
    else:
        max_dd = np.nan

    period = bt[bt.index >= backtest_start]
    bh_return = ((period["Close"].iloc[-1]/period["Close"].iloc[0])-1)*100 if len(period)>1 else np.nan
    bh_curve = period["Close"]/period["Close"].iloc[0] if len(period)>1 else pd.Series(dtype=float)
    bh_dd = float((bh_curve/bh_curve.cummax()-1).min()*100) if len(bh_curve) else np.nan

    years = (period.index[-1]-period.index[0]).days/365.25 if len(period)>1 else np.nan
    strat_cagr = ((1+strategy_return/100)**(1/years)-1)*100 if pd.notna(strategy_return) and years and strategy_return>-100 else np.nan
    bh_cagr = ((period["Close"].iloc[-1]/period["Close"].iloc[0])**(1/years)-1)*100 if years else np.nan
    alpha = strategy_return - bh_return if pd.notna(strategy_return) else np.nan

    close = float(bt["Close"].iloc[-1])
    ema = bt["Close"].ewm(span=200, adjust=False).mean()
    ema_now = float(ema.iloc[-1])
    atr_pct = float(atr(bt,14).iloc[-1]/close*100)
    _,_,adx_s = dmi_adx(bt,14,14)
    adx_now = float(adx_s.iloc[-1]) if pd.notna(adx_s.iloc[-1]) else np.nan
    liq_b = float(((bt["Close"]*bt["Volume"]).rolling(20).mean().iloc[-1])/1e9)

    pf_score = 0 if pd.isna(pf) else clamp((pf-1)*15,0,15)
    exp_score = 0 if pd.isna(expectancy) else clamp(expectancy/3*10,0,10)
    alpha_score = 0 if pd.isna(alpha) else clamp(alpha/20*10,0,10)
    wl_score = 0 if pd.isna(wl) else clamp((wl-1)*10,0,10)
    sample_score = clamp(total/15*5,0,5)
    abs_dd = abs(max_dd) if pd.notna(max_dd) else 999
    dd_score = 10 if abs_dd<=15 else 0 if abs_dd>=40 else (40-abs_dd)/25*10
    adx_score = 0 if pd.isna(adx_now) else clamp((adx_now-15)/15*10,0,10)
    ema_score = (5 if close>ema_now else 0)+(5 if len(ema)>20 and ema.iloc[-1]>ema.iloc[-21] else 0)
    atr_score = 10 if 1<=atr_pct<=6 else clamp(atr_pct*10,0,10) if atr_pct<1 else clamp((10.5-atr_pct)/(10.5-6)*10,0,10)
    liq_score = clamp(liq_b/50*10,0,10)
    fit = clamp(pf_score+exp_score+alpha_score+wl_score+sample_score+dd_score+adx_score+ema_score+atr_score+liq_score,0,100)

    return {
        "Fit Score":fit,"Profit Factor":pf,"Expectancy %":expectancy,
        "Strategy Return %":strategy_return,"Strategy CAGR %":strat_cagr,
        "Buy&Hold %":bh_return,"Buy&Hold CAGR %":bh_cagr,
        "Alpha %":alpha,"Max DD %":max_dd,"Buy&Hold Max DD %":bh_dd,
        "ADX":adx_now,"ATR %":atr_pct,"Liquidity B/day":liq_b,"EMA200":ema_now
    }

def analyze_stock(ticker, left, right, data_start, backtest_start, near_pct=3.0):
    df = download_stock(ticker, data_start)
    if df.empty or len(df)<250:
        return None
    bt = pine_like_pivots(df,left,right)
    trades,running = standalone_trade_history(bt,backtest_start)
    stats = fit_score_current(bt,trades,backtest_start)

    close = float(bt["Close"].iloc[-1])
    ph = float(bt["pivotHigh"].iloc[-1]) if pd.notna(bt["pivotHigh"].iloc[-1]) else np.nan
    pl = float(bt["pivotLow"].iloc[-1]) if pd.notna(bt["pivotLow"].iloc[-1]) else np.nan
    dist = (ph-close)/close*100 if pd.notna(ph) else np.nan
    setup_risk = (ph-pl)/ph*100 if pd.notna(ph) and pd.notna(pl) and ph>0 else np.nan
    extension = (close/stats["EMA200"]-1)*100 if stats["EMA200"] else np.nan

    # Opportunity-first status. Running positions are separated from fresh opportunities.
    if running:
        setup="IN POSITION"
    elif bool(bt["breakoutBull"].iloc[-1]):
        setup="READY TO BUY"
    elif pd.notna(extension) and extension>15:
        setup="OVEREXTENDED"
    elif pd.notna(setup_risk) and setup_risk>10:
        setup="HIGH RISK"
    elif pd.notna(dist) and 0<=dist<=2.0:
        setup="NEAR ENTRY"
    elif pd.notna(dist) and 2.0<dist<=7.0:
        setup="WATCH"
    elif stats["Fit Score"]<35:
        setup="AVOID"
    else:
        setup="NOT READY"

    # Setup Score ranks CURRENT opportunity, while Fit Score measures historical quality.
    # Distance is intentionally the largest component so the radar surfaces actionable setups.
    distance_score = 0.0
    if pd.notna(dist) and dist >= 0:
        distance_score = clamp((max(near_pct*2,6.0)-dist)/max(near_pct*2,6.0)*40, 0, 40)
    trend_score = 15 if close > stats["EMA200"] else 0
    ema_series = bt["Close"].ewm(span=200, adjust=False).mean()
    ema_rising = len(ema_series) > 20 and ema_series.iloc[-1] > ema_series.iloc[-21]
    trend_score += 10 if ema_rising else 0
    fit_component = clamp(stats["Fit Score"],0,100) * 0.25
    liq_component = clamp(stats["Liquidity B/day"]/50*10,0,10) if pd.notna(stats["Liquidity B/day"]) else 0
    risk_penalty = 10 if pd.notna(setup_risk) and setup_risk>8 else 0
    setup_score = clamp(distance_score + trend_score + fit_component + liq_component - risk_penalty, 0, 100)
    if setup == "READY TO BUY":
        setup_score = max(setup_score, 95)

    wins = trades[trades["Return %"]>0] if not trades.empty else pd.DataFrame()
    wr = len(wins)/len(trades)*100 if len(trades) else np.nan
    avg_hold = trades["Holding Days"].mean() if len(trades) else np.nan

    result = {
        "Ticker":ticker.replace(".JK",""),"Setup":setup,"Opportunity Score":setup_score,"Setup Score":setup_score,"Fit Score":stats["Fit Score"],
        "Close":close,"Pivot High":ph,"Pivot Low":pl,"Distance Entry %":dist,
        "Setup Risk %":setup_risk,"Trades":len(trades),"Win Rate %":wr,
        "Profit Factor":stats["Profit Factor"],"Expectancy %":stats["Expectancy %"],
        "Strategy Return %":stats["Strategy Return %"],"Strategy CAGR %":stats["Strategy CAGR %"],
        "Buy&Hold %":stats["Buy&Hold %"],"Buy&Hold CAGR %":stats["Buy&Hold CAGR %"],
        "Alpha %":stats["Alpha %"],"Max DD %":stats["Max DD %"],
        "Buy&Hold Max DD %":stats["Buy&Hold Max DD %"],"Average Holding Days":avg_hold,
        "ADX":stats["ADX"],"ATR %":stats["ATR %"],"Liquidity B/day":stats["Liquidity B/day"],
        "Trend":"BULLISH" if close>stats["EMA200"] else "BEARISH",
        "_bt":bt,"_trades":trades,"_running":running
    }
    return result

def status_badge(status):
    cls = {
        "READY TO BUY":"ready","NEAR ENTRY":"near","WATCH":"near","IN POSITION":"position",
        "NOT READY":"wait","WAIT":"wait","HIGH RISK":"risk","OVEREXTENDED":"risk","AVOID":"avoid"
    }.get(status,"wait")
    return f'<span class="bo-pill {cls}">{status}</span>'

# SIDEBAR
st.sidebar.title("BO Stock Settings")
page = st.sidebar.radio("Page",["Dashboard","Scanner","Universe","Stock Master","Stock Detail","Portfolio","Watchlist"])
left = st.sidebar.number_input("Pivot Left",min_value=1,value=4,step=1)
right = st.sidebar.number_input("Pivot Right",min_value=1,value=4,step=1)
initial_capital = st.sidebar.number_input("Initial Capital (IDR)",min_value=1_000_000,value=100_000_000,step=10_000_000)
min_fit = st.sidebar.slider("Minimum Fit Score",0,100,45)
max_positions = st.sidebar.number_input("Max Positions",min_value=1,max_value=20,value=5,step=1)

st.sidebar.markdown("### AUTO Scanner Filter")
auto_price_min = st.sidebar.number_input("Minimum Price", min_value=0.0, value=100.0, step=50.0)
auto_liq_min_b = st.sidebar.number_input("Minimum Avg Value / Day (Rp B)", min_value=0.0, value=5.0, step=5.0)
auto_atr_min = st.sidebar.number_input("Minimum ATR %", min_value=0.0, value=1.0, step=0.25)
auto_history_days = st.sidebar.number_input("Minimum Trading Days", min_value=60, value=200, step=20)
auto_prefilter_limit = st.sidebar.slider("Max Candidates After Pre-filter", 20, 250, 100, 10)

backtest_start_input = st.sidebar.date_input("Backtest Start",value=pd.Timestamp("2017-08-20"))
backtest_start = pd.Timestamp(backtest_start_input)
universe_mode = st.sidebar.selectbox("Universe Mode",["AUTO","MANUAL","AUTO+MANUAL"])
manual_text = st.sidebar.text_area("Manual Tickers",value="ANTM, INCO, ADRO, PTBA, BMRI")
manual = [normalize_ticker(x) for x in manual_text.replace("\n",",").split(",")]
manual = [x for x in manual if x]

idx_master, master_source = load_idx_stock_master()
st.sidebar.caption(f"IDX Stock Master: {len(idx_master)} codes • {master_source}")

if universe_mode=="MANUAL":
    universe=manual
elif universe_mode=="AUTO+MANUAL":
    universe=list(dict.fromkeys(DEFAULT_AUTO+manual))
else:
    universe=DEFAULT_AUTO

st.sidebar.caption(f"{len(universe)} scanner tickers active")
data_start="2005-01-01"


@st.cache_data(ttl=3600, show_spinner=False)
def quick_prefilter_idx(master_codes_tuple, price_min, liq_min_b, atr_min_pct, min_days, max_candidates):
    """
    Fast pre-filter using only recent data.
    This is intentionally lighter than full historical BO backtest.
    """
    rows = []
    tickers = [normalize_ticker(x) for x in master_codes_tuple if x]
    tickers = [x for x in tickers if x]

    # Process in chunks to reduce Yahoo request overhead / failure risk.
    chunk_size = 60
    for start in range(0, len(tickers), chunk_size):
        chunk = tickers[start:start+chunk_size]
        try:
            raw = yf.download(
                chunk,
                period="18mo",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception:
            continue

        for ticker in chunk:
            try:
                if len(chunk) == 1:
                    d = raw.copy()
                else:
                    if ticker not in raw.columns.get_level_values(0):
                        continue
                    d = raw[ticker].copy()

                if d.empty or "Close" not in d.columns:
                    continue
                d = d[["Open","High","Low","Close","Volume"]].dropna()
                if len(d) < min_days:
                    continue

                close = float(d["Close"].iloc[-1])
                if close < price_min:
                    continue

                avg_value_b = float((d["Close"] * d["Volume"]).tail(20).mean() / 1e9)
                if avg_value_b < liq_min_b:
                    continue

                atr_pct_now = float(atr(d, 14).iloc[-1] / close * 100)
                if pd.isna(atr_pct_now) or atr_pct_now < atr_min_pct:
                    continue

                ema200 = d["Close"].ewm(span=200, adjust=False).mean()
                ema_now = float(ema200.iloc[-1])
                ema_rising = len(ema200) > 20 and ema200.iloc[-1] > ema200.iloc[-21]
                above_ema = close > ema_now

                # Lightweight score only for candidate ordering.
                trend_score = (10 if above_ema else 0) + (10 if ema_rising else 0)
                liq_score = clamp(avg_value_b / 50.0 * 10.0, 0, 10)
                atr_score = clamp(atr_pct_now / 3.0 * 10.0, 0, 10)
                pre_score = trend_score + liq_score + atr_score

                rows.append({
                    "Ticker": ticker.replace(".JK",""),
                    "Close": close,
                    "Avg Value B/day": avg_value_b,
                    "ATR %": atr_pct_now,
                    "Above EMA200": above_ema,
                    "EMA200 Rising": ema_rising,
                    "Pre Score": pre_score,
                })
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["Pre Score","Avg Value B/day"],
        ascending=[False,False]
    ).head(max_candidates).reset_index(drop=True)
    out.insert(0,"Pre Rank",np.arange(1,len(out)+1))
    return out

@st.cache_data(ttl=3600, show_spinner="Running full BO analysis...")
def full_auto_scan(candidate_tuple, left, right, data_start, backtest_start_str):
    return scan_universe(
        tuple(normalize_ticker(x) for x in candidate_tuple),
        left, right, data_start, backtest_start_str
    )

@st.cache_data(ttl=300,show_spinner="Scanning universe...")
def scan_universe(universe_tuple,left,right,data_start,backtest_start_str):
    rows=[]
    for ticker in universe_tuple:
        try:
            res=analyze_stock(ticker,left,right,data_start,pd.Timestamp(backtest_start_str))
            if res:
                rows.append({k:v for k,v in res.items() if not k.startswith("_")})
        except Exception:
            pass
    df=pd.DataFrame(rows)
    if not df.empty:
        priority={"READY TO BUY":0,"NEAR ENTRY":1,"WATCH":2,"IN POSITION":3,"NOT READY":4,"WAIT":4,"HIGH RISK":5,"OVEREXTENDED":6,"AVOID":7}
        df["_priority"]=df["Setup"].map(priority).fillna(9)
        df=df.sort_values(["_priority","Setup Score","Distance Entry %","Fit Score"],ascending=[True,False,True,False],na_position="last").drop(columns="_priority").reset_index(drop=True)
        df.insert(0,"Rank",np.arange(1,len(df)+1))
    return df

scanner=scan_universe(tuple(universe),left,right,data_start,str(backtest_start.date()))

# Automatic one-time retry for transient Yahoo/cache failures.
if scanner.empty:
    try:
        scan_universe.clear()
        scanner=scan_universe(tuple(universe),left,right,data_start,str(backtest_start.date()))
    except Exception:
        scanner=pd.DataFrame()

# Manual recovery control. It also clears per-stock Yahoo cache.
if st.sidebar.button("↻ Refresh Main Scanner", use_container_width=True):
    try:
        scan_universe.clear()
        download_stock.clear()
        full_auto_scan.clear()
    except Exception:
        pass
    st.rerun()

st.title("BO Stock Analytics v8.0")
st.caption("IDX Opportunity Radar • Candlestick Stock Detail • BO vs Buy & Hold vs IHSG equity curves • Pivot 4/4")

scanner_available = not scanner.empty
if not scanner_available and page in ["Dashboard","Scanner","Universe","Portfolio"]:
    st.warning(
        "Main Scanner belum berhasil dimuat. Page lain tetap dapat digunakan. "
        "Tekan 'Refresh Main Scanner' di sidebar atau jalankan Smart AUTO Scanner."
    )

if page=="Dashboard":
    if scanner.empty:
        st.subheader("Dashboard")
        st.info(
            "Main Scanner belum tersedia, tetapi Stock Master, Stock Detail, Watchlist, "
            "dan Smart AUTO Scanner tetap aktif."
        )
        st.markdown("### Quick Access")
        c1,c2,c3 = st.columns(3)
        c1.metric("IDX Stock Master", len(idx_master))
        c2.metric("Watchlist", len(st.session_state.watchlist))
        pre_tmp = st.session_state.get("auto_prefilter", pd.DataFrame())
        c3.metric("AUTO Candidates", len(pre_tmp) if not pre_tmp.empty else 0)
    else:
        ready_count=int(scanner["Setup"].isin(["READY TO BUY","NEAR ENTRY","WATCH"]).sum())
        inpos=int((scanner["Setup"]=="IN POSITION").sum())
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Scanner Universe",len(scanner))
        c2.metric("Radar Candidates",ready_count)
        c3.metric("In Position",inpos)
        c4.metric("Average Fit",f"{scanner['Fit Score'].mean():.1f}")

        auto_full_dash = st.session_state.get("auto_full_scan", pd.DataFrame())
        if not auto_full_dash.empty:
            st.subheader("Smart AUTO Highlights")
            auto_high = auto_full_dash[
                (auto_full_dash["Fit Score"] >= min_fit) &
                (auto_full_dash["Setup"].isin(["READY TO BUY","NEAR ENTRY","IN POSITION"]))
            ].head(10)
            if not auto_high.empty:
                st.dataframe(
                    auto_high[["Rank","Ticker","Setup","Fit Score","Close","Pivot High","Distance Entry %","Profit Factor","Alpha %"]].round(2),
                    use_container_width=True,
                    hide_index=True
                )

        st.subheader("Top Opportunities")
        cols=["Rank","Ticker","Setup","Fit Score","Close","Pivot High","Distance Entry %","Trades","Win Rate %","Profit Factor","Expectancy %","Max DD %","Trend"]
        st.dataframe(scanner[cols].head(12).round(2),use_container_width=True,hide_index=True)

        st.subheader("Watchlist")
        wrows=[]
        for t in st.session_state.watchlist:
            try:
                r=analyze_stock(t,left,right,data_start,backtest_start)
                if r: wrows.append({k:v for k,v in r.items() if not k.startswith("_")})
            except: pass
        if wrows:
            st.dataframe(pd.DataFrame(wrows)[["Ticker","Setup","Fit Score","Close","Pivot High","Distance Entry %","Profit Factor","Alpha %"]].round(2),use_container_width=True,hide_index=True)
elif page=="Scanner":
    st.subheader("Scanner — Breakout Radar")
    st.caption("V6 memprioritaskan setup BARU: READY → NEAR → WATCH. IN POSITION dipisahkan agar tidak menutupi peluang entry baru.")

    tab_auto, tab_batch, tab_main = st.tabs(["Smart AUTO Scanner","Custom Batch","Main Universe"])

    with tab_auto:
        st.markdown("### Step 1 — Pre-filter seluruh Stock Master")
        st.caption(
            "Pre-filter memakai recent price, liquidity, ATR, dan EMA trend. "
            "Hasilnya hanya shortlist kandidat untuk analisis BO penuh."
        )

        run_pre = st.button("Run Smart Pre-filter", type="primary")
        if run_pre:
            with st.spinner("Pre-filtering IDX stock master..."):
                prefilter = quick_prefilter_idx(
                    tuple(idx_master["code"].tolist()),
                    auto_price_min,
                    auto_liq_min_b,
                    auto_atr_min,
                    auto_history_days,
                    auto_prefilter_limit,
                )
            st.session_state["auto_prefilter"] = prefilter

        prefilter = st.session_state.get("auto_prefilter", pd.DataFrame())
        if not prefilter.empty:
            c1,c2,c3 = st.columns(3)
            c1.metric("Stock Master", len(idx_master))
            c2.metric("Candidates", len(prefilter))
            c3.metric("Top Pre Score", f"{prefilter['Pre Score'].max():.1f}")

            st.dataframe(prefilter.round(2), use_container_width=True, hide_index=True)

            st.markdown("### Step 2 — Full BO 4/4 analysis")
            full_count = st.slider(
                "How many top candidates to fully backtest?",
                10, min(100, len(prefilter)), min(30, len(prefilter)), 5
            )
            run_full = st.button("Run Full BO Scanner")

            if run_full:
                selected_candidates = prefilter["Ticker"].head(full_count).tolist()
                auto_full = full_auto_scan(
                    tuple(selected_candidates),
                    left, right, data_start, str(backtest_start.date())
                )
                st.session_state["auto_full_scan"] = auto_full

            auto_full = st.session_state.get("auto_full_scan", pd.DataFrame())
            if not auto_full.empty:
                st.success(f"Full BO analysis completed for {len(auto_full)} stocks.")

                st.markdown("#### Breakout Radar")
                radar_tab, new_tab, pos_tab, all_tab = st.tabs(
                    ["🔥 READY / NEAR / WATCH","⚡ New Breakouts","🔵 Active Positions","All Results"]
                )
                radar_cols=[
                    "Rank","Ticker","Setup","Opportunity Score","Fit Score","Close","Pivot High",
                    "Distance Entry %","Setup Risk %","Trades","Win Rate %","Profit Factor",
                    "Expectancy %","Alpha %","Trend"
                ]

                with radar_tab:
                    radar=auto_full[
                        auto_full["Setup"].isin(["READY TO BUY","NEAR ENTRY","WATCH"])
                    ].copy()
                    radar=radar.sort_values(
                        ["Opportunity Score","Distance Entry %","Fit Score"],
                        ascending=[False,True,False],
                        na_position="last"
                    )
                    if radar.empty:
                        st.info("Belum ada READY / NEAR / WATCH pada batch ini. Coba tambah jumlah Full BO candidates.")
                    else:
                        st.dataframe(
                            radar[[c for c in radar_cols if c in radar.columns]].round(2),
                            use_container_width=True,
                            hide_index=True
                        )

                with new_tab:
                    fresh=auto_full[auto_full["Setup"]=="READY TO BUY"].copy()
                    if fresh.empty:
                        st.info("Belum ada breakout baru pada candle terbaru.")
                    else:
                        st.dataframe(
                            fresh[[c for c in radar_cols if c in fresh.columns]].round(2),
                            use_container_width=True,
                            hide_index=True
                        )

                with pos_tab:
                    active=auto_full[auto_full["Setup"]=="IN POSITION"].copy()
                    if active.empty:
                        st.info("Tidak ada posisi strategy yang sedang aktif pada batch ini.")
                    else:
                        st.dataframe(
                            active[[c for c in radar_cols if c in active.columns]].round(2),
                            use_container_width=True,
                            hide_index=True
                        )

                with all_tab:
                    st.dataframe(auto_full.round(2), use_container_width=True, hide_index=True)

    with tab_batch:
        st.caption("Pilih saham manual dari Stock Master untuk scanner batch.")
        batch_labels = st.multiselect(
            "Choose stocks from IDX master",
            idx_master["label"].tolist(),
            default=[],
            help="Sebaiknya 10–50 saham per batch agar stabil."
        )
        batch_limit = st.slider("Maximum stocks per batch", 5, 100, 30, 5, key="batch_limit")
        run_batch = st.button("Run Custom Batch")

        if run_batch and batch_labels:
            batch_tickers = [
                normalize_ticker(master_code_from_label(x))
                for x in batch_labels[:batch_limit]
            ]
            custom_df = scan_universe(
                tuple(batch_tickers), left, right, data_start, str(backtest_start.date())
            )
            if custom_df.empty:
                st.warning("No results from selected batch.")
            else:
                st.success(f"Scanned {len(custom_df)} stocks")
                st.dataframe(custom_df.round(2), use_container_width=True, hide_index=True)

    with tab_main:
        st.caption("Universe scanner rutin dari AUTO / MANUAL / AUTO+MANUAL di sidebar.")
        if scanner.empty:
            st.warning("Main Universe belum termuat. Gunakan tombol Refresh Main Scanner di sidebar.")
            st.dataframe(
                pd.DataFrame({"Configured Ticker":[x.replace(".JK","") for x in universe]}),
                use_container_width=True,
                hide_index=True
            )
        else:
            status_filter = st.multiselect(
                "Setup filter",
                sorted(scanner["Setup"].dropna().unique()),
                default=sorted(scanner["Setup"].dropna().unique())
            )
            view = scanner[scanner["Setup"].isin(status_filter)].copy()
            rtab, ptab, alltab = st.tabs(["🔥 Opportunity Radar","🔵 In Position","All Results"])
            with rtab:
                rv=view[view["Setup"].isin(["READY TO BUY","NEAR ENTRY","WATCH"])].copy()
                if rv.empty: st.info("Belum ada fresh setup di Main Universe.")
                else: st.dataframe(rv.round(2),use_container_width=True,hide_index=True)
            with ptab:
                pv=view[view["Setup"]=="IN POSITION"].copy()
                st.dataframe(pv.round(2),use_container_width=True,hide_index=True) if not pv.empty else st.info("Tidak ada posisi aktif.")
            with alltab:
                st.dataframe(view.round(2),use_container_width=True,hide_index=True)

elif page=="Universe":
    st.subheader("Universe")
    st.write(f"Mode aktif: **{universe_mode}**")
    st.caption("Universe di sini hanya mengatur sumber scanner rutin. Portfolio personal dibangun di halaman Portfolio dari seluruh 951 saham IDX.")
    if scanner.empty:
        st.warning("Data analitik Main Universe belum tersedia. Daftar ticker tetap ditampilkan.")
        st.dataframe(
            pd.DataFrame({"Ticker":[x.replace(".JK","") for x in universe]}),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.dataframe(
            scanner[["Rank","Ticker","Setup","Setup Score","Fit Score","Strategy Return %","Buy&Hold %","Alpha %","Max DD %","Trades","Win Rate %","Profit Factor"]].round(2),
            use_container_width=True,
            hide_index=True
        )

elif page=="Stock Master":
    st.subheader("IDX Stock Master")
    st.caption("Master emiten untuk pencarian seluruh saham. Primary source adalah CSV repository, sehingga tidak perlu scraping KSEI saat aplikasi dibuka.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Stock Codes Loaded", len(idx_master))
    c2.metric("Master Source", master_source)
    c3.metric("Scanner Universe", len(universe))
    c4.metric("Watchlist", len(st.session_state.watchlist))

    query = st.text_input("Search code / company name", placeholder="Contoh: ANTM, Petrosea, Bank")
    master_view = idx_master.copy()
    if query.strip():
        q = query.strip().lower()
        master_view = master_view[
            master_view["code"].str.lower().str.contains(q, na=False) |
            master_view["name"].str.lower().str.contains(q, na=False)
        ]

    display_cols = [c for c in ["code","name","listingDate","listingBoard"] if c in master_view.columns]
    st.dataframe(
        master_view[display_cols].head(1000),
        use_container_width=True,
        hide_index=True
    )

    if master_source == "FALLBACK":
        st.error("Full master gagal dimuat. Aplikasi sedang memakai fallback list.")
    else:
        st.success(f"Stock Master loaded from {master_source}.")

    st.info(
        "Stock Master hanya daftar pencarian. Scanner rutin sengaja memakai subset agar dashboard tetap cepat. "
        "Untuk analisis saham apa pun, buka Stock Detail atau pilih Custom Batch di Scanner."
    )

elif page=="Stock Detail":
    st.subheader("Stock Detail — Search Any IDX Ticker")
    st.info("Pilih saham dari IDX Stock Master atau ketik ticker manual. Analisis dilakukan on-demand sehingga Stock Detail tidak dibatasi scanner universe.")

    search_mode = st.radio("Search method", ["IDX Stock Master","Manual ticker"], horizontal=True)

    if search_mode == "IDX Stock Master":
        default_idx = 0
        labels = idx_master["label"].tolist()
        selected_label = st.selectbox(
            "Search stock",
            labels,
            index=default_idx,
            help="Ketik kode atau nama emiten untuk mencari."
        )
        ticker_input = master_code_from_label(selected_label)
    else:
        ticker_input = st.text_input("Ticker IDX", value="ANTM", placeholder="Contoh: PTRO")

    ticker_full=normalize_ticker(ticker_input)
    if ticker_full:
        with st.spinner(f"Analyzing {ticker_full}..."):
            res=analyze_stock(ticker_full,left,right,data_start,backtest_start)

        if res is None:
            st.error("Ticker tidak ditemukan / data Yahoo Finance tidak cukup. Pastikan kode saham benar.")
        else:
            st.markdown(status_badge(res["Setup"]),unsafe_allow_html=True)
            st.markdown(f"### {res['Ticker']}")

            c1,c2,c3,c4,c5=st.columns(5)
            c1.metric("Close",f"{res['Close']:,.0f}")
            c2.metric("Fit Score",f"{res['Fit Score']:.1f}")
            c3.metric("Pivot High",f"{res['Pivot High']:,.0f}" if pd.notna(res["Pivot High"]) else "N/A")
            c4.metric("Distance Entry",f"{res['Distance Entry %']:.2f}%" if pd.notna(res["Distance Entry %"]) else "N/A")
            c5.metric("Trend",res["Trend"])

            c6,c7,c8,c9,c10=st.columns(5)
            c6.metric("BO Return",f"{res['Strategy Return %']:.1f}%" if pd.notna(res["Strategy Return %"]) else "N/A")
            c7.metric("Buy & Hold",f"{res['Buy&Hold %']:.1f}%" if pd.notna(res["Buy&Hold %"]) else "N/A")
            c8.metric("Alpha",f"{res['Alpha %']:.1f}%" if pd.notna(res["Alpha %"]) else "N/A")
            c9.metric("Profit Factor",f"{res['Profit Factor']:.2f}" if pd.notna(res["Profit Factor"]) else "N/A")
            c10.metric("Win Rate",f"{res['Win Rate %']:.1f}%" if pd.notna(res["Win Rate %"]) else "N/A")

            col_add,col_remove=st.columns(2)
            if col_add.button("➕ Add to Watchlist"):
                if ticker_full not in st.session_state.watchlist:
                    st.session_state.watchlist.append(ticker_full)
                    st.success("Added to watchlist.")
            if col_remove.button("➖ Remove from Watchlist"):
                if ticker_full in st.session_state.watchlist:
                    st.session_state.watchlist.remove(ticker_full)
                    st.success("Removed from watchlist.")

            st.subheader("Candlestick + Pivot Stair")
            chart_bars = st.select_slider(
                "Chart window",
                options=[100,200,300,500,800],
                value=300,
                key=f"chart_window_{res['Ticker']}"
            )
            bt_chart=res["_bt"].tail(chart_bars).copy()

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=bt_chart.index,
                open=bt_chart["Open"],
                high=bt_chart["High"],
                low=bt_chart["Low"],
                close=bt_chart["Close"],
                name="Price",
                increasing_line_color="#22c55e",
                decreasing_line_color="#ef4444"
            ))
            fig.add_trace(go.Scatter(
                x=bt_chart.index,
                y=bt_chart["pivotHigh"],
                mode="lines",
                name="Pivot High",
                line=dict(color="#22d3ee", width=2, shape="hv")
            ))
            fig.add_trace(go.Scatter(
                x=bt_chart.index,
                y=bt_chart["pivotLow"],
                mode="lines",
                name="Pivot Low",
                line=dict(color="#f97316", width=2, shape="hv")
            ))

            if res["_running"]:
                ep = float(res["_running"]["Entry Price"])
                fig.add_hline(
                    y=ep,
                    line_dash="dot",
                    annotation_text=f"BO Entry {ep:,.0f}",
                    annotation_position="top left"
                )

            fig.update_layout(
                height=620,
                margin=dict(l=10,r=10,t=35,b=10),
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h"),
                hovermode="x unified"
            )
            st.plotly_chart(fig,use_container_width=True)

            st.subheader("Equity Curve — BO vs Buy & Hold vs IHSG")
            full_bt = res["_bt"].copy()
            bo_curve = build_bo_daily_equity(
                full_bt,
                res["_trades"],
                res["_running"],
                backtest_start,
                initial_capital
            )
            stock_bh_curve = build_buyhold_curve(
                full_bt["Close"],
                backtest_start,
                initial_capital
            )

            ihsg_df = download_stock("^JKSE", data_start)
            if not ihsg_df.empty:
                ihsg_curve = build_buyhold_curve(
                    ihsg_df["Close"],
                    backtest_start,
                    initial_capital
                )
            else:
                ihsg_curve = pd.Series(dtype=float)

            eq_df = pd.concat([
                bo_curve.rename("BO Strategy"),
                stock_bh_curve.rename(f"{res['Ticker']} Buy & Hold"),
                ihsg_curve.rename("IHSG Buy & Hold")
            ],axis=1).sort_index().ffill().dropna(how="all")

            if eq_df.empty:
                st.info("Equity curve belum tersedia untuk periode ini.")
            else:
                # Align to first common available values and normalize each series independently
                norm_df = pd.DataFrame(index=eq_df.index)
                for col in eq_df.columns:
                    s = eq_df[col].dropna()
                    if not s.empty:
                        norm_df[col] = s / float(s.iloc[0]) * float(initial_capital)
                norm_df = norm_df.ffill()

                eq_fig = go.Figure()
                for col in norm_df.columns:
                    eq_fig.add_trace(go.Scatter(
                        x=norm_df.index,
                        y=norm_df[col],
                        mode="lines",
                        name=col
                    ))
                eq_fig.update_layout(
                    height=500,
                    margin=dict(l=10,r=10,t=30,b=10),
                    yaxis_title="Equity (IDR, same starting capital)",
                    hovermode="x unified",
                    legend=dict(orientation="h")
                )
                st.plotly_chart(eq_fig,use_container_width=True)

                latest = {}
                for col in norm_df.columns:
                    s = norm_df[col].dropna()
                    latest[col] = float(s.iloc[-1]) if not s.empty else np.nan

                ec1,ec2,ec3 = st.columns(3)
                bo_final = latest.get("BO Strategy",np.nan)
                bh_final = latest.get(f"{res['Ticker']} Buy & Hold",np.nan)
                ihsg_final = latest.get("IHSG Buy & Hold",np.nan)

                ec1.metric(
                    "BO Final Equity",
                    f"Rp{bo_final/1_000_000:,.1f}M" if pd.notna(bo_final) else "N/A",
                    f"{(bo_final/initial_capital-1)*100:.1f}%" if pd.notna(bo_final) else None
                )
                ec2.metric(
                    "Stock B&H Final Equity",
                    f"Rp{bh_final/1_000_000:,.1f}M" if pd.notna(bh_final) else "N/A",
                    f"{(bh_final/initial_capital-1)*100:.1f}%" if pd.notna(bh_final) else None
                )
                ec3.metric(
                    "IHSG Final Equity",
                    f"Rp{ihsg_final/1_000_000:,.1f}M" if pd.notna(ihsg_final) else "N/A",
                    f"{(ihsg_final/initial_capital-1)*100:.1f}%" if pd.notna(ihsg_final) else None
                )

            st.subheader("Breakout Strategy vs Buy & Hold")
            ihsg_period = download_stock("^JKSE",data_start)
            ihsg_period = ihsg_period[ihsg_period.index>=backtest_start] if not ihsg_period.empty else ihsg_period
            ihsg_return=np.nan
            ihsg_cagr=np.nan
            ihsg_dd=np.nan
            if len(ihsg_period)>1:
                ihsg_return=(ihsg_period["Close"].iloc[-1]/ihsg_period["Close"].iloc[0]-1)*100
                yrs=(ihsg_period.index[-1]-ihsg_period.index[0]).days/365.25
                if yrs>0:
                    ihsg_cagr=((ihsg_period["Close"].iloc[-1]/ihsg_period["Close"].iloc[0])**(1/yrs)-1)*100
                ihsg_curve_raw=ihsg_period["Close"]/ihsg_period["Close"].iloc[0]
                ihsg_dd=(ihsg_curve_raw/ihsg_curve_raw.cummax()-1).min()*100

            comparison=pd.DataFrame({
                "Metric":["Total Return %","CAGR %","Max Drawdown %"],
                "BO Strategy":[res["Strategy Return %"],res["Strategy CAGR %"],res["Max DD %"]],
                f"{res['Ticker']} Buy & Hold":[res["Buy&Hold %"],res["Buy&Hold CAGR %"],res["Buy&Hold Max DD %"]],
                "IHSG Buy & Hold":[ihsg_return,ihsg_cagr,ihsg_dd],
            })
            comparison["BO vs Stock B&H"]=comparison["BO Strategy"]-comparison[f"{res['Ticker']} Buy & Hold"]
            comparison["BO vs IHSG"]=comparison["BO Strategy"]-comparison["IHSG Buy & Hold"]
            st.dataframe(comparison.round(2),use_container_width=True,hide_index=True)

            st.subheader("Trade Statistics")
            stats=pd.DataFrame({
                "Metric":["Closed Trades","Win Rate %","Profit Factor","Expectancy %","Average Holding Days","ATR %","ADX","Liquidity B/day"],
                "Value":[res["Trades"],res["Win Rate %"],res["Profit Factor"],res["Expectancy %"],res["Average Holding Days"],res["ATR %"],res["ADX"],res["Liquidity B/day"]]
            })
            st.dataframe(stats.round(2),use_container_width=True,hide_index=True)

            if res["_running"]:
                st.subheader("Current BO Position")
                st.dataframe(pd.DataFrame([res["_running"]]).round(2),use_container_width=True,hide_index=True)

            st.subheader("Last 50 Closed Trades")
            st.dataframe(res["_trades"].tail(50).round(2),use_container_width=True,hide_index=True)

elif page=="Watchlist":
    st.subheader("My Watchlist")
    st.caption("Watchlist tersimpan selama session aplikasi aktif.")
    wrows=[]
    for t in st.session_state.watchlist:
        try:
            r=analyze_stock(t,left,right,data_start,backtest_start)
            if r: wrows.append({k:v for k,v in r.items() if not k.startswith("_")})
        except: pass
    if wrows:
        wdf=pd.DataFrame(wrows)
        st.dataframe(wdf[["Ticker","Setup","Fit Score","Close","Pivot High","Distance Entry %","Strategy Return %","Buy&Hold %","Alpha %","Profit Factor"]].round(2),use_container_width=True,hide_index=True)
    else:
        st.info("Watchlist masih kosong.")

elif page=="Portfolio":
    st.subheader("Portfolio Builder")
    st.caption(
        "Portfolio manual benar-benar personal: pilih saham dari seluruh 951 IDX Stock Master. "
        "Saham dapat ditambahkan tanpa harus lolos filter AUTO."
    )

    build_tab, analyze_tab = st.tabs(["➕ Build My Portfolio","📊 Analyze Portfolio"])

    with build_tab:
        st.markdown("### Search Stock")
        selected_label = st.selectbox(
            "Search from IDX Stock Master",
            idx_master["label"].tolist(),
            help="Ketik kode saham atau nama emiten."
        )
        selected_code = master_code_from_label(selected_label)
        selected_ticker = normalize_ticker(selected_code)

        with st.spinner(f"Loading {selected_code}..."):
            preview = analyze_stock(selected_ticker,left,right,data_start,backtest_start)

        if preview is None:
            st.warning("Data saham tidak tersedia / histori belum cukup.")
        else:
            st.markdown(f"### {preview['Ticker']} — Stock Preview")
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Close",f"{preview['Close']:,.0f}")
            c2.metric("Setup",preview["Setup"])
            c3.metric("Opportunity",f"{preview['Opportunity Score']:.1f}")
            c4.metric("Fit Score",f"{preview['Fit Score']:.1f}")
            c5.metric(
                "Distance Entry",
                f"{preview['Distance Entry %']:.2f}%" if pd.notna(preview["Distance Entry %"]) else "N/A"
            )

            c6,c7,c8,c9 = st.columns(4)
            c6.metric("Pivot High",f"{preview['Pivot High']:,.0f}" if pd.notna(preview["Pivot High"]) else "N/A")
            c7.metric("BO Return",f"{preview['Strategy Return %']:.1f}%" if pd.notna(preview["Strategy Return %"]) else "N/A")
            c8.metric("Buy & Hold",f"{preview['Buy&Hold %']:.1f}%" if pd.notna(preview["Buy&Hold %"]) else "N/A")
            c9.metric("Profit Factor",f"{preview['Profit Factor']:.2f}" if pd.notna(preview["Profit Factor"]) else "N/A")

            b1,b2 = st.columns(2)
            if b1.button("➕ Tambahkan ke Portfolio",type="primary",use_container_width=True):
                if selected_ticker not in st.session_state.manual_portfolio:
                    st.session_state.manual_portfolio.append(selected_ticker)
                    st.success(f"{selected_code} ditambahkan.")
                    st.rerun()
                else:
                    st.info(f"{selected_code} sudah ada di portfolio.")

            if b2.button("⭐ Tambahkan ke Watchlist",use_container_width=True):
                if selected_ticker not in st.session_state.watchlist:
                    st.session_state.watchlist.append(selected_ticker)
                    st.success(f"{selected_code} ditambahkan ke Watchlist.")

        st.divider()
        st.markdown("### My Manual Portfolio")

        if not st.session_state.manual_portfolio:
            st.info("Portfolio manual masih kosong.")
        else:
            rows = []
            for ticker in st.session_state.manual_portfolio:
                try:
                    r=analyze_stock(ticker,left,right,data_start,backtest_start)
                    if r:
                        rows.append({
                            "Ticker":r["Ticker"],
                            "Setup":r["Setup"],
                            "Opportunity Score":r["Opportunity Score"],
                            "Fit Score":r["Fit Score"],
                            "Close":r["Close"],
                            "Pivot High":r["Pivot High"],
                            "Distance Entry %":r["Distance Entry %"],
                            "BO Return %":r["Strategy Return %"],
                            "BuyHold %":r["Buy&Hold %"],
                            "Alpha %":r["Alpha %"],
                            "Profit Factor":r["Profit Factor"]
                        })
                except Exception:
                    pass

            if rows:
                manual_df=pd.DataFrame(rows)
                st.dataframe(manual_df.round(2),use_container_width=True,hide_index=True)

            remove_code=st.selectbox(
                "Remove stock",
                [x.replace(".JK","") for x in st.session_state.manual_portfolio]
            )
            r1,r2=st.columns(2)
            if r1.button("🗑 Remove Selected",use_container_width=True):
                rt=normalize_ticker(remove_code)
                if rt in st.session_state.manual_portfolio:
                    st.session_state.manual_portfolio.remove(rt)
                    st.rerun()
            if r2.button("Clear Manual Portfolio",use_container_width=True):
                st.session_state.manual_portfolio=[]
                st.rerun()

    with analyze_tab:
        st.markdown("### Portfolio Source")
        source_mode=st.radio(
            "Choose source",
            ["MY MANUAL PORTFOLIO","AUTO RADAR CANDIDATES","AUTO + MANUAL"],
            horizontal=True
        )

        manual_codes=[x.replace(".JK","") for x in st.session_state.manual_portfolio]
        auto_full=st.session_state.get("auto_full_scan",pd.DataFrame())

        if not auto_full.empty:
            auto_candidates=auto_full[
                auto_full["Setup"].isin(["READY TO BUY","NEAR ENTRY","WATCH"])
            ].sort_values(
                ["Opportunity Score","Distance Entry %"],
                ascending=[False,True]
            )["Ticker"].tolist()
        else:
            auto_candidates=[]

        if source_mode=="MY MANUAL PORTFOLIO":
            portfolio_options=manual_codes
        elif source_mode=="AUTO RADAR CANDIDATES":
            portfolio_options=auto_candidates
        else:
            portfolio_options=list(dict.fromkeys(auto_candidates+manual_codes))

        if not portfolio_options:
            st.info(
                "Belum ada saham pada source ini. Tambahkan saham manual atau jalankan "
                "Smart AUTO Scanner → Full BO Scanner."
            )
        else:
            selected_names=st.multiselect(
                "Select stocks for analysis",
                portfolio_options,
                default=portfolio_options[:max_positions]
            )

            rows=[]
            for name in selected_names:
                try:
                    r=analyze_stock(normalize_ticker(name),left,right,data_start,backtest_start)
                    if r:
                        rows.append(r)
                except Exception:
                    pass

            if rows:
                ptab=pd.DataFrame([{
                    "Ticker":r["Ticker"],
                    "Setup":r["Setup"],
                    "Opportunity Score":r["Opportunity Score"],
                    "Fit Score":r["Fit Score"],
                    "BO Return %":r["Strategy Return %"],
                    "BuyHold %":r["Buy&Hold %"],
                    "Alpha %":r["Alpha %"],
                    "BO Max DD %":r["Max DD %"],
                    "BuyHold Max DD %":r["Buy&Hold Max DD %"],
                    "PF":r["Profit Factor"]
                } for r in rows])

                st.dataframe(ptab.round(2),use_container_width=True,hide_index=True)

                ihsg=download_stock("^JKSE",data_start)
                ihsg=ihsg[ihsg.index>=backtest_start] if not ihsg.empty else ihsg
                ihsg_ret=((ihsg["Close"].iloc[-1]/ihsg["Close"].iloc[0]-1)*100) if len(ihsg)>1 else np.nan

                m1,m2,m3,m4=st.columns(4)
                m1.metric("Selected Stocks",len(rows))
                m2.metric("Avg BO Return",f"{ptab['BO Return %'].mean():.1f}%")
                m3.metric("Avg Buy & Hold",f"{ptab['BuyHold %'].mean():.1f}%")
                m4.metric("IHSG Buy & Hold",f"{ihsg_ret:.1f}%" if pd.notna(ihsg_ret) else "N/A")

                st.info(
                    "V7 sudah memakai builder personal + AUTO Radar source. "
                    "Full dynamic daily-MTM portfolio engine akan menjadi tahap berikutnya."
                )
            else:
                st.warning("Belum ada saham yang berhasil dianalisis.")

elif page=="Watchlist":
    st.subheader("My Watchlist")
    st.caption("Watchlist tersimpan selama session aplikasi aktif.")
    wrows=[]
    for t in st.session_state.watchlist:
        try:
            r=analyze_stock(t,left,right,data_start,backtest_start)
            if r: wrows.append({k:v for k,v in r.items() if not k.startswith("_")})
        except: pass
    if wrows:
        wdf=pd.DataFrame(wrows)
        st.dataframe(wdf[["Ticker","Setup","Fit Score","Close","Pivot High","Distance Entry %","Strategy Return %","Buy&Hold %","Alpha %","Profit Factor"]].round(2),use_container_width=True,hide_index=True)
    else:
        st.info("Watchlist masih kosong.")

elif page=="Portfolio":
    st.subheader("Portfolio Research")
    if scanner.empty:
        portfolio_options = list(dict.fromkeys(
            [x.replace(".JK","") for x in manual] +
            [x.replace(".JK","") for x in st.session_state.watchlist]
        ))
        default_portfolio = portfolio_options[:max_positions]
        st.info("Main Scanner belum tersedia. Portfolio sementara memakai Manual Tickers + Watchlist.")
    else:
        qualified=scanner[scanner["Fit Score"]>=min_fit].copy()
        portfolio_options=scanner["Ticker"].tolist()
        default_portfolio=qualified["Ticker"].head(max_positions).tolist()

    selected_names=st.multiselect("Select stocks",portfolio_options,default=default_portfolio)
    if not selected_names:
        st.warning("Pilih minimal satu saham."); st.stop()

    rows=[]
    for name in selected_names:
        r=analyze_stock(normalize_ticker(name),left,right,data_start,backtest_start)
        if r:
            rows.append(r)
    ptab=pd.DataFrame([{
        "Ticker":r["Ticker"],"Fit Score":r["Fit Score"],"Setup":r["Setup"],
        "BO Return %":r["Strategy Return %"],"BuyHold %":r["Buy&Hold %"],
        "Alpha %":r["Alpha %"],"BO Max DD %":r["Max DD %"],
        "BuyHold Max DD %":r["Buy&Hold Max DD %"],"PF":r["Profit Factor"]
    } for r in rows])
    if ptab.empty:
        st.error("Tidak ada saham portfolio yang berhasil dianalisis saat ini.")
        st.stop()
    st.dataframe(ptab.round(2),use_container_width=True,hide_index=True)

    ihsg=download_stock("^JKSE",data_start)
    ihsg=ihsg[ihsg.index>=backtest_start] if not ihsg.empty else ihsg
    ihsg_ret=((ihsg["Close"].iloc[-1]/ihsg["Close"].iloc[0]-1)*100) if len(ihsg)>1 else np.nan

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Selected Stocks",len(rows))
    c2.metric("Avg BO Return",f"{ptab['BO Return %'].mean():.1f}%")
    c3.metric("Avg Buy & Hold",f"{ptab['BuyHold %'].mean():.1f}%")
    c4.metric("IHSG Buy & Hold",f"{ihsg_ret:.1f}%" if pd.notna(ihsg_ret) else "N/A")

    st.warning("Portfolio v2 masih berupa research comparison. Full dynamic daily-MTM portfolio engine dari notebook akan dipindahkan ke web pada upgrade berikutnya.")
