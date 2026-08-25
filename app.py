
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(
    page_title="BO Stock Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# THEME / CSS
# =========================
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: #151515;
    border: 1px solid #333;
    padding: 12px;
    border-radius: 10px;
}
div[data-testid="stDataFrame"] {border: 1px solid #333; border-radius: 8px;}
.small-note {color:#9ca3af; font-size:0.85rem;}
.status-ready {color:#22c55e; font-weight:700;}
.status-near {color:#84cc16; font-weight:700;}
.status-wait {color:#e5e7eb; font-weight:700;}
.status-risk {color:#f97316; font-weight:700;}
.status-avoid {color:#ef4444; font-weight:700;}
</style>
""", unsafe_allow_html=True)

# =========================
# DEFAULTS
# =========================
DEFAULT_AUTO = [
    "ANTM.JK","PTBA.JK","ADRO.JK","INCO.JK","MDKA.JK","ITMG.JK","MEDC.JK","PGAS.JK",
    "BMRI.JK","BBRI.JK","BBCA.JK","BBNI.JK","TLKM.JK","ASII.JK","UNTR.JK","ICBP.JK",
    "INDF.JK","KLBF.JK","CPIN.JK","JPFA.JK"
]

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
    raw = yf.download(ticker, start=data_start, auto_adjust=False, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open","High","Low","Close","Volume"]].dropna().copy()
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

def fit_score_current(bt, trades, backtest_start):
    hist = trades.copy()
    total = len(hist)
    wins = hist[hist["Return %"] > 0] if total else pd.DataFrame()
    losses = hist[hist["Return %"] < 0] if total else pd.DataFrame()

    avg_win = wins["Return %"].mean() if len(wins) else np.nan
    avg_loss = abs(losses["Return %"].mean()) if len(losses) else np.nan
    wl = avg_win/avg_loss if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0 else np.nan
    expectancy = hist["Return %"].mean() if total else np.nan
    gp = wins["Return %"].sum() if len(wins) else 0
    gl = abs(losses["Return %"].sum()) if len(losses) else 0
    pf = gp/gl if gl > 0 else (999 if gp > 0 else np.nan)

    eq = 1.0
    eqs = []
    for r in hist["Return %"] if total else []:
        eq *= (1 + r/100)
        eqs.append(eq)
    strategy_return = (eq-1)*100 if total else np.nan
    if eqs:
        arr = np.array(eqs)
        peak = np.maximum.accumulate(arr)
        max_dd = float(((arr/peak)-1).min()*100)
    else:
        max_dd = np.nan

    period = bt[bt.index >= backtest_start]
    buy_hold = (
        (float(period["Close"].iloc[-1])/float(period["Close"].iloc[0])-1)*100
        if len(period) > 1 else np.nan
    )
    alpha = strategy_return - buy_hold if pd.notna(strategy_return) else np.nan

    close = float(bt["Close"].iloc[-1])
    ema = bt["Close"].ewm(span=200, adjust=False).mean()
    ema_now = float(ema.iloc[-1])
    atr_pct = float(atr(bt,14).iloc[-1] / close * 100)
    _,_,adx_s = dmi_adx(bt,14,14)
    adx_now = float(adx_s.iloc[-1]) if pd.notna(adx_s.iloc[-1]) else np.nan
    liq_b = float(((bt["Close"]*bt["Volume"]).rolling(20).mean().iloc[-1])/1e9)

    pf_score = 0 if pd.isna(pf) else clamp((pf-1)*15,0,15)
    exp_score = 0 if pd.isna(expectancy) else clamp(expectancy/3*10,0,10)
    alpha_score = 0 if pd.isna(alpha) else clamp(alpha/20*10,0,10)
    wl_score = 0 if pd.isna(wl) else clamp((wl-1)*10,0,10)
    sample_score = clamp(total/15*5,0,5)

    abs_dd = abs(max_dd) if pd.notna(max_dd) else 999
    dd_score = 10 if abs_dd <= 15 else 0 if abs_dd >= 40 else (40-abs_dd)/25*10
    adx_score = 0 if pd.isna(adx_now) else clamp((adx_now-15)/15*10,0,10)
    ema_score = (5 if close > ema_now else 0) + (5 if len(ema)>20 and ema.iloc[-1] > ema.iloc[-21] else 0)

    if 1 <= atr_pct <= 6:
        atr_score = 10
    elif atr_pct < 1:
        atr_score = clamp(atr_pct*10,0,10)
    else:
        atr_score = clamp((10.5-atr_pct)/(10.5-6)*10,0,10)

    liq_score = clamp(liq_b/50*10,0,10)
    fit = clamp(pf_score+exp_score+alpha_score+wl_score+sample_score+dd_score+adx_score+ema_score+atr_score+liq_score,0,100)

    return {
        "Fit Score": fit,
        "Profit Factor": pf,
        "Expectancy %": expectancy,
        "Strategy Return %": strategy_return,
        "Buy&Hold %": buy_hold,
        "Alpha %": alpha,
        "Max DD %": max_dd,
        "ADX": adx_now,
        "ATR %": atr_pct,
        "Liquidity B/day": liq_b,
        "EMA200": ema_now
    }

def analyze_stock(ticker, left, right, data_start, backtest_start, near_pct=3.0):
    df = download_stock(ticker, data_start)
    if df.empty or len(df) < 250:
        return None

    bt = pine_like_pivots(df, left, right)
    trades, running = standalone_trade_history(bt, backtest_start)
    stats = fit_score_current(bt, trades, backtest_start)

    close = float(bt["Close"].iloc[-1])
    ph = float(bt["pivotHigh"].iloc[-1]) if pd.notna(bt["pivotHigh"].iloc[-1]) else np.nan
    pl = float(bt["pivotLow"].iloc[-1]) if pd.notna(bt["pivotLow"].iloc[-1]) else np.nan
    dist = (ph-close)/close*100 if pd.notna(ph) else np.nan
    setup_risk = (ph-pl)/ph*100 if pd.notna(ph) and pd.notna(pl) and ph>0 else np.nan
    extension = (close/stats["EMA200"]-1)*100 if stats["EMA200"] else np.nan

    if running:
        setup = "IN POSITION"
    elif stats["Fit Score"] < 45:
        setup = "AVOID"
    elif pd.notna(extension) and extension > 15:
        setup = "OVEREXTENDED"
    elif pd.notna(setup_risk) and setup_risk > 10:
        setup = "HIGH RISK"
    elif bool(bt["breakoutBull"].iloc[-1]):
        setup = "READY TO BUY"
    elif pd.notna(dist) and 0 <= dist <= near_pct:
        setup = "NEAR ENTRY"
    else:
        setup = "WAIT"

    wins = trades[trades["Return %"]>0] if not trades.empty else pd.DataFrame()
    wr = len(wins)/len(trades)*100 if len(trades) else np.nan

    return {
        "Ticker": ticker.replace(".JK",""),
        "Setup": setup,
        "Fit Score": stats["Fit Score"],
        "Close": close,
        "Pivot High": ph,
        "Pivot Low": pl,
        "Distance Entry %": dist,
        "Setup Risk %": setup_risk,
        "Trades": len(trades),
        "Win Rate %": wr,
        "Profit Factor": stats["Profit Factor"],
        "Expectancy %": stats["Expectancy %"],
        "Strategy Return %": stats["Strategy Return %"],
        "Buy&Hold %": stats["Buy&Hold %"],
        "Alpha %": stats["Alpha %"],
        "Max DD %": stats["Max DD %"],
        "ADX": stats["ADX"],
        "ATR %": stats["ATR %"],
        "Liquidity B/day": stats["Liquidity B/day"],
        "Trend": "BULLISH" if close > stats["EMA200"] else "BEARISH",
        "_bt": bt,
        "_trades": trades,
        "_running": running
    }

# =========================
# SIDEBAR SETTINGS
# =========================
st.sidebar.title("BO Stock Settings")
page = st.sidebar.radio("Page", ["Dashboard","Scanner","Universe","Stock Detail","Portfolio"])

left = st.sidebar.number_input("Pivot Left", min_value=1, value=4, step=1)
right = st.sidebar.number_input("Pivot Right", min_value=1, value=4, step=1)
initial_capital = st.sidebar.number_input("Initial Capital (IDR)", min_value=1_000_000, value=100_000_000, step=10_000_000)
min_fit = st.sidebar.slider("Minimum Fit Score", 0, 100, 45)
max_positions = st.sidebar.number_input("Max Positions", min_value=1, max_value=20, value=5, step=1)
backtest_start_input = st.sidebar.date_input("Backtest Start", value=pd.Timestamp("2017-08-20"))
backtest_start = pd.Timestamp(backtest_start_input)

universe_mode = st.sidebar.selectbox("Universe Mode", ["AUTO","MANUAL","AUTO+MANUAL"])
manual_text = st.sidebar.text_area("Manual Tickers", value="ANTM, INCO, ADRO, PTBA, BMRI")
manual = [normalize_ticker(x) for x in manual_text.replace("\n",",").split(",")]
manual = [x for x in manual if x]

if universe_mode == "MANUAL":
    universe = manual
elif universe_mode == "AUTO+MANUAL":
    universe = list(dict.fromkeys(DEFAULT_AUTO + manual))
else:
    universe = DEFAULT_AUTO

st.sidebar.caption(f"{len(universe)} tickers active")
data_start = "2005-01-01"

# =========================
# SCAN
# =========================
@st.cache_data(ttl=3600, show_spinner="Scanning universe...")
def scan_universe(universe_tuple, left, right, data_start, backtest_start_str):
    rows = []
    objects = {}
    for ticker in universe_tuple:
        try:
            res = analyze_stock(ticker, left, right, data_start, pd.Timestamp(backtest_start_str))
            if res:
                objects[ticker] = res
                rows.append({k:v for k,v in res.items() if not k.startswith("_")})
        except Exception:
            pass
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Fit Score","Distance Entry %"], ascending=[False,True]).reset_index(drop=True)
        df.insert(0,"Rank",np.arange(1,len(df)+1))
    return df, objects

scanner, objects = scan_universe(tuple(universe), left, right, data_start, str(backtest_start.date()))

# =========================
# HEADER
# =========================
st.title("BO Stock Analytics")
st.caption("Breakout Pivot 4/4 research dashboard • Python/Yahoo Finance • No order execution")

if scanner.empty:
    st.error("No scanner data available. Try another universe or refresh later.")
    st.stop()

ready_count = int(scanner["Setup"].isin(["READY TO BUY","NEAR ENTRY"]).sum())
inpos_count = int((scanner["Setup"]=="IN POSITION").sum())
avg_fit = float(scanner["Fit Score"].mean())

# =========================
# DASHBOARD
# =========================
if page == "Dashboard":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Universe", len(scanner))
    c2.metric("Ready / Near", ready_count)
    c3.metric("In Position", inpos_count)
    c4.metric("Avg Fit", f"{avg_fit:.1f}")

    st.subheader("Top Opportunities")
    cols = ["Rank","Ticker","Setup","Fit Score","Close","Pivot High","Distance Entry %","Trades","Win Rate %","Profit Factor","Expectancy %","Max DD %","Trend"]
    st.dataframe(scanner[cols].head(10).round(2), use_container_width=True, hide_index=True)

    st.subheader("Fit Score Distribution")
    fig, ax = plt.subplots(figsize=(10,3.5))
    ax.hist(scanner["Fit Score"].dropna(), bins=10)
    ax.set_xlabel("Fit Score")
    ax.set_ylabel("Stocks")
    ax.grid(True, alpha=0.2)
    st.pyplot(fig, use_container_width=True)

# =========================
# SCANNER
# =========================
elif page == "Scanner":
    st.subheader("Scanner")
    status_filter = st.multiselect(
        "Setup filter",
        sorted(scanner["Setup"].dropna().unique().tolist()),
        default=sorted(scanner["Setup"].dropna().unique().tolist())
    )
    min_fit_view = st.slider("View minimum Fit Score", 0, 100, int(min_fit))
    view = scanner[(scanner["Setup"].isin(status_filter)) & (scanner["Fit Score"] >= min_fit_view)]
    st.dataframe(view.round(2), use_container_width=True, hide_index=True)

# =========================
# UNIVERSE
# =========================
elif page == "Universe":
    st.subheader("Universe")
    st.write(f"Mode aktif: **{universe_mode}**")
    st.dataframe(scanner[[
        "Rank","Ticker","Setup","Fit Score","Strategy Return %","Buy&Hold %","Alpha %",
        "Max DD %","Trades","Win Rate %","Profit Factor"
    ]].round(2), use_container_width=True, hide_index=True)

    st.info("Untuk mengganti saham manual, edit daftar di sidebar. Mode AUTO+MANUAL menggabungkan shortlist default dengan pilihan Anda.")

# =========================
# STOCK DETAIL
# =========================
elif page == "Stock Detail":
    selected = st.selectbox("Choose stock", scanner["Ticker"].tolist())
    ticker_full = normalize_ticker(selected)
    res = objects.get(ticker_full)

    if res:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Fit Score", f"{res['Fit Score']:.1f}")
        c2.metric("Setup", res["Setup"])
        c3.metric("BO Return", f"{res['Strategy Return %']:.1f}%")
        c4.metric("Buy & Hold", f"{res['Buy&Hold %']:.1f}%")

        c5,c6,c7,c8 = st.columns(4)
        c5.metric("Alpha vs B&H", f"{res['Alpha %']:.1f}%")
        c6.metric("Profit Factor", f"{res['Profit Factor']:.2f}" if pd.notna(res['Profit Factor']) else "N/A")
        c7.metric("Trades", res["Trades"])
        c8.metric("Win Rate", f"{res['Win Rate %']:.1f}%")

        st.subheader("Price + Pivot")
        bt = res["_bt"].tail(300)
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(bt.index, bt["Close"], label="Close")
        ax.plot(bt.index, bt["pivotHigh"], label="Pivot High")
        ax.plot(bt.index, bt["pivotLow"], label="Pivot Low")
        ax.legend()
        ax.grid(True, alpha=0.2)
        st.pyplot(fig, use_container_width=True)

        st.subheader("Breakout vs Buy & Hold")
        comp = pd.DataFrame({
            "Metric":["Total Return %","Alpha %","Max DD %","Trades","Win Rate %","Profit Factor"],
            "Breakout":[res["Strategy Return %"],res["Alpha %"],res["Max DD %"],res["Trades"],res["Win Rate %"],res["Profit Factor"]],
            "Buy & Hold":[res["Buy&Hold %"],0,np.nan,np.nan,np.nan,np.nan]
        })
        st.dataframe(comp.round(2), use_container_width=True, hide_index=True)

        st.subheader("Completed Trades")
        st.dataframe(res["_trades"].tail(50).round(2), use_container_width=True, hide_index=True)

# =========================
# PORTFOLIO
# =========================
elif page == "Portfolio":
    st.subheader("Portfolio Snapshot")

    qualified = scanner[scanner["Fit Score"] >= min_fit].copy()
    selected_names = st.multiselect(
        "Select stocks for portfolio comparison",
        scanner["Ticker"].tolist(),
        default=qualified["Ticker"].head(max_positions).tolist()
    )

    if not selected_names:
        st.warning("Select at least one stock.")
        st.stop()

    # Simple equal-weight strategy-equity comparison using standalone compounded trade returns.
    # Full historical dynamic-MTM engine remains the next backend upgrade.
    per_stock = []
    for name in selected_names:
        r = objects.get(normalize_ticker(name))
        if r:
            per_stock.append(r)

    bo_returns = [r["Strategy Return %"] for r in per_stock if pd.notna(r["Strategy Return %"])]
    bh_returns = [r["Buy&Hold %"] for r in per_stock if pd.notna(r["Buy&Hold %"])]

    avg_bo = np.mean(bo_returns) if bo_returns else np.nan
    avg_bh = np.mean(bh_returns) if bh_returns else np.nan

    # IHSG benchmark
    ihsg = download_stock("^JKSE", data_start)
    ihsg = ihsg[ihsg.index >= backtest_start] if not ihsg.empty else ihsg
    ihsg_ret = ((ihsg["Close"].iloc[-1]/ihsg["Close"].iloc[0]-1)*100) if len(ihsg)>1 else np.nan

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Selected Stocks", len(per_stock))
    c2.metric("Avg BO Return", f"{avg_bo:.1f}%" if pd.notna(avg_bo) else "N/A")
    c3.metric("Avg Buy & Hold", f"{avg_bh:.1f}%" if pd.notna(avg_bh) else "N/A")
    c4.metric("IHSG Buy & Hold", f"{ihsg_ret:.1f}%" if pd.notna(ihsg_ret) else "N/A")

    st.subheader("Selected Universe Comparison")
    ptab = pd.DataFrame([{
        "Ticker":r["Ticker"],
        "Fit Score":r["Fit Score"],
        "Setup":r["Setup"],
        "BO Return %":r["Strategy Return %"],
        "BuyHold %":r["Buy&Hold %"],
        "Alpha %":r["Alpha %"],
        "Max DD %":r["Max DD %"],
        "PF":r["Profit Factor"],
    } for r in per_stock])
    st.dataframe(ptab.round(2), use_container_width=True, hide_index=True)

    st.warning(
        "Portfolio page v1.0 currently shows portfolio research snapshot. "
        "The exact dynamic daily-MTM engine from notebook v0.7 will be integrated in the next backend upgrade."
    )
