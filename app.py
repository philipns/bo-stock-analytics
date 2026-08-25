
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import StringIO
from pathlib import Path

st.set_page_config(
    page_title="BO Stock Analytics v4",
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

    if running: setup="IN POSITION"
    elif stats["Fit Score"]<45: setup="AVOID"
    elif pd.notna(extension) and extension>15: setup="OVEREXTENDED"
    elif pd.notna(setup_risk) and setup_risk>10: setup="HIGH RISK"
    elif bool(bt["breakoutBull"].iloc[-1]): setup="READY TO BUY"
    elif pd.notna(dist) and 0<=dist<=near_pct: setup="NEAR ENTRY"
    else: setup="WAIT"

    wins = trades[trades["Return %"]>0] if not trades.empty else pd.DataFrame()
    wr = len(wins)/len(trades)*100 if len(trades) else np.nan
    avg_hold = trades["Holding Days"].mean() if len(trades) else np.nan

    result = {
        "Ticker":ticker.replace(".JK",""),"Setup":setup,"Fit Score":stats["Fit Score"],
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
        "READY TO BUY":"ready","NEAR ENTRY":"near","IN POSITION":"position",
        "WAIT":"wait","HIGH RISK":"risk","OVEREXTENDED":"risk","AVOID":"avoid"
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

@st.cache_data(ttl=3600,show_spinner="Scanning universe...")
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
        df=df.sort_values(["Fit Score","Distance Entry %"],ascending=[False,True]).reset_index(drop=True)
        df.insert(0,"Rank",np.arange(1,len(df)+1))
    return df

scanner=scan_universe(tuple(universe),left,right,data_start,str(backtest_start.date()))

st.title("BO Stock Analytics v4")
st.caption("IDX breakout research • Repository-backed Stock Master • Pivot 4/4 • Python/Yahoo Finance")

if page in ["Dashboard","Scanner","Universe","Portfolio"] and scanner.empty:
    st.error("Scanner data unavailable. Try another universe or refresh later.")
    st.stop()

if page=="Dashboard":
    ready_count=int(scanner["Setup"].isin(["READY TO BUY","NEAR ENTRY"]).sum())
    inpos=int((scanner["Setup"]=="IN POSITION").sum())
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Scanner Universe",len(scanner))
    c2.metric("Ready / Near",ready_count)
    c3.metric("In Position",inpos)
    c4.metric("Average Fit",f"{scanner['Fit Score'].mean():.1f}")

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
    st.subheader("Scanner")
    st.caption("Default scanner memakai universe ringan. Gunakan Custom Batch untuk memilih saham dari seluruh IDX Stock Master tanpa mengubah universe utama.")

    with st.expander("Custom Batch Scanner"):
        batch_labels = st.multiselect(
            "Choose stocks from IDX master",
            idx_master["label"].tolist(),
            default=[],
            help="Sebaiknya 10–50 saham per batch agar stabil di Streamlit Free."
        )
        batch_limit = st.slider("Maximum stocks per batch", 5, 100, 30, 5)
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

    st.divider()
    st.subheader("Main Scanner Universe")
    status_filter=st.multiselect("Setup filter",sorted(scanner["Setup"].dropna().unique()),default=sorted(scanner["Setup"].dropna().unique()))
    view=scanner[(scanner["Setup"].isin(status_filter))&(scanner["Fit Score"]>=min_fit)]
    st.dataframe(view.round(2),use_container_width=True,hide_index=True)

elif page=="Universe":
    st.subheader("Universe")
    st.write(f"Mode aktif: **{universe_mode}**")
    st.caption("Scanner Universe adalah subset untuk ranking rutin. Stock Detail tidak dibatasi oleh daftar ini.")
    st.dataframe(scanner[["Rank","Ticker","Setup","Fit Score","Strategy Return %","Buy&Hold %","Alpha %","Max DD %","Trades","Win Rate %","Profit Factor"]].round(2),use_container_width=True,hide_index=True)

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

            st.subheader("Price + Pivot Stair")
            bt=res["_bt"].tail(400)
            fig,ax=plt.subplots(figsize=(12,5))
            ax.plot(bt.index,bt["Close"],label="Close")
            ax.plot(bt.index,bt["pivotHigh"],label="Pivot High")
            ax.plot(bt.index,bt["pivotLow"],label="Pivot Low")
            ax.grid(True,alpha=.2); ax.legend()
            st.pyplot(fig,use_container_width=True)

            st.subheader("Breakout Strategy vs Buy & Hold")
            comparison=pd.DataFrame({
                "Metric":["Total Return %","CAGR %","Max Drawdown %"],
                "Breakout":[res["Strategy Return %"],res["Strategy CAGR %"],res["Max DD %"]],
                "Buy & Hold":[res["Buy&Hold %"],res["Buy&Hold CAGR %"],res["Buy&Hold Max DD %"]],
            })
            comparison["BO - B&H"]=comparison["Breakout"]-comparison["Buy & Hold"]
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
    st.subheader("Portfolio Research")
    qualified=scanner[scanner["Fit Score"]>=min_fit].copy()
    selected_names=st.multiselect("Select stocks",scanner["Ticker"].tolist(),default=qualified["Ticker"].head(max_positions).tolist())
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
