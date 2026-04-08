# ══════════════════════════════════════════════════════════════
#  FILE 3 — Streamlit Dashboard
#  Interactive stock search + validation + quantum decisions
#  streamlit run 03_dashboard.py
#
#  pip install streamlit plotly yfinance
# ══════════════════════════════════════════════════════════════

import json, os, subprocess, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import yfinance as yf

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Portfolio Engine",
    page_icon="Q", layout="wide",
    initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    /* The FinAssist Palette */
    --bg-dark: #1a162b;
    --bg-light: #412a6e;
    --bg-sidebar: #12101f;
    
    --card: rgba(20, 20, 25, 0.6);
    --border: rgba(255,255,255,0.06);
    
    --cyan: #00d4ff;
    --green: #00e588;
    --amber: #ff9f0a;
    --red: #ff453a;
    --purple: #9d4edd;
    --text: #ffffff;
    --muted: #8e8e93;
}

/* SHATTER THE STREAMLIT GREY BOX AND REMOVE WATERMARKS/SHORTCUTS */
[data-testid="stAppViewContainer"] {
    background-color: transparent !important;
}
#MainMenu, footer {
    visibility: hidden !important;
    display: none !important;
}
[data-testid="stHeader"] {
    background-color: transparent !important;
}
/* HIDE THE "PRESS ENTER" KEYBOARD INSTRUCTION */
div[data-testid="InputInstructions"] {
    display: none !important;
    visibility: hidden !important;
}

/* THE GLOBAL FINASSIST BACKGROUND */
.stApp {
    background-color: var(--bg-dark);
    background-image: 
        radial-gradient(circle at 10% 80%, rgba(157, 78, 221, 0.15) 0%, transparent 50%),
        radial-gradient(circle at 90% 20%, rgba(219, 39, 119, 0.1) 0%, transparent 50%);
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: var(--text);
}

/* BASE TYPOGRAPHY */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: var(--text);
    -webkit-font-smoothing: antialiased;
}
h1, h2, h3 { font-family: 'Inter', -apple-system, sans-serif !important; font-weight: 600; }

/* THE SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif !important;
}

/* TABS (Segmented Control) */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(10, 10, 15, 0.5);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.8rem; color: var(--muted); border-radius: 8px; padding: 8px 16px; transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(157,78,221,0.2), rgba(0,212,255,0.1)) !important;
    border: 1px solid rgba(157,78,221,0.4);
    color: #ffffff !important;
}

/* GLOWING METRIC CARDS (Replaces standard kcard/metric-card) */
.kcard, .metric-card {
    background: var(--card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    text-align: left;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    position: relative;
    overflow: hidden;
    margin-bottom: 8px;
}
.kcard:hover, .metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255,255,255,0.15);
}

/* Specific glows for top metrics */
.glow-value { border-top: 2px solid var(--purple); box-shadow: inset 0 20px 20px -20px rgba(157,78,221,0.4); }
.glow-buy   { border-top: 2px solid var(--green);  box-shadow: inset 0 20px 20px -20px rgba(0,229,136,0.4); }
.glow-hold  { border-top: 2px solid var(--amber);  box-shadow: inset 0 20px 20px -20px rgba(255,159,10,0.4); }
.glow-sell  { border-top: 2px solid var(--red);    box-shadow: inset 0 20px 20px -20px rgba(255,69,58,0.4); }
.glow-upd   { border-top: 2px solid var(--cyan);   box-shadow: inset 0 20px 20px -20px rgba(0,212,255,0.4); }

.klabel, .metric-label {
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--muted); margin-bottom: 8px;
    display: block;
}
.kval, .metric-value {
    font-size: 1.6rem; font-weight: 700; color: #ffffff;
}

/* Specialized Colors */
.pos { color: var(--green); }
.neg { color: var(--red);  }
.neu { color: var(--cyan); }

/* Buttons */
.stButton>button {
    font-family: 'Inter', sans-serif; font-weight: 600;
    background: linear-gradient(135deg, rgba(157,78,221,0.2), rgba(219,39,119,0.2));
    color: #ffffff;
    border: 1px solid rgba(157,78,221,0.5);
    border-radius: 12px; padding: 10px 24px;
    transition: all 0.2s;
}
.stButton>button:hover {
    transform: translateY(-2px);
    border-color: rgba(219,39,119,0.8);
    box-shadow: 0 4px 20px rgba(219,39,119,0.3);
}

/* Banner / Headers */
.banner {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(157, 78, 221, 0.15) 100%);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 24px;
    padding: 30px 40px; margin-bottom: 20px;
    box-shadow: 0 10px 40px -10px rgba(0,212,255,0.2), inset 0 0 20px rgba(0,212,255,0.1);
}
.banner-title {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #ffffff 30%, var(--purple));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 8px 0;
}
.shead {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted);
    border-bottom: 1px solid var(--border); padding-bottom: 8px; margin: 24px 0 16px 0;
}
</style>""", unsafe_allow_html=True)

PALETTE   = ["#00d4ff", "#00e588", "#ff9f0a", "#9d4edd",
             "#ff453a", "#fb923c", "#34d399", "#60a5fa"]
DEC_COLOR = {"BUY":"#00e588","HOLD":"#ff9f0a","SELL":"#ff453a"}
DEC_EMOJI = {"BUY":"BUY","HOLD":"HOLD","SELL":"SELL"}
RISK_COLOR= {"Low":"#00e588","Medium":"#ff9f0a","High":"#ff453a"}

PT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif",color="#8e8e93",size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)",linecolor="rgba(255,255,255,0.1)",
               tickfont=dict(color="#8e8e93")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)",linecolor="rgba(255,255,255,0.1)",
               tickfont=dict(color="#8e8e93")),
    legend=dict(bgcolor="rgba(20,25,35,0.5)",bordercolor="rgba(255,255,255,0.1)",
                borderwidth=1),
    margin=dict(l=40,r=20,t=44,b=40),
)

# ── Session state ─────────────────────────────────────────────
for key, val in [
    ("selected_stocks", {}),
    ("portfolio_value", 100000.0),
    ("quota_used", 0),
    ("val_result", None),
    ("bulk_results", None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Helpers ───────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def validate_ticker(ticker: str):
    try:
        ticker = ticker.strip().upper()
        if "." not in ticker:
            ticker = ticker + ".NS"
            
        # yfinance .info hangs or fails often. Bypassing it for reliable download
        hist = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        if hist.empty:
            return {"valid":False,"ticker":ticker,"error":"No data on yFinance"}
            
        # yf > 0.2.40 multi-index fix
        close_data = hist["Close"].iloc[-1]
        price = float(close_data.iloc[0]) if isinstance(close_data, pd.Series) else float(close_data)
        return {
            "valid"    : True,
            "ticker"   : ticker,
            "name"     : ticker.replace(".NS", ""),
            "price"    : price,
            "exchange" : "NSE",
            "sector"   : "—",
            "currency" : "INR",
            "market_cap": 0,
        }
    except Exception as e:
        return {"valid":False,"ticker":ticker,"error":str(e)}

@st.cache_data(ttl=180, show_spinner=False)
def fetch_prices(tickers, period):
    frames = {}
    for name, ticker in tickers.items():
        try:
            raw = yf.download(ticker, period=period,
                               auto_adjust=True, progress=False)
            if not raw.empty:
                frames[name] = raw["Close"].squeeze()
        except Exception:
            pass
    try:
        n = yf.download("^NSEI", period=period,
                          auto_adjust=True, progress=False)
        if not n.empty:
            frames["NIFTY"] = n["Close"].squeeze()
    except Exception:
        pass
    return pd.DataFrame(frames).dropna()

def load_results():
    if os.path.exists("live_results.json"):
        with open("live_results.json") as f:
            return json.load(f)
    return None

@st.cache_data(ttl=900, show_spinner=False)
def scan_market_movers():
    nifty50_tickers = [
        "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
        "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS",
        "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
        "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
        "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS",
        "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
        "LTIM.NS", "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS",
        "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
        "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
        "TECHM.NS", "TITAN.NS", "UPL.NS", "ULTRACEMCO.NS", "WIPRO.NS"
    ]
    import concurrent.futures
    try:
        def fetch_single(t):
            try:
                raw = yf.Ticker(t).history(period="5d")
                if len(raw) >= 2:
                    close_arr = raw["Close"].to_numpy().flatten()
                    hi_arr = raw["High"].to_numpy().flatten()
                    lo_arr = raw["Low"].to_numpy().flatten()
                    op_arr = raw["Open"].to_numpy().flatten()
                    
                    pct = (close_arr[-1] - close_arr[-2]) / close_arr[-2] * 100
                    vol = ((hi_arr[-1] - lo_arr[-1]) / op_arr[-1] * 100) if op_arr[-1] != 0 else 0
                    return t, pct, vol
            except: pass
            return t, None, None

        pct_dict = {}
        vol_dict = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            for t, pct, vol in executor.map(fetch_single, nifty50_tickers):
                if pct is not None:
                    pct_dict[t] = pct
                    vol_dict[t] = vol
                    
        if len(pct_dict) < 2:
            raise ValueError("No data")
            
        pct_series = pd.Series(pct_dict)
        vol_series = pd.Series(vol_dict)
        
        gainers = pct_series.nlargest(7)
        losers = pct_series.nsmallest(7)
        most_volatile = vol_series.nlargest(10)
        
        selected = {}
        for t in gainers.index: selected[t.replace(".NS", "")] = t
        for t in losers.index:
            if t not in selected.values(): selected[t.replace(".NS", "")] = t
        for t in most_volatile.index:
            if t not in selected.values():
                selected[t.replace(".NS", "")] = t
                if len(selected) >= 20: break
                
        if not selected: raise ValueError("Empty selection")
        return selected

    except Exception as e:
        return {"RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "INFY": "INFY.NS", "HDFCBANK": "HDFCBANK.NS", "ICICIBANK": "ICICIBANK.NS"}


# ════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="banner">
    <div class="badge">HYBRID QUANTUM-CLASSICAL · IBM FREE PLAN</div>
    <div class="banner-title"> Quantum Portfolio Engine</div>
    <p class="banner-sub">
        Bloomberg Training &nbsp;|&nbsp;
        yFinance Live &nbsp;|&nbsp;
        NewsAPI Sentiment &nbsp;|&nbsp;
        IBM VQC Decision &nbsp;|&nbsp;
        Performance Metrics
    </p>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='margin-top:-20px;margin-bottom:30px;'>
        <div style='font-family:Inter, sans-serif; font-size:1.6rem; font-weight:700;
                    background:linear-gradient(135deg, #7c3aed, #db2777); 
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            Quantum Engine
        </div>
        <div style='font-size:0.6rem; letter-spacing:0.15em; color:#8e8e93; text-transform:uppercase; margin-top:4px;'>
            HYBRID PORTFOLIO INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    # IBM Quota
    st.markdown('<div class="shead" style="margin-top:10px;">IBM QUOTA</div>',
                unsafe_allow_html=True)
    qu    = st.session_state.quota_used
    qpct  = min(qu/600, 1.0)
    qcol  = "#00e588" if qpct<0.5 else \
            "#ff9f0a" if qpct<0.8 else "#ff453a"
    st.markdown(f"""
    <div style='background:rgba(20,20,30,0.5); border:1px solid rgba(255,255,255,0.05); 
                border-radius:12px; padding:16px; margin-bottom:12px; box-shadow:0 4px 15px rgba(0,0,0,0.2);'>
        <div style='display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:8px;'>
            <div style='font-family:Inter, sans-serif; font-size:0.65rem; color:#8e8e93; font-weight:600; letter-spacing:0.05em;'>QPU USAGE</div>
            <div style='font-family:Inter, sans-serif; font-size:0.9rem; font-weight:700; color:{qcol};'>
                {qu}<span style='font-size:0.65rem; color:#64748b; font-weight:500;'> / 600s</span>
            </div>
        </div>
        <div style='background:rgba(0,0,0,0.3); border-radius:6px; height:8px; width:100%; overflow:hidden; border:1px solid rgba(255,255,255,0.02);'>
            <div style='background:linear-gradient(90deg, {qcol}88, {qcol}); width:{qpct*100:.1f}%; height:100%; border-radius:6px; box-shadow:0 0 10px {qcol}44;'></div>
        </div>
        <div style='font-family:Inter, sans-serif; font-size:0.6rem; color:#64748b; margin-top:10px; text-align:right;'>
            ≈ {int((600-qu)/5)} runs remaining
        </div>
    </div>""", unsafe_allow_html=True)
    if st.button("Reset Quota", use_container_width=True):
        st.session_state.quota_used = 0
        st.rerun()


    # Portfolio value
    st.markdown('<div class="shead">PORTFOLIO VALUE</div>',
                unsafe_allow_html=True)
    port_input = st.number_input(
        "Rs. Amount", min_value=10000,
        max_value=100000000,
        value=int(st.session_state.portfolio_value),
        step=10000, format="%d",
        label_visibility="collapsed")
    st.session_state.portfolio_value = float(port_input)
    st.caption(f"Rs. {port_input:,.0f}")

    # Stock search
    st.markdown('<div class="shead">SEARCH STOCKS</div>',
                unsafe_allow_html=True)
    st.caption("Type NSE ticker — .NS added automatically")
    search = st.text_input(
        "Search", placeholder="e.g. TCS or INFY.NS",
        label_visibility="collapsed", key="search_box")

    if search.strip():
        t = search.strip().upper()
        if "." not in t: t = t + ".NS"
        with st.spinner(f"Checking {t}..."):
            r = validate_ticker(t)
        if r["valid"]:
            st.success(f"[OK] {r['ticker']}")
            st.caption(f"Rs.{r.get('price',0):,.2f} | "
                       f"{r.get('exchange','NSE')}")
            label = st.text_input(
                "Label", value=t.replace(".NS",""),
                key=f"lbl_{t}",
                label_visibility="collapsed")
                
            def add_stock(l, tk):
                st.session_state.selected_stocks[l] = tk
                st.session_state.search_box = ""
                
            st.button("Add to Portfolio",
                      key=f"add_{t}",
                      on_click=add_stock,
                      args=(label, r["ticker"]),
                      use_container_width=True)
        else:
            st.error(f"[X] {t} not found")
            st.caption(r.get("error","Invalid ticker"))

    # ── Selected stocks ───────────────────────────────────────
    st.markdown('<div class="shead">PORTFOLIO STOCKS</div>',
                unsafe_allow_html=True)
    if not st.session_state.selected_stocks:
        st.caption("No stocks added yet")
        if st.button("Scan Today's Movers",
                      use_container_width=True):
            with st.spinner("Scanning NIFTY 50 for gainers, losers & volatility..."):
                st.session_state.selected_stocks = scan_market_movers()
            st.rerun()
    else:
        for name, ticker in list(
                st.session_state.selected_stocks.items()):
            c1,c2 = st.columns([4,1])
            c1.markdown(
                f"<span style='color:#00d4ff;font-size:.72rem;'>"
                f"{name}</span> "
                f"<span style='color:#64748b;font-size:.62rem;'>"
                f"{ticker}</span>",
                unsafe_allow_html=True)
            if c2.button("X", key=f"del_{name}"):
                del st.session_state.selected_stocks[name]
                st.rerun()
        if st.button("Clear All", use_container_width=True):
            st.session_state.selected_stocks = {}
            st.rerun()

    # Settings
    st.markdown('<div class="shead">SETTINGS</div>',
                unsafe_allow_html=True)
    period = st.selectbox("Period",
                           ["1mo","3mo","6mo","1y"], index=1)

    st.markdown("---")
    n_sel   = len(st.session_state.selected_stocks)
    run_btn = st.button(
        f"Run Analysis",
        use_container_width=True,
        disabled=(n_sel == 0 and not os.path.exists("uploaded_portfolio.csv")))

# ════════════════════════════════════════════════════════════════
#  RUN ANALYSIS
# ════════════════════════════════════════════════════════════════
if st.session_state.get("force_run", False):
    run_btn = True
    st.session_state.force_run = False

if run_btn and (st.session_state.selected_stocks or os.path.exists("uploaded_portfolio.csv")):
    cfg = {
        "stocks"         : st.session_state.selected_stocks,
        "portfolio_value": st.session_state.portfolio_value,
        "period"         : period,
        "timestamp"      : datetime.today().isoformat(),
    }
    with open("selected_stocks.json","w") as f:
        json.dump(cfg, f, indent=2)

    st.info(f" Submitting 1 IBM job with "
            f"{n_sel} circuits ({n_sel*1024} shots total)...")
    progress = st.progress(0, text="Initializing...")

    try:
        import sys
        proc = subprocess.Popen(
            [sys.executable, "02_live_quantum_system.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        log_lines = []
        step_map  = {
            "loading"  : (15,"Loading models..."),
            "fetching" : (30,"Fetching yFinance data..."),
            "sentiment": (45,"Computing sentiment..."),
            "feature"  : (60,"Engineering features..."),
            "mlp"      : (75,"Running MLP..."),
            "quantum"  : (85,"Running IBM quantum job..."),
            "job compl": (95,"Processing results..."),
            "saved"    : (99,"Saving results..."),
        }

        with st.expander("Live Log", expanded=True):
            log_area = st.empty()
            for line in proc.stdout:
                line = line.rstrip()
                log_lines.append(line)
                log_area.code(
                    "\n".join(log_lines[-25:]),
                    language="bash")
                for key,(pct,msg) in step_map.items():
                    if key in line.lower():
                        progress.progress(pct, text=msg)

        proc.wait()
        if proc.returncode == 0:
            progress.progress(100, text=" Complete!")
            # Update quota tracker
            st.session_state.quota_used += n_sel * 5
            st.success(
                f" Done! Used ~{n_sel*5}s QPU. "
                f"Total this month: "
                f"~{st.session_state.quota_used}s / 600s")
            st.cache_data.clear()
            time.sleep(1)
            if st.session_state.get('return_to_tab4_soon', False):
                st.session_state.return_to_tab4 = True
                st.session_state.return_to_tab4_soon = False
            st.rerun()
        else:
            progress.empty()
            st.error(" Analysis failed. Check log above.")
    except FileNotFoundError:
        progress.empty()
        st.error(" 02_live_quantum_system.py not found. "
                  "Put both files in same folder.")

# ════════════════════════════════════════════════════════════════
#  DISPLAY RESULTS
# ════════════════════════════════════════════════════════════════
res = load_results()

if res is None:
    # Landing page
    st.markdown('<div class="shead">HOW TO USE</div>',
                unsafe_allow_html=True)
    steps = [
        ("1","Search","Type any NSE ticker in sidebar"),
        ("2","Validate","Live yFinance check"),
        ("3","Add","Add to your portfolio"),
        ("4","Amount","Set Rs. to invest"),
        ("5","Run","1 IBM job -> BUY/HOLD/SELL"),
    ]
    cols = st.columns(5)
    for col,(num,title,sub) in zip(cols,steps):
        col.markdown(f"""
        <div style='background:rgba(20, 25, 35, 0.45);border:1px solid #1e293b;
                    border-radius:12px;padding:14px;
                    text-align:center;'>
            <div style='font-size:1.3rem;margin-bottom:5px;
                        color:#00d4ff;font-family:Inter, sans-serif;'>
            {num}</div>
            <div style='font-family:Inter, sans-serif;font-size:.8rem;
                        color:#00d4ff;margin-bottom:3px;'>
            {title}</div>
            <div style='font-size:.7rem;color:#64748b;'>
            {sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="shead">POPULAR NSE STOCKS</div>',
                unsafe_allow_html=True)
    
    popular = [
        ("RELIANCE","RELIANCE.NS","Energy"),
        ("TCS","TCS.NS","IT"),
        ("INFY","INFY.NS","IT"),
        ("HDFCBANK","HDFCBANK.NS","Banking"),
        ("ICICIBANK","ICICIBANK.NS","Banking"),
        ("WIPRO","WIPRO.NS","IT"),
        ("BAJFINANCE","BAJFINANCE.NS","Finance"),
        ("MARUTI","MARUTI.NS","Auto"),
        ("SUNPHARMA","SUNPHARMA.NS","Pharma"),
        ("TATASTEEL","TATASTEEL.NS","Metal"),
    ]
    pcols = st.columns(5)
    for i,(name,ticker,sector) in enumerate(popular):
        pcols[i%5].markdown(f"""
        <div style='background:rgba(20, 25, 35, 0.45);border:1px solid #1e293b;
                    border-radius:8px;padding:8px 10px;
                    margin-bottom:6px;'>
            <div style='font-family:Inter, sans-serif;font-size:.73rem;
                        color:#00d4ff;'>{name}</div>
            <div style='font-size:.63rem;color:#64748b;'>
                {ticker} · {sector}</div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ── Parse ─────────────────────────────────────────────────────
positions    = res.get("positions",[])
qd_all       = res.get("quantum_decisions",{})
mlp_all      = res.get("mlp_predictions",{})
sent_all     = res.get("sentiment_scores",{})
user_tickers = res.get("user_stocks",{})
port_val     = res.get("portfolio_value",100000)
total_inv    = res.get("total_invested",0)
backend      = res.get("backend","").upper()
ts           = res.get("timestamp","")[:16].replace("T"," ")
shots        = res.get("shots",1024)
qpu_used     = res.get("qpu_seconds_used",0)
buy_thresh   = res.get("buy_threshold",0.55)
sell_thresh  = res.get("sell_threshold",0.45)

buys  = sum(1 for p in positions if p["decision"]=="BUY")
holds = sum(1 for p in positions if p["decision"]=="HOLD")
sells = sum(1 for p in positions if p["decision"]=="SELL")

# ── KPI bar ───────────────────────────────────────────────────
k = st.columns(7)
kpis = [
    ("UPDATED",    ts,                     "neu"),
    ("BACKEND",    backend,                "neu"),
    ("SHOTS",      f"{shots:,}",           "neu"),
    ("🟢 BUY",     str(buys),              "pos"),
    ("🟡 HOLD",    str(holds),             "neu"),
    ("🔴 SELL",    str(sells),             "neg"),
    ("INVESTED",   f"Rs.{total_inv:,.0f}",  "neu"),
]
for col,(lbl,val,cls) in zip(k,kpis):
    g_cls = "glow-buy" if cls=="pos" else ("glow-sell" if cls=="neg" else "glow-upd")
    if lbl == "🟡 HOLD": g_cls = "glow-hold"
    
    col.markdown(f"""
    <div class="kcard {g_cls}">
        <div class="klabel">{lbl}</div>
        <div class="kval {cls}">{val}</div>
    </div>""", unsafe_allow_html=True)

# QPU usage bar
qpu_pct   = min(qpu_used/600, 1.0)
qpu_color = "#00ff9d" if qpu_pct<0.5 \
       else "#f59e0b" if qpu_pct<0.8 \
       else "#f87171"
st.markdown(f"""
<div style='margin-top:8px;background:var(--card);
            border:1px solid var(--border);border-radius:8px;
            padding:7px 16px;display:flex;
            align-items:center;gap:12px;'>
    <span style='font-family:Inter, sans-serif;font-size:.62rem;
                 color:#64748b;'>QPU THIS RUN</span>
    <div style='flex:1;background:#1e293b;
                border-radius:4px;height:5px;'>
        <div style='background:{qpu_color};
                    width:{qpu_pct*100:.1f}%;
                    height:5px;border-radius:4px;'></div>
    </div>
    <span style='font-family:Inter, sans-serif;font-size:.62rem;
                 color:{qpu_color};'>{qpu_used}s used</span>
</div>""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
t1,t2,t3,t4,t5,t6,t7,t8 = st.tabs([
    "Quantum Decisions",
    "Performance Metrics",
    "Model Comparison",
    "Portfolio Advisor",
    "Live Prices",
    "ML + Sentiment",
    "Position Sizing",
    "Stock Validator",
])

if st.session_state.get('return_to_tab4', False):
    st.session_state.return_to_tab4 = False
    st.components.v1.html("""
        <script>
        const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
        if(tabs.length > 3) { tabs[3].click(); }
        </script>
    """, height=0)

# ══════════════════════════════════════════════════════════════
#  TAB 1: Quantum Decisions
# ══════════════════════════════════════════════════════════════
with t1:
    st.markdown('<div class="shead">AI PLAIN ENGLISH BREAKDOWN</div>', unsafe_allow_html=True)
    _ins = []
    if buys == 0 and sells > 0:
        _ins.append("The quantum engine is deeply pessimistic about this selection. No buying opportunities were found, suggesting significant downward momentum across the board.")
    elif buys > (sells + holds):
        _ins.append("The model detected extremely strong upward momentum in your scanned assets. Consider locking in positions on the strongest BUY signals.")
    
    for p in positions:
        if p["signal"] > 0.8:
            _ins.append(f"<b>{p['name']}</b> is showing massive breakout potential (Signal: {p['signal']:.2f}). The hybrid model has very high confidence in its short-term rally.")
        elif p["signal"] < 0.2:
            _ins.append(f"<b>{p['name']}</b> is facing severe selling pressure. Our quantum model detects strong statistical risk of a continued slide.")
    
    if not _ins:
        _ins.append("The quantum model suggests a mixed, relatively stable market environment for these assets. Expect moderate volatility and watch for moving averages.")
        
    st.markdown(f"""
    <div style='background:rgba(20,20,30,0.6); padding:20px; border-radius:12px; border:1px solid rgba(0, 212, 255, 0.2); border-left:4px solid #00d4ff; margin-bottom:20px; box-shadow:0 4px 15px rgba(0,0,0,0.2);'>
        <h4 style='margin-top:0; color:#00d4ff; font-family:Inter, sans-serif; font-size:1.0rem;'>Key Takeaways</h4>
        <div style='color:#e2e8f0; line-height:1.6; font-family:Inter, sans-serif; font-size:0.85rem;'>
            {"<br><br>".join(_ins)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="shead">BUY / HOLD / SELL</div>',
                unsafe_allow_html=True)
    n      = len(positions)
    n_cols = min(n,5)
    for rs in range(0,n,n_cols):
        batch = positions[rs:rs+n_cols]
        dcols = st.columns(len(batch))
        for col,pos in zip(dcols,batch):
            dec   = pos["decision"]
            color = DEC_COLOR[dec]
            emoji = DEC_EMOJI[dec]
            col.markdown(f"""
            <div class="dc-{dec.lower()}">
                <div style='font-family:Inter, sans-serif;
                            font-size:.7rem;color:#64748b;
                            margin-bottom:5px;'>
                {pos['name']}</div>
                <div class="dc-label" style='color:{color};'>
                    {dec}</div>
                <div style='font-size:.72rem;color:#94a3b8;
                            margin-top:6px;'>
                    Signal: {pos['signal']:.3f}<br>
                    Conf: {pos['confidence']:.1f}%<br>
                    P(UP): {pos['prob_up']:.3f}<br>
                    Timeline: {pos.get('timeline_days','--')}d
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="shead">SIGNAL STRENGTH</div>',
                unsafe_allow_html=True)
    
    fig_s = go.Figure(go.Bar(
        x=[p["name"]   for p in positions],
        y=[p["signal"] for p in positions],
        marker_color=[DEC_COLOR[p["decision"]] for p in positions],
        text=[f"{p['signal']:.3f}" for p in positions],
        textposition="outside", showlegend=False))
    fig_s.add_hline(y=buy_thresh, line_dash="dash",
                    line_color="#00ff9d",
                    annotation_text=f"BUY>{buy_thresh:.2f}")
    fig_s.add_hline(y=sell_thresh, line_dash="dash",
                    line_color="#f87171",
                    annotation_text=f"SELL<{sell_thresh:.2f}")
    fig_s.add_hline(y=0.5, line_dash="dot",
                    line_color="#334155")
    fig_s.update_layout(
        **{k:v for k,v in PT.items() if k!="yaxis"},
        height=320,
        yaxis=dict(range=[0,1.15], gridcolor="#1e293b",
                   linecolor="#1e293b",
                   tickfont=dict(color="#64748b")),
        title=dict(
            text=f"Quantum Signal [{shots} shots] "
                 f"[0=Sell to 1=Buy]",
            font=dict(family="Inter, sans-serif",size=12,
                      color="#00d4ff")))
    st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('<div class="shead">PORTFOLIO WEIGHTS</div>',
                unsafe_allow_html=True)
    fig_w = go.Figure(go.Bar(
        x=[p["name"]   for p in positions],
        y=[p["weight"] for p in positions],
        marker_color=[DEC_COLOR[p["decision"]] for p in positions],
        text=[f"{p['weight']*100:.1f}%" for p in positions],
        textposition="outside"))
    fig_w.update_layout(
        **PT, height=270,
        title=dict(text="Final Weights (Quantum x MLP)",
                   font=dict(family="Inter, sans-serif",size=12,
                             color="#00d4ff")))
    st.plotly_chart(fig_w, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  TAB 2: Performance Metrics
# ══════════════════════════════════════════════════════════════
with t2:
    st.markdown('<div class="shead">PERFORMANCE METRICS + EXPECTED TIMELINE</div>',
                unsafe_allow_html=True)

    # Summary cards per stock
    for pos in positions:
        dec   = pos["decision"]
        color = DEC_COLOR[dec]
        rcol  = RISK_COLOR.get(pos.get("risk_level","Medium"),
                                "#f59e0b")
        exp_r = pos.get("expected_return", 0)
        exp_c = "#00ff9d" if exp_r >= 0 else "#f87171"

        st.markdown(f"""
        <div style='background:rgba(20, 25, 35, 0.45);border:1px solid #1e293b;
                    border-left:3px solid {color};
                    border-radius:10px;padding:14px 18px;
                    margin-bottom:10px;'>
            <div style='display:flex;justify-content:space-between;
                        align-items:center;margin-bottom:10px;'>
                <div style='font-family:Inter, sans-serif;font-size:.95rem;
                            color:#e2e8f0;font-weight:600;'>
                {pos['name']}</div>
                <div style='display:flex;gap:8px;'>
                    <span style='background:{color}22;
                                 border:1px solid {color};
                                 color:{color};
                                 font-family:Inter, sans-serif;
                                 font-size:.65rem;
                                 padding:2px 10px;
                                 border-radius:20px;'>
                    {dec}</span>
                    <span style='background:{rcol}22;
                                 border:1px solid {rcol};
                                 color:{rcol};
                                 font-family:Inter, sans-serif;
                                 font-size:.65rem;
                                 padding:2px 10px;
                                 border-radius:20px;'>
                    {pos.get("risk_level","--")} Risk</span>
                </div>
            </div>
            <div style='display:grid;
                        grid-template-columns:repeat(4,1fr);
                        gap:8px;'>
                <div class="metric-card">
                    <div class="metric-label">Timeline</div>
                    <div class="metric-value" style='color:#00d4ff;'>
                    {pos.get("timeline_days","--")} days</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Expected Return</div>
                    <div class="metric-value" style='color:{exp_c};'>
                    {exp_r:+.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Win Probability</div>
                    <div class="metric-value" style='color:#00d4ff;'>
                    {pos.get("win_prob","--")}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value" style='color:{color};'>
                    {pos.get("confidence","--")}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value"
                         style='color:{"#00ff9d" if pos.get("sharpe",0)>0 else "#f87171"};'>
                    {pos.get("sharpe","--")}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value" style='color:#f87171;'>
                    {pos.get("max_drawdown","--")}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Volatility ({pos.get("timeline_days","--")}d)</div>
                    <div class="metric-value" style='color:#f59e0b;'>
                    {pos.get("volatility_pct","--")}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Ann. Volatility</div>
                    <div class="metric-value" style='color:#f59e0b;'>
                    {pos.get("ann_vol_pct","--")}%</div>
                </div>
            </div>
            <div style='display:grid;
                        grid-template-columns:repeat(2,1fr);
                        gap:8px;margin-top:8px;'>
                <div class="metric-card">
                    <div class="metric-label">Target Price</div>
                    <div class="metric-value" style='color:#00ff9d;'>
                    Rs.{pos.get("target_price","--"):,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Stop Loss</div>
                    <div class="metric-value" style='color:#f87171;'>
                    Rs.{pos.get("stop_loss","--"):,.2f}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
        
    st.markdown('<div class="shead">PORTFOLIO SUMMARY</div>',
            unsafe_allow_html=True)

    # Weighted portfolio metrics
    weights     = [p["weight"] for p in positions]
    exp_rets    = [p.get("expected_return", 0) for p in positions]
    sharpes     = [p.get("sharpe", 0) for p in positions]
    win_probs   = [p.get("win_prob", 0) for p in positions]
    vols        = [p.get("ann_vol_pct", 0) for p in positions]
    
    port_exp_ret = float(np.dot(weights, exp_rets))
    port_sharpe  = float(np.dot(weights, sharpes))
    port_win     = float(np.dot(weights, win_probs))
    port_vol     = float(np.dot(weights, vols))
    
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Portfolio Exp Return", f"{port_exp_ret:+.2f}%")
    m2.metric("Portfolio Sharpe",     f"{port_sharpe:.3f}")
    m3.metric("Portfolio Win Prob",   f"{port_win:.1f}%")
    m4.metric("Portfolio Ann Vol",    f"{port_vol:.2f}%")

    # Timeline comparison chart
    st.markdown('<div class="shead">TIMELINE COMPARISON</div>',
                unsafe_allow_html=True)
    fig_tl = go.Figure()
    fig_tl.add_trace(go.Bar(
        name="Timeline (days)",
        x=[p["name"] for p in positions],
        y=[p.get("timeline_days",0) for p in positions],
        marker_color=[DEC_COLOR[p["decision"]] for p in positions],
        text=[f"{p.get('timeline_days',0)}d"
              for p in positions],
        textposition="outside"))
    fig_tl.update_layout(
        **PT, height=280,
        title=dict(
            text="Expected Decision Timeline (days)",
            font=dict(family="Inter, sans-serif",size=12,
                      color="#00d4ff")))
    st.plotly_chart(fig_tl, use_container_width=True)

    # Metrics comparison chart
    st.markdown('<div class="shead">METRICS COMPARISON</div>',
                unsafe_allow_html=True)
    fig_met = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Expected Return %",
                        "Win Probability %",
                        "Sharpe Ratio"))
    colors = [DEC_COLOR[p["decision"]] for p in positions]
    names  = [p["name"] for p in positions]

    fig_met.add_trace(go.Bar(
        x=names,
        y=[p.get("expected_return",0) for p in positions],
        marker_color=colors, showlegend=False,
        text=[f"{p.get('expected_return',0):+.1f}%"
              for p in positions],
        textposition="outside"), row=1, col=1)

    fig_met.add_trace(go.Bar(
        x=names,
        y=[p.get("win_prob",0) for p in positions],
        marker_color=colors, showlegend=False,
        text=[f"{p.get('win_prob',0):.1f}%"
              for p in positions],
        textposition="outside"), row=1, col=2)

    fig_met.add_trace(go.Bar(
        x=names,
        y=[p.get("sharpe",0) for p in positions],
        marker_color=colors, showlegend=False,
        text=[f"{p.get('sharpe',0):.2f}"
              for p in positions],
        textposition="outside"), row=1, col=3)

    fig_met.update_layout(
        **PT, height=320,
        title=dict(
            text="Performance Metrics Comparison",
            font=dict(family="Inter, sans-serif",size=12,
                      color="#00d4ff")))
    st.plotly_chart(fig_met, use_container_width=True)

    # Full metrics table
    st.markdown('<div class="shead">FULL METRICS TABLE</div>',
                unsafe_allow_html=True)
    tbl_rows = []
    for pos in positions:
        tbl_rows.append({
            "Stock"       : pos["name"],
            "Decision"    : pos["decision"],
            "Confidence"  : f"{pos.get('confidence',0):.1f}%",
            "Timeline"    : f"{pos.get('timeline_days','--')}d",
            "Exp Return"  : f"{pos.get('expected_return',0):+.2f}%",
            "Target Rs."  : f"Rs.{pos.get('target_price',0):,.2f}",
            "Stop Loss"   : f"Rs.{pos.get('stop_loss',0):,.2f}",
            "Win Prob"    : f"{pos.get('win_prob',0):.1f}%",
            "Sharpe"      : f"{pos.get('sharpe',0):.3f}",
            "Max DD"      : f"{pos.get('max_drawdown',0):.2f}%",
            "Vol (period)": f"{pos.get('volatility_pct',0):.2f}%",
            "Ann Vol"     : f"{pos.get('ann_vol_pct',0):.2f}%",
            "Risk"        : pos.get("risk_level","--"),
        })
    st.dataframe(
        pd.DataFrame(tbl_rows).set_index("Stock"),
        use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  TAB 3: Model Comparison
# ══════════════════════════════════════════════════════════════
with t3:
    st.markdown('<div class="shead">MODEL COMPARISON (BACKTEST RESULTS)</div>', unsafe_allow_html=True)
    if os.path.exists("model_comparison.json"):
        with open("model_comparison.json", "r") as f:
            models_data = json.load(f)
        
        bench_ret = models_data.get("benchmark_return", 0)
        st.markdown(f"**Benchmark Return (NIFTY): {bench_ret:.2f}%**")
        
        m_stats = models_data.get("models", {})
        if m_stats:
            m_df = pd.DataFrame(m_stats).T
            m_df = m_df[["return", "sharpe", "dd", "hit_rate", "agreement"]]
            st.dataframe(m_df, use_container_width=True)
            
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                fig1 = go.Figure(go.Bar(x=m_df.index, y=m_df['return'], marker_color=PALETTE[:len(m_df)]))
                fig1.update_layout(**PT, title=dict(text="Return %", font=dict(family="Inter, sans-serif", color="#00d4ff")))
                st.plotly_chart(fig1, use_container_width=True)
            with mc2:
                fig2 = go.Figure(go.Bar(x=m_df.index, y=m_df['sharpe'], marker_color=PALETTE[:len(m_df)]))
                fig2.update_layout(**PT, title=dict(text="Sharpe Ratio", font=dict(family="Inter, sans-serif", color="#00d4ff")))
                st.plotly_chart(fig2, use_container_width=True)
            with mc3:
                fig3 = go.Figure(go.Bar(x=m_df.index, y=m_df['dd'], marker_color=PALETTE[:len(m_df)]))
                fig3.update_layout(**PT, title=dict(text="Max Drawdown %", font=dict(family="Inter, sans-serif", color="#00d4ff")))
                st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No model comparison data found yet. Run an analysis.")

# ══════════════════════════════════════════════════════════════
#  TAB 4: Portfolio Advisor
# ══════════════════════════════════════════════════════════════
with t4:
    st.markdown('<div class="banner-title" style="font-size: 2.2rem; margin-top:20px; background:linear-gradient(135deg, #00d4ff, #9d4edd); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">Portfolio Advisor</div>', unsafe_allow_html=True)
    
    def trigger_run():
        st.session_state.force_run = True
        st.session_state.return_to_tab4_soon = True
    
    a1, a2 = st.columns([1, 1])
    with a1:
        st.markdown("""
        <div style='font-size:0.75rem; color:#94a3b8; margin-bottom:8px; line-height:1.4;'>
        <b>Format (Example):</b><br/>
        <code>Ticker, Quantity, Avg_Price</code><br/>
        <code>RELIANCE, 25, 2980.50</code><br/>
        <code>TCS, 10, 3950.00</code>
        </div>
        """, unsafe_allow_html=True)
        up_file = st.file_uploader("Upload CSV Portfolio", type=['csv'])
        if up_file:
            pd.read_csv(up_file).to_csv("uploaded_portfolio.csv", index=False)
            st.success("Uploaded successfully!")
            st.button("Run Portfolio Analysis", on_click=trigger_run, key="btn_run_csv", use_container_width=True)
        elif st.button("Use Current Selection instead of Uploaded CSV"):
            if os.path.exists("uploaded_portfolio.csv"): os.remove("uploaded_portfolio.csv")
            st.success("Removed uploaded CSV. Will process selected stocks instead.")
            
    with a2:
        tgt_ret = st.number_input("Target Annual Return (%)", min_value=1.0, max_value=100.0, value=8.0, step=0.5)
        if st.button("Set Target Return"):
            if os.path.exists("selected_stocks.json"):
                with open("selected_stocks.json", "r") as f: s_opt = json.load(f)
            else:
                s_opt = {}
            s_opt["target_return"] = tgt_ret / 100.0
            with open("selected_stocks.json", "w") as f: json.dump(s_opt, f)
            st.success(f"Set target return to {tgt_ret}%")
    
    if os.path.exists("advisor_results.json"):
        with open("advisor_results.json", "r") as f:
            adv_res = json.load(f)
            
        recs = adv_res.get("recommendations", [])
        
        st.markdown('<div class="shead">AI PORTFOLIO X-RAY</div>', unsafe_allow_html=True)
        r_buys = sum(1 for r in recs if r["decision"] == "BUY")
        r_sells = sum(1 for r in recs if r["decision"] == "SELL")
        r_ins = []
        if r_sells > r_buys:
            r_ins.append("Your portfolio is carrying significant dead-weight. The Quantum Advisor strongly recommends shedding the highlighted SELL assets to prevent continued drawdown.")
        elif r_buys >= r_sells and len(recs) > 0:
            r_ins.append("Your portfolio shows strong fundamental health. The Quantum Advisor suggests holding your core assets and potentially averaging up on the strong BUY signals.")
            
        st.markdown(f"""
        <div style='background:rgba(20,20,30,0.6); padding:20px; border-radius:12px; border:1px solid rgba(157, 78, 221, 0.3); border-left:4px solid #9d4edd; margin-bottom:20px; box-shadow:0 4px 15px rgba(0,0,0,0.2);'>
            <h4 style='margin-top:0; color:#9d4edd; font-family:Inter, sans-serif; font-size:1.0rem;'>Overall Portfolio Health</h4>
            <div style='color:#e2e8f0; line-height:1.6; font-family:Inter, sans-serif; font-size:0.85rem;'>
                {"<br><br>".join(r_ins)}<br><br>
                <b>Outperformer Expected:</b> <span style='color:#00d4ff;'>{adv_res.get("best_stock")}</span> &nbsp;|&nbsp; 
                <b>Laggard:</b> <span style='color:#ff453a;'>{adv_res.get("worst_stock")}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Actionable Recommendations")
        recs = adv_res.get("recommendations", [])
        ins = adv_res.get("insights", {})
        for r in recs:
            rl = r.get("risk_level", "--")
            rcol = RISK_COLOR.get(rl, "#64748b")
            st.markdown(f"""
            <div style='background:rgba(20, 25, 35, 0.45);border-left:3px solid {DEC_COLOR[r['decision']]};padding:10px;margin-bottom:8px;border-radius:5px;'>
                <b>{r['name']}</b>: <span style='color:{DEC_COLOR[r['decision']]};'>{r['decision']}</span><br/>
                <span style='color:{rcol}; font-size: 0.8rem'>{rl} Risk</span> | <span style='color:#00d4ff; font-size: 0.8rem'>Expected Return: {r.get('expected_return',0)}%</span><br/>
                <i>{ins.get(r['name'], "")}</i>
            </div>""", unsafe_allow_html=True)
            
        alt_recs = adv_res.get("alternative_suggestions", [])
        if alt_recs:
            st.markdown("### 💡 Quantum Alternative Suggestions")
            tar_pct = adv_res.get("target_return_ann", 0.08) * 100
            st.markdown(f"<div style='font-size:0.8rem; color:#8e8e93; margin-bottom:12px;'>These external market assets have statistically robust BUY signals and are optimized specifically to reach your {tar_pct:.1f}% Annual Target Return boundary.</div>", unsafe_allow_html=True)
            for r in alt_recs:
                rl = r.get("risk_level", "--")
                rcol = RISK_COLOR.get(rl, "#64748b")
                st.markdown(f"""
                <div style='background:rgba(0, 212, 255, 0.05);border-left:3px solid #00d4ff;padding:10px;margin-bottom:8px;border-radius:5px;'>
                    <b>{r['name']}</b>: <span style='color:#00d4ff;'>{r['decision']}</span><br/>
                    <span style='color:{rcol}; font-size: 0.8rem'>{rl} Risk</span> | <span style='color:#00e588; font-size: 0.8rem'>Est. {r.get('expected_return',0)}% in {r.get('timeline_days', '--')} days ({(r.get('expected_return',0)*(252/(r.get('timeline_days', 252) if r.get('timeline_days', 0)>0 else 252))):.1f}% Annually)</span><br/>
                    <i style='color:#94a3b8;'>{ins.get(r['name'], "")}</i>
                </div>""", unsafe_allow_html=True)
                
            table_html = "<table style='width:100%; border-collapse: collapse; text-align: left; margin-top:10px; color:#e2e8f0; font-size:0.85rem;'>"
            table_html += "<tr style='border-bottom: 2px solid #5b21b6; color:#9d4edd;'><th>Asset</th><th>Timeline</th><th>Est. Return</th><th>Annualized Value</th><th>Sharpe</th></tr>"
            for r in alt_recs:
                ann = r.get('expected_return',0)*(252/(r.get('timeline_days', 252) if r.get('timeline_days', 0)>0 else 252))
                table_html += f"<tr style='border-bottom: 1px solid rgba(255,255,255,0.1);'>"
                table_html += f"<td style='padding:8px;'><b>{r['name']}</b></td>"
                table_html += f"<td style='padding:8px;'>{r.get('timeline_days', '--')} Days</td>"
                table_html += f"<td style='padding:8px; color:#00d4ff;'>{r.get('expected_return',0)}%</td>"
                table_html += f"<td style='padding:8px; color:#00e588;'>{ann:.1f}%</td>"
                table_html += f"<td style='padding:8px;'>{r.get('sharpe',0):.2f}</td>"
                table_html += "</tr>"
            table_html += "</table>"
            
            st.markdown(f"""
            <div style='background:rgba(20, 25, 35, 0.45); border:1px solid rgba(157, 78, 221, 0.3); border-radius:10px; padding:15px; margin-top:20px; box-shadow:0 4px 15px rgba(0,0,0,0.2);'>
                <h4 style='color:#9d4edd; margin-top:0; margin-bottom:10px; font-size:1.0rem;'>Mathematical Path to {tar_pct:.1f}% (Annually)</h4>
                <div style='font-size:0.75rem; color:#8e8e93; margin-bottom:15px;'>*Annualized Value verifies the asset crosses the {tar_pct:.1f}% limit when scaled to 1 year. The Quantum Engine sorts these by the highest Sharpe Ratio to ensure you hit the target with minimal volatility.*</div>
                {table_html}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No advisor recommendations found yet.")

# ══════════════════════════════════════════════════════════════
#  TAB 5: Live Prices
# ══════════════════════════════════════════════════════════════
with t5:
    st.markdown('<div class="shead">LIVE PRICE CHARTS</div>',
                unsafe_allow_html=True)
    with st.spinner("Fetching..."):
        price_df = fetch_prices(user_tickers, period)

    if not price_df.empty:
        fig_p = go.Figure()
        for i,col in enumerate(price_df.columns):
            p     = price_df[col]
            color = "#64748b" if col=="NIFTY" \
                else PALETTE[i%len(PALETTE)]
            dash  = "dash" if col=="NIFTY" else "solid"
            fig_p.add_trace(go.Scatter(
                x=price_df.index, y=p/p.iloc[0]*100,
                name=col,
                line=dict(color=color,width=1.8,dash=dash)))
        fig_p.add_hline(y=100,line_dash="dot",
                        line_color="#334155")
        fig_p.update_layout(
            **PT, height=380,
            title=dict(
                text=f"Normalised Price Base=100 ({period})",
                font=dict(family="Inter, sans-serif",size=12,
                          color="#00d4ff")))
        st.plotly_chart(fig_p, use_container_width=True)

        rows = []
        for pos in positions:
            name = pos["name"]
            if name in price_df.columns:
                curr = float(price_df[name].iloc[-1])
                prev = float(price_df[name].iloc[-2]) \
                    if len(price_df)>1 else curr
                chg  = (curr-prev)/prev*100
                rows.append({
                    "Stock"   : name,
                    "Price"   : f"Rs.{curr:,.2f}",
                    "Change %" : f"{chg:+.2f}%",
                    "Decision": pos["decision"],
                    "Signal"  : f"{pos['signal']:.3f}",
                    "P(UP)"   : f"{pos['prob_up']:.3f}",
                })
        if rows:
            st.dataframe(
                pd.DataFrame(rows).set_index("Stock"),
                use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  TAB 6: ML + Sentiment
# ══════════════════════════════════════════════════════════════
with t6:
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="shead">MLP P(UP)</div>',
                    unsafe_allow_html=True)
        probs = [p["prob_up"] for p in positions]
        fig_m = go.Figure(go.Bar(
            x=[p["name"] for p in positions], y=probs,
            marker_color=["#00ff9d" if v>0.5 else "#f87171"
                          for v in probs],
            text=[f"{v:.3f}" for v in probs],
            textposition="outside"))
        fig_m.add_hline(y=0.5,line_dash="dot",
                        line_color="#64748b")
        fig_m.update_layout(
            **{k:v for k,v in PT.items() if k!="yaxis"},
            height=300,
            yaxis=dict(range=[0,1.15],gridcolor="#1e293b",
                       linecolor="#1e293b",
                       tickfont=dict(color="#64748b")),
            title=dict(text="MLP Direction Probability",
                       font=dict(family="Inter, sans-serif",size=12,
                                 color="#00d4ff")))
        st.plotly_chart(fig_m, use_container_width=True)

    with c2:
        st.markdown('<div class="shead">SENTIMENT</div>',
                    unsafe_allow_html=True)
        sent_v = [sent_all.get(p["name"],0) for p in positions]
        fig_se = go.Figure(go.Bar(
            x=[p["name"] for p in positions], y=sent_v,
            marker_color=["#00ff9d" if v>0 else "#f87171"
                          for v in sent_v],
            text=[f"{v:+.3f}" for v in sent_v],
            textposition="outside"))
        fig_se.add_hline(y=0,line_dash="dot",
                         line_color="#64748b")
        fig_se.update_layout(
            **{k:v for k,v in PT.items() if k!="yaxis"},
            height=300,
            yaxis=dict(range=[-1.2,1.2],gridcolor="#1e293b",
                       linecolor="#1e293b",
                       tickfont=dict(color="#64748b")),
            title=dict(text="NewsAPI + VADER Sentiment",
                       font=dict(family="Inter, sans-serif",size=12,
                                 color="#00d4ff")))
        st.plotly_chart(fig_se, use_container_width=True)

    rows_c = [{"Stock":p["name"],
               "P(UP)":round(p["prob_up"],3),
               "Q-Signal":round(p["signal"],3),
               "Sentiment":round(sent_all.get(p["name"],0),3),
               "Confidence":f"{p['confidence']:.1f}%",
               "Decision":p["decision"]}
              for p in positions]
    st.dataframe(
        pd.DataFrame(rows_c).set_index("Stock"),
        use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  TAB 7: Position Sizing
# ══════════════════════════════════════════════════════════════
with t7:
    st.markdown('<div class="shead">INTEGER SHARE ALLOCATION</div>',
                unsafe_allow_html=True)
    rows_p = [{"Stock":p["name"],
               "Weight":f"{p['weight']*100:+.1f}%",
               "Alloc":f"Rs.{p['alloc']:,.0f}",
               "Price":f"Rs.{p['price']:,.2f}",
               "Shares":p["shares"],
               "Invested":f"Rs.{p['invested']:,.0f}",
               "Cash Left":f"Rs.{p['alloc']-p['invested']:,.0f}",
               "Action":p["decision"]}
              for p in positions]
    st.dataframe(
        pd.DataFrame(rows_p).set_index("Stock"),
        use_container_width=True)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Portfolio Value", f"Rs.{port_val:,.0f}")
    m2.metric("Total Invested",  f"Rs.{total_inv:,.0f}")
    m3.metric("Remaining Cash",  f"Rs.{port_val-total_inv:,.0f}")
    m4.metric("Stocks to Buy",   str(buys))

    c1,c2 = st.columns(2)
    with c1:
        pie = [(p["name"],p["invested"])
               for p in positions if p["shares"]>0]
        if pie:
            fig_pie = go.Figure(go.Pie(
                labels=[d[0] for d in pie],
                values=[d[1] for d in pie],
                marker=dict(colors=PALETTE[:len(pie)]),
                hole=0.42))
            fig_pie.update_layout(
                **PT, height=300,
                title=dict(text="Allocation",
                           font=dict(family="Inter, sans-serif",
                                     size=12,color="#00d4ff")))
            st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        fig_sh = go.Figure(go.Bar(
            x=[p["name"]   for p in positions],
            y=[p["shares"] for p in positions],
            marker_color=[DEC_COLOR[p["decision"]]
                          for p in positions],
            text=[str(p["shares"]) for p in positions],
            textposition="outside"))
        fig_sh.update_layout(
            **PT, height=300,
            title=dict(text="Shares to Buy",
                       font=dict(family="Inter, sans-serif",
                                 size=12,color="#00d4ff")))
        st.plotly_chart(fig_sh, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  TAB 8: Stock Validator
# ══════════════════════════════════════════════════════════════
with t8:
    st.markdown('<div class="shead">VALIDATE ANY TICKER</div>',
                unsafe_allow_html=True)
    vc1,vc2 = st.columns([3,1])
    val_t   = vc1.text_input(
        "Ticker", placeholder="BAJAJFINSV or BAJAJFINSV.NS",
        label_visibility="collapsed", key="val_inp")
    val_btn = vc2.button("Check", key="val_b")

    if val_btn and val_t.strip():
        with st.spinner("Checking yFinance..."):
            r = validate_ticker(val_t.strip())
        st.session_state.val_result = r

    if st.session_state.val_result:
        r = st.session_state.val_result
        if r["valid"]:
            st.success(f"[OK] {r['ticker']} is valid")
            ic = st.columns(4)
            ic[0].metric("Name",     r.get("name","")[:20])
            ic[1].metric("Price",    f"Rs.{r.get('price',0):,.2f}")
            ic[2].metric("Exchange", r.get("exchange","NSE"))
            ic[3].metric("Sector",   r.get("sector","--"))
            if st.button(f"Add {r['ticker']} to Portfolio",
                         key="add_validated"):
                name = r["ticker"].replace(
                    ".NS","").replace(".BO","")
                st.session_state.selected_stocks[name] = \
                    r["ticker"]
                st.session_state.val_result = None
                st.success(f"Added {name}!")
                st.rerun()
        else:
            st.error(f"{val_t.strip()} -- not found")
            st.write(f"**Reason:** {r.get('error','')}")

    st.markdown('<div class="shead">BULK VALIDATE</div>',
                unsafe_allow_html=True)
    bulk = st.text_area(
        "Tickers (one per line)",
        placeholder="TCS\nINFY\nWIPRO",
        label_visibility="collapsed",
        height=100)

    if st.button("Validate All"):
        tickers = [t.strip() for t in bulk.strip().split("\n")
                   if t.strip()]
        if tickers:
            rows_b = []
            pb = st.progress(0)
            for i, t in enumerate(tickers):
                r = validate_ticker(t)
                rows_b.append({
                    "Ticker"   : r["ticker"],
                    "Valid"    : "yes" if r["valid"] else "no",
                    "Name"     : r.get("name","")[:25]
                                  if r["valid"] else
                                  r.get("error","")[:25],
                    "Price"    : f"Rs.{r.get('price',0):,.2f}"
                                  if r["valid"] else "--",
                    "Exchange" : r.get("exchange","--")
                                  if r["valid"] else "--",
                    "_ticker"  : r["ticker"],
                    "_valid"   : r["valid"],
                })
                pb.progress((i+1)/len(tickers))
            pb.empty()
            st.session_state.bulk_results = rows_b

    if st.session_state.bulk_results:
        rows_b = st.session_state.bulk_results
        display = [{k:v for k,v in r.items()
                    if not k.startswith("_")}
                   for r in rows_b]
        st.dataframe(
            pd.DataFrame(display).set_index("Ticker"),
            use_container_width=True)
        valid = [r for r in rows_b if r["_valid"]]
        if valid:
            st.info(f"{len(valid)} valid stocks found")
            if st.button(
                    f"Add all {len(valid)} valid stocks",
                    key="add_all_valid"):
                for r in valid:
                    n = r["_ticker"].replace(
                        ".NS","").replace(".BO","")
                    st.session_state.selected_stocks[n] = \
                        r["_ticker"]
                st.session_state.bulk_results = None
                st.success(f"Added {len(valid)} stocks!")
                st.rerun()
            if st.button("Clear Results", key="clear_bulk"):
                st.session_state.bulk_results = None
                st.rerun()

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style='border-top:1px solid #1e293b;margin-top:40px;
            padding-top:14px;display:flex;
            justify-content:space-between;'>
    <div style='font-family:Inter, sans-serif;font-size:.62rem;
                color:#334155;'>
    HYBRID QUANTUM-CLASSICAL PORTFOLIO ENGINE v3.0</div>
    <div style='font-family:Inter, sans-serif;font-size:.62rem;
                color:#334155;'>
    Bloomberg · yFinance · NewsAPI · IBM Qiskit</div>
</div>""", unsafe_allow_html=True)
