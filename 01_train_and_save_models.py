# ══════════════════════════════════════════════════════════════
#  FILE 1 — Train & Save Universal Models
#  Run ONCE on Bloomberg Data.xlsx
#  Uses AerSimulator ONLY — no IBM connection needed
#
#  pip install pandas numpy scikit-learn arch joblib
#       vaderSentiment newsapi-python qiskit qiskit-aer
# ══════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd, joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, precision_score, recall_score, f1_score
)
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler

# ══════════════════════════════════════════════════════════════
#  CONFIG — edit before running
# ══════════════════════════════════════════════════════════════
BLOOMBERG_FILE = "Data.xlsx"
SAVE_DIR       = "saved_models"
NEWS_API_KEY   = "ead1b5aa12e6421c818fc5e8ce2fad69"  # ← NewsAPI key

ALL_ASSETS       = ["^NSEI", "ADANIPOWER", "BAJFINANCE", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY", "HDFCBANK", "HINDUNILVR", "ICICIBANK", "ITC", "JSWSTEEL", "LT", "M&M", "MARUTI", "NTPC", "ONGC", "RELIANCE", "SBIN", "SUNPHARMA", "TATASTEEL", "TCS"]
PORTFOLIO_ASSETS = ["ADANIPOWER", "BAJFINANCE", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY", "HDFCBANK", "HINDUNILVR", "ICICIBANK", "ITC", "JSWSTEEL", "LT", "M&M", "MARUTI", "NTPC", "ONGC", "RELIANCE", "SBIN", "SUNPHARMA", "TATASTEEL", "TCS"]

ASSET_QUERIES = {
    "^NSEI"     : "NIFTY OR Nifty50 OR Indian stock market",
    "ADANIPOWER": "Adani Power stock India",
    "BAJFINANCE": "Bajaj Finance stock India",
    "BHARTIARTL": "Bharti Airtel stock India",
    "CIPLA"     : "Cipla pharmaceutical India",
    "COALINDIA" : "Coal India stock",
    "DRREDDY"   : "Dr Reddys Laboratories stock India",
    "HDFCBANK"  : "HDFC Bank stock India",
    "HINDUNILVR": "Hindustan Unilever HUL India",
    "ICICIBANK" : "ICICI Bank stock India",
    "ITC"       : "ITC Limited stock India",
    "JSWSTEEL"  : "JSW Steel stock India",
    "LT"        : "Larsen Toubro stock India",
    "M&M"       : "Mahindra Mahindra stock India",
    "MARUTI"    : "Maruti Suzuki stock India",
    "NTPC"      : "NTPC Limited stock India",
    "ONGC"      : "ONGC Oil Natural Gas stock India",
    "RELIANCE"  : "Reliance Industries stock India",
    "SBIN"      : "State Bank of India SBI stock",
    "SUNPHARMA" : "Sun Pharmaceutical stock India",
    "TATASTEEL" : "Tata Steel stock India",
    "TCS"       : "Tata Consultancy Services TCS stock India",
}

FEATURE_COLS = [
    "MA10_ratio", "MA50_ratio", "RSI", "GARCH_Vol",
    "Return_Lag1", "Return_Lag2", "Return_Lag3",
    "Momentum_pct", "Volume_ratio", "Volatility_Change",
    "Beta", "NIFTY_corr", "Sentiment"
]

N_QUBITS = 5
N_PARAMS = 10
SHOTS    = 1024

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Save directory : {SAVE_DIR}/")
print(f"ARCH available : {ARCH_AVAILABLE}")
print(f"Backend        : AerSimulator (local, no IBM needed)")

# ════════════════════════════════════════════════════════════════
#  STEP 1: LOAD & CLEAN BLOOMBERG DATA
# ════════════════════════════════════════════════════════════════
print("\n[1/7] Loading Bloomberg data...")

df = pd.read_excel(BLOOMBERG_FILE)
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)
df = df.dropna(subset=["Date"]).drop_duplicates()
df = df.sort_values("Date").set_index("Date")
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.ffill().bfill()
print(f"  Rows  : {df.shape[0]}")
print(f"  Cols  : {df.shape[1]}")
print(f"  Range : {df.index[0].date()} to {df.index[-1].date()}")

# ════════════════════════════════════════════════════════════════
#  STEP 2: SENTIMENT (NewsAPI + VADER + price proxy)
# ════════════════════════════════════════════════════════════════
print("\n[2/7] Computing sentiment...")

analyzer = SentimentIntensityAnalyzer()

def price_proxy(df, asset):
    mom  = df[f"{asset}_LAST"].pct_change(5)
    vol  = df[f"{asset}_VOL"].pct_change(5)
    raw  = mom * (1 + vol.clip(lower=0))
    rmax = raw.rolling(60).max().replace(0, 1e-9)
    rmin = raw.rolling(60).min()
    return (2*(raw - rmin)/(rmax - rmin + 1e-9) - 1).clip(-1, 1)

def fetch_news(query):
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        today   = datetime.today()
        resp    = newsapi.get_everything(
            q          = query,
            from_param = (today - timedelta(days=25)).strftime("%Y-%m-%d"),
            to         = today.strftime("%Y-%m-%d"),
            language   = "en",
            sort_by    = "publishedAt",
            page_size  = 100
        )
        scores = []
        for art in resp.get("articles", []):
            txt = (art.get("title","") or "") + " " + \
                  (art.get("description","") or "")
            scores.append(analyzer.polarity_scores(txt)["compound"])
        return float(np.mean(scores)) if scores else None
    except Exception as e:
        print(f"    NewsAPI error: {e}")
        return None

sent_df = pd.DataFrame(index=df.index)
for asset in ALL_ASSETS:
    sent_df[f"{asset}_Sentiment"] = price_proxy(df, asset)
    score = fetch_news(ASSET_QUERIES[asset])
    if score is not None:
        sent_df[f"{asset}_Sentiment"].iloc[-25:] = score
        print(f"  {asset:6s}: NewsAPI={score:+.3f}")
    else:
        print(f"  {asset:6s}: price proxy only")

sent_df["Portfolio_Sentiment"] = sent_df[
    [f"{a}_Sentiment" for a in PORTFOLIO_ASSETS]].mean(axis=1)
df = pd.concat([df, sent_df], axis=1)

# ════════════════════════════════════════════════════════════════
#  STEP 3: GENERIC FEATURE ENGINEERING
#  Ratio-based — works for ANY stock (universal model)
# ════════════════════════════════════════════════════════════════
print("\n[3/7] Engineering generic features...")

all_rows = []

for asset in ALL_ASSETS:
    lc   = f"{asset}_LAST"
    vc   = f"{asset}_VOL"
    pr   = (df[lc]/df[lc].shift(1)).replace(0, np.nan).clip(lower=1e-9)
    lr   = np.log(pr)
    ma10 = df[lc].rolling(10).mean()
    ma50 = df[lc].rolling(50).mean()
    vol30= df[vc].rolling(30).mean().replace(0, 1e-9)

    # GARCH volatility
    if ARCH_AVAILABLE:
        try:
            r100 = (lr.dropna()*100).replace(
                [np.inf,-np.inf], np.nan).dropna()
            fit  = arch_model(r100, p=1, q=1,
                               rescale=False).fit(disp="off")
            garch_vol = pd.Series(
                fit.conditional_volatility.values / 100,
                index=r100.index)
        except Exception:
            garch_vol = lr.rolling(10).std()
    else:
        garch_vol = lr.rolling(10).std()

    # RSI
    gain = lr.clip(lower=0).rolling(14).mean()
    loss = (-lr.clip(upper=0)).rolling(14).mean().replace(0, 1e-9)
    rsi  = 100 - (100/(1 + gain/loss))

    # NIFTY correlation & beta
    nifty_lr  = np.log((df["^NSEI_LAST"]/df["^NSEI_LAST"].shift(1))
                        .replace(0, np.nan).clip(lower=1e-9))
    nifty_var = nifty_lr.rolling(30).var().replace(0, 1e-9)
    beta      = lr.rolling(30).cov(nifty_lr) / nifty_var
    corr      = lr.rolling(30).corr(nifty_lr)

    feat = pd.DataFrame({
        "MA10_ratio"       : df[lc] / ma10.replace(0, 1e-9),
        "MA50_ratio"       : df[lc] / ma50.replace(0, 1e-9),
        "RSI"              : rsi,
        "GARCH_Vol"        : garch_vol,
        "Return_Lag1"      : lr.shift(1),
        "Return_Lag2"      : lr.shift(2),
        "Return_Lag3"      : lr.shift(3),
        "Momentum_pct"     : df[lc].pct_change(5),
        "Volume_ratio"     : df[vc] / vol30,
        "Volatility_Change": garch_vol.pct_change(fill_method=None),
        "Beta"             : beta,
        "NIFTY_corr"       : corr,
        "Sentiment"        : df[f"{asset}_Sentiment"],
        "Log_Return"       : lr,
        "Target_Direction" : (lr.shift(-1) > 0).astype(int),
        "Asset"            : asset,
    }, index=df.index)

    all_rows.append(feat)

universal_df = pd.concat(all_rows).dropna()
print(f"  Universal dataset : {universal_df.shape}")
print(f"  Stocks stacked    : {ALL_ASSETS}")

# ════════════════════════════════════════════════════════════════
#  STEP 4: TRAIN UNIVERSAL MLP
# ════════════════════════════════════════════════════════════════
print("\n[4/7] Training universal MLP...")

dates = universal_df.index.unique().sort_values()
split = dates[int(len(dates)*0.80)]
train = universal_df[universal_df.index <= split]
test  = universal_df[universal_df.index >  split]

X_train = train[FEATURE_COLS]
y_train = train["Target_Direction"]
X_test  = test[FEATURE_COLS]
y_test  = test["Target_Direction"]

scaler  = StandardScaler()
Xtr_s   = scaler.fit_transform(X_train)
Xte_s   = scaler.transform(X_test)

model   = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    max_iter=5000, early_stopping=True,
    n_iter_no_change=30, random_state=42
)
model.fit(Xtr_s, y_train)

preds = model.predict(Xte_s)
probs = model.predict_proba(Xte_s)[:,1]
acc   = accuracy_score(y_test, preds)
auc   = roc_auc_score(y_test, probs)
print(f"  Overall Accuracy : {acc:.3f}")
print(f"  Overall AUC-ROC  : {auc:.3f}")

print("\n  Per-asset accuracy:")
for asset in ALL_ASSETS:
    mask = test["Asset"] == asset
    if mask.sum() == 0: continue
    a = accuracy_score(y_test[mask], preds[mask])
    print(f"    {asset:6s}: {a:.3f}")

# ════════════════════════════════════════════════════════════════
#  EVALUATION — Confusion Matrix + Classification Metrics
#  Computed on TEST SET ONLY — pure evaluation, no leakage
# ════════════════════════════════════════════════════════════════
print("\n[4b/7] Computing detailed evaluation metrics (test set)...")

cm        = confusion_matrix(y_test, preds)
TN, FP, FN, TP = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
precision = float(precision_score(y_test, preds, zero_division=0))
recall    = float(recall_score(y_test, preds, zero_division=0))
f1        = float(f1_score(y_test, preds, zero_division=0))

print(f"  Confusion Matrix  : TP={TP} FP={FP} TN={TN} FN={FN}")
print(f"  Precision         : {precision:.3f}")
print(f"  Recall            : {recall:.3f}")
print(f"  F1 Score          : {f1:.3f}")

model_metrics = {
    "accuracy"        : round(float(acc),      4),
    "auc"             : round(float(auc),      4),
    "precision"       : round(precision,       4),
    "recall"          : round(recall,          4),
    "f1"              : round(f1,              4),
    "confusion_matrix": {"TP": TP, "FP": FP, "TN": TN, "FN": FN},
    "n_test_samples"  : int(len(y_test)),
    "split_date"      : str(split.date()),
    "per_asset_accuracy": {},
}
for asset in ALL_ASSETS:
    mask = test["Asset"] == asset
    if mask.sum() == 0: continue
    model_metrics["per_asset_accuracy"][asset] = round(
        float(accuracy_score(y_test[mask], preds[mask])), 4)

import json as _json
with open(f"{SAVE_DIR}/model_metrics.json", "w") as _f:
    _json.dump(model_metrics, _f, indent=2)
print(f"  Saved: model_metrics.json")

# ════════════════════════════════════════════════════════════════
#  STEP 5: CLASSICAL PORTFOLIO WEIGHTS
# ════════════════════════════════════════════════════════════════
print("\n[5/7] Computing classical portfolio weights...")

port_df   = universal_df[universal_df["Asset"].isin(PORTFOLIO_ASSETS)]
ret_pivot = port_df.pivot_table(
    index=port_df.index, columns="Asset", values="Log_Return"
).dropna()

# Expected returns from MLP
mu = []
for asset in PORTFOLIO_ASSETS:
    mask = test["Asset"] == asset
    if mask.sum() == 0:
        mu.append(0.0)
        continue
    p   = probs[mask.values]
    act = test.loc[test["Asset"]==asset,
                   "Log_Return"].values[:len(p)]
    mu.append(float(np.mean(p * act)) * 0.3)
mu = np.array(mu)
print(f"  Expected returns (mu): {np.round(mu, 6)}")

lw = LedoitWolf()
lw.fit(ret_pivot)
cov_matrix = lw.covariance_

# Risk Parity
vols          = np.sqrt(np.diag(cov_matrix))
risk_parity_w = (1/vols) / np.sum(1/vols)

# Max-Sharpe
def neg_sharpe(w, mu, cov):
    r   = np.dot(w, mu)
    vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
    return -r/vol if vol > 0 else 0.0

n   = len(PORTFOLIO_ASSETS)
res = minimize(neg_sharpe, np.ones(n)/n, args=(mu, cov_matrix),
               method="SLSQP", bounds=[(-0.3, 0.6)]*n,
               constraints={"type":"eq","fun":lambda w:np.sum(w)-1})
max_sharpe_w = res.x

print(f"  Risk Parity : {np.round(risk_parity_w, 3)}")
print(f"  Max-Sharpe  : {np.round(max_sharpe_w,  3)}")

# ════════════════════════════════════════════════════════════════
#  STEP 6: TRAIN QUANTUM VQC + SPSA
#  AerSimulator ONLY — unlimited, no IBM quota used
# ════════════════════════════════════════════════════════════════
print("\n[6/7] Training Quantum VQC with SPSA (AerSimulator)...")
print("  AerSimulator used for training (unlimited, free)")
print("  IBM hardware used only for live inference in 02_")

features_pv = ParameterVector("f", N_QUBITS)
theta_pv    = ParameterVector("th", N_PARAMS)

def build_vqc():
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    # Layer 1: Feature encoding
    for i in range(N_QUBITS):
        qc.ry(features_pv[i], i)
    qc.barrier()
    # Layer 2: Trainable layer 1
    for i in range(N_QUBITS):
        qc.ry(theta_pv[i], i)
    qc.barrier()
    # Layer 3: Entanglement ring
    for i in range(N_QUBITS):
        qc.cx(i, (i+1) % N_QUBITS)
    qc.barrier()
    # Layer 4: Trainable layer 2
    for i in range(N_QUBITS):
        qc.ry(theta_pv[i + N_QUBITS], i)
    qc.barrier()
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc

vqc = build_vqc()
print(f"\n  Circuit depth    : {vqc.depth()}")
print(f"  Trainable params : {N_PARAMS}")
print(f"  Shots per eval   : {SHOTS}")

# Global sampler instantiation (moved outside the loop for massive speedup)
# Running on CPU is optimal for 5-qubit circuits as GPU memory transfer overhead hurts small circuits
GLOBAL_SAMPLER = AerSampler(backend_options={"method": "statevector"})

def run_vqc_aer(theta_vals, feat_row):
    """Run VQC on AerSimulator, return mean qubit signal."""
    b = {}
    for i,p in enumerate(features_pv): b[p] = float(feat_row[i])
    for i,p in enumerate(theta_pv):    b[p] = float(theta_vals[i])
    bound  = vqc.assign_parameters(b)
    # Using the global sampler massively reduces overhead per call
    job    = GLOBAL_SAMPLER.run(bound, shots=SHOTS)
    quasi  = job.result().quasi_dists[0]
    counts = {format(k, f"0{N_QUBITS}b"): int(v*SHOTS)
              for k,v in quasi.items()}
    total  = sum(counts.values())
    qprobs = np.zeros(N_QUBITS)
    for bs, cnt in counts.items():
        for i, bit in enumerate(bs[::-1]):
            if i < N_QUBITS and bit == "1":
                qprobs[i] += cnt/total
    weights = np.array([0.10, 0.20, 0.10, 0.35, 0.25])
    return float(np.dot(qprobs, weights))

def get_q_inputs(df_sub):
    """Build normalized [0,π] quantum inputs for independent individual stock events."""
    n_rv = df_sub["^NSEI_LAST"].pct_change().rolling(10).std()
    n_rt = np.log((df_sub["^NSEI_LAST"]/df_sub["^NSEI_LAST"].shift(1)).replace(0, np.nan).clip(lower=1e-9))
    
    X_list, y_list = [], []
    for a in PORTFOLIO_ASSETS:
        lc = f"{a}_LAST"
        if lc not in df_sub.columns: continue
        lr = np.log((df_sub[lc]/df_sub[lc].shift(1)).replace(0, np.nan).clip(lower=1e-9))
        a_rv = lr.rolling(10).std()
        sent = df_sub[f"{a}_Sentiment"]
        fwd  = lr.shift(-1)
        
        tmp = pd.DataFrame({'n_rv': n_rv, 'n_rt': n_rt, 'a_rv': a_rv, 'a_rt': lr, 'sent': sent, 'fwd': fwd}).dropna()
        if len(tmp) == 0: continue
        X_list.append(tmp[['n_rv', 'n_rt', 'a_rv', 'a_rt', 'sent']].values)
        y_list.append(tmp['fwd'].values)
        
    X_raw = np.vstack(X_list) if X_list else np.empty((0,5))
    y_raw = np.concatenate(y_list) if y_list else np.empty(0)
    
    sc = MinMaxScaler(feature_range=(0, np.pi))
    return sc.fit_transform(X_raw) if len(X_raw)>0 else X_raw, sc, y_raw

X_all, q_sc, y_all = get_q_inputs(df)
N         = len(X_all)
train_end = int(N * 0.70)
val_end   = int(N * 0.85)

train_q, train_y = X_all[:train_end], y_all[:train_end]
val_q,   val_y   = X_all[train_end:val_end], y_all[train_end:val_end]

def sharpe_obj_q(theta_vals, y_true, q_inputs):
    """SPSA objective: Evaluate direct trade direction accuracy on batches."""
    if len(q_inputs) == 0: return 0.0
    idx     = np.random.choice(len(q_inputs), min(3, len(q_inputs)), replace=False)
    
    signals = [run_vqc_aer(theta_vals, q_inputs[i]) for i in idx]
    returns = y_true[idx]
    
    strats  = [(s - 0.5) * r for s, r in zip(signals, returns)]
    vol     = np.std(strats)
    return -np.mean(strats)/(vol + 1e-9)

np.random.seed(42)
best_theta   = None
best_val_obj = np.inf

print("\n  Running 1 SPSA restart (40000 iterations)...")

for restart in range(1):
    print(f"\n  ── Restart {restart+1}/1 ──")
    theta  = np.random.uniform(-np.pi, np.pi, size=N_PARAMS)
    best_t = theta.copy()
    best_o = sharpe_obj_q(theta, train_y, train_q)
    print(f"     Init obj: {best_o:.4f}")

    for k in range(1, 40001):
        ak    = 0.05 / (k**0.602)
        ck    = 0.10 / (k**0.101)
        delta = np.random.choice([-1,1], size=N_PARAMS).astype(float)
        fp    = sharpe_obj_q(theta+ck*delta, train_y, train_q)
        fm    = sharpe_obj_q(theta-ck*delta, train_y, train_q)
        g     = (fp - fm) / (2*ck*delta)
        theta = np.clip(theta - ak*g, -np.pi, np.pi)
        obj   = sharpe_obj_q(theta, train_y, train_q)
        if obj < best_o:
            best_t, best_o = theta.copy(), obj
        if k % 1000 == 0:
            print(f"     [{k}/40000] best: {best_o:.4f}")

    val_obj = sharpe_obj_q(best_t, val_y, val_q)
    print(f"     Val obj: {val_obj:.4f}")
    if val_obj < best_val_obj:
        best_val_obj = val_obj
        best_theta   = best_t.copy()

print(f"\n  ✅ Best theta found | val obj: {best_val_obj:.4f}")

# ════════════════════════════════════════════════════════════════
#  STEP 7: SAVE ALL MODELS
# ════════════════════════════════════════════════════════════════
print("\n[7/7] Saving all models...")

joblib.dump(model,  f"{SAVE_DIR}/universal_mlp.pkl")
joblib.dump(scaler, f"{SAVE_DIR}/universal_scaler.pkl")
joblib.dump(q_sc,   f"{SAVE_DIR}/q_scaler.pkl")
np.save(f"{SAVE_DIR}/quantum_theta.npy",  best_theta)
np.save(f"{SAVE_DIR}/max_sharpe_w.npy",   max_sharpe_w)
np.save(f"{SAVE_DIR}/risk_parity_w.npy",  risk_parity_w)
np.save(f"{SAVE_DIR}/mu.npy",             mu)

config = {
    "FEATURE_COLS"     : FEATURE_COLS,
    "ALL_ASSETS"       : ALL_ASSETS,
    "PORTFOLIO_ASSETS" : PORTFOLIO_ASSETS,
    "ASSET_QUERIES"    : ASSET_QUERIES,
    "split_date"       : str(split.date()),
    "N_QUBITS"         : N_QUBITS,
    "N_PARAMS"         : N_PARAMS,
    "SHOTS"            : SHOTS,
    "train_accuracy"   : float(acc),
    "train_auc"        : float(auc),
    "train_precision"  : float(precision),
    "train_recall"     : float(recall),
    "train_f1"         : float(f1),
}
joblib.dump(config, f"{SAVE_DIR}/feature_config.pkl")

print("\n" + "="*55)
print("  ✅ ALL MODELS SAVED SUCCESSFULLY")
print("="*55)
print(f"  universal_mlp.pkl     — works for ANY NSE stock")
print(f"  universal_scaler.pkl  — normalizes live features")
print(f"  quantum_theta.npy     — {N_PARAMS} trained VQC params")
print(f"  max_sharpe_w.npy      — classical optimal weights")
print(f"  risk_parity_w.npy     — risk parity weights")
print(f"  feature_config.pkl    — all config saved")
print(f"  MLP Accuracy  : {acc:.3f}")
print(f"  MLP AUC-ROC   : {auc:.3f}")
print(f"  Precision     : {precision:.3f}")
print(f"  Recall        : {recall:.3f}")
print(f"  F1 Score      : {f1:.3f}")
print(f"  VQC Val Obj   : {best_val_obj:.4f}")
print(f"  IBM used      : 0 minutes (AerSimulator only)")
print(f"  model_metrics.json saved")
print(f"\n  Next: python 02_live_quantum_system.py")
