import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

st.set_page_config(
    page_title="MIRA — Satellite Anomaly Detector",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SPACE_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #020818 !important;
    color: #e0e8ff !important;
}
[data-testid="stSidebar"] {
    background-color: #050d2e !important;
    border-right: 1px solid #1a2a6c;
}
[data-testid="stHeader"] { background: transparent !important; }

h1, h2, h3, h4, h5, h6 { color: #7eb8f7 !important; }
label, .stMarkdown, p { color: #b8cef7 !important; }

.stSelectbox > div > div,
.stSlider > div,
.stNumberInput > div > div {
    background-color: #0b1640 !important;
    border: 1px solid #1e3a8a !important;
    color: #e0e8ff !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1e3a8a, #3b5de7);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.4rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3b5de7, #5e7fff);
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(94, 127, 255, 0.4);
}
.stMetric { background: #0b1640; border-radius: 10px; padding: 12px; border: 1px solid #1e3a8a; }
.stMetric label { color: #7eb8f7 !important; }
.stMetric [data-testid="stMetricValue"] { color: #e0e8ff !important; }

hr { border-color: #1a2a6c !important; }

.js-plotly-plot { border-radius: 12px; }

.starfield {
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none; z-index: 0;
    overflow: hidden;
}
.star {
    position: absolute;
    background: white;
    border-radius: 50%;
    animation: twinkle linear infinite;
}
@keyframes twinkle {
    0%   { opacity: 0.1; transform: scale(1);   }
    50%  { opacity: 1;   transform: scale(1.3); }
    100% { opacity: 0.1; transform: scale(1);   }
}

.rocket-track {
    position: fixed;
    bottom: -60px;
    right: 60px;
    z-index: 2;
    animation: rocketFly 8s linear infinite;
    font-size: 2.4rem;
    filter: drop-shadow(0 0 10px #5e7fff);
}
@keyframes rocketFly {
    0%   { bottom: -60px; right: 60px;  opacity: 0;   transform: rotate(-45deg); }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { bottom: 110vh; right: 55vw; opacity: 0; transform: rotate(-45deg); }
}

@keyframes alertPulse {
    0%   { box-shadow: 0 0 0 0   rgba(239, 68, 68, 0.8); background: #1a0808; }
    50%  { box-shadow: 0 0 40px 8px rgba(239, 68, 68, 0.5); background: #2d0a0a; }
    100% { box-shadow: 0 0 0 0   rgba(239, 68, 68, 0.0); background: #1a0808; }
}
.red-alert {
    animation: alertPulse 1.2s ease-in-out 4;
    border: 2px solid #ef4444 !important;
    border-radius: 12px;
    padding: 16px;
}
.alert-banner {
    background: #1a0808;
    border: 2px solid #ef4444;
    border-radius: 10px;
    padding: 14px 20px;
    color: #fca5a5;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.3px;
    animation: alertPulse 1.2s ease-in-out infinite;
    margin-bottom: 12px;
}
.normal-banner {
    background: #021a0a;
    border: 2px solid #22c55e;
    border-radius: 10px;
    padding: 14px 20px;
    color: #86efac;
    font-weight: 600;
    font-size: 1.05rem;
    margin-bottom: 12px;
}
.warn-banner {
    background: #1a1202;
    border: 2px solid #f59e0b;
    border-radius: 10px;
    padding: 14px 20px;
    color: #fcd34d;
    font-weight: 600;
    font-size: 1.05rem;
    margin-bottom: 12px;
}

.mode-badge-real {
    display: inline-block;
    background: #0a1f0a;
    border: 1.5px solid #22c55e;
    border-radius: 6px;
    padding: 3px 12px;
    color: #86efac;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.mode-badge-sim {
    display: inline-block;
    background: #0f0f1a;
    border: 1.5px solid #f59e0b;
    border-radius: 6px;
    padding: 3px 12px;
    color: #fcd34d;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 6px;
}

.cause-card {
    background: #080f2e;
    border: 1px solid #1e3a8a;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0;
    color: #b8cef7;
    font-size: 0.92rem;
    line-height: 1.6;
}
.cause-card .cause-title {
    color: #7eb8f7;
    font-weight: 700;
    font-size: 0.98rem;
    margin-bottom: 4px;
}
.cause-high  { border-left: 4px solid #ef4444; }
.cause-med   { border-left: 4px solid #f59e0b; }
.cause-low   { border-left: 4px solid #22c55e; }

.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #7eb8f7;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 18px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #1a2a6c;
}
.dataset-info {
    background: #060d20;
    border: 1px solid #1e3a8a;
    border-radius: 10px;
    padding: 12px 16px;
    color: #7eb8f7;
    font-size: 0.85rem;
    line-height: 1.7;
    margin-bottom: 10px;
}
</style>

<div class="starfield" id="stars"></div>
<div class="rocket-track">🚀</div>

<script>
(function() {
    var sf = document.getElementById('stars');
    if (!sf) return;
    for (var i = 0; i < 120; i++) {
        var s = document.createElement('div');
        s.className = 'star';
        var sz = Math.random() * 2.5 + 0.5;
        s.style.cssText = [
            'width:'  + sz + 'px',
            'height:' + sz + 'px',
            'left:'   + (Math.random()*100) + 'vw',
            'top:'    + (Math.random()*100) + 'vh',
            'animation-duration:' + (Math.random()*4+2) + 's',
            'animation-delay:'    + (Math.random()*5)   + 's'
        ].join(';');
        sf.appendChild(s);
    }
})();
</script>
"""

st.markdown(SPACE_CSS, unsafe_allow_html=True)

OPSSAT_META_COLS = {"segment", "anomaly", "train", "channel", "sampling", "duration", "len"}

OPSSAT_ROOT_CAUSE_LIBRARY = {
    "mean": {"high": ("📉 Abnormal Signal Mean", "The mean telemetry value deviates significantly from nominal."), "medium": ("📊 Elevated Mean Drift", "A moderate mean offset is present.")},
    "var": {"high": ("🔀 Critical Variance Spike", "Signal variance has increased dramatically."), "medium": ("〰️ Elevated Signal Variance", "Above-nominal variance detected.")},
    "std": {"high": ("📡 High Signal Dispersion", "Standard deviation far outside training norms."), "medium": ("〰️ Moderate Signal Dispersion", "Mildly elevated standard deviation.")},
    "smooth10_n_peaks": {"high": ("🏔️ Abnormal Peak Count (10-pt smooth)", "Number of peaks in the lightly-smoothed signal is anomalous."), "medium": ("🏔️ Elevated Peak Count (10-pt smooth)", "More peaks than expected.")},
    "smooth20_n_peaks": {"high": ("🏔️ Abnormal Peak Count (20-pt smooth)", "Highly smoothed signal still shows anomalous peak count."), "medium": ("🏔️ Elevated Peak Count (20-pt smooth)", "Above-nominal peaks in the 20-point smoothed signal.")},
    "diff_peaks": {"high": ("🔺 Anomalous First-Difference Peak Count", "The first-difference series has an abnormal number of peaks."), "medium": ("🔺 Elevated First-Difference Peaks", "Moderately high rate of change reversals.")},
    "diff2_peaks": {"high": ("🔻 Anomalous Second-Difference Peak Count", "Acceleration of the signal has an abnormal peak count."), "medium": ("🔻 Elevated Second-Difference Peaks", "Moderate spike in second-difference peaks.")},
    "diff_var": {"high": ("⚡ High First-Difference Variance", "The variance of the signal's first difference is critically elevated."), "medium": ("⚡ Elevated Rate-of-Change Variance", "First-difference variance above nominal baseline.")},
    "diff2_var": {"high": ("🌊 High Second-Difference Variance", "The second-difference variance is anomalously high."), "medium": ("🌊 Elevated Signal Acceleration Variance", "Moderate increase in second-difference variance.")},
    "gaps_squared": {"high": ("🕳️ Anomalous Sampling Gap Structure", "The squared-gap metric deviates significantly."), "medium": ("🕳️ Irregular Sampling Gaps", "Moderate deviation in sampling gap structure.")},
    "len_weighted": {"high": ("📏 Anomalous Length-Weighted Metric", "Length-weighted feature significantly off-nominal."), "medium": ("📏 Elevated Length-Weighted Deviation", "Moderate length-weighted anomaly.")},
    "var_div_duration": {"high": ("⏱️ Critical Variance-per-Second", "Variance normalised by duration is critically high."), "medium": ("⏱️ Elevated Variance Rate", "Above-nominal variance-per-second detected.")},
    "var_div_len": {"high": ("📐 Critical Variance-per-Sample", "Per-sample variance is critically elevated."), "medium": ("📐 Elevated Per-Sample Variance", "Moderate per-sample variance elevation.")},
}

SIM_FEATURES = [
    "Battery Voltage (V)", "Solar Panel Output (W)", "CPU Temperature (°C)",
    "Signal Strength (dBm)", "Attitude Error (deg)", "Thruster Fuel (%)",
    "Memory Usage (%)", "Downlink Rate (Mbps)",
]

SIM_ROOT_CAUSE_LIBRARY = {
    "Battery Voltage (V)": {"high": ("⚡ Critical Power Failure", "Battery voltage is severely out of range."), "medium": ("🔋 Battery Stress", "Voltage deviation detected.")},
    "Solar Panel Output (W)": {"high": ("☀️ Solar Array Fault", "Output has dropped critically."), "medium": ("🌑 Reduced Solar Efficiency", "Minor output degradation.")},
    "CPU Temperature (°C)": {"high": ("🌡️ Thermal Runaway Risk", "CPU temperature exceeds safe operating range."), "medium": ("🔥 CPU Thermal Stress", "Temperature approaching upper limit.")},
    "Signal Strength (dBm)": {"high": ("📡 Communication Link Loss", "Signal strength critically low."), "medium": ("📶 Signal Degradation", "Sub-nominal signal detected.")},
    "Attitude Error (deg)": {"high": ("🔄 Attitude Control System Failure", "Large attitude error indicates possible reaction wheel failure."), "medium": ("↔️ Attitude Drift", "Moderate pointing error detected.")},
    "Thruster Fuel (%)": {"high": ("🛑 Fuel Depletion Alert", "Fuel level critically low or anomalous reading detected."), "medium": ("⛽ Propellant Concern", "Fuel consumption rate is above nominal.")},
    "Memory Usage (%)": {"high": ("💾 Memory Overflow Risk", "Memory utilization critically high."), "medium": ("🗄️ High Memory Utilization", "Memory nearing capacity.")},
    "Downlink Rate (Mbps)": {"high": ("📉 Downlink Failure", "Downlink rate severely degraded or zero."), "medium": ("📊 Reduced Downlink Throughput", "Throughput below nominal.")},
}

@st.cache_data(show_spinner=False)
def load_opssat_dataset(path: str = "dataset.csv"):
    if not os.path.exists(path):
        return None, [], f"dataset.csv not found at `{os.path.abspath(path)}`."
    try:
        df = pd.read_csv(path)
        feature_cols = [c for c in df.columns if c not in OPSSAT_META_COLS]
        feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        if not feature_cols:
            return None, [], "No numeric feature columns found after excluding metadata."
        return df, feature_cols, None
    except Exception as e:
        return None, [], str(e)

def calculate_risk_thresholds(scores, preds):
    anomaly_scores = scores[preds == -1]
    if len(anomaly_scores) == 0:
        return {"critical": None, "high": None, "medium": None}
    return {
        "critical": float(np.percentile(anomaly_scores, 20)),
        "high": float(np.percentile(anomaly_scores, 50)),
        "medium": float(np.percentile(anomaly_scores, 80))
    }

def risk_level(score, thresholds):
    if thresholds["critical"] is None:
        return "LOW"
    if score <= thresholds["critical"]:
        return "HIGH"
    if score <= thresholds["high"]:
        return "MEDIUM"
    return "LOW"

def train_ocsvm(df: pd.DataFrame, features: list, nu: float, kernel: str, gamma: str):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X)
    return model, scaler

def run_predict(model, scaler, df: pd.DataFrame, features: list):
    X = scaler.transform(df[features])
    preds = model.predict(X)
    scores = model.score_samples(X)
    return preds, scores

def explain_anomaly(row: pd.Series, normal_stats: pd.DataFrame, features: list, root_cause_lib: dict) -> list:
    causes = []
    for feat in features:
        val = row[feat]
        mean = normal_stats.loc["mean", feat]
        std = normal_stats.loc["std", feat]
        z = abs(val - mean) / max(std, 1e-9)
        if z > 3.0: severity = "high"
        elif z > 2.0: severity = "medium"
        else: continue
        lib_entry = root_cause_lib.get(feat, {}).get(severity)
        if lib_entry is None:
            for key in root_cause_lib:
                if feat.startswith(key) or key in feat:
                    lib_entry = root_cause_lib[key].get(severity)
                    break
        if lib_entry:
            title, desc = lib_entry
        else:
            title = f"⚠️ Feature Deviation: {feat}"
            desc = f"Value {val:.4g} deviates {z:.2f}σ from the nominal mean ({mean:.4g})."
        causes.append({"feature": feat, "value": val, "z_score": z, "severity": severity, "title": title, "desc": desc})
    causes.sort(key=lambda c: c["z_score"], reverse=True)
    return causes[:3]

def make_radar_chart(row: pd.Series, normal_stats: pd.DataFrame, features: list):
    short = [f[:18] + "…" if len(f) > 18 else f for f in features]
    cats = short + [short[0]]
    z_scores = [abs(row[f] - normal_stats.loc["mean", f]) / max(normal_stats.loc["std", f], 1e-9) for f in features]
    z_closed = z_scores + [z_scores[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[1] * (len(features) + 1), theta=cats, fill="toself", name="Normal Boundary",
                                   line=dict(color="#3b5de7", dash="dash"), fillcolor="rgba(59,93,231,0.1)"))
    fig.add_trace(go.Scatterpolar(r=z_closed, theta=cats, fill="toself", name="Current Reading",
                                   line=dict(color="#ef4444"), fillcolor="rgba(239,68,68,0.2)"))
    fig.update_layout(polar=dict(bgcolor="#0b1640", radialaxis=dict(visible=True, range=[0, max(5, max(z_scores) + 1)], gridcolor="#1e3a8a", color="#7eb8f7"), angularaxis=dict(gridcolor="#1e3a8a", color="#7eb8f7")),
                      paper_bgcolor="#020818", plot_bgcolor="#020818", font=dict(color="#b8cef7", size=11),
                      legend=dict(bgcolor="#050d2e", bordercolor="#1e3a8a"), margin=dict(l=40, r=40, t=30, b=20), height=360)
    return fig

def make_score_timeline(scores, preds, x_labels=None):
    colors = ["#ef4444" if p == -1 else "#22c55e" for p in preds]
    x = x_labels if x_labels is not None else list(range(len(scores)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=scores, mode="lines+markers", line=dict(color="#3b5de7", width=1.5),
                             marker=dict(color=colors, size=5), name="Decision Score"))
    fig.update_layout(xaxis=dict(title="Telemetry Segment", gridcolor="#0d1f4a", color="#7eb8f7"),
                      yaxis=dict(title="Anomaly Score", gridcolor="#0d1f4a", color="#7eb8f7"),
                      paper_bgcolor="#020818", plot_bgcolor="#020818", font=dict(color="#b8cef7"),
                      margin=dict(l=40, r=20, t=20, b=40), height=300, legend=dict(bgcolor="#050d2e"))
    return fig

def make_pca_scatter(df: pd.DataFrame, preds, scaler, features: list):
    X = scaler.transform(df[features])
    pca = PCA(n_components=2, random_state=0)
    X2d = pca.fit_transform(X)
    labels = ["Anomaly" if p == -1 else "Normal" for p in preds]
    fig = go.Figure()
    for label, color in [("Normal", "#3b5de7"), ("Anomaly", "#ef4444")]:
        mask = np.array([l == label for l in labels])
        fig.add_trace(go.Scatter(x=X2d[mask, 0], y=X2d[mask, 1], mode="markers", name=label,
                                 marker=dict(color=color, size=6, opacity=0.75)))
    fig.update_layout(xaxis=dict(title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", gridcolor="#0d1f4a", color="#7eb8f7"),
                      yaxis=dict(title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", gridcolor="#0d1f4a", color="#7eb8f7"),
                      paper_bgcolor="#020818", plot_bgcolor="#020818", font=dict(color="#b8cef7"),
                      margin=dict(l=40, r=20, t=20, b=40), height=320, legend=dict(bgcolor="#050d2e", bordercolor="#1e3a8a"))
    return fig

def generate_normal_data(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({"Battery Voltage (V)": rng.normal(28.0, 0.4, n), "Solar Panel Output (W)": rng.normal(120.0, 5.0, n),
                          "CPU Temperature (°C)": rng.normal(45.0, 3.0, n), "Signal Strength (dBm)": rng.normal(-70.0, 2.0, n),
                          "Attitude Error (deg)": rng.normal(0.05, 0.02, n), "Thruster Fuel (%)": rng.normal(75.0, 2.0, n),
                          "Memory Usage (%)": rng.normal(55.0, 5.0, n), "Downlink Rate (Mbps)": rng.normal(50.0, 3.0, n)})

def inject_anomalies(df: pd.DataFrame, anomaly_rate: float = 0.08) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = df.copy()
    n = len(df)
    n_anom = max(1, int(n * anomaly_rate))
    idx = rng.choice(n, n_anom, replace=False)
    for i in idx:
        col = rng.choice(SIM_FEATURES)
        magnitude = rng.uniform(3.0, 6.0)
        direction = rng.choice([-1, 1])
        df.at[i, col] += direction * magnitude * df[col].std()
    return df

def render_alert_banner(n_anomalies, risk_counts):
    has_high = risk_counts.get("HIGH", 0) > 0
    if has_high:
        st.markdown(f"<div class='alert-banner'>🚨 CRITICAL ALERT — {risk_counts['HIGH']} HIGH-RISK anomal{'y' if risk_counts['HIGH']==1 else 'ies'} detected! Immediate mission review required.</div>", unsafe_allow_html=True)
    elif risk_counts.get("MEDIUM", 0) > 0:
        st.markdown(f"<div class='warn-banner'>⚠️ HIGH PRIORITY — {risk_counts['MEDIUM']} MEDIUM-RISK anomal{'y' if risk_counts['MEDIUM']==1 else 'ies'} detected. Immediate investigation recommended.</div>", unsafe_allow_html=True)
    elif n_anomalies > 0:
        st.markdown(f"<div class='warn-banner'>⚠️ CAUTION — {n_anomalies} anomal{'y' if n_anomalies==1 else 'ies'} detected at LOW risk. Monitor closely.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='normal-banner'>✅ ALL SYSTEMS NOMINAL — No anomalies detected.</div>", unsafe_allow_html=True)

def render_kpi_row(preds, scores, risk_counts):
    n_total = len(preds)
    n_anomalies = int((preds == -1).sum())
    n_normal = int((preds == 1).sum())
    anom_pct = n_anomalies / n_total * 100
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📊 Segments Analysed", f"{n_total:,}")
    m2.metric("✅ Normal", f"{n_normal:,}")
    m3.metric("⚠️ Anomalies", f"{n_anomalies:,}", delta=f"{anom_pct:.1f}%", delta_color="inverse")
    m4.metric("🔴 High Risk", risk_counts.get("HIGH", 0), delta_color="inverse")
    m5.metric("🟡 Medium Risk", risk_counts.get("MEDIUM", 0), delta_color="off")

def render_charts(test_df, preds, scores, scaler, features, x_labels=None):
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("<div class='section-title'>📈 Anomaly Score Timeline</div>", unsafe_allow_html=True)
        st.plotly_chart(make_score_timeline(scores, preds, x_labels), use_container_width=True, key="timeline")
    with col_right:
        st.markdown("<div class='section-title'>🔵 PCA Feature Space</div>", unsafe_allow_html=True)
        st.plotly_chart(make_pca_scatter(test_df, preds, scaler, features), use_container_width=True, key="pca")

def render_inspector(test_df, preds, scores, normal_stats, features, root_cause_lib, thresholds, frame_label_prefix="Segment"):
    st.markdown("<div class='section-title'>🔬 AI Mission Anomaly Investigator</div>", unsafe_allow_html=True)
    anomaly_indices = np.where(preds == -1)[0]
    if len(anomaly_indices) == 0:
        st.info("No anomalies to inspect.")
        return
    frame_labels = {int(i): f"{frame_label_prefix} {i:04d}  —  " + risk_level(scores[i], thresholds) + "  risk  (score: {:.3f})".format(scores[i]) for i in anomaly_indices}
    sorted_indices = sorted(anomaly_indices, key=lambda i: scores[i])
    selected_frame = st.selectbox("Select anomaly frame to inspect:", options=sorted_indices, format_func=lambda i: frame_labels[i])
    row = test_df.iloc[selected_frame]
    s = scores[selected_frame]
    rlevel = risk_level(s, thresholds)
    causes = explain_anomaly(row, normal_stats, features, root_cause_lib)
    badge_color = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}[rlevel]
    frame_css = "red-alert" if rlevel == "HIGH" else ""
    st.markdown(f"<div class='{frame_css}' style='background:#080f2e;border:2px solid {badge_color};border-radius:12px;padding:16px 20px;margin-bottom:12px;'><span style='color:{badge_color};font-weight:700;font-size:1rem;'>{'🚨' if rlevel=='HIGH' else '⚠️' if rlevel=='MEDIUM' else '🔵'} {frame_label_prefix} {selected_frame:04d} — {rlevel} RISK</span><span style='color:#57606a;font-size:0.88rem;margin-left:16px;'>Decision score: {s:.4f}</span></div>", unsafe_allow_html=True)
    detail_col, radar_col = st.columns([1, 1])
    with detail_col:
        if causes:
            st.markdown("**Identified Root Causes:**")
            for c in causes:
                css_cls = ("cause-high" if c["severity"] == "high" else "cause-med" if c["severity"] == "medium" else "cause-low")
                st.markdown(f"<div class='cause-card {css_cls}'><div class='cause-title'>{c['title']}</div><b>{c['feature']}</b>: {c['value']:.4g} (z-score: {c['z_score']:.2f})<br>{c['desc']}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='cause-card cause-low'><div class='cause-title'>ℹ️ Subtle Multi-Feature Deviation</div>No single feature exceeds the 2σ threshold.</div>", unsafe_allow_html=True)
        st.markdown("<br>**Feature Values vs. Nominal Baseline:**", unsafe_allow_html=True)
        tbl_rows = []
        for f in features:
            val = row[f]
            mean = normal_stats.loc["mean", f]
            std = normal_stats.loc["std", f]
            z = (val - mean) / max(std, 1e-9)
            tbl_rows.append({"Feature": f, "Value": round(float(val), 4), "Nominal Mean": round(float(mean), 4), "Δ (σ)": round(float(z), 2)})
        tbl = pd.DataFrame(tbl_rows).set_index("Feature")
        st.dataframe(tbl, use_container_width=True)
    with radar_col:
        st.markdown("**Deviation Radar:**")
        st.plotly_chart(make_radar_chart(row, normal_stats, features), use_container_width=True, key=f"radar_{selected_frame}")

def render_heatmap(df, features, key_suffix=""):
    with st.expander("📊 Full Telemetry Heatmap", expanded=False):
        heat_df = df[features].copy()
        heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min() + 1e-9)
        fig_heat = px.imshow(heat_norm.T, labels=dict(x="Segment", y="Feature", color="Normalised Value"), color_continuous_scale="RdBu_r", aspect="auto")
        fig_heat.update_layout(paper_bgcolor="#020818", plot_bgcolor="#020818", font=dict(color="#b8cef7"),
                               height=max(260, min(60 * len(features), 600)), margin=dict(l=20, r=20, t=10, b=20),
                               coloraxis_colorbar=dict(tickfont=dict(color="#b8cef7"), title_font=dict(color="#b8cef7")))
        st.plotly_chart(fig_heat, use_container_width=True, key=f"heatmap{key_suffix}")

with st.sidebar:
    st.markdown("## 🛰️ MIRA Controls")
    st.markdown("<hr>", unsafe_allow_html=True)
    analysis_mode = st.radio("Analysis Mode", options=["🛰️ Real OPS-SAT Data", "🔬 Mission Simulation"], index=0)
    IS_REAL = analysis_mode.startswith("🛰️")
    st.markdown("<hr>", unsafe_allow_html=True)
    if IS_REAL:
        st.markdown("<div style='background:#0a1a0a;border:1px solid #22c55e;border-radius:8px;padding:10px 14px;margin-bottom:10px;'><span style='color:#86efac;font-weight:700;font-size:0.85rem;'>🛰️ REAL OPS-SAT DATA</span><br><span style='color:#57606a;font-size:0.78rem;line-height:1.6;'>ESA OPS-SAT-1 mission telemetry.<br>Model trains on nominal segments (train=1, anomaly=0).<br><b>Fixed config:</b> OneClassSVM(rbf, nu=0.22)</span></div>", unsafe_allow_html=True)
        nu_val = 0.22
        kernel_val = "rbf"
        gamma_val = "scale"
        st.caption(f"**Kernel:** {kernel_val}  |  **Nu:** {nu_val}  |  **Gamma:** {gamma_val}")
    else:
        st.markdown("<div style='background:#1a1202;border:1px solid #f59e0b;border-radius:8px;padding:10px 14px;margin-bottom:10px;'><span style='color:#fcd34d;font-weight:700;font-size:0.85rem;'>🔬 MISSION SIMULATION</span><br><span style='color:#57606a;font-size:0.78rem;line-height:1.6;'>All values are <b>synthetic</b>. No real satellite data.<br>Adjust parameters freely for demonstration.</span></div>", unsafe_allow_html=True)
        st.markdown("### 🤖 Model Hyperparameters")
        nu_val = st.slider("Nu (outlier fraction)", 0.01, 0.50, 0.08, 0.01)
        kernel_val = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
        gamma_val = st.selectbox("Gamma", ["scale", "auto"], index=0)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📡 Telemetry Simulation")
        n_frames = st.slider("Training frames", 100, 1000, 300, 50)
        anomaly_pct = st.slider("Injected anomaly %", 1, 30, 8, 1)
    st.markdown("<hr>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Run MIRA Analysis", use_container_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div style='color:#57606a;font-size:0.78rem;line-height:1.6'>MIRA uses <b>OneClassSVM</b> trained on nominal telemetry.<br>Anomalies are frames where the satellite's behaviour deviates beyond the learned decision boundary.<br><br><b>IBM AI Builders Challenge</b><br>Advance Space Exploration with AI<br>Built with <b>IBM Bob</b></div>", unsafe_allow_html=True)

st.markdown("<div style='text-align:center;padding:24px 0 8px 0;'><span style='font-size:2.8rem;'>🛰️</span><h1 style='font-size:2.5rem;margin:4px 0 0 0;color:#7eb8f7;letter-spacing:2px;'>MIRA</h1><p style='color:#57606a;font-size:1rem;letter-spacing:4px;margin-top:2px;'>MISSION INTELLIGENCE &amp; RISK ANALYZER</p></div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

if not run_btn:
    if IS_REAL:
        sub = "Select <b style='color:#86efac;'>🛰️ Real OPS-SAT Data</b> mode and click"
    else:
        sub = "Configure the simulation parameters and click"
    st.markdown(f"<div style='text-align:center;padding:60px 20px;color:#57606a;'><div style='font-size:4rem;margin-bottom:16px;'>🌌</div><h3 style='color:#3b5de7;'>Awaiting Mission Start</h3><p>{sub} <b style='color:#7eb8f7;'>🚀 Run MIRA Analysis</b> to begin satellite monitoring.</p></div>", unsafe_allow_html=True)
    st.stop()

if IS_REAL:
    st.markdown("<div style='text-align:center;margin-bottom:8px;'><span class='mode-badge-real'>🛰️ REAL OPS-SAT DATA — ESA OPS-SAT-1 Mission Telemetry</span></div>", unsafe_allow_html=True)
    with st.spinner("📂 Loading OPS-SAT dataset…"):
        raw_df, feature_cols, load_err = load_opssat_dataset("dataset.csv")
    if load_err:
        st.error(f"**Dataset Error:** {load_err}")
        st.markdown("<div class='cause-card cause-high'><div class='cause-title'>📁 dataset.csv not found</div>Place <code>dataset.csv</code> in the same directory as <code>app.py</code> before running the app.</div>", unsafe_allow_html=True)
        st.stop()
    n_train_nominal = int(((raw_df["train"] == 1) & (raw_df["anomaly"] == 0)).sum()) if "train" in raw_df.columns and "anomaly" in raw_df.columns else len(raw_df)
    channels = raw_df["channel"].unique().tolist() if "channel" in raw_df.columns else ["—"]
    ch_str = ", ".join(str(c) for c in channels[:6]) + ("…" if len(channels) > 6 else "")
    st.markdown(f"<div class='dataset-info'><b>Dataset:</b> ESA OPS-SAT-1 Telemetry &nbsp;|&nbsp; <b>Total segments:</b> {len(raw_df):,} &nbsp;|&nbsp; <b>Nominal training segments:</b> {n_train_nominal:,} &nbsp;|&nbsp; <b>Features used:</b> {len(feature_cols)} &nbsp;|&nbsp; <b>Channels:</b> {ch_str}</div>", unsafe_allow_html=True)
    if "train" in raw_df.columns and "anomaly" in raw_df.columns:
        train_df = raw_df[(raw_df["train"] == 1) & (raw_df["anomaly"] == 0)].copy()
        test_df = raw_df[raw_df["train"] == 0].copy()
    else:
        train_df = raw_df.copy()
        test_df = raw_df.copy()
    if len(train_df) == 0:
        st.error("No nominal training segments found (train=1, anomaly=0). Check your dataset.")
        st.stop()
    with st.spinner("🤖 Training OneClassSVM on nominal OPS-SAT segments…"):
        model, scaler = train_ocsvm(train_df, feature_cols, nu=0.22, kernel="rbf", gamma="scale")
    with st.spinner("🔍 Scanning all telemetry segments…"):
        preds, scores = run_predict(model, scaler, test_df, feature_cols)
    normal_stats = train_df[feature_cols].describe().loc[["mean", "std"]]
    n_anomalies = int((preds == -1).sum())
    # Calibrate operational risk levels from detected anomaly scores
    thresholds = calculate_risk_thresholds(scores, preds)
    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for s_val, p in zip(scores, preds):
        if p == -1:
            risk = risk_level(s_val, thresholds)
            risk_counts[risk] += 1
    has_labels = "anomaly" in test_df.columns
    if has_labels:
        true_labels = test_df["anomaly"].values
        n_true_anom = int((true_labels == 1).sum())
        tp = int(((preds == -1) & (true_labels == 1)).sum())
        fp = int(((preds == -1) & (true_labels == 0)).sum())
        fn = int(((preds == 1) & (true_labels == 1)).sum())
        tn = int(((preds == 1) & (true_labels == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy = accuracy_score(true_labels, (preds == -1).astype(int))
        f1_score_val = f1_score(true_labels, (preds == -1).astype(int))
    render_alert_banner(n_anomalies, risk_counts)
    render_kpi_row(preds, scores, risk_counts)
    if has_labels:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>✅ Ground-Truth Validation</div>", unsafe_allow_html=True)
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("📋 True Anomalies", n_true_anom)
        g2.metric("🎯 True Positives", tp)
        g3.metric("🎯 Precision", f"{precision:.2%}")
        g4.metric("🔁 Recall", f"{recall:.2%}")
        st.markdown("<br>", unsafe_allow_html=True)
        g5, g6, g7 = st.columns(3)
        g5.metric("📊 Accuracy", f"{accuracy:.2%}")
        g6.metric("🎯 F1 Score", f"{f1:.2%}")
        g7.metric("📋 Confusion Matrix", f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    st.markdown("<hr>", unsafe_allow_html=True)
    x_labels = test_df["segment"].tolist() if "segment" in test_df.columns else None
    render_charts(test_df, preds, scores, scaler, feature_cols, x_labels)
    st.markdown("<hr>", unsafe_allow_html=True)
    render_inspector(test_df, preds, scores, normal_stats, feature_cols, OPSSAT_ROOT_CAUSE_LIBRARY, thresholds, "Segment")
    st.markdown("<hr>", unsafe_allow_html=True)
    render_heatmap(test_df, feature_cols, key_suffix="_real")
else:
    st.markdown("<div style='text-align:center;margin-bottom:8px;'><span class='mode-badge-sim'>🔬 MISSION SIMULATION — All values are SYNTHETIC</span></div>", unsafe_allow_html=True)
    with st.spinner("🛰️ Generating synthetic telemetry stream…"):
        normal_df = generate_normal_data(n_frames)
        test_df = inject_anomalies(normal_df.copy(), anomaly_rate=anomaly_pct / 100)
        normal_stats = normal_df.describe().loc[["mean", "std"]]
    with st.spinner("🤖 Training OneClassSVM on synthetic nominal data…"):
        model, scaler = train_ocsvm(normal_df, SIM_FEATURES, nu=nu_val, kernel=kernel_val, gamma=gamma_val)
    with st.spinner("🔍 Scanning synthetic telemetry frames…"):
        preds, scores = run_predict(model, scaler, test_df, SIM_FEATURES)
    n_anomalies = int((preds == -1).sum())
    # Calibrate operational risk levels from detected anomaly scores
    thresholds = calculate_risk_thresholds(scores, preds)
    risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for s_val, p in zip(scores, preds):
        if p == -1:
            risk = risk_level(s_val, thresholds)
            risk_counts[risk] += 1
    render_alert_banner(n_anomalies, risk_counts)
    render_kpi_row(preds, scores, risk_counts)
    st.markdown("<hr>", unsafe_allow_html=True)
    render_charts(test_df, preds, scores, scaler, SIM_FEATURES)
    st.markdown("<hr>", unsafe_allow_html=True)
    render_inspector(test_df, preds, scores, normal_stats, SIM_FEATURES, SIM_ROOT_CAUSE_LIBRARY, thresholds, "Frame")
    st.markdown("<hr>", unsafe_allow_html=True)
    render_heatmap(test_df, SIM_FEATURES, key_suffix="_sim")

st.markdown("<div style='text-align:center;color:#57606a;font-size:0.78rem;padding:20px 0 8px 0;border-top:1px solid #1a2a6c;margin-top:16px;'>MIRA — Mission Intelligence &amp; Risk Analyzer &nbsp;|&nbsp; OneClassSVM &nbsp;|&nbsp; OPS-SAT dataset © ESA &nbsp;|&nbsp; Application built with <b style='color:#3b5de7;'>IBM Bob</b></div>", unsafe_allow_html=True)
