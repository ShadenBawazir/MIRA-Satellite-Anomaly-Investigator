"""
MIRA — Mission Intelligence & Risk Analyzer
Satellite anomaly detection powered by OneClassSVM.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import random

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MIRA — Satellite Anomaly Detector",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark space theme + rocket animation ─────────────────────────────────────
SPACE_CSS = """
<style>
/* ── Global background ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #020818 !important;
    color: #e0e8ff !important;
}
[data-testid="stSidebar"] {
    background-color: #050d2e !important;
    border-right: 1px solid #1a2a6c;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 { color: #7eb8f7 !important; }
label, .stMarkdown, p { color: #b8cef7 !important; }

/* ── Widgets ── */
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

/* ── Dividers ── */
hr { border-color: #1a2a6c !important; }

/* ── Plotly chart container ── */
.js-plotly-plot { border-radius: 12px; }

/* ── Starfield ── */
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

/* ── Rocket ── */
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

/* ── Red alert pulse ── */
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

/* ── Root-cause card ── */
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

/* ── Section titles ── */
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
</style>

<!-- Starfield -->
<div class="starfield" id="stars"></div>
<!-- Rocket -->
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

# ─── Feature definitions ─────────────────────────────────────────────────────
FEATURES = [
    "Battery Voltage (V)",
    "Solar Panel Output (W)",
    "CPU Temperature (°C)",
    "Signal Strength (dBm)",
    "Attitude Error (deg)",
    "Thruster Fuel (%)",
    "Memory Usage (%)",
    "Downlink Rate (Mbps)",
]

ROOT_CAUSE_LIBRARY = {
    "Battery Voltage (V)": {
        "high":   ("⚡ Critical Power Failure",
                   "Battery voltage is severely out of range. Possible causes: charging circuit fault, "
                   "deep discharge event, or cell degradation. Recommend immediate power-system diagnostics."),
        "medium": ("🔋 Battery Stress",
                   "Voltage deviation detected. Could indicate partial cell failure or thermal stress from "
                   "eclipse transitions. Monitor charge cycles closely."),
    },
    "Solar Panel Output (W)": {
        "high":   ("☀️ Solar Array Fault",
                   "Output has dropped critically. Potential causes: panel shadowing, micrometeorite strike, "
                   "or deployment mechanism failure. Verify panel orientation telemetry."),
        "medium": ("🌑 Reduced Solar Efficiency",
                   "Minor output degradation. May be caused by panel degradation over time or off-nominal "
                   "sun-pointing angle. Review attitude control data."),
    },
    "CPU Temperature (°C)": {
        "high":   ("🌡️ Thermal Runaway Risk",
                   "CPU temperature exceeds safe operating range. Possible causes: thermal control system "
                   "failure, heater malfunction, or excessive computational load. Initiate thermal safe mode."),
        "medium": ("🔥 CPU Thermal Stress",
                   "Temperature approaching upper limit. Could result from high task scheduling load or "
                   "partial heat-pipe degradation. Reduce non-critical processes."),
    },
    "Signal Strength (dBm)": {
        "high":   ("📡 Communication Link Loss",
                   "Signal strength critically low. Potential causes: antenna pointing error, hardware fault "
                   "in RF chain, or atmospheric interference. Check ground station alignment."),
        "medium": ("📶 Signal Degradation",
                   "Sub-nominal signal detected. May be caused by partial antenna obstruction or "
                   "transponder aging. Log for trend analysis."),
    },
    "Attitude Error (deg)": {
        "high":   ("🔄 Attitude Control System Failure",
                   "Large attitude error indicates possible reaction wheel failure, gyroscope drift, or "
                   "magnetic torquer anomaly. Satellite stability is at risk."),
        "medium": ("↔️ Attitude Drift",
                   "Moderate pointing error detected. Could be due to disturbance torques or sensor "
                   "calibration drift. Review ACS actuator health."),
    },
    "Thruster Fuel (%)": {
        "high":   ("🛑 Fuel Depletion Alert",
                   "Fuel level critically low or anomalous reading detected. Potential propellant leak or "
                   "incorrect fuel gauge calibration. Suspend maneuvers and audit fuel budget."),
        "medium": ("⛽ Propellant Concern",
                   "Fuel consumption rate is above nominal. Review recent maneuver history and thruster "
                   "valve cycling for micro-leaks."),
    },
    "Memory Usage (%)": {
        "high":   ("💾 Memory Overflow Risk",
                   "Memory utilization critically high. Could cause data loss, process crashes, or safe-mode "
                   "entry. Flush non-critical buffers and dump telemetry immediately."),
        "medium": ("🗄️ High Memory Utilization",
                   "Memory nearing capacity. Review active payload data buffering and clear stale logs "
                   "to prevent overflow."),
    },
    "Downlink Rate (Mbps)": {
        "high":   ("📉 Downlink Failure",
                   "Downlink rate severely degraded or zero. Possible modem fault, transponder overheating, "
                   "or ground station tracking loss. Switch to backup downlink channel."),
        "medium": ("📊 Reduced Downlink Throughput",
                   "Throughput below nominal. May be caused by link margin erosion or scheduling conflicts "
                   "at the ground station. Re-optimize contact window plan."),
    },
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def generate_normal_data(n: int = 300) -> pd.DataFrame:
    """Simulate nominal satellite telemetry."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Battery Voltage (V)":    rng.normal(28.0, 0.4, n),
        "Solar Panel Output (W)": rng.normal(120.0, 5.0, n),
        "CPU Temperature (°C)":   rng.normal(45.0, 3.0, n),
        "Signal Strength (dBm)":  rng.normal(-70.0, 2.0, n),
        "Attitude Error (deg)":   rng.normal(0.05, 0.02, n),
        "Thruster Fuel (%)":      rng.normal(75.0, 2.0, n),
        "Memory Usage (%)":       rng.normal(55.0, 5.0, n),
        "Downlink Rate (Mbps)":   rng.normal(50.0, 3.0, n),
    })


def inject_anomalies(df: pd.DataFrame, anomaly_rate: float = 0.08) -> pd.DataFrame:
    """Randomly corrupt a fraction of rows to simulate anomalies."""
    rng = np.random.default_rng(7)
    df = df.copy()
    n = len(df)
    n_anom = max(1, int(n * anomaly_rate))
    idx = rng.choice(n, n_anom, replace=False)
    for i in idx:
        col = rng.choice(FEATURES)
        magnitude = rng.uniform(3.0, 6.0)
        direction = rng.choice([-1, 1])
        df.at[i, col] += direction * magnitude * df[col].std()
    return df


def train_model(df: pd.DataFrame, nu: float, kernel: str, gamma: str):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X)
    return model, scaler


def predict(model, scaler, df: pd.DataFrame):
    X = scaler.transform(df[FEATURES])
    preds   = model.predict(X)           # +1 normal, -1 anomaly
    scores  = model.score_samples(X)     # higher = more normal
    return preds, scores


def risk_level(score: float, score_min: float, score_max: float) -> str:
    """Map decision score to LOW / MEDIUM / HIGH risk."""
    norm = (score - score_min) / max(score_max - score_min, 1e-9)
    if norm < 0.25:
        return "HIGH"
    if norm < 0.50:
        return "MEDIUM"
    return "LOW"


def explain_anomaly(row: pd.Series, normal_stats: pd.DataFrame) -> list[dict]:
    """Generate root-cause explanations by comparing row to training statistics."""
    causes = []
    for feat in FEATURES:
        val  = row[feat]
        mean = normal_stats.loc["mean", feat]
        std  = normal_stats.loc["std",  feat]
        z    = abs(val - mean) / max(std, 1e-9)
        if z > 3.0:
            severity = "high"
        elif z > 2.0:
            severity = "medium"
        else:
            continue
        lib_entry = ROOT_CAUSE_LIBRARY.get(feat, {}).get(severity)
        if lib_entry:
            title, desc = lib_entry
            causes.append({
                "feature":   feat,
                "value":     val,
                "z_score":   z,
                "severity":  severity,
                "title":     title,
                "desc":      desc,
            })
    causes.sort(key=lambda c: c["z_score"], reverse=True)
    return causes


def make_radar_chart(row: pd.Series, normal_stats: pd.DataFrame):
    categories = FEATURES + [FEATURES[0]]
    z_scores = [
        abs(row[f] - normal_stats.loc["mean", f]) / max(normal_stats.loc["std", f], 1e-9)
        for f in FEATURES
    ]
    z_scores_closed = z_scores + [z_scores[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[1] * (len(FEATURES) + 1),
        theta=categories,
        fill="toself",
        name="Normal Boundary",
        line=dict(color="#3b5de7", dash="dash"),
        fillcolor="rgba(59,93,231,0.1)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=z_scores_closed,
        theta=categories,
        fill="toself",
        name="Current Reading",
        line=dict(color="#ef4444"),
        fillcolor="rgba(239,68,68,0.2)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0b1640",
            radialaxis=dict(visible=True, range=[0, max(5, max(z_scores) + 1)],
                            gridcolor="#1e3a8a", color="#7eb8f7"),
            angularaxis=dict(gridcolor="#1e3a8a", color="#7eb8f7"),
        ),
        paper_bgcolor="#020818",
        plot_bgcolor="#020818",
        font=dict(color="#b8cef7"),
        legend=dict(bgcolor="#050d2e", bordercolor="#1e3a8a"),
        margin=dict(l=40, r=40, t=30, b=20),
        height=350,
    )
    return fig


def make_score_timeline(scores, preds):
    colors = ["#ef4444" if p == -1 else "#22c55e" for p in preds]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(scores))),
        y=scores,
        mode="lines+markers",
        line=dict(color="#3b5de7", width=1.5),
        marker=dict(color=colors, size=5),
        name="Decision Score",
    ))
    fig.update_layout(
        xaxis=dict(title="Telemetry Frame", gridcolor="#0d1f4a", color="#7eb8f7"),
        yaxis=dict(title="Anomaly Score", gridcolor="#0d1f4a", color="#7eb8f7"),
        paper_bgcolor="#020818",
        plot_bgcolor="#020818",
        font=dict(color="#b8cef7"),
        margin=dict(l=40, r=20, t=20, b=40),
        height=300,
        legend=dict(bgcolor="#050d2e"),
    )
    return fig


def make_pca_scatter(df: pd.DataFrame, preds, scaler):
    X = scaler.transform(df[FEATURES])
    pca = PCA(n_components=2, random_state=0)
    X2d = pca.fit_transform(X)
    colors = ["#ef4444" if p == -1 else "#3b5de7" for p in preds]
    labels = ["Anomaly" if p == -1 else "Normal" for p in preds]
    fig = go.Figure()
    for label, color in [("Normal", "#3b5de7"), ("Anomaly", "#ef4444")]:
        mask = [l == label for l in labels]
        fig.add_trace(go.Scatter(
            x=X2d[mask, 0], y=X2d[mask, 1],
            mode="markers",
            name=label,
            marker=dict(color=color, size=6, opacity=0.75),
        ))
    fig.update_layout(
        xaxis=dict(title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                   gridcolor="#0d1f4a", color="#7eb8f7"),
        yaxis=dict(title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                   gridcolor="#0d1f4a", color="#7eb8f7"),
        paper_bgcolor="#020818",
        plot_bgcolor="#020818",
        font=dict(color="#b8cef7"),
        margin=dict(l=40, r=20, t=20, b=40),
        height=320,
        legend=dict(bgcolor="#050d2e", bordercolor="#1e3a8a"),
    )
    return fig


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛰️ MIRA Controls")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### 🤖 Model Hyperparameters")
    nu = st.slider("Nu (outlier fraction)", 0.01, 0.50, 0.08, 0.01,
                   help="Upper bound on training anomaly fraction.")
    kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
    gamma  = st.selectbox("Gamma", ["scale", "auto"], index=0)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📡 Telemetry Simulation")
    n_frames    = st.slider("Training frames", 100, 1000, 300, 50)
    anomaly_pct = st.slider("Injected anomaly %", 1, 30, 8, 1)

    st.markdown("<hr>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Run MIRA Analysis", use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#57606a;font-size:0.8rem;line-height:1.6'>"
        "MIRA uses <b>OneClassSVM</b> trained on nominal telemetry.<br>"
        "Anomalies are frames where the satellite's behaviour deviates "
        "beyond the learned decision boundary."
        "</div>",
        unsafe_allow_html=True,
    )

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center;padding:24px 0 8px 0;'>
        <span style='font-size:2.8rem;'>🛰️</span>
        <h1 style='font-size:2.5rem;margin:4px 0 0 0;color:#7eb8f7;letter-spacing:2px;'>MIRA</h1>
        <p style='color:#57606a;font-size:1rem;letter-spacing:4px;margin-top:2px;'>
            MISSION INTELLIGENCE &amp; RISK ANALYZER
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

# ─── Main logic ──────────────────────────────────────────────────────────────
if not run_btn:
    st.markdown(
        """
        <div style='text-align:center;padding:60px 20px;color:#57606a;'>
            <div style='font-size:4rem;margin-bottom:16px;'>🌌</div>
            <h3 style='color:#3b5de7;'>Awaiting Mission Start</h3>
            <p>Configure the model parameters in the sidebar and click
               <b style='color:#7eb8f7;'>🚀 Run MIRA Analysis</b> to begin satellite monitoring.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Data generation ──────────────────────────────────────────────────────────
with st.spinner("🛰️ Generating telemetry stream…"):
    normal_df  = generate_normal_data(n_frames)
    test_df    = inject_anomalies(normal_df.copy(), anomaly_rate=anomaly_pct / 100)
    normal_stats = normal_df.describe().loc[["mean", "std"]]

with st.spinner("🤖 Training OneClassSVM…"):
    model, scaler = train_model(normal_df, nu=nu, kernel=kernel, gamma=gamma)

with st.spinner("🔍 Scanning telemetry frames…"):
    preds, scores = predict(model, scaler, test_df)

# ── Summary metrics ───────────────────────────────────────────────────────────
n_anomalies = int((preds == -1).sum())
n_normal    = int((preds ==  1).sum())
anom_pct    = n_anomalies / len(preds) * 100
score_min, score_max = scores.min(), scores.max()

risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
for s, p in zip(scores, preds):
    if p == -1:
        risk_counts[risk_level(s, score_min, score_max)] += 1

has_high_risk = risk_counts["HIGH"] > 0

# ── Red alert banner ──────────────────────────────────────────────────────────
if has_high_risk:
    st.markdown(
        f"<div class='alert-banner'>🚨 RED ALERT — {risk_counts['HIGH']} HIGH-RISK "
        f"anomal{'y' if risk_counts['HIGH']==1 else 'ies'} detected! "
        f"Immediate mission review recommended.</div>",
        unsafe_allow_html=True,
    )
elif n_anomalies > 0:
    st.markdown(
        f"<div class='warn-banner'>⚠️ CAUTION — {n_anomalies} anomal"
        f"{'y' if n_anomalies==1 else 'ies'} detected at MEDIUM/LOW risk. "
        f"Monitor closely.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='normal-banner'>✅ ALL SYSTEMS NOMINAL — No anomalies detected.</div>",
        unsafe_allow_html=True,
    )

# ── KPI row ───────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("📊 Frames Analysed", f"{len(preds):,}")
m2.metric("✅ Normal",          f"{n_normal:,}")
m3.metric("⚠️ Anomalies",       f"{n_anomalies:,}", delta=f"{anom_pct:.1f}%",
          delta_color="inverse")
m4.metric("🔴 High Risk",       risk_counts["HIGH"],   delta_color="inverse")
m5.metric("🟡 Medium Risk",     risk_counts["MEDIUM"], delta_color="off")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Charts row ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("<div class='section-title'>📈 Anomaly Score Timeline</div>",
                unsafe_allow_html=True)
    st.plotly_chart(make_score_timeline(scores, preds),
                    use_container_width=True, key="timeline")

with col_right:
    st.markdown("<div class='section-title'>🔵 PCA Feature Space</div>",
                unsafe_allow_html=True)
    st.plotly_chart(make_pca_scatter(test_df, preds, scaler),
                    use_container_width=True, key="pca")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Root-cause analysis ───────────────────────────────────────────────────────
st.markdown("<div class='section-title'>🔬 Root-Cause Anomaly Inspector</div>",
            unsafe_allow_html=True)

anomaly_indices = np.where(preds == -1)[0]

if len(anomaly_indices) == 0:
    st.info("No anomalies to inspect.")
else:
    frame_labels = {
        int(i): f"Frame {i:04d}  —  "
                + risk_level(scores[i], score_min, score_max)
                + " risk  (score: {:.3f})".format(scores[i])
        for i in anomaly_indices
    }
    # Sort high-risk first
    sorted_indices = sorted(
        anomaly_indices,
        key=lambda i: scores[i],   # lower score = worse
    )
    selected_frame = st.selectbox(
        "Select anomaly frame to inspect:",
        options=sorted_indices,
        format_func=lambda i: frame_labels[i],
    )

    row    = test_df.iloc[selected_frame]
    s      = scores[selected_frame]
    rlevel = risk_level(s, score_min, score_max)
    causes = explain_anomaly(row, normal_stats)

    # ── Alert box per frame ────────────────────────────────────────────────
    badge_color = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}[rlevel]
    frame_css   = "red-alert" if rlevel == "HIGH" else ""
    st.markdown(
        f"<div class='{frame_css}' style='background:#080f2e;border:2px solid {badge_color};"
        f"border-radius:12px;padding:16px 20px;margin-bottom:12px;'>"
        f"<span style='color:{badge_color};font-weight:700;font-size:1rem;'>"
        f"{'🚨' if rlevel=='HIGH' else '⚠️' if rlevel=='MEDIUM' else '🔵'} "
        f"Frame {selected_frame:04d} — {rlevel} RISK</span>"
        f"<span style='color:#57606a;font-size:0.88rem;margin-left:16px;'>"
        f"Decision score: {s:.4f}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    detail_col, radar_col = st.columns([1, 1])

    with detail_col:
        if causes:
            st.markdown("**Identified Root Causes:**")
            for c in causes:
                css_cls = f"cause-{'high' if c['severity']=='high' else 'med' if c['severity']=='medium' else 'low'}"
                st.markdown(
                    f"<div class='cause-card {css_cls}'>"
                    f"<div class='cause-title'>{c['title']}</div>"
                    f"<b>{c['feature']}</b>: {c['value']:.3f} "
                    f"(z-score: {c['z_score']:.2f})<br>"
                    f"{c['desc']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div class='cause-card cause-low'>"
                "<div class='cause-title'>ℹ️ Subtle Multi-Feature Deviation</div>"
                "No single feature exceeds the 2σ threshold. The anomaly is driven by a "
                "combination of minor deviations across multiple sensors. Review all channels "
                "in the radar chart for compound effects."
                "</div>",
                unsafe_allow_html=True,
            )

        # Raw telemetry table
        st.markdown("<br>**Raw Telemetry vs. Normal Baseline:**", unsafe_allow_html=True)
        tbl_rows = []
        for f in FEATURES:
            val  = row[f]
            mean = normal_stats.loc["mean", f]
            std  = normal_stats.loc["std",  f]
            z    = (val - mean) / max(std, 1e-9)
            tbl_rows.append({"Feature": f,
                             "Value": round(val, 3),
                             "Nominal Mean": round(mean, 3),
                             "Δ (σ)": round(z, 2)})
        tbl = pd.DataFrame(tbl_rows).set_index("Feature")
        st.dataframe(tbl, use_container_width=True)

    with radar_col:
        st.markdown("**Deviation Radar:**")
        st.plotly_chart(make_radar_chart(row, normal_stats),
                        use_container_width=True, key=f"radar_{selected_frame}")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Full telemetry heatmap ────────────────────────────────────────────────────
with st.expander("📊 Full Telemetry Heatmap", expanded=False):
    heat_df = test_df[FEATURES].copy()
    # Normalise each column to [0,1] for visual clarity
    heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min() + 1e-9)
    fig_heat = px.imshow(
        heat_norm.T,
        labels=dict(x="Frame", y="Feature", color="Normalised Value"),
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    fig_heat.update_layout(
        paper_bgcolor="#020818",
        plot_bgcolor="#020818",
        font=dict(color="#b8cef7"),
        height=300,
        margin=dict(l=20, r=20, t=10, b=20),
        coloraxis_colorbar=dict(tickfont=dict(color="#b8cef7"),
                                title_font=dict(color="#b8cef7")),
    )
    st.plotly_chart(fig_heat, use_container_width=True, key="heatmap")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#57606a;font-size:0.78rem;"
    "padding:20px 0 8px 0;border-top:1px solid #1a2a6c;margin-top:16px;'>"
    "MIRA — Mission Intelligence &amp; Risk Analyzer &nbsp;|&nbsp; "
    "OneClassSVM &nbsp;|&nbsp; Made with IBM Bob"
    "</div>",
    unsafe_allow_html=True,
)
