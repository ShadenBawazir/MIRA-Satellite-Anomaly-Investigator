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

def apply_scenario(df, scenario):
    df = df.copy()
    if scenario == "Power Failure":
        df["Battery Voltage (V)"] -= 3.0
        df["Solar Panel Output (W)"] *= 0.55
    elif scenario == "Thermal Stress":
        df["CPU Temperature (°C)"] += 18
    elif scenario == "Communication Loss":
        df["Signal Strength (dBm)"] -= 20
        df["Downlink Rate (Mbps)"] *= 0.2
    elif scenario == "Attitude Drift":
        df["Attitude Error (deg)"] += 0.5
    elif scenario == "Multi-Subsystem Failure":
        df["Battery Voltage (V)"] -= 3
        df["CPU Temperature (°C)"] += 15
        df["Signal Strength (dBm)"] -= 15
    return df

MISSION_IMPACT_LIBRARY = {
    "Battery Voltage (V)": {"high": "Power availability may become insufficient for non-essential spacecraft operations.", "medium": "Power margin is reduced and should be monitored."},
    "Solar Panel Output (W)": {"high": "Reduced generation may accelerate battery depletion.", "medium": "Reduced generation could lower the available power margin."},
    "CPU Temperature (°C)": {"high": "Thermal stress may affect onboard computing reliability.", "medium": "Thermal conditions should be monitored."},
    "Signal Strength (dBm)": {"high": "Communication reliability may be compromised.", "medium": "Communication margin is reduced."},
    "Attitude Error (deg)": {"high": "Pointing instability may affect payload operations and communication.", "medium": "Attitude stability requires monitoring."},
    "Thruster Fuel (%)": {"high": "Reduced propellant margin may constrain future attitude or orbit-control operations.", "medium": "Propellant margin should be monitored."},
    "Memory Usage (%)": {"high": "Memory exhaustion could disrupt onboard software processes.", "medium": "High utilization may increase software reliability risk."},
    "Downlink Rate (Mbps)": {"high": "Reduced downlink capacity may delay or prevent telemetry and payload data transmission.", "medium": "Reduced throughput may affect data delivery."}
}

MISSION_ACTION_LIBRARY = {
    "Battery Voltage (V)": {"high": ["Verify solar-array power generation.", "Reduce non-essential payload activity.", "Monitor battery recovery in the next telemetry window."], "medium": ["Monitor battery voltage trend.", "Check solar-array output."]},
    "Solar Panel Output (W)": {"high": ["Verify solar-array orientation.", "Check for power-generation degradation.", "Monitor battery state."], "medium": ["Monitor solar output trend.", "Check spacecraft attitude."]},
    "CPU Temperature (°C)": {"high": ["Check onboard processor workload.", "Review thermal telemetry.", "Reduce non-essential processing if required."], "medium": ["Monitor CPU temperature trend.", "Review processor workload."]},
    "Signal Strength (dBm)": {"high": ["Verify ground-link availability.", "Check spacecraft attitude and antenna pointing.", "Prioritize critical telemetry transmission."], "medium": ["Monitor communication strength.", "Verify antenna pointing."]},
    "Attitude Error (deg)": {"high": ["Verify attitude-control telemetry.", "Check reaction-wheel or actuator behavior.", "Prioritize spacecraft stabilization."], "medium": ["Monitor attitude trend.", "Review attitude-control subsystem telemetry."]},
    "Thruster Fuel (%)": {"high": ["Verify propellant telemetry.", "Review recent maneuver history.", "Reassess remaining maneuver capability."], "medium": ["Monitor propellant consumption.", "Review recent maneuvers."]},
    "Memory Usage (%)": {"high": ["Inspect onboard software memory usage.", "Identify abnormal process growth.", "Consider restarting non-critical processes if operationally safe."], "medium": ["Monitor memory utilization.", "Review onboard process activity."]},
    "Downlink Rate (Mbps)": {"high": ["Verify communication link health.", "Check antenna and pointing status.", "Prioritize mission-critical data."], "medium": ["Monitor downlink throughput.", "Review communication conditions."]}
}

def get_mission_status(risk_counts):
    if risk_counts.get("HIGH", 0) > 0:
        return "CRITICAL", "🚨 Immediate mission review required."
    elif risk_counts.get("MEDIUM", 0) > 0:
        return "ATTENTION", "⚠️ High-risk telemetry requires operator review."
    elif risk_counts.get("LOW", 0) > 0:
        return "MONITOR", "🟡 Anomalous behavior detected. Continue monitoring."
    else:
        return "NOMINAL", "✅ Telemetry is within learned nominal behavior."

def generate_mission_recommendation(causes):
    if not causes:
        return {"impact": "No dominant subsystem risk identified.", "actions": ["Continue nominal monitoring."]}
    primary = causes[0]
    feature = primary["feature"]
    severity = primary["severity"]
    impact = MISSION_IMPACT_LIBRARY.get(feature, {}).get(severity, "Potential subsystem degradation detected.")
    actions = MISSION_ACTION_LIBRARY.get(feature, {}).get(severity, ["Continue monitoring the affected telemetry."])
    return {"primary_cause": primary, "impact": impact, "actions": actions}

def build_mission_summary(test_df, preds, scores):
    n_total = len(preds)
    n_anomalies = int((preds == -1).sum())
    if n_anomalies == 0:
        return {"total": n_total, "anomalies": 0, "message": "No anomalous behavior detected."}
    anomaly_idx = np.where(preds == -1)[0]
    worst_idx = anomaly_idx[np.argmin(scores[anomaly_idx])]
    return {"total": n_total, "anomalies": n_anomalies, "worst_index": int(worst_idx), "worst_score": float(scores[worst_idx])}

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
                st.markdown(f"<div class='cause-card {css
