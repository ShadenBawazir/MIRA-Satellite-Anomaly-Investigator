"""
MIRA — Mission Intelligence & Risk Analyzer
Satellite anomaly detection powered by OneClassSVM.

Supports two analysis modes:
  1. REAL OPS-SAT DATA  — trained on the ESA OPS-SAT-1 telemetry dataset (dataset.csv)
  2. MISSION SIMULATION — fully synthetic telemetry for interactive demonstration

Dataset credit: ESA OPS-SAT-1 mission telemetry, published via Zenodo.
IBM Bob assisted in building this application (app architecture,
UI design, and ML pipeline). IBM Bob did not create the dataset.
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials

# Load environment variables
load_dotenv()

# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═════════════════════════════════════════════════════════════════════════════

if 'run_clicked' not in st.session_state:
    st.session_state.run_clicked = False
if 'last_mode' not in st.session_state:
    st.session_state.last_mode = None
if 'results' not in st.session_state:
    st.session_state.results = None


# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MIRA — Satellite Anomaly Detector",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═════════════════════════════════════════════════════════════════════════════
# CSS
# ═════════════════════════════════════════════════════════════════════════════

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
    color: #ffffff; border: none; border-radius: 8px;
    padding: 0.5rem 1.4rem; font-weight: 600;
    letter-spacing: 0.5px; transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3b5de7, #5e7fff);
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(94,127,255,0.4);
}
.stMetric {
    background: #0b1640; border-radius: 10px; padding: 12px;
    border: 1px solid #1e3a8a;
    box-shadow: 0 2px 8px rgba(59,93,231,0.15);
}
.stMetric label { color: #7eb8f7 !important; }
.stMetric [data-testid="stMetricValue"] { color: #e0e8ff !important; }
hr { border-color: #1a2a6c !important; }
.js-plotly-plot { border-radius: 12px; }
.stTabs [data-baseweb="tab-list"] {
    background-color: #050d2e;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1a2a6c;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border-radius: 8px;
    color: #7eb8f7;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 6px 14px;
}
.stTabs [aria-selected="true"] {
    background-color: #1e3a8a !important;
    color: #ffffff !important;
}
.starfield {
    position: fixed; top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none; z-index: 0; overflow: hidden;
}
.star {
    position: absolute; background: white;
    border-radius: 50%; animation: twinkle linear infinite;
}
@keyframes twinkle {
    0%   { opacity: 0.1; transform: scale(1);   }
    50%  { opacity: 1;   transform: scale(1.3); }
    100% { opacity: 0.1; transform: scale(1);   }
}
@keyframes alertPulse {
    0%   { box-shadow: 0 0 0 0   rgba(239,68,68,0.8); background:#1a0808; }
    50%  { box-shadow: 0 0 40px 8px rgba(239,68,68,0.5); background:#2d0a0a; }
    100% { box-shadow: 0 0 0 0   rgba(239,68,68,0.0); background:#1a0808; }
}
.alert-banner {
    background:#1a0808; border:2px solid #ef4444; border-radius:10px;
    padding:14px 20px; color:#fca5a5; font-weight:700; font-size:1.05rem;
    animation:alertPulse 1.2s ease-in-out infinite; margin-bottom:12px;
}
.normal-banner {
    background:#021a0a; border:2px solid #22c55e; border-radius:10px;
    padding:14px 20px; color:#86efac; font-weight:600;
    font-size:1.05rem; margin-bottom:12px;
}
.warn-banner {
    background:#1a1202; border:2px solid #f59e0b; border-radius:10px;
    padding:14px 20px; color:#fcd34d; font-weight:600;
    font-size:1.05rem; margin-bottom:12px;
}
.mode-badge-real {
    display:inline-block; background:#0a1f0a; border:1.5px solid #22c55e;
    border-radius:6px; padding:3px 12px; color:#86efac;
    font-size:0.82rem; font-weight:700; letter-spacing:1px; margin-bottom:6px;
}
.mode-badge-sim {
    display:inline-block; background:#0f0f1a; border:1.5px solid #f59e0b;
    border-radius:6px; padding:3px 12px; color:#fcd34d;
    font-size:0.82rem; font-weight:700; letter-spacing:1px; margin-bottom:6px;
}
.cause-card {
    background:#080f2e; border:1px solid #1e3a8a; border-radius:10px;
    padding:14px 18px; margin:6px 0; color:#b8cef7;
    font-size:0.92rem; line-height:1.6;
}
.cause-card .cause-title { color:#7eb8f7; font-weight:700; font-size:0.98rem; margin-bottom:4px; }
.cause-high { border-left:4px solid #ef4444; }
.cause-med  { border-left:4px solid #f59e0b; }
.cause-low  { border-left:4px solid #22c55e; }
.section-title {
    font-size:1.1rem; font-weight:700; color:#7eb8f7;
    text-transform:uppercase; letter-spacing:1px;
    margin:18px 0 8px 0; padding-bottom:4px;
    border-bottom:1px solid #1a2a6c;
}
.dataset-info {
    background:#060d20; border:1px solid #1e3a8a; border-radius:10px;
    padding:12px 16px; color:#7eb8f7; font-size:0.85rem;
    line-height:1.7; margin-bottom:10px;
}
.model-card {
    background:#060d20; border:1px solid #1e3a8a; border-radius:10px;
    padding:16px 20px; color:#b8cef7; font-size:0.9rem;
    line-height:1.7; margin-bottom:10px;
}
.summary-card {
    background:linear-gradient(135deg,#0b1640,#1e3a8a);
    border:2px solid #3b5de7; border-radius:12px;
    padding:16px 20px; margin-bottom:12px; color:#e0e8ff;
}
.impact-card {
    background:#080f2e; border-left:4px solid #f59e0b;
    border-radius:8px; padding:12px 16px; margin:8px 0;
    color:#b8cef7; font-size:0.9rem;
}
.action-card {
    background:#080f2e; border-left:4px solid #22c55e;
    border-radius:8px; padding:10px 14px; margin:4px 0;
    color:#b8cef7; font-size:0.88rem;
}
.mini-grid {
    display:grid; grid-template-columns:repeat(4,1fr);
    gap:8px; margin-top:8px;
}
.mini-cell {
    background:#080f2e; padding:8px; border-radius:6px;
}
.mini-cell .lbl { font-size:0.72rem; color:#57606a; }
.mini-cell .val { font-size:1.05rem; }
</style>
"""

st.markdown(SPACE_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="starfield" id="stars"></div>
    <script>
    (function(){
        var sf=document.getElementById('stars');
        if(!sf)return;
        for(var i=0;i<120;i++){
            var s=document.createElement('div');
            s.className='star';
            var sz=Math.random()*2.5+0.5;
            s.style.cssText=[
                'width:'+sz+'px','height:'+sz+'px',
                'left:'+(Math.random()*100)+'vw',
                'top:'+(Math.random()*100)+'vh',
                'animation-duration:'+(Math.random()*4+2)+'s',
                'animation-delay:'+(Math.random()*5)+'s'
            ].join(';');
            sf.appendChild(s);
        }
    })();
    </script>
    """,
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

OPSSAT_META_COLS = {"segment", "anomaly", "train", "channel", "sampling", "duration", "len"}

OPSSAT_ROOT_CAUSE_LIBRARY = {
    "mean":             {"high": ("📉 Abnormal Signal Mean",              "The mean telemetry value deviates significantly from nominal."),
                         "medium": ("📊 Elevated Mean Drift",             "A moderate mean offset is present.")},
    "var":              {"high": ("🔀 Critical Variance Spike",           "Signal variance has increased dramatically."),
                         "medium": ("〰️ Elevated Signal Variance",        "Above-nominal variance detected.")},
    "std":              {"high": ("📡 High Signal Dispersion",            "Standard deviation far outside training norms."),
                         "medium": ("〰️ Moderate Signal Dispersion",      "Mildly elevated standard deviation.")},
    "smooth10_n_peaks": {"high": ("🏔️ Abnormal Peak Count (10-pt)",      "Peaks in lightly-smoothed signal are anomalous."),
                         "medium": ("🏔️ Elevated Peak Count (10-pt)",    "More peaks than expected.")},
    "smooth20_n_peaks": {"high": ("🏔️ Abnormal Peak Count (20-pt)",      "Heavily smoothed signal still shows anomalous peak count."),
                         "medium": ("🏔️ Elevated Peak Count (20-pt)",    "Above-nominal peaks in the 20-pt smoothed signal.")},
    "diff_peaks":       {"high": ("🔺 Anomalous 1st-Diff Peak Count",    "First-difference has abnormal peaks."),
                         "medium": ("🔺 Elevated 1st-Diff Peaks",        "Moderately high rate-of-change reversals.")},
    "diff2_peaks":      {"high": ("🔻 Anomalous 2nd-Diff Peak Count",    "Signal acceleration has abnormal peak count."),
                         "medium": ("🔻 Elevated 2nd-Diff Peaks",        "Moderate spike in second-difference peaks.")},
    "diff_var":         {"high": ("⚡ High 1st-Diff Variance",           "First-difference variance critically elevated."),
                         "medium": ("⚡ Elevated Rate-of-Change Var",    "First-diff variance above nominal baseline.")},
    "diff2_var":        {"high": ("🌊 High 2nd-Diff Variance",           "Second-difference variance anomalously high."),
                         "medium": ("🌊 Elevated Accel Variance",        "Moderate increase in second-diff variance.")},
    "gaps_squared":     {"high": ("🕳️ Anomalous Sampling Gaps",          "Squared-gap metric deviates significantly."),
                         "medium": ("🕳️ Irregular Sampling Gaps",        "Moderate deviation in gap structure.")},
    "len_weighted":     {"high": ("📏 Anomalous Length-Weighted Metric", "Length-weighted feature significantly off-nominal."),
                         "medium": ("📏 Elevated Length-Weighted Dev",   "Moderate length-weighted anomaly.")},
    "var_div_duration": {"high": ("⏱️ Critical Variance-per-Second",     "Variance/duration is critically high."),
                         "medium": ("⏱️ Elevated Variance Rate",         "Above-nominal variance-per-second.")},
    "var_div_len":      {"high": ("📐 Critical Variance-per-Sample",     "Per-sample variance critically elevated."),
                         "medium": ("📐 Elevated Per-Sample Variance",   "Moderate per-sample variance elevation.")},
}

SIM_FEATURES = [
    "Battery Voltage (V)", "Solar Panel Output (W)", "CPU Temperature (°C)",
    "Signal Strength (dBm)", "Attitude Error (deg)", "Thruster Fuel (%)",
    "Memory Usage (%)", "Downlink Rate (Mbps)",
]

SIM_ROOT_CAUSE_LIBRARY = {
    "Battery Voltage (V)":    {"high": ("⚡ Critical Power Failure",         "Battery voltage severely out of range."),
                               "medium": ("🔋 Battery Stress",               "Voltage deviation detected.")},
    "Solar Panel Output (W)": {"high": ("☀️ Solar Array Fault",             "Output has dropped critically."),
                               "medium": ("🌑 Reduced Solar Efficiency",     "Minor output degradation.")},
    "CPU Temperature (°C)":   {"high": ("🌡️ Thermal Runaway Risk",          "CPU temperature exceeds safe range."),
                               "medium": ("🔥 CPU Thermal Stress",           "Temperature approaching upper limit.")},
    "Signal Strength (dBm)":  {"high": ("📡 Communication Link Loss",       "Signal strength critically low."),
                               "medium": ("📶 Signal Degradation",           "Sub-nominal signal detected.")},
    "Attitude Error (deg)":   {"high": ("🔄 Attitude Control Failure",      "Large attitude error detected."),
                               "medium": ("↔️ Attitude Drift",               "Moderate pointing error detected.")},
    "Thruster Fuel (%)":      {"high": ("🛑 Fuel Depletion Alert",          "Fuel level critically low."),
                               "medium": ("⛽ Propellant Concern",           "Fuel consumption above nominal.")},
    "Memory Usage (%)":       {"high": ("💾 Memory Overflow Risk",          "Memory utilization critically high."),
                               "medium": ("🗄️ High Memory Utilization",     "Memory nearing capacity.")},
    "Downlink Rate (Mbps)":   {"high": ("📉 Downlink Failure",              "Downlink rate severely degraded."),
                               "medium": ("📊 Reduced Downlink Throughput", "Throughput below nominal.")},
}

SUBSYSTEM_LIBRARY = {
    "Battery Voltage (V)": "⚡ Power", "Solar Panel Output (W)": "☀️ Power",
    "CPU Temperature (°C)": "🌡️ Thermal", "Signal Strength (dBm)": "📡 Communication",
    "Attitude Error (deg)": "🔄 Attitude Control", "Thruster Fuel (%)": "🚀 Propulsion",
    "Memory Usage (%)": "💾 Computing", "Downlink Rate (Mbps)": "📡 Communication",
}

MISSION_IMPACT_LIBRARY = {
    "Battery Voltage (V)":    {"high": "Power may become insufficient for spacecraft operations. Risk of safe mode entry.",
                               "medium": "Power margin is reduced. Monitor battery state closely."},
    "Solar Panel Output (W)": {"high": "Reduced generation may accelerate battery depletion.",
                               "medium": "Reduced generation lowers available power margin."},
    "CPU Temperature (°C)":   {"high": "Thermal stress may affect computing reliability. Risk of hardware damage.",
                               "medium": "Thermal conditions should be monitored."},
    "Signal Strength (dBm)":  {"high": "Communication reliability compromised. Risk of telemetry loss.",
                               "medium": "Communication margin is reduced."},
    "Attitude Error (deg)":   {"high": "Pointing instability may affect payload and comms. Risk of mission failure.",
                               "medium": "Attitude stability requires monitoring."},
    "Thruster Fuel (%)":      {"high": "Reduced propellant margin may constrain future maneuvers.",
                               "medium": "Propellant margin should be monitored."},
    "Memory Usage (%)":       {"high": "Memory exhaustion could disrupt onboard software. Risk of crash and data loss.",
                               "medium": "High utilization may increase reliability risk."},
    "Downlink Rate (Mbps)":   {"high": "Reduced capacity may delay telemetry transmission.",
                               "medium": "Reduced throughput may affect data delivery."},
}

MISSION_ACTION_LIBRARY = {
    "Battery Voltage (V)":    {"high": ["Verify solar-array generation", "Reduce non-essential payload activity",
                                        "Monitor battery recovery in next telemetry window"],
                               "medium": ["Monitor battery voltage trend", "Check solar-array output"]},
    "Solar Panel Output (W)": {"high": ["Verify solar-array orientation", "Check for power degradation", "Monitor battery state"],
                               "medium": ["Monitor solar output trend", "Check spacecraft attitude"]},
    "CPU Temperature (°C)":   {"high": ["Check processor workload", "Review thermal telemetry", "Reduce non-essential processing"],
                               "medium": ["Monitor CPU temperature trend", "Review processor workload"]},
    "Signal Strength (dBm)":  {"high": ["Verify ground-link availability", "Check antenna pointing", "Prioritize critical telemetry"],
                               "medium": ["Monitor communication strength", "Verify antenna pointing"]},
    "Attitude Error (deg)":   {"high": ["Verify ACS telemetry", "Check reaction-wheel behavior", "Prioritize stabilization"],
                               "medium": ["Monitor attitude trend", "Review ACS subsystem telemetry"]},
    "Thruster Fuel (%)":      {"high": ["Verify propellant telemetry", "Review maneuver history", "Suspend non-critical maneuvers"],
                               "medium": ["Monitor propellant consumption", "Review recent maneuvers"]},
    "Memory Usage (%)":       {"high": ["Inspect memory usage", "Identify abnormal process growth", "Restart non-critical processes if safe"],
                               "medium": ["Monitor memory utilization", "Review onboard process activity"]},
    "Downlink Rate (Mbps)":   {"high": ["Verify communication link health", "Check antenna status", "Prioritize mission-critical data"],
                               "medium": ["Monitor downlink throughput", "Review communication conditions"]},
}

SIMULATION_SCENARIOS = {
    "Normal Operation":        {"description": "All systems nominal.", "modifications": {}},
    "Power Failure":           {"description": "Battery voltage drops, solar output decreases.",
                                "modifications": {"Battery Voltage (V)": -3.0, "Solar Panel Output (W)": -54.0}},
    "Thermal Stress":          {"description": "CPU temperature rises due to cooling failure.",
                                "modifications": {"CPU Temperature (°C)": 18.0}},
    "Communication Loss":      {"description": "Signal strength drops, downlink rate falls.",
                                "modifications": {"Signal Strength (dBm)": -20.0, "Downlink Rate (Mbps)": -40.0}},
    "Attitude Drift":          {"description": "Attitude control shows significant pointing error.",
                                "modifications": {"Attitude Error (deg)": 0.5}},
    "Memory Overflow":         {"description": "Memory usage increases critically.",
                                "modifications": {"Memory Usage (%)": 30.0}},
    "Multi-Subsystem Failure": {"description": "Multiple subsystems simultaneously affected.",
                                "modifications": {"Battery Voltage (V)": -2.5, "CPU Temperature (°C)": 12.0,
                                                  "Signal Strength (dBm)": -15.0, "Attitude Error (deg)": 0.3}},
    "Propellant Leak":         {"description": "Thruster fuel drops due to suspected leak.",
                                "modifications": {"Thruster Fuel (%)": -25.0}},
}


# ═════════════════════════════════════════════════════════════════════════════
# ML HELPERS
# ═════════════════════════════════════════════════════════════════════════════

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


def risk_level(score: float, score_min: float, score_max: float) -> str:
    norm = (score - score_min) / max(score_max - score_min, 1e-9)
    if norm < 0.10: return "CRITICAL"
    if norm < 0.25: return "HIGH"
    if norm < 0.50: return "MEDIUM"
    return "LOW"


def train_ocsvm(df, features, nu, kernel, gamma):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X)
    return model, scaler


def run_predict(model, scaler, df, features):
    X = scaler.transform(df[features])
    return model.predict(X), model.score_samples(X)


def classify_subsystem(feature: str) -> str:
    if feature in SUBSYSTEM_LIBRARY:
        return SUBSYSTEM_LIBRARY[feature]
    prefixes = {"mean": "📈 Stats", "var": "📈 Stats", "std": "📈 Stats",
                "smooth": "🎛️ Processing", "diff": "⚡ Dynamics",
                "gap": "🕳️ Sampling", "len": "📏 Segment", "duration": "⏱️ Temporal"}
    for p, s in prefixes.items():
        if feature.startswith(p) or p in feature:
            return s
    return "🔧 General"


def explain_anomaly(row, normal_stats, features, root_cause_lib):
    causes = []
    for feat in features:
        val  = row[feat]
        mean = normal_stats.loc["mean", feat]
        std  = normal_stats.loc["std",  feat]
        z    = abs(val - mean) / max(std, 1e-9)
        if z > 3.0:   severity = "high"
        elif z > 2.0: severity = "medium"
        else:         continue
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
            desc  = f"Value {val:.4g} deviates {z:.2f}σ from nominal mean ({mean:.4g})."
        confidence = max(0.3, min(1.0, (z / 6.0) * (0.9 if severity == "high" else 0.75)))
        causes.append({"feature": feat, "value": val, "z_score": z, "severity": severity,
                       "title": title, "desc": desc, "confidence": confidence,
                       "subsystem": classify_subsystem(feat)})
    causes.sort(key=lambda c: c["z_score"], reverse=True)
    return causes[:3]


def get_mission_status(risk_counts):
    if risk_counts.get("CRITICAL", 0) > 0:
        return "CRITICAL", "🚨 Immediate mission review required. Consider safe mode."
    if risk_counts.get("HIGH", 0) > 0:
        return "HIGH", "⚠️ High-risk anomalies detected. Mission priorities need reassessment."
    if risk_counts.get("MEDIUM", 0) > 0:
        return "MEDIUM", "🟡 Medium-risk anomalies detected. Continue close monitoring."
    if risk_counts.get("LOW", 0) > 0:
        return "LOW", "🔵 Low-risk anomalies detected. Monitor for escalation."
    return "NOMINAL", "✅ All systems operating within nominal parameters."


def generate_mission_recommendation(causes):
    if not causes:
        return {"impact": "No dominant subsystem risk identified.",
                "actions": ["Continue nominal monitoring."],
                "subsystem": "🔧 No specific subsystem"}
    primary  = causes[0]
    feature  = primary["feature"]
    severity = primary["severity"]
    impact   = MISSION_IMPACT_LIBRARY.get(feature, {}).get(severity, "Potential subsystem degradation.")
    actions  = MISSION_ACTION_LIBRARY.get(feature, {}).get(severity, ["Continue monitoring."])
    return {"impact": impact, "actions": actions,
            "subsystem": classify_subsystem(feature), "primary_cause": primary}
# ═════════════════════════════════════════════════════════════════════════════
# GENERATIVE AI - MISSION BRIEF GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_watsonx_model():
    """
    Initialize IBM watsonx.ai model for Mission Brief generation.
    Returns None if credentials are not configured.
    """
    api_key = os.getenv("WATSONX_APIKEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

    if not api_key or not project_id:
        return None

    credentials = Credentials(
        url=url,
        api_key=api_key
    )

    model = ModelInference(
        model_id="ibm/granite-3-8b-instruct",
        credentials=credentials,
        project_id=project_id
    )

    return model


def generate_ai_mission_brief(
    mission_status,
    anomaly_count,
    primary_cause,
    subsystem,
    impact,
    actions
):
    """
    Generate an AI-powered Mission Intelligence Brief using IBM Granite.
    The AI is grounded in structured outputs from the ML pipeline.
    """
    model = get_watsonx_model()

    if model is None:
        return (
            "⚠️ **Generative AI is not configured.**\n\n"
            "To enable the AI Mission Brief, add the following environment variables:\n"
            "- `WATSONX_APIKEY`\n"
            "- `WATSONX_PROJECT_ID`\n\n"
            "The ML anomaly detection and mission assessment are still fully functional."
        )

    actions_text = "\n".join(f"- {a}" for a in actions)

    prompt = f"""
You are a spacecraft mission intelligence assistant.

Generate a concise operational mission brief based ONLY on
the telemetry analysis provided below.

Mission status: {mission_status}
Detected anomalies: {anomaly_count}
Primary root cause: {primary_cause}
Subsystem: {subsystem}
Mission impact: {impact}

Recommended actions:
{actions_text}

Your response must contain exactly these sections:

MISSION ASSESSMENT
IMPACT
RECOMMENDED ACTIONS
CONFIDENCE NOTE

Do not invent telemetry values.
Do not claim that the spacecraft has failed unless the evidence
explicitly supports that conclusion.
Clearly distinguish detected anomalies from possible causes.
Keep the response concise and suitable for a flight operations team.
"""

    try:
        response = model.generate(
            prompt=prompt,
            params={
                "max_new_tokens": 300,
                "temperature": 0.2,
                "top_p": 0.9
            }
        )

        return response["results"][0]["generated_text"]

    except Exception as e:
        return f"⚠️ **AI Mission Brief unavailable:** {str(e)}"

def apply_scenario(df, scenario):
    if scenario not in SIMULATION_SCENARIOS:
        return df
    df = df.copy()
    for feat, change in SIMULATION_SCENARIOS[scenario]["modifications"].items():
        if feat in df.columns:
            df[feat] += change
    return df


def generate_normal_data(n=300):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Battery Voltage (V)":    rng.normal(28.0,  0.4,  n),
        "Solar Panel Output (W)": rng.normal(120.0, 5.0,  n),
        "CPU Temperature (°C)":   rng.normal(45.0,  3.0,  n),
        "Signal Strength (dBm)":  rng.normal(-70.0, 2.0,  n),
        "Attitude Error (deg)":   rng.normal(0.05,  0.02, n),
        "Thruster Fuel (%)":      rng.normal(75.0,  2.0,  n),
        "Memory Usage (%)":       rng.normal(55.0,  5.0,  n),
        "Downlink Rate (Mbps)":   rng.normal(50.0,  3.0,  n),
    })


def inject_anomalies(df, anomaly_rate=0.08):
    rng = np.random.default_rng(7)
    df  = df.copy()
    n   = len(df)
    idx = rng.choice(n, max(1, int(n * anomaly_rate)), replace=False)
    for i in idx:
        col = rng.choice(SIM_FEATURES)
        df.at[i, col] += rng.choice([-1, 1]) * rng.uniform(3.0, 6.0) * df[col].std()
    return df


# ═════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def make_radar_chart(row, normal_stats, features):
    short    = [f[:16] + "…" if len(f) > 16 else f for f in features]
    cats     = short + [short[0]]
    z_scores = [abs(row[f] - normal_stats.loc["mean", f]) / max(normal_stats.loc["std", f], 1e-9) for f in features]
    z_closed = z_scores + [z_scores[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[1]*(len(features)+1), theta=cats, fill="toself",
                                  name="Normal Boundary", line=dict(color="#3b5de7", dash="dash"),
                                  fillcolor="rgba(59,93,231,0.1)"))
    fig.add_trace(go.Scatterpolar(r=z_closed, theta=cats, fill="toself", name="Current Reading",
                                  line=dict(color="#ef4444"), fillcolor="rgba(239,68,68,0.2)"))
    fig.update_layout(
        polar=dict(bgcolor="#0b1640",
                   radialaxis=dict(visible=True, range=[0, max(5, max(z_scores)+1)],
                                   gridcolor="#1e3a8a", color="#7eb8f7"),
                   angularaxis=dict(gridcolor="#1e3a8a", color="#7eb8f7")),
        paper_bgcolor="#020818", plot_bgcolor="#020818",
        font=dict(color="#b8cef7", size=11),
        legend=dict(bgcolor="#050d2e", bordercolor="#1e3a8a"),
        margin=dict(l=40, r=40, t=30, b=20), height=340)
    return fig


def make_score_timeline(scores, preds, x_labels=None):
    colors = ["#ef4444" if p == -1 else "#22c55e" for p in preds]
    x = x_labels if x_labels is not None else list(range(len(scores)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=scores, mode="lines+markers",
                             line=dict(color="#3b5de7", width=1.5),
                             marker=dict(color=colors, size=5), name="Decision Score"))
    fig.update_layout(xaxis=dict(title="Telemetry Segment", gridcolor="#0d1f4a", color="#7eb8f7"),
                      yaxis=dict(title="Anomaly Score",     gridcolor="#0d1f4a", color="#7eb8f7"),
                      paper_bgcolor="#020818", plot_bgcolor="#020818", font=dict(color="#b8cef7"),
                      margin=dict(l=40, r=20, t=20, b=40), height=320, legend=dict(bgcolor="#050d2e"))
    return fig


def make_pca_scatter(df, preds, scaler, features):
    X2d  = PCA(n_components=2, random_state=0).fit_transform(scaler.transform(df[features]))
    pca  = PCA(n_components=2, random_state=0).fit(scaler.transform(df[features]))
    labels = ["Anomaly" if p == -1 else "Normal" for p in preds]
    fig = go.Figure()
    for label, color in [("Normal", "#3b5de7"), ("Anomaly", "#ef4444")]:
        mask = np.array([l == label for l in labels])
        fig.add_trace(go.Scatter(x=X2d[mask,0], y=X2d[mask,1], mode="markers", name=label,
                                 marker=dict(color=color, size=6, opacity=0.75)))
    fig.update_layout(xaxis=dict(title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", gridcolor="#0d1f4a", color="#7eb8f7"),
                      yaxis=dict(title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", gridcolor="#0d1f4a", color="#7eb8f7"),
                      paper_bgcolor="#020818", plot_bgcolor="#020818", font=dict(color="#b8cef7"),
                      margin=dict(l=40, r=20, t=20, b=40), height=320,
                      legend=dict(bgcolor="#050d2e", bordercolor="#1e3a8a"))
    return fig


def make_heatmap(df, features, key_suffix=""):
    with st.expander("📊 Full Telemetry Heatmap", expanded=False):
        heat_df = df[features].copy()
        heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min() + 1e-9)
        fig = px.imshow(heat_norm.T,
                        labels=dict(x="Segment", y="Feature", color="Normalised Value"),
                        color_continuous_scale="RdBu_r", aspect="auto")
        fig.update_layout(paper_bgcolor="#020818", plot_bgcolor="#020818", font=dict(color="#b8cef7"),
                          height=max(260, min(60*len(features), 600)),
                          margin=dict(l=20, r=20, t=10, b=20),
                          coloraxis_colorbar=dict(tickfont=dict(color="#b8cef7"),
                                                  title_font=dict(color="#b8cef7")))
        st.plotly_chart(fig, use_container_width=True, key=f"heatmap{key_suffix}")


# ═════════════════════════════════════════════════════════════════════════════
# RENDER HELPERS
# ═════════════════════════════════════════════════════════════════════════════

_STATUS_COLORS = {"CRITICAL": "#ef4444", "HIGH": "#f59e0b",
                  "MEDIUM": "#eab308", "LOW": "#3b82f6", "NOMINAL": "#22c55e"}
_STATUS_ICONS  = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "NOMINAL": "🟢"}


def render_alert_banner(n_anomalies, risk_counts):
    if risk_counts.get("CRITICAL", 0) > 0:
        c = risk_counts["CRITICAL"]
        st.markdown(f"<div class='alert-banner'>🚨 CRITICAL ALERT — {c} CRITICAL-RISK "
                    f"{'anomaly' if c==1 else 'anomalies'} detected! Immediate review required.</div>",
                    unsafe_allow_html=True)
    elif risk_counts.get("HIGH", 0) > 0:
        c = risk_counts["HIGH"]
        st.markdown(f"<div class='alert-banner' style='background:#1a1202;border-color:#f59e0b;'>"
                    f"⚠️ HIGH-RISK ALERT — {c} HIGH-RISK {'anomaly' if c==1 else 'anomalies'} detected!</div>",
                    unsafe_allow_html=True)
    elif risk_counts.get("MEDIUM", 0) > 0:
        c = risk_counts["MEDIUM"]
        st.markdown(f"<div class='warn-banner'>⚠️ {c} MEDIUM-RISK "
                    f"{'anomaly' if c==1 else 'anomalies'} detected. Monitor closely.</div>",
                    unsafe_allow_html=True)
    elif n_anomalies > 0:
        st.markdown(f"<div class='warn-banner'>🔵 {n_anomalies} LOW-risk "
                    f"{'anomaly' if n_anomalies==1 else 'anomalies'} detected.</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='normal-banner'>✅ ALL SYSTEMS NOMINAL — No anomalies detected.</div>",
                    unsafe_allow_html=True)


def render_mission_status(risk_counts, n_anomalies=0):
    status, message = get_mission_status(risk_counts)
    color = _STATUS_COLORS.get(status, "#7eb8f7")
    icon  = _STATUS_ICONS.get(status, "🛰️")
    anim  = "animation:alertPulse 2s ease-in-out infinite;" if status in ("CRITICAL","HIGH") else ""
    st.markdown(
        f"<div style='background:{color}15;border:2px solid {color};border-radius:12px;"
        f"padding:16px 20px;margin-bottom:12px;{anim}'>"
        f"<span style='font-size:1.15rem;font-weight:700;color:{color};'>{icon} MISSION STATUS: {status}</span>"
        f"<div style='font-size:0.9rem;color:{color}DD;margin-top:6px;'>{message}</div>"
        f"<div style='font-size:0.78rem;color:{color}88;margin-top:6px;'>"
        f"Anomalies: {n_anomalies} &nbsp;|&nbsp; "
        f"CRITICAL: {risk_counts.get('CRITICAL',0)} &nbsp;|&nbsp; "
        f"HIGH: {risk_counts.get('HIGH',0)} &nbsp;|&nbsp; "
        f"MEDIUM: {risk_counts.get('MEDIUM',0)} &nbsp;|&nbsp; "
        f"LOW: {risk_counts.get('LOW',0)}</div></div>",
        unsafe_allow_html=True)


def render_kpi_row(preds, risk_counts):
    n_total     = len(preds)
    n_anomalies = int((preds == -1).sum())
    n_normal    = int((preds ==  1).sum())
    anom_pct    = n_anomalies / n_total * 100
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("📊 Segments",    f"{n_total:,}")
    m2.metric("✅ Normal",       f"{n_normal:,}")
    m3.metric("⚠️ Anomalies",   f"{n_anomalies:,}", delta=f"{anom_pct:.1f}%", delta_color="inverse")
    m4.metric("🔴 CRITICAL",    risk_counts.get("CRITICAL", 0), delta_color="inverse")
    m5.metric("🟠 HIGH",        risk_counts.get("HIGH", 0),     delta_color="inverse")
    m6.metric("🟡 MEDIUM",      risk_counts.get("MEDIUM", 0),   delta_color="off")


def render_summary_card(preds, risk_counts, n_total):
    n_anomalies = int((preds == -1).sum())
    anomaly_rate = n_anomalies / n_total * 100
    status, message = get_mission_status(risk_counts)
    color = _STATUS_COLORS.get(status, "#7eb8f7")

    if n_anomalies == 0:
        summary = "All systems are operating within nominal parameters. No anomalous behaviour detected."
        recommendation = "Continue routine monitoring."
    elif risk_counts.get("CRITICAL", 0) > 0:
        summary = f"CRITICAL situation: {risk_counts['CRITICAL']} critical anomalies detected. Immediate intervention required."
        recommendation = "Initiate contingency procedures. Consider safe mode."
    elif risk_counts.get("HIGH", 0) > 0:
        summary = f"High-risk situation: {risk_counts['HIGH']} high-risk anomalies detected."
        recommendation = "Prioritise investigation and adjust mission timeline."
    elif risk_counts.get("MEDIUM", 0) > 0:
        summary = f"Medium-risk: {risk_counts['MEDIUM']} anomalies require attention."
        recommendation = "Increase monitoring frequency and prepare contingency plans."
    else:
        summary = f"Low-risk: {risk_counts.get('LOW',0)} minor anomalies detected."
        recommendation = "Continue monitoring and review periodically."

    st.markdown(
        f"<div class='summary-card'>"
        f"<div style='font-size:1rem;font-weight:700;color:{color};margin-bottom:8px;'>📊 MISSION INTELLIGENCE SUMMARY</div>"
        f"<div style='font-size:0.92rem;color:#b8cef7;line-height:1.6;margin-bottom:10px;'>{summary}</div>"
        f"<div class='mini-grid'>"
        f"<div class='mini-cell'><div class='lbl'>Segments</div><div class='val' style='color:#7eb8f7;'>{n_total:,}</div></div>"
        f"<div class='mini-cell'><div class='lbl'>Anomalies</div><div class='val' style='color:#ef4444;'>{n_anomalies}</div></div>"
        f"<div class='mini-cell'><div class='lbl'>Rate</div><div class='val' style='color:#f59e0b;'>{anomaly_rate:.1f}%</div></div>"
        f"<div class='mini-cell'><div class='lbl'>Status</div><div class='val' style='color:{color};'>{status}</div></div>"
        f"</div>"
        f"<div style='font-size:0.88rem;color:#b8cef7;margin-top:10px;padding-top:8px;border-top:1px solid #1e3a8a;'>"
        f"💡 <b>Recommendation:</b> {recommendation}</div></div>",
        unsafe_allow_html=True)


def render_inspector_tab(test_df, preds, scores, normal_stats, features, root_cause_lib, prefix):
    st.markdown("<div class='section-title'>🔬 Root-Cause Anomaly Inspector</div>", unsafe_allow_html=True)
    anomaly_indices = np.where(preds == -1)[0]
    score_min, score_max = scores.min(), scores.max()

    if len(anomaly_indices) == 0:
        st.info("No anomalies detected — nothing to inspect.")
        return

    frame_labels = {
        int(i): f"{prefix} {i:04d} — {risk_level(scores[i], score_min, score_max)} risk  (score: {scores[i]:.3f})"
        for i in anomaly_indices
    }
    selected = st.selectbox("Select anomaly to inspect:",
                            options=sorted(anomaly_indices, key=lambda i: scores[i]),
                            format_func=lambda i: frame_labels[i])
    row    = test_df.iloc[selected]
    s      = scores[selected]
    rlevel = risk_level(s, score_min, score_max)
    causes = explain_anomaly(row, normal_stats, features, root_cause_lib)

    badge  = _STATUS_COLORS.get(rlevel, "#7eb8f7")
    icon   = "🚨" if rlevel in ("CRITICAL","HIGH") else ("🟡" if rlevel == "MEDIUM" else "🔵")
    pulse  = "animation:alertPulse 1.2s ease-in-out 4;" if rlevel in ("CRITICAL","HIGH") else ""
    st.markdown(
        f"<div style='background:#080f2e;border:2px solid {badge};border-radius:12px;"
        f"padding:14px 18px;margin-bottom:12px;{pulse}'>"
        f"<span style='color:{badge};font-weight:700;'>{icon} {prefix} {selected:04d} — {rlevel} RISK</span>"
        f"<span style='color:#57606a;font-size:0.85rem;margin-left:14px;'>Score: {s:.4f}</span></div>",
        unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        if causes:
            st.markdown("**Root Causes:**")
            for c in causes:
                cls = "cause-high" if c["severity"]=="high" else ("cause-med" if c["severity"]=="medium" else "cause-low")
                conf = c.get("confidence", 0.7) * 100
                st.markdown(
                    f"<div class='cause-card {cls}'>"
                    f"<div class='cause-title'>{c['title']}</div>"
                    f"<b>{c['feature']}</b>: {c['value']:.4g} &nbsp;(z={c['z_score']:.2f})<br>"
                    f"{c['desc']}<br>"
                    f"<span style='color:#57606a;font-size:0.78rem;'>📍 {c['subsystem']} &nbsp;|&nbsp; 🎯 Confidence: {conf:.0f}%</span>"
                    f"</div>", unsafe_allow_html=True)

            rec = generate_mission_recommendation(causes)
            st.markdown("**🎯 Mission Impact & Actions:**")
            st.markdown(
                f"<div class='impact-card'><b>Subsystem:</b> {rec['subsystem']}<br><b>Impact:</b> {rec['impact']}</div>",
                unsafe_allow_html=True)
            for action in rec["actions"]:
                st.markdown(f"<div class='action-card'>• {action}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='cause-card cause-low'><div class='cause-title'>ℹ️ Subtle Multi-Feature Deviation</div>"
                "No single feature exceeds 2σ. The anomaly results from compound minor deviations across multiple channels.</div>",
                unsafe_allow_html=True)

        st.markdown("<br><b>Feature Table:</b>", unsafe_allow_html=True)
        tbl_rows = []
        for f in features:
            val  = row[f]
            mean = normal_stats.loc["mean", f]
            std  = normal_stats.loc["std",  f]
            z    = (val - mean) / max(std, 1e-9)
            tbl_rows.append({"Feature": f, "Value": round(float(val),4),
                             "Nominal Mean": round(float(mean),4), "Δ (σ)": round(float(z),2)})
        st.dataframe(pd.DataFrame(tbl_rows).set_index("Feature"), use_container_width=True)

    with right:
        st.markdown("**Deviation Radar:**")
        st.plotly_chart(make_radar_chart(row, normal_stats, features),
                        use_container_width=True, key=f"radar_{selected}")


def render_charts_tab(test_df, preds, scores, scaler, features, x_labels=None, key_suffix=""):
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("<div class='section-title'>📈 Anomaly Score Timeline</div>", unsafe_allow_html=True)
        st.plotly_chart(make_score_timeline(scores, preds, x_labels),
                        use_container_width=True, key=f"timeline{key_suffix}")
    with c2:
        st.markdown("<div class='section-title'>🔵 PCA Feature Space</div>", unsafe_allow_html=True)
        st.plotly_chart(make_pca_scatter(test_df, preds, scaler, features),
                        use_container_width=True, key=f"pca{key_suffix}")
    make_heatmap(test_df, features, key_suffix=key_suffix)


def render_model_info_tab(model_type, nu, kernel, gamma, n_features, n_train, mode,
                          preds=None, true_labels=None, has_labels=False,
                          tp=0, fp=0, fn=0, tn=0, precision=0.0, recall=0.0):
    # Model card
    st.markdown("<div class='section-title'>📋 Model Card</div>", unsafe_allow_html=True)
    cells = [("Model", "OneClassSVM"), ("Kernel", kernel), ("Nu", f"{nu:.2f}"),
             ("Gamma", gamma), ("Features", str(n_features)), ("Train Samples", f"{n_train:,}"),
             ("Mode", mode), ("Preprocessing", "StandardScaler")]
    grid_html = "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:8px;'>"
    for lbl, val in cells:
        grid_html += (f"<div style='background:#080f2e;padding:8px;border-radius:6px;'>"
                      f"<div style='font-size:0.72rem;color:#57606a;'>{lbl}</div>"
                      f"<div style='font-size:0.9rem;color:#e0e8ff;'>{val}</div></div>")
    grid_html += "</div>"
    st.markdown(f"<div class='model-card'>{grid_html}</div>", unsafe_allow_html=True)

    # Validation / confusion matrix
    if has_labels and preds is not None and true_labels is not None:
        st.markdown("<div class='section-title'>✅ Ground-Truth Validation</div>", unsafe_allow_html=True)
        n_true_anom = int((true_labels == 1).sum())
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("📋 True Anomalies", n_true_anom)
        g2.metric("🎯 True Positives", tp)
        g3.metric("🎯 Precision",      f"{precision:.2%}")
        g4.metric("🔁 Recall",         f"{recall:.2%}")

        st.markdown("<div class='section-title'>📊 Confusion Matrix</div>", unsafe_allow_html=True)
        fig = go.Figure(data=go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Predicted Normal", "Predicted Anomaly"],
            y=["Actual Normal", "Actual Anomaly"],
            text=[[tn, fp], [fn, tp]], texttemplate="%{text}",
            textfont={"size": 16, "color": "#e0e8ff"},
            colorscale=[[0,"#0b1640"],[0.5,"#1e3a8a"],[1,"#3b5de7"]], showscale=False))
        fig.update_layout(title="Confusion Matrix", title_font_color="#7eb8f7",
                          paper_bgcolor="#020818", plot_bgcolor="#020818",
                          font=dict(color="#b8cef7"), height=300,
                          margin=dict(l=40,r=20,t=40,b=20))
        col_cm, col_stats = st.columns([2,1])
        with col_cm:
            st.plotly_chart(fig, use_container_width=True, key="cm")
        with col_stats:
            st.markdown("**Performance:**")
            a1,a2 = st.columns(2)
            a1.metric("Accuracy",  f"{(tp+tn)/max(tp+tn+fp+fn,1):.2%}")
            a2.metric("F1-Score",  f"{2*precision*recall/max(precision+recall,1e-9):.2%}")
            st.markdown(
                f"<div style='background:#080f2e;padding:10px;border-radius:8px;margin-top:8px;font-size:0.82rem;'>"
                f"<b style='color:#7eb8f7;'>Counts:</b><br>"
                f"<span style='color:#b8cef7;'>TN={tn} &nbsp; FP={fp}<br>FN={fn} &nbsp; TP={tp}</span></div>",
                unsafe_allow_html=True)


def render_limitations_tab():
    items = [
        ("🔴", "One-Class Nature", "Only learns from normal data; cannot distinguish anomaly types."),
        ("🟡", "Data Quality",     "Results depend on training dataset completeness and quality."),
        ("🟠", "Seasonal Drift",   "Normal seasonal variations may be flagged as anomalies."),
        ("🔵", "Threshold Heuristics", "Risk thresholds are heuristic and may need per-mission tuning."),
        ("🟣", "Feature Engineering", "Raw telemetry may contain additional information not captured."),
        ("🟤", "Interpretability", "OneClassSVM provides limited interpretability vs. deep learning."),
        ("⚪", "Computational Cost", "Training on large datasets can be computationally expensive."),
        ("🟥", "Ground Truth",     "In real missions, ground truth labels are often incomplete."),
        ("🟨", "Context Awareness", "The model doesn't understand mission context or constraints."),
        ("🟩", "Anomaly Labelling", "Novel anomaly types unseen in training may be missed."),
    ]
    st.markdown(
        "<div style='background:#080f2e;border:1px solid #f59e0b;border-radius:12px;padding:16px 20px;'>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:1.05rem;font-weight:700;color:#f59e0b;margin-bottom:12px;'>⚠️ LIMITATIONS & CONSTRAINTS</div>",
        unsafe_allow_html=True)
    for icon, title, desc in items:
        st.markdown(
            f"<div style='display:flex;align-items:flex-start;gap:10px;margin-bottom:10px;'>"
            f"<span style='font-size:1.1rem;'>{icon}</span>"
            f"<div><b style='color:#7eb8f7;font-size:0.9rem;'>{title}:</b> "
            f"<span style='color:#b8cef7;font-size:0.87rem;'>{desc}</span></div></div>",
            unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🛰️ MIRA Controls")
    st.markdown("<hr>", unsafe_allow_html=True)

    analysis_mode = st.radio(
        "Analysis Mode",
        options=["🛰️ Real OPS-SAT Data", "🔬 Mission Simulation"],
        index=0,
        help="Choose between real ESA OPS-SAT-1 telemetry or synthetic simulation.")
    IS_REAL = analysis_mode.startswith("🛰️")
    st.markdown("<hr>", unsafe_allow_html=True)

    if IS_REAL:
        st.markdown(
            "<div style='background:#0a1a0a;border:1px solid #22c55e;border-radius:8px;"
            "padding:10px 14px;margin-bottom:10px;'>"
            "<span style='color:#86efac;font-weight:700;font-size:0.85rem;'>🛰️ REAL OPS-SAT DATA</span><br>"
            "<span style='color:#57606a;font-size:0.78rem;line-height:1.6;'>"
            "ESA OPS-SAT-1 mission telemetry.<br>"
            "Trains on nominal segments (train=1, anomaly=0).<br>"
            "<b>Fixed config:</b> OneClassSVM(rbf, nu=0.22)</span></div>",
            unsafe_allow_html=True)
        nu_val     = 0.22
        kernel_val = "rbf"
        gamma_val  = "scale"
        st.caption(f"**Kernel:** {kernel_val}  |  **Nu:** {nu_val}  |  **Gamma:** {gamma_val}")
        selected_scenario = "Normal Operation"
        n_frames, anomaly_pct = 300, 8
    else:
        st.markdown(
            "<div style='background:#1a1202;border:1px solid #f59e0b;border-radius:8px;"
            "padding:10px 14px;margin-bottom:10px;'>"
            "<span style='color:#fcd34d;font-weight:700;font-size:0.85rem;'>🔬 MISSION SIMULATION</span><br>"
            "<span style='color:#57606a;font-size:0.78rem;line-height:1.6;'>"
            "All values are <b>synthetic</b>. No real satellite data.</span></div>",
            unsafe_allow_html=True)
        st.markdown("### 🤖 Hyperparameters")
        nu_val     = st.slider("Nu (outlier fraction)", 0.01, 0.50, 0.08, 0.01)
        kernel_val = st.selectbox("Kernel", ["rbf","linear","poly","sigmoid"], index=0)
        gamma_val  = st.selectbox("Gamma",  ["scale","auto"], index=0)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📡 Simulation")
        n_frames    = st.slider("Training frames", 100, 1000, 300, 50)
        anomaly_pct = st.slider("Injected anomaly %", 1, 30, 8, 1)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🎯 Scenario")
        selected_scenario = st.selectbox("Select Scenario", list(SIMULATION_SCENARIOS.keys()), index=0)
        if selected_scenario != "Normal Operation":
            st.markdown(
                f"<div style='background:#080f2e;border:1px solid #1e3a8a;border-radius:6px;"
                f"padding:8px 12px;font-size:0.8rem;color:#b8cef7;'>"
                f"📖 {SIMULATION_SCENARIOS[selected_scenario]['description']}</div>",
                unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    
    # استخدام session state للزر
    run_btn = st.button("🚀 Run MIRA Analysis", use_container_width=True, key="run_btn")
    if run_btn:
        st.session_state.run_clicked = True
        st.session_state.last_mode = analysis_mode
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#57606a;font-size:0.75rem;line-height:1.6;'>"
        "MIRA uses <b>OneClassSVM</b> trained on nominal telemetry.<br><br>"
        "<b>IBM AI Builders Challenge</b><br>"
        "Advance Space Exploration with AI<br>"
        "Built with <b>IBM Bob</b></div>",
        unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style="text-align:center;padding:16px 0 6px 0;">
        <span style="font-size:2.2rem;">🛰️</span>
        <div style="font-size:2rem;font-weight:700;color:#7eb8f7;letter-spacing:3px;margin:2px 0;">MIRA</div>
        <div style="color:#57606a;font-size:0.85rem;letter-spacing:4px;">MISSION INTELLIGENCE &amp; RISK ANALYZER</div>
        <div style="width:60px;height:2px;background:#3b5de7;margin:10px auto 0 auto;border-radius:2px;"></div>
    </div>
    """,
    unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# IDLE SCREEN
# ═════════════════════════════════════════════════════════════════════════════

if not st.session_state.run_clicked:
    sub = ("Select <b style='color:#86efac;'>🛰️ Real OPS-SAT Data</b> mode and click"
           if IS_REAL else "Configure the simulation parameters and click")
    st.markdown(
        f"<div style='text-align:center;padding:60px 20px;color:#57606a;'>"
        f"<div style='font-size:3.5rem;margin-bottom:12px;'>🌌</div>"
        f"<h3 style='color:#3b5de7;'>Awaiting Mission Start</h3>"
        f"<p>{sub} <b style='color:#7eb8f7;'>🚀 Run MIRA Analysis</b> to begin satellite monitoring.</p>"
        f"</div>",
        unsafe_allow_html=True)
    st.stop()

# Reset run_clicked when mode changes
if st.session_state.last_mode != analysis_mode:
    st.session_state.run_clicked = False
    st.session_state.last_mode = analysis_mode
    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# MODE A — REAL OPS-SAT DATA
# ═════════════════════════════════════════════════════════════════════════════

if IS_REAL:
    st.markdown(
        "<div style='text-align:center;margin-bottom:8px;'>"
        "<span class='mode-badge-real'>🛰️ REAL OPS-SAT DATA — ESA OPS-SAT-1 Mission Telemetry</span></div>",
        unsafe_allow_html=True)

    with st.spinner("📂 Loading OPS-SAT dataset…"):
        raw_df, feature_cols, load_err = load_opssat_dataset("dataset.csv")

    if load_err:
        st.error(f"**Dataset Error:** {load_err}")
        st.markdown(
            "<div class='cause-card cause-high'><div class='cause-title'>📁 dataset.csv not found</div>"
            "Place <code>dataset.csv</code> in the same directory as <code>mira_app.py</code>.</div>",
            unsafe_allow_html=True)
        st.stop()

    n_train_nominal = (int(((raw_df["train"]==1)&(raw_df["anomaly"]==0)).sum())
                       if "train" in raw_df.columns and "anomaly" in raw_df.columns else len(raw_df))
    channels = raw_df["channel"].unique().tolist() if "channel" in raw_df.columns else ["—"]
    ch_str = ", ".join(str(c) for c in channels[:6]) + ("…" if len(channels) > 6 else "")
    st.markdown(
        f"<div class='dataset-info'><b>Dataset:</b> ESA OPS-SAT-1 &nbsp;|&nbsp; "
        f"<b>Segments:</b> {len(raw_df):,} &nbsp;|&nbsp; "
        f"<b>Nominal train:</b> {n_train_nominal:,} &nbsp;|&nbsp; "
        f"<b>Features:</b> {len(feature_cols)} &nbsp;|&nbsp; <b>Channels:</b> {ch_str}</div>",
        unsafe_allow_html=True)

    if "train" in raw_df.columns and "anomaly" in raw_df.columns:
        train_df = raw_df[(raw_df["train"]==1)&(raw_df["anomaly"]==0)].copy()
        test_df  = raw_df.copy()
    else:
        train_df = raw_df.copy()
        test_df  = raw_df.copy()

    if len(train_df) == 0:
        st.error("No nominal training segments found (train=1, anomaly=0).")
        st.stop()

    with st.spinner("🤖 Training OneClassSVM…"):
        model, scaler = train_ocsvm(train_df, feature_cols, nu=0.22, kernel="rbf", gamma="scale")
    with st.spinner("🔍 Scanning telemetry…"):
        preds, scores = run_predict(model, scaler, test_df, feature_cols)

    normal_stats    = train_df[feature_cols].describe().loc[["mean","std"]]
    score_min, score_max = scores.min(), scores.max()
    n_anomalies     = int((preds == -1).sum())
    n_total         = len(preds)
    risk_counts     = {"CRITICAL":0,"HIGH":0,"MEDIUM":0,"LOW":0}
    for sv, p in zip(scores, preds):
        if p == -1:
            risk_counts[risk_level(sv, score_min, score_max)] += 1

    has_labels  = "anomaly" in test_df.columns
    true_labels = test_df["anomaly"].values if has_labels else None
    tp = fp = fn = tn = 0
    precision = recall = 0.0
    if has_labels:
        tp = int(((preds==-1)&(true_labels==1)).sum())
        fp = int(((preds==-1)&(true_labels==0)).sum())
        fn = int(((preds== 1)&(true_labels==1)).sum())
        tn = int(((preds== 1)&(true_labels==0)).sum())
        precision = tp / max(tp+fp, 1)
        recall    = tp / max(tp+fn, 1)

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🔬 Anomaly Inspector",
        "📈 Charts", "📋 Model Info", "⚠️ Limitations"])

    with tab1:
        render_mission_status(risk_counts, n_anomalies)
        render_alert_banner(n_anomalies, risk_counts)
        render_kpi_row(preds, risk_counts)
        st.markdown("<hr>", unsafe_allow_html=True)
        render_summary_card(preds, risk_counts, n_total)

    with tab2:
        render_inspector_tab(test_df, preds, scores, normal_stats,
                             feature_cols, OPSSAT_ROOT_CAUSE_LIBRARY, "Segment")

    with tab3:
        x_labels = test_df["segment"].tolist() if "segment" in test_df.columns else None
        render_charts_tab(test_df, preds, scores, scaler, feature_cols, x_labels, "_real")

    with tab4:
        render_model_info_tab(
            "OneClassSVM", nu_val, kernel_val, gamma_val,
            len(feature_cols), len(train_df), "🛰️ Real OPS-SAT",
            preds=preds, true_labels=true_labels, has_labels=has_labels,
            tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall)

    with tab5:
        render_limitations_tab()


# ═════════════════════════════════════════════════════════════════════════════
# MODE B — MISSION SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

else:
    st.markdown(
        "<div style='text-align:center;margin-bottom:8px;'>"
        "<span class='mode-badge-sim'>🔬 MISSION SIMULATION — All values are SYNTHETIC</span></div>",
        unsafe_allow_html=True)

    if selected_scenario != "Normal Operation":
        st.markdown(
            f"<div class='dataset-info'><b>🎯 Active Scenario:</b> {selected_scenario} &nbsp;|&nbsp; "
            f"{SIMULATION_SCENARIOS[selected_scenario]['description']}</div>",
            unsafe_allow_html=True)

    with st.spinner("🛰️ Generating synthetic telemetry…"):
        normal_df = generate_normal_data(n_frames)
        if selected_scenario != "Normal Operation":
            normal_df = apply_scenario(normal_df, selected_scenario)
        test_df      = inject_anomalies(normal_df.copy(), anomaly_rate=anomaly_pct/100)
        normal_stats = normal_df.describe().loc[["mean","std"]]

    with st.spinner("🤖 Training OneClassSVM…"):
        model, scaler = train_ocsvm(normal_df, SIM_FEATURES, nu=nu_val, kernel=kernel_val, gamma=gamma_val)
    with st.spinner("🔍 Scanning telemetry…"):
        preds, scores = run_predict(model, scaler, test_df, SIM_FEATURES)

    score_min, score_max = scores.min(), scores.max()
    n_anomalies = int((preds == -1).sum())
    n_total     = len(preds)
    risk_counts = {"CRITICAL":0,"HIGH":0,"MEDIUM":0,"LOW":0}
    for sv, p in zip(scores, preds):
        if p == -1:
            risk_counts[risk_level(sv, score_min, score_max)] += 1

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🔬 Anomaly Inspector",
        "📈 Charts", "📋 Model Info", "⚠️ Limitations"])

    with tab1:
        render_mission_status(risk_counts, n_anomalies)
        render_alert_banner(n_anomalies, risk_counts)
        render_kpi_row(preds, risk_counts)
        st.markdown("<hr>", unsafe_allow_html=True)
        render_summary_card(preds, risk_counts, n_total)

    with tab2:
        render_inspector_tab(test_df, preds, scores, normal_stats,
                             SIM_FEATURES, SIM_ROOT_CAUSE_LIBRARY, "Frame")

    with tab3:
        render_charts_tab(test_df, preds, scores, scaler, SIM_FEATURES, key_suffix="_sim")

    with tab4:
        render_model_info_tab(
            "OneClassSVM", nu_val, kernel_val, gamma_val,
            len(SIM_FEATURES), len(normal_df), "🔬 Simulation")

    with tab5:
        render_limitations_tab()


# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<div style='text-align:center;color:#57606a;font-size:0.75rem;"
    "padding:16px 0 6px 0;border-top:1px solid #1a2a6c;margin-top:16px;'>"
    "MIRA — Mission Intelligence &amp; Risk Analyzer &nbsp;|&nbsp; "
    "OneClassSVM &nbsp;|&nbsp; "
    "OPS-SAT dataset © ESA &nbsp;|&nbsp; "
    "Built with <b style='color:#3b5de7;'>IBM Bob</b>"
    "</div>",
    unsafe_allow_html=True)
