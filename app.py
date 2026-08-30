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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


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
# DARK SPACE THEME
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

[data-testid="stHeader"] {
    background: transparent !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #7eb8f7 !important;
}

label, .stMarkdown, p {
    color: #b8cef7 !important;
}

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

.stMetric {
    background: #0b1640;
    border-radius: 10px;
    padding: 12px;
    border: 1px solid #1e3a8a;
}

.stMetric label {
    color: #7eb8f7 !important;
}

.stMetric [data-testid="stMetricValue"] {
    color: #e0e8ff !important;
}

hr {
    border-color: #1a2a6c !important;
}

.js-plotly-plot {
    border-radius: 12px;
}

.starfield {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

.star {
    position: absolute;
    background: white;
    border-radius: 50%;
    animation: twinkle linear infinite;
}

@keyframes twinkle {
    0% {
        opacity: 0.1;
        transform: scale(1);
    }

    50% {
        opacity: 1;
        transform: scale(1.3);
    }

    100% {
        opacity: 0.1;
        transform: scale(1);
    }
}

@keyframes alertPulse {
    0% {
        box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.8);
        background: #1a0808;
    }

    50% {
        box-shadow: 0 0 40px 8px rgba(239, 68, 68, 0.5);
        background: #2d0a0a;
    }

    100% {
        box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.0);
        background: #1a0808;
    }
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

.cause-high {
    border-left: 4px solid #ef4444;
}

.cause-med {
    border-left: 4px solid #f59e0b;
}

.cause-low {
    border-left: 4px solid #22c55e;
}

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

.model-card {
    background: #060d20;
    border: 1px solid #1e3a8a;
    border-radius: 10px;
    padding: 16px 20px;
    color: #b8cef7;
    font-size: 0.9rem;
    line-height: 1.7;
    margin-bottom: 10px;
}

.limitation-card {
    background: #080f2e;
    border: 1px solid #f59e0b;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #b8cef7;
    font-size: 0.88rem;
    line-height: 1.6;
}

.summary-card {
    background: linear-gradient(135deg, #0b1640, #1e3a8a);
    border: 2px solid #3b5de7;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    color: #e0e8ff;
}

.impact-card {
    background: #080f2e;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #b8cef7;
    font-size: 0.9rem;
}

.action-card {
    background: #080f2e;
    border-left: 4px solid #22c55e;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #b8cef7;
    font-size: 0.9rem;
}

</style>
"""

st.markdown(SPACE_CSS, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# STARFIELD
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="starfield" id="stars"></div>
    <script>
    (function() {
        var sf = document.getElementById('stars');
        if (!sf) return;
        for (var i = 0; i < 120; i++) {
            var s = document.createElement('div');
            s.className = 'star';
            var sz = Math.random() * 2.5 + 0.5;
            s.style.cssText = [
                'width:' + sz + 'px',
                'height:' + sz + 'px',
                'left:' + (Math.random() * 100) + 'vw',
                'top:' + (Math.random() * 100) + 'vh',
                'animation-duration:' + (Math.random() * 4 + 2) + 's',
                'animation-delay:' + (Math.random() * 5) + 's'
            ].join(';');
            sf.appendChild(s);
        }
    })();
    </script>
    """,
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════════════
# OPS-SAT DATASET CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

OPSSAT_META_COLS = {"segment", "anomaly", "train", "channel", "sampling", "duration", "len"}

OPSSAT_ROOT_CAUSE_LIBRARY = {
    "mean": {
        "high": ("📉 Abnormal Signal Mean", "The mean telemetry value deviates significantly from nominal."),
        "medium": ("📊 Elevated Mean Drift", "A moderate mean offset is present."),
    },
    "var": {
        "high": ("🔀 Critical Variance Spike", "Signal variance has increased dramatically."),
        "medium": ("〰️ Elevated Signal Variance", "Above-nominal variance detected."),
    },
    "std": {
        "high": ("📡 High Signal Dispersion", "Standard deviation far outside training norms."),
        "medium": ("〰️ Moderate Signal Dispersion", "Mildly elevated standard deviation."),
    },
    "smooth10_n_peaks": {
        "high": ("🏔️ Abnormal Peak Count (10-pt smooth)", "Number of peaks in the lightly-smoothed signal is anomalous."),
        "medium": ("🏔️ Elevated Peak Count (10-pt smooth)", "More peaks than expected."),
    },
    "smooth20_n_peaks": {
        "high": ("🏔️ Abnormal Peak Count (20-pt smooth)", "Highly smoothed signal still shows anomalous peak count."),
        "medium": ("🏔️ Elevated Peak Count (20-pt smooth)", "Above-nominal peaks in the 20-point smoothed signal."),
    },
    "diff_peaks": {
        "high": ("🔺 Anomalous First-Difference Peak Count", "The first-difference series has an abnormal number of peaks."),
        "medium": ("🔺 Elevated First-Difference Peaks", "Moderately high rate of change reversals."),
    },
    "diff2_peaks": {
        "high": ("🔻 Anomalous Second-Difference Peak Count", "Acceleration of the signal has an abnormal peak count."),
        "medium": ("🔻 Elevated Second-Difference Peaks", "Moderate spike in second-difference peaks."),
    },
    "diff_var": {
        "high": ("⚡ High First-Difference Variance", "The variance of the signal's first difference is critically elevated."),
        "medium": ("⚡ Elevated Rate-of-Change Variance", "First-difference variance above nominal baseline."),
    },
    "diff2_var": {
        "high": ("🌊 High Second-Difference Variance", "The second-difference variance is anomalously high."),
        "medium": ("🌊 Elevated Signal Acceleration Variance", "Moderate increase in second-difference variance."),
    },
    "gaps_squared": {
        "high": ("🕳️ Anomalous Sampling Gap Structure", "The squared-gap metric deviates significantly."),
        "medium": ("🕳️ Irregular Sampling Gaps", "Moderate deviation in sampling gap structure."),
    },
    "len_weighted": {
        "high": ("📏 Anomalous Length-Weighted Metric", "Length-weighted feature significantly off-nominal."),
        "medium": ("📏 Elevated Length-Weighted Deviation", "Moderate length-weighted anomaly."),
    },
    "var_div_duration": {
        "high": ("⏱️ Critical Variance-per-Second", "Variance normalised by duration is critically high."),
        "medium": ("⏱️ Elevated Variance Rate", "Above-nominal variance-per-second detected."),
    },
    "var_div_len": {
        "high": ("📐 Critical Variance-per-Sample", "Per-sample variance is critically elevated."),
        "medium": ("📐 Elevated Per-Sample Variance", "Moderate per-sample variance elevation."),
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# SYNTHETIC SIMULATION FEATURES
# ═════════════════════════════════════════════════════════════════════════════

SIM_FEATURES = [
    "Battery Voltage (V)",
    "Solar Panel Output (W)",
    "CPU Temperature (°C)",
    "Signal Strength (dBm)",
    "Attitude Error (deg)",
    "Thruster Fuel (%)",
    "Memory Usage (%)",
    "Downlink Rate (Mbps)",
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

# ═════════════════════════════════════════════════════════════════════════════
# SUBSYSTEM CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════

SUBSYSTEM_LIBRARY = {
    "Battery Voltage (V)": "⚡ Power Subsystem",
    "Solar Panel Output (W)": "☀️ Power Subsystem",
    "CPU Temperature (°C)": "🌡️ Thermal Subsystem",
    "Signal Strength (dBm)": "📡 Communication Subsystem",
    "Attitude Error (deg)": "🔄 Attitude Control Subsystem",
    "Thruster Fuel (%)": "🚀 Propulsion Subsystem",
    "Memory Usage (%)": "💾 Onboard Computing Subsystem",
    "Downlink Rate (Mbps)": "📡 Communication Subsystem",
}

# OPS-SAT feature prefix mapping
OPS_SAT_SUBSYSTEM_MAP = {
    "mean": "📈 Signal Statistics",
    "var": "📈 Signal Statistics",
    "std": "📈 Signal Statistics",
    "smooth": "🎛️ Signal Processing",
    "diff": "⚡ Signal Dynamics",
    "gap": "🕳️ Sampling Integrity",
    "len": "📏 Segment Characteristics",
    "duration": "⏱️ Temporal Characteristics",
}

# ═════════════════════════════════════════════════════════════════════════════
# MISSION IMPACT AND ACTIONS
# ═════════════════════════════════════════════════════════════════════════════

MISSION_IMPACT_LIBRARY = {
    "Battery Voltage (V)": {
        "high": "Power availability may become insufficient for non-essential spacecraft operations. Risk of spacecraft safe mode entry.",
        "medium": "Power margin is reduced and should be monitored. May affect long-term mission planning."
    },
    "Solar Panel Output (W)": {
        "high": "Reduced generation may accelerate battery depletion. Could lead to complete power loss if not addressed.",
        "medium": "Reduced generation could lower the available power margin. Monitor battery state closely."
    },
    "CPU Temperature (°C)": {
        "high": "Thermal stress may affect onboard computing reliability. Potential for hardware damage if sustained.",
        "medium": "Thermal conditions should be monitored. May affect processing performance."
    },
    "Signal Strength (dBm)": {
        "high": "Communication reliability may be compromised. Risk of telemetry loss and mission data gaps.",
        "medium": "Communication margin is reduced. May affect downlink throughput."
    },
    "Attitude Error (deg)": {
        "high": "Pointing instability may affect payload operations and communication. Risk of mission objective failure.",
        "medium": "Attitude stability requires monitoring. May affect pointing accuracy."
    },
    "Thruster Fuel (%)": {
        "high": "Reduced propellant margin may constrain future attitude or orbit-control operations. Risk of mission extension limitation.",
        "medium": "Propellant margin should be monitored. May affect upcoming maneuvers."
    },
    "Memory Usage (%)": {
        "high": "Memory exhaustion could disrupt onboard software processes. Risk of system crash and data loss.",
        "medium": "High utilization may increase software reliability risk. Monitor for anomalies."
    },
    "Downlink Rate (Mbps)": {
        "high": "Reduced downlink capacity may delay or prevent telemetry and payload data transmission. Risk of mission data loss.",
        "medium": "Reduced throughput may affect data delivery. Monitor communication status."
    }
}

MISSION_ACTION_LIBRARY = {
    "Battery Voltage (V)": {
        "high": ["Verify solar-array power generation", "Reduce non-essential payload activity", "Monitor battery recovery in the next telemetry window", "Consider power reconfiguration"],
        "medium": ["Monitor battery voltage trend", "Check solar-array output", "Review power budget"]
    },
    "Solar Panel Output (W)": {
        "high": ["Verify solar-array orientation", "Check for power-generation degradation", "Monitor battery state", "Consider panel retargeting"],
        "medium": ["Monitor solar output trend", "Check spacecraft attitude", "Review power generation schedule"]
    },
    "CPU Temperature (°C)": {
        "high": ["Check onboard processor workload", "Review thermal telemetry", "Reduce non-essential processing if required", "Consider thermal control adjustments"],
        "medium": ["Monitor CPU temperature trend", "Review processor workload", "Check thermal control system"]
    },
    "Signal Strength (dBm)": {
        "high": ["Verify ground-link availability", "Check spacecraft attitude and antenna pointing", "Prioritize critical telemetry transmission", "Consider backup communication channel"],
        "medium": ["Monitor communication strength", "Verify antenna pointing", "Review link budget"]
    },
    "Attitude Error (deg)": {
        "high": ["Verify attitude-control telemetry", "Check reaction-wheel or actuator behavior", "Prioritize spacecraft stabilization", "Consider attitude safe mode"],
        "medium": ["Monitor attitude trend", "Review attitude-control subsystem telemetry", "Check reaction wheel health"]
    },
    "Thruster Fuel (%)": {
        "high": ["Verify propellant telemetry", "Review recent maneuver history", "Reassess remaining maneuver capability", "Suspend non-critical maneuvers"],
        "medium": ["Monitor propellant consumption", "Review recent maneuvers", "Check for potential leaks"]
    },
    "Memory Usage (%)": {
        "high": ["Inspect onboard software memory usage", "Identify abnormal process growth", "Consider restarting non-critical processes if operationally safe", "Flush unnecessary buffers"],
        "medium": ["Monitor memory utilization", "Review onboard process activity", "Check for memory leaks"]
    },
    "Downlink Rate (Mbps)": {
        "high": ["Verify communication link health", "Check antenna and pointing status", "Prioritize mission-critical data", "Consider backup downlink"],
        "medium": ["Monitor downlink throughput", "Review communication conditions", "Check ground station status"]
    }
}

# ═════════════════════════════════════════════════════════════════════════════
# SIMULATION SCENARIOS
# ═════════════════════════════════════════════════════════════════════════════

SIMULATION_SCENARIOS = {
    "Normal Operation": {
        "description": "All systems operating within nominal parameters.",
        "modifications": {}
    },
    "Power Failure": {
        "description": "Battery voltage drops and solar panel output decreases significantly.",
        "modifications": {
            "Battery Voltage (V)": -3.0,
            "Solar Panel Output (W)": -0.45  # 55% decrease
        }
    },
    "Thermal Stress": {
        "description": "CPU temperature rises significantly due to cooling system failure.",
        "modifications": {
            "CPU Temperature (°C)": +18.0
        }
    },
    "Communication Loss": {
        "description": "Signal strength drops and downlink rate decreases.",
        "modifications": {
            "Signal Strength (dBm)": -20.0,
            "Downlink Rate (Mbps)": -0.8  # 80% decrease
        }
    },
    "Attitude Drift": {
        "description": "Attitude control system shows significant pointing error.",
        "modifications": {
            "Attitude Error (deg)": +0.5
        }
    },
    "Memory Overflow": {
        "description": "Memory usage increases critically, risking system crash.",
        "modifications": {
            "Memory Usage (%)": +30.0
        }
    },
    "Multi-Subsystem Failure": {
        "description": "Multiple subsystems simultaneously affected - critical scenario.",
        "modifications": {
            "Battery Voltage (V)": -2.5,
            "CPU Temperature (°C)": +12.0,
            "Signal Strength (dBm)": -15.0,
            "Attitude Error (deg)": +0.3
        }
    },
    "Propellant Leak": {
        "description": "Thruster fuel level drops rapidly due to suspected leak.",
        "modifications": {
            "Thruster Fuel (%)": -25.0
        }
    }
}

# ═════════════════════════════════════════════════════════════════════════════
# DATASET LOADER
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


# ═════════════════════════════════════════════════════════════════════════════
# RISK LEVEL
# ═════════════════════════════════════════════════════════════════════════════

def risk_level(score: float, score_min: float, score_max: float) -> str:
    norm = (
        (score - score_min)
        / max(score_max - score_min, 1e-9)
    )
    
    if norm < 0.10:
        return "CRITICAL"
    if norm < 0.25:
        return "HIGH"
    if norm < 0.50:
        return "MEDIUM"
    return "LOW"


# ═════════════════════════════════════════════════════════════════════════════
# MISSION STATUS
# ═════════════════════════════════════════════════════════════════════════════

def get_mission_status(risk_counts):
    if risk_counts.get("CRITICAL", 0) > 0:
        return "CRITICAL", "🚨 IMMEDIATE MISSION REVIEW REQUIRED. Multiple critical anomalies detected. Consider safe mode."
    elif risk_counts.get("HIGH", 0) > 0:
        return "HIGH", "⚠️ High-risk anomalies detected. Mission priorities need reassessment."
    elif risk_counts.get("MEDIUM", 0) > 0:
        return "MEDIUM", "🟡 Medium-risk anomalies detected. Continue close monitoring."
    elif risk_counts.get("LOW", 0) > 0:
        return "LOW", "🔵 Low-risk anomalies detected. Monitor for escalation."
    else:
        return "NOMINAL", "✅ All systems operating within nominal parameters."


def render_mission_status(risk_counts, n_anomalies=0):
    status, message = get_mission_status(risk_counts)
    
    status_colors = {
        "CRITICAL": "#ef4444",
        "HIGH": "#f59e0b", 
        "MEDIUM": "#eab308",
        "LOW": "#3b82f6",
        "NOMINAL": "#22c55e"
    }
    
    status_icons = {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🔵",
        "NOMINAL": "🟢"
    }
    
    color = status_colors.get(status, "#7eb8f7")
    icon = status_icons.get(status, "🛰️")
    
    st.markdown(
        f"""
        <div style='background: {color}15; border: 2px solid {color}; 
                    border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;
                    animation: alertPulse 2s ease-in-out infinite;'>
            <span style='font-size: 1.2rem; font-weight: 700; color: {color};'>
                {icon} MISSION STATUS: {status}
            </span>
            <div style='font-size: 0.95rem; color: {color}DD; margin-top: 6px;'>
                {message}
            </div>
            <div style='font-size: 0.8rem; color: {color}99; margin-top: 8px;'>
                Detected anomalies: {n_anomalies} | 
                CRITICAL: {risk_counts.get("CRITICAL", 0)} | 
                HIGH: {risk_counts.get("HIGH", 0)} | 
                MEDIUM: {risk_counts.get("MEDIUM", 0)} | 
                LOW: {risk_counts.get("LOW", 0)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ═════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════════

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


def classify_subsystem(feature: str) -> str:
    """Classify a feature into its subsystem."""
    # First check exact match in SIM_FEATURES
    if feature in SUBSYSTEM_LIBRARY:
        return SUBSYSTEM_LIBRARY[feature]
    
    # Then check OPS-SAT feature prefixes
    for prefix, subsystem in OPS_SAT_SUBSYSTEM_MAP.items():
        if feature.startswith(prefix) or prefix in feature:
            return subsystem
    
    return "🔧 General Telemetry"


def calculate_confidence(z_score: float, severity: str) -> float:
    """Calculate confidence based on z-score and severity."""
    base_confidence = min(1.0, z_score / 6.0)  # 6σ = 100% confidence
    
    if severity == "high":
        confidence = base_confidence * 0.9  # High severity slightly reduces confidence
    elif severity == "medium":
        confidence = base_confidence * 0.75  # Medium severity reduces confidence more
    else:
        confidence = base_confidence * 0.6  # Low severity reduces confidence most
    
    return max(0.3, min(1.0, confidence))  # Clamp between 30% and 100%


def explain_anomaly(row: pd.Series, normal_stats: pd.DataFrame, features: list, root_cause_lib: dict):
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
        
        # Calculate confidence
        confidence = calculate_confidence(z, severity)
        subsystem = classify_subsystem(feat)
        
        causes.append({
            "feature": feat,
            "value": val,
            "z_score": z,
            "severity": severity,
            "title": title,
            "desc": desc,
            "confidence": confidence,
            "subsystem": subsystem,
        })
    
    causes.sort(key=lambda c: c["z_score"], reverse=True)
    return causes[:3]  # Top 3


def generate_mission_recommendation(causes):
    """Generate mission impact and recommended actions based on root causes."""
    if not causes:
        return {
            "impact": "No dominant subsystem risk identified.",
            "actions": ["Continue nominal monitoring."],
            "subsystem": "🔧 No specific subsystem affected"
        }
    
    primary = causes[0]
    feature = primary["feature"]
    severity = primary["severity"]
    
    # Get mission impact
    impact = MISSION_IMPACT_LIBRARY.get(feature, {}).get(severity, 
                "Potential subsystem degradation detected. Monitor closely.")
    
    # Get recommended actions
    actions = MISSION_ACTION_LIBRARY.get(feature, {}).get(severity, 
                ["Continue monitoring the affected telemetry."])
    
    # Get subsystem
    subsystem = classify_subsystem(feature)
    
    return {
        "impact": impact,
        "actions": actions,
        "subsystem": subsystem,
        "primary_cause": primary
    }


# ═════════════════════════════════════════════════════════════════════════════
# MISSION INTELLIGENCE SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

def generate_mission_summary(preds, scores, risk_counts, n_anomalies, n_total):
    """Generate a comprehensive mission intelligence summary."""
    
    # Calculate key metrics
    anomaly_rate = (n_anomalies / n_total) * 100 if n_total > 0 else 0
    critical_count = risk_counts.get("CRITICAL", 0)
    high_count = risk_counts.get("HIGH", 0)
    medium_count = risk_counts.get("MEDIUM", 0)
    low_count = risk_counts.get("LOW", 0)
    
    # Determine mission status
    status, message = get_mission_status(risk_counts)
    
    # Generate summary text
    if n_anomalies == 0:
        summary_text = "All systems are operating within nominal parameters. No anomalous behavior detected."
        recommendation = "Continue routine monitoring and maintain current operational tempo."
    elif critical_count > 0:
        summary_text = f"⚠️ CRITICAL SITUATION: {critical_count} critical-risk anomalies detected. Multiple subsystems may be affected. Immediate mission review is required."
        recommendation = "Initiate contingency procedures, consider safe mode, and prepare for emergency recovery operations."
    elif high_count > 0:
        summary_text = f"🚨 HIGH-RISK: {high_count} high-risk anomalies detected. Mission objectives may be at risk without immediate intervention."
        recommendation = "Prioritize investigation of high-risk anomalies and adjust mission timeline accordingly."
    elif medium_count > 0:
        summary_text = f"🟡 MEDIUM-RISK: {medium_count} medium-risk anomalies detected. Continue monitoring and plan for potential escalations."
        recommendation = "Increase monitoring frequency and prepare contingency plans for potential escalations."
    else:
        summary_text = f"🔵 LOW-RISK: {low_count} low-risk anomalies detected. No immediate action required but continue monitoring."
        recommendation = "Continue routine monitoring and review low-risk anomalies periodically."
    
    return {
        "status": status,
        "message": message,
        "summary_text": summary_text,
        "recommendation": recommendation,
        "metrics": {
            "total_frames": n_total,
            "anomaly_count": n_anomalies,
            "anomaly_rate": anomaly_rate,
            "critical": critical_count,
            "high": high_count,
            "medium": medium_count,
            "low": low_count,
            "normal": n_total - n_anomalies
        }
    }


def render_mission_summary(summary_data):
    """Render mission intelligence summary in UI."""
    
    status = summary_data["status"]
    metrics = summary_data["metrics"]
    
    # Color based on status
    color_map = {
        "CRITICAL": "#ef4444",
        "HIGH": "#f59e0b",
        "MEDIUM": "#eab308",
        "LOW": "#3b82f6",
        "NOMINAL": "#22c55e"
    }
    
    color = color_map.get(status, "#7eb8f7")
    
    st.markdown(
        f"""
        <div class='summary-card'>
            <div style='font-size: 1.1rem; font-weight: 700; color: {color}; margin-bottom: 8px;'>
                📊 MISSION INTELLIGENCE SUMMARY
            </div>
            <div style='font-size: 0.95rem; color: #b8cef7; line-height: 1.6; margin-bottom: 12px;'>
                {summary_data["summary_text"]}
            </div>
            <div style='font-size: 0.9rem; color: #7eb8f7; margin-top: 10px;'>
                <b>📋 Key Metrics:</b>
            </div>
            <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 8px;'>
                <div style='background: #080f2e; padding: 8px; border-radius: 6px;'>
                    <div style='font-size: 0.75rem; color: #57606a;'>Frames</div>
                    <div style='font-size: 1.1rem; color: #7eb8f7;'>{metrics["total_frames"]:,}</div>
                </div>
                <div style='background: #080f2e; padding: 8px; border-radius: 6px;'>
                    <div style='font-size: 0.75rem; color: #57606a;'>Anomalies</div>
                    <div style='font-size: 1.1rem; color: #ef4444;'>{metrics["anomaly_count"]}</div>
                </div>
                <div style='background: #080f2e; padding: 8px; border-radius: 6px;'>
                    <div style='font-size: 0.75rem; color: #57606a;'>Rate</div>
                    <div style='font-size: 1.1rem; color: #f59e0b;'>{metrics["anomaly_rate"]:.1f}%</div>
                </div>
                <div style='background: #080f2e; padding: 8px; border-radius: 6px;'>
                    <div style='font-size: 0.75rem; color: #57606a;'>Status</div>
                    <div style='font-size: 1.1rem; color: {color};'>{status}</div>
                </div>
            </div>
            <div style='font-size: 0.9rem; color: #b8cef7; margin-top: 12px; padding-top: 10px; border-top: 1px solid #1e3a8a;'>
                <b>💡 Recommended Action:</b> {summary_data["recommendation"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ═════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX
# ═════════════════════════════════════════════════════════════════════════════

def render_confusion_matrix(preds, true_labels):
    """Render confusion matrix with metrics."""
    
    if true_labels is None:
        return
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, preds, labels=[1, -1])
    
    # Extract values
    tn, fp = cm[0]  # True Normal, False Anomaly
    fn, tp = cm[1]  # False Normal, True Anomaly
    
    # Calculate metrics
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = f1_score(true_labels, preds, labels=[1, -1], zero_division=0)
    
    # Render confusion matrix as plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Predicted Normal", "Predicted Anomaly"],
        y=["Actual Normal", "Actual Anomaly"],
        text=[[tn, fp], [fn, tp]],
        texttemplate="%{text}",
        textfont={"size": 16, "color": "#e0e8ff"},
        colorscale=[[0, "#0b1640"], [0.5, "#1e3a8a"], [1, "#3b5de7"]],
        showscale=False,
        hovertemplate="<b>%{y} → %{x}</b><br>Count: %{z}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        title_font_color="#7eb8f7",
        paper_bgcolor="#020818",
        plot_bgcolor="#020818",
        font=dict(color="#b8cef7"),
        height=300,
        margin=dict(l=40, r=20, t=40, b=20)
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True, key="confusion_matrix")
    
    with col2:
        st.markdown("**📈 Model Performance:**")
        
        m1, m2 = st.columns(2)
        m1.metric("🎯 Accuracy", f"{accuracy:.2%}")
        m2.metric("🎯 Precision", f"{precision:.2%}")
        
        m3, m4 = st.columns(2)
        m3.metric("🔁 Recall", f"{recall:.2%}")
        m4.metric("📊 F1-Score", f"{f1:.2%}")
        
        st.markdown(f"""
        <div style='background: #080f2e; padding: 10px; border-radius: 8px; margin-top: 10px; font-size: 0.85rem;'>
            <b style='color: #7eb8f7;'>Interpretation:</b><br>
            <span style='color: #b8cef7;'>
            True Normal (TN): {tn}<br>
            False Anomaly (FP): {fp}<br>
            False Normal (FN): {fn}<br>
            True Anomaly (TP): {tp}
            </span>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL CARD
# ═════════════════════════════════════════════════════════════════════════════

def render_model_card(model_type, nu, kernel, gamma, features_count, n_train_samples, mode):
    """Render model card with model information."""
    
    st.markdown(
        f"""
        <div class='model-card'>
            <div style='font-size: 1.1rem; font-weight: 700; color: #7eb8f7; margin-bottom: 12px;'>
                📋 MODEL CARD
            </div>
            <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;'>
                <div style='background: #080f2e; padding: 8px; border-radius: 6px;'>
                    <div style='font-size: 0.75rem; color: #57606a;'>Model Type</div>
                    <div style='font-size: 0.95rem; color: #e0e8ff;'>OneClassSVM</div>
                </div>
                <div style='background: #080f2e; padding: 8px; border-radius: 6px;'>
                    <div style='font-size: 0.75rem; color: #57606a;'>Kernel</div>
                    <div style='font-size: 0.95rem; color: #e0e8ff;'>{kernel}</div>
                </div>
                <div style='background: #080f2e; padding: 8px; border-radius: 6px;'>
                    <div style='font-size: 0.75rem; color: #57606a;'>Nu Parameter</div>
                    <div style='font-size: 0.95rem; color: #e0e8ff;'>{nu:.2f}</div>
                </div>
                <div style='background: #080f2e; padding: 8px; border-radius: 6px;'>
                    <div style='font-size: 0.75rem; color: #57606a;'>Gamma</div>
                    <div style='font-size: 0.95rem; color: #e0e8ff;'>{gamma}</div>
                </div>
                <div style='background: #080f2e; padding: 8px; border-radius: 6px;'>
                    <div style='font-size: 0.75rem; color: #57606a;'>Features Used</div>
                    <div style='font-size: 0.95rem; color: #e0e8ff;'>{features_count}</div>
                </div>
                <div style='background: #080f2e; padding: 8px; border-radius: 6px;'>
                    <div style='font-size: 0.75rem; color: #57606a;'>Training Samples</div>
                    <div style='font-size: 0.95rem; color: #e0e8ff;'>{n_train_samples:,}</div>
                </div>
                <div style='background: #080f2e; padding: 8px; border-radius: 6px;'>
                    <div style='font-size: 0.75rem; color: #57606a;'>Mode</div>
                    <div style='font-size: 0.95rem; color: #e0e8ff;'>{mode}</div>
                </div>
                <div style='background: #080f2e; padding: 8px; border-radius: 6px;'>
                    <div style='font-size: 0.75rem; color: #57606a;'>Preprocessing</div>
                    <div style='font-size: 0.95rem; color: #e0e8ff;'>StandardScaler</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ═════════════════════════════════════════════════════════════════════════════
# LIMITATIONS
# ═════════════════════════════════════════════════════════════════════════════

def render_limitations():
    """Render limitations section."""
    
    limitations = [
        "🔴 **One-Class Nature**: The model only learns from normal data and cannot distinguish between different types of anomalies.",
        "🟡 **Data Quality Dependency**: Results heavily depend on the quality and completeness of the training dataset.",
        "🟠 **Seasonal/Environmental Changes**: The model may flag normal seasonal variations as anomalies.",
        "🔵 **Label Imbalance**: Limited anomaly samples may affect threshold calibration.",
        "🟣 **Feature Engineering**: The model relies on engineered features; raw telemetry might contain additional useful information.",
        "🟤 **Model Interpretability**: OneClassSVM provides limited interpretability compared to deep learning approaches.",
        "⚪ **Computational Cost**: Training on large datasets can be computationally expensive.",
        "🟥 **Ground Truth Availability**: In real mission scenarios, ground truth labels are often unavailable or incomplete.",
        "🟨 **Threshold Sensitivity**: Risk level thresholds are heuristic and may require tuning per mission.",
        "🟩 **Contextual Understanding**: The model doesn't understand mission context or operational constraints.",
    ]
    
    st.markdown(
        """
        <div style='background: #080f2e; border: 1px solid #f59e0b; border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;'>
            <div style='font-size: 1.1rem; font-weight: 700; color: #f59e0b; margin-bottom: 12px;'>
                ⚠️ LIMITATIONS & CONSTRAINTS
            </div>
        """,
        unsafe_allow_html=True
    )
    
    for limitation in limitations:
        st.markdown(
            f"<div style='font-size: 0.88rem; color: #b8cef7; margin-bottom: 8px; padding-left: 16px;'>{limitation}</div>",
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# AI MISSION REPORT
# ═════════════════════════════════════════════════════════════════════════════

def render_ai_report(summary_data, causes_list, risk_counts):
    """Render comprehensive AI mission report."""
    
    st.markdown(
        "<div style='background: linear-gradient(135deg, #0b1640, #1e3a8a); "
        "border: 2px solid #3b5de7; border-radius: 12px; padding: 20px; margin-bottom: 16px;'>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<div style='font-size: 1.3rem; font-weight: 700; color: #7eb8f7; margin-bottom: 16px;'>"
        "🤖 AI MISSION REPORT"
        "</div>",
        unsafe_allow_html=True
    )
    
    # Section 1: Mission Status
    status = summary_data["status"]
    color_map = {
        "CRITICAL": "#ef4444",
        "HIGH": "#f59e0b",
        "MEDIUM": "#eab308",
        "LOW": "#3b82f6",
        "NOMINAL": "#22c55e"
    }
    color = color_map.get(status, "#7eb8f7")
    
    st.markdown(
        f"""
        <div style='font-size: 0.95rem; margin-bottom: 12px;'>
            <span style='color: {color}; font-weight: 700;'>■ MISSION STATUS: {status}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Section 2: Summary
    st.markdown(
        "<div style='font-size: 0.9rem; color: #b8cef7; margin-bottom: 12px;'>"
        f"<b>📝 Summary:</b> {summary_data['summary_text']}"
        "</div>",
        unsafe_allow_html=True
    )
    
    # Section 3: Key Metrics
    metrics = summary_data["metrics"]
    st.markdown(
        f"""
        <div style='font-size: 0.9rem; color: #b8cef7; margin-bottom: 12px;'>
            <b>📊 Key Metrics:</b><br>
            Total Frames: {metrics["total_frames"]:,} | 
            Anomalies: {metrics["anomaly_count"]} | 
            Rate: {metrics["anomaly_rate"]:.1f}%<br>
            CRITICAL: {metrics["critical"]} | 
            HIGH: {metrics["high"]} | 
            MEDIUM: {metrics["medium"]} | 
            LOW: {metrics["low"]}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Section 4: Root Causes
    if causes_list:
        st.markdown(
            "<div style='font-size: 0.9rem; color: #b8cef7; margin-bottom: 12px;'>"
            "<b>🔍 Top Root Causes:</b>"
            "</div>",
            unsafe_allow_html=True
        )
        
        for i, cause in enumerate(causes_list[:3]):
            confidence_pct = cause.get("confidence", 0.7) * 100
            st.markdown(
                f"""
                <div style='font-size: 0.88rem; color: #b8cef7; margin-left: 16px; margin-bottom: 8px;'>
                    {i+1}. {cause["title"]} 
                    <span style='color: #57606a;'>(Confidence: {confidence_pct:.1f}%)</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Section 5: Recommendation
    st.markdown(
        f"""
        <div style='font-size: 0.9rem; color: #b8cef7; margin-top: 12px; padding-top: 12px; border-top: 1px solid #1e3a8a;'>
            <b>💡 Recommended Action:</b> {summary_data["recommendation"]}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# RADAR CHART
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
# SCORE TIMELINE
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
# PCA SCATTER
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
# SIMULATION DATA
# ═════════════════════════════════════════════════════════════════════════════

def generate_normal_data(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({"Battery Voltage (V)": rng.normal(28.0, 0.4, n),
                          "Solar Panel Output (W)": rng.normal(120.0, 5.0, n),
                          "CPU Temperature (°C)": rng.normal(45.0, 3.0, n),
                          "Signal Strength (dBm)": rng.normal(-70.0, 2.0, n),
                          "Attitude Error (deg)": rng.normal(0.05, 0.02, n),
                          "Thruster Fuel (%)": rng.normal(75.0, 2.0, n),
                          "Memory Usage (%)": rng.normal(55.0, 5.0, n),
                          "Downlink Rate (Mbps)": rng.normal(50.0, 3.0, n)})


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


def apply_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Apply a simulation scenario to the dataframe."""
    if scenario not in SIMULATION_SCENARIOS:
        return df
    
    modifications = SIMULATION_SCENARIOS[scenario]["modifications"]
    df = df.copy()
    
    for feature, change in modifications.items():
        if feature in df.columns:
            if change < 0:
                # For percentage-like changes (e.g., 0.45 means 45% decrease)
                if abs(change) < 1:
                    df[feature] *= (1 + change)
                else:
                    df[feature] += change
            else:
                df[feature] += change
    
    return df


# ═════════════════════════════════════════════════════════════════════════════
# ALERT BANNER
# ═════════════════════════════════════════════════════════════════════════════

def render_alert_banner(n_anomalies, risk_counts):
    has_critical = risk_counts.get("CRITICAL", 0) > 0
    has_high = risk_counts.get("HIGH", 0) > 0
    has_medium = risk_counts.get("MEDIUM", 0) > 0
    
    if has_critical:
        count = risk_counts.get("CRITICAL", 0)
        word = "anomaly" if count == 1 else "anomalies"
        st.markdown(
            f"<div class='alert-banner' style='background:#1a0000;border-color:#ef4444;'>"
            f"🚨 CRITICAL ALERT — {count} CRITICAL-RISK {word} detected! "
            f"Immediate mission review required."
            f"</div>",
            unsafe_allow_html=True,
        )
    elif has_high:
        count = risk_counts.get("HIGH", 0)
        word = "anomaly" if count == 1 else "anomalies"
        st.markdown(
            f"<div class='alert-banner' style='background:#1a1202;border-color:#f59e0b;'>"
            f"⚠️ HIGH-RISK ALERT — {count} HIGH-RISK {word} detected! "
            f"Immediate investigation recommended."
            f"</div>",
            unsafe_allow_html=True,
        )
    elif has_medium:
        count = risk_counts.get("MEDIUM", 0)
        word = "anomaly" if count == 1 else "anomalies"
        st.markdown(
            f"<div class='warn-banner'>"
            f"⚠️ {count} MEDIUM-RISK {word} detected. Monitor closely."
            f"</div>",
            unsafe_allow_html=True,
        )
    elif n_anomalies > 0:
        word = "anomaly" if n_anomalies == 1 else "anomalies"
        st.markdown(
            f"<div class='warn-banner'>"
            f"🔵 {n_anomalies} {word} detected at LOW risk. Continue monitoring."
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='normal-banner'>"
            "✅ ALL SYSTEMS NOMINAL — No anomalies detected."
            "</div>",
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ═════════════════════════════════════════════════════════════════════════════

def render_kpi_row(preds, scores, risk_counts):
    n_total = len(preds)
    n_anomalies = int((preds == -1).sum())
    n_normal = int((preds == 1).sum())
    anom_pct = n_anomalies / n_total * 100
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    m1.metric("📊 Frames Analysed", f"{n_total:,}")
    m2.metric("✅ Normal", f"{n_normal:,}")
    m3.metric("⚠️ Anomalies", f"{n_anomalies:,}", delta=f"{anom_pct:.1f}%", delta_color="inverse")
    m4.metric("🔴 CRITICAL", risk_counts.get("CRITICAL", 0), delta_color="inverse")
    m5.metric("🟠 HIGH", risk_counts.get("HIGH", 0), delta_color="inverse")
    m6.metric("🟡 MEDIUM", risk_counts.get("MEDIUM", 0), delta_color="off")


# ═════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═════════════════════════════════════════════════════════════════════════════

def render_charts(test_df, preds, scores, scaler, features, x_labels=None):
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("<div class='section-title'>📈 Anomaly Score Timeline</div>", unsafe_allow_html=True)
        st.plotly_chart(make_score_timeline(scores, preds, x_labels), use_container_width=True, key="timeline")
    with col_right:
        st.markdown("<div class='section-title'>🔵 PCA Feature Space</div>", unsafe_allow_html=True)
        st.plotly_chart(make_pca_scatter(test_df, preds, scaler, features), use_container_width=True, key="pca")


# ═════════════════════════════════════════════════════════════════════════════
# ANOMALY INSPECTOR
# ═════════════════════════════════════════════════════════════════════════════

def render_inspector(test_df, preds, scores, normal_stats, features, root_cause_lib, frame_label_prefix="Frame"):
    st.markdown("<div class='section-title'>🔬 Root-Cause Anomaly Inspector</div>", unsafe_allow_html=True)
    anomaly_indices = np.where(preds == -1)[0]
    score_min = scores.min()
    score_max = scores.max()
    if len(anomaly_indices) == 0:
        st.info("No anomalies to inspect.")
        return
    frame_labels = {
        int(i): (f"{frame_label_prefix} {i:04d} — " + risk_level(scores[i], score_min, score_max) + f" risk (score: {scores[i]:.3f})")
        for i in anomaly_indices
    }
    sorted_indices = sorted(anomaly_indices, key=lambda i: scores[i])
    selected_frame = st.selectbox("Select anomaly frame to inspect:", options=sorted_indices, format_func=lambda i: frame_labels[i])
    row = test_df.iloc[selected_frame]
    s = scores[selected_frame]
    rlevel = risk_level(s, score_min, score_max)
    causes = explain_anomaly(row, normal_stats, features, root_cause_lib)
    
    badge_color = {"CRITICAL": "#ef4444", "HIGH": "#f59e0b", "MEDIUM": "#eab308", "LOW": "#3b82f6"}[rlevel]
    frame_css = "red-alert" if rlevel == "HIGH" or rlevel == "CRITICAL" else ""
    icon = "🚨" if rlevel == "CRITICAL" else ("⚠️" if rlevel == "HIGH" else ("🟡" if rlevel == "MEDIUM" else "🔵"))
    
    st.markdown(
        f"<div class='{frame_css}' style='background:#080f2e;border:2px solid {badge_color};border-radius:12px;padding:16px 20px;margin-bottom:12px;'>"
        f"<span style='color:{badge_color};font-weight:700;font-size:1rem;'>{icon} {frame_label_prefix} {selected_frame:04d} — {rlevel} RISK</span>"
        f"<span style='color:#57606a;font-size:0.88rem;margin-left:16px;'>Decision score: {s:.4f}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    
    detail_col, radar_col = st.columns([1, 1])
    
    with detail_col:
        if causes:
            st.markdown("**Identified Root Causes:**")
            for c in causes:
                css_cls = "cause-high" if c["severity"] == "high" else "cause-med" if c["severity"] == "medium" else "cause-low"
                confidence_pct = c.get("confidence", 0.7) * 100
                subsystem = c.get("subsystem", "🔧 General")
                st.markdown(
                    f"<div class='cause-card {css_cls}'>"
                    f"<div class='cause-title'>{c['title']}</div>"
                    f"<b>{c['feature']}</b>: {c['value']:.4g} (z-score: {c['z_score']:.2f})<br>"
                    f"{c['desc']}<br>"
                    f"<span style='color:#57606a;font-size:0.8rem;'>"
                    f"📍 {subsystem} | 🎯 Confidence: {confidence_pct:.1f}%"
                    f"</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            
            # Mission Impact and Recommended Actions
            recommendation = generate_mission_recommendation(causes)
            
            st.markdown("**🎯 Mission Impact:**")
            st.markdown(
                f"<div class='impact-card'>"
                f"<b>📌 Affected Subsystem:</b> {recommendation['subsystem']}<br>"
                f"<b>Impact:</b> {recommendation['impact']}"
                f"</div>",
                unsafe_allow_html=True
            )
            
            st.markdown("**✅ Recommended Actions:**")
            for action in recommendation["actions"]:
                st.markdown(
                    f"<div class='action-card'>"
                    f"• {action}"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown("<div class='cause-card cause-low'><div class='cause-title'>ℹ️ Subtle Multi-Feature Deviation</div>No single feature exceeds the 2σ threshold. The anomaly is driven by a combination of minor deviations across multiple channels.</div>", unsafe_allow_html=True)
    
    with radar_col:
        st.markdown("**Deviation Radar:**")
        st.plotly_chart(make_radar_chart(row, normal_stats, features), use_container_width=True, key=f"radar_{selected_frame}")


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

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
        st.caption(f"**Kernel:** {kernel_val} | **Nu:** {nu_val} | **Gamma:** {gamma_val}")
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
        
        # Add scenario selection
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🎯 Simulation Scenario")
        selected_scenario = st.selectbox(
            "Select Scenario",
            options=list(SIMULATION_SCENARIOS.keys()),
            index=0,
            help="Apply a specific fault scenario to the simulated telemetry"
        )
        
        if selected_scenario != "Normal Operation":
            st.markdown(
                f"<div style='background:#080f2e;border:1px solid #1e3a8a;border-radius:6px;padding:8px 12px;margin-bottom:8px;font-size:0.8rem;color:#b8cef7;'>"
                f"<b>📖 Description:</b> {SIMULATION_SCENARIOS[selected_scenario]['description']}"
                f"</div>",
                unsafe_allow_html=True
            )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Run MIRA Analysis", use_container_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div style='color:#57606a;font-size:0.78rem;line-height:1.6'>MIRA uses <b>OneClassSVM</b> trained on nominal telemetry.<br>Anomalies are frames where the satellite's behaviour deviates beyond the learned decision boundary.<br><br><b>IBM AI Builders Challenge</b><br>Advance Space Exploration with AI<br>Built with <b>IBM Bob</b></div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style="text-align:center;padding:24px 0 8px 0;">
        <div style="font-size:2.8rem;line-height:1;margin-bottom:8px;">🛰️</div>
        <div style="font-size:2.5rem;font-weight:700;color:#7eb8f7;letter-spacing:2px;line-height:1.2;margin:0;">MIRA</div>
        <div style="color:#57606a;font-size:1rem;letter-spacing:4px;margin-top:8px;line-height:1.5;">MISSION INTELLIGENCE &amp; RISK ANALYZER</div>
        <div style="width:80px;height:2px;background:#3b5de7;margin:14px auto 0 auto;border-radius:2px;"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# IDLE SCREEN
# ═════════════════════════════════════════════════════════════════════════════

if not run_btn:
    if IS_REAL:
        sub = "Select <b style='color:#86efac;'>🛰️ Real OPS-SAT Data</b> mode and click"
    else:
        sub = "Configure the simulation parameters and click"
    st.markdown(f"<div style='text-align:center;padding:60px 20px;color:#57606a;'><div style='font-size:4rem;margin-bottom:16px;'>🌌</div><h3 style='color:#3b5de7;'>Awaiting Mission Start</h3><p>{sub} <b style='color:#7eb8f7;'>🚀 Run MIRA Analysis</b> to begin satellite monitoring.</p></div>", unsafe_allow_html=True)
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# MODE A — REAL OPS-SAT DATA
# ═════════════════════════════════════════════════════════════════════════════

if IS_REAL:
    st.markdown("<div style='text-align:center;margin-bottom:8px;'><span class='mode-badge-real'>🛰️ REAL OPS-SAT DATA — ESA OPS-SAT-1 Mission Telemetry</span></div>", unsafe_allow_html=True)
    
    with st.spinner("📂 Loading OPS-SAT dataset…"):
        raw_df, feature_cols, load_err = load_opssat_dataset("dataset.csv")
    
    if load_err:
        st.error(f"**Dataset Error:** {load_err}")
        st.markdown("<div class='cause-card cause-high'><div class='cause-title'>📁 dataset.csv not found</div>Place <code>dataset.csv</code> in the same directory as <code>mira_app.py</code> before running the app.</div>", unsafe_allow_html=True)
        st.stop()
    
    n_train_nominal = int(((raw_df["train"] == 1) & (raw_df["anomaly"] == 0)).sum()) if "train" in raw_df.columns and "anomaly" in raw_df.columns else len(raw_df)
    channels = raw_df["channel"].unique().tolist() if "channel" in raw_df.columns else ["—"]
    ch_str = ", ".join(str(c) for c in channels[:6]) + ("…" if len(channels) > 6 else "")
    
    st.markdown(f"<div class='dataset-info'><b>Dataset:</b> ESA OPS-SAT-1 Telemetry &nbsp;|&nbsp; <b>Total segments:</b> {len(raw_df):,} &nbsp;|&nbsp; <b>Nominal training segments:</b> {n_train_nominal:,} &nbsp;|&nbsp; <b>Features used:</b> {len(feature_cols)} &nbsp;|&nbsp; <b>Channels:</b> {ch_str}</div>", unsafe_allow_html=True)
    
    if "train" in raw_df.columns and "anomaly" in raw_df.columns:
        train_df = raw_df[(raw_df["train"] == 1) & (raw_df["anomaly"] == 0)].copy()
        test_df = raw_df.copy()
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
    score_min = scores.min()
    score_max = scores.max()
    n_anomalies = int((preds == -1).sum())
    n_total = len(preds)
    
    risk_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for s_val, p in zip(scores, preds):
        if p == -1:
            risk_counts[risk_level(s_val, score_min, score_max)] += 1
    
    has_labels = "anomaly" in test_df.columns
    true_labels = None
    if has_labels:
        true_labels = test_df["anomaly"].values
    
    # Render Mission Status
    render_mission_status(risk_counts, n_anomalies)
    
    # Render Alert Banner
    render_alert_banner(n_anomalies, risk_counts)
    
    # Render KPI Row
    render_kpi_row(preds, scores, risk_counts)
    
    # Render Mission Intelligence Summary
    st.markdown("<hr>", unsafe_allow_html=True)
    summary_data = generate_mission_summary(preds, scores, risk_counts, n_anomalies, n_total)
    render_mission_summary(summary_data)
    
    # Render AI Mission Report
    st.markdown("<hr>", unsafe_allow_html=True)
    # Get top causes from worst anomalies
    anomaly_indices = np.where(preds == -1)[0]
    all_causes = []
    if len(anomaly_indices) > 0:
        sorted_idx = sorted(anomaly_indices, key=lambda i: scores[i])[:10]
        for idx in sorted_idx:
            row = test_df.iloc[idx]
            causes = explain_anomaly(row, normal_stats, feature_cols, OPSSAT_ROOT_CAUSE_LIBRARY)
            all_causes.extend(causes)
    render_ai_report(summary_data, all_causes[:3], risk_counts)
    
    # Ground Truth Validation
    if has_labels:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>✅ Ground-Truth Validation</div>", unsafe_allow_html=True)
        g1, g2, g3, g4 = st.columns(4)
        n_true_anom = int((true_labels == 1).sum())
        tp = int(((preds == -1) & (true_labels == 1)).sum())
        fp = int(((preds == -1) & (true_labels == 0)).sum())
        fn = int(((preds == 1) & (true_labels == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        g1.metric("📋 True Anomalies", n_true_anom)
        g2.metric("🎯 True Positives", tp)
        g3.metric("🎯 Precision", f"{precision:.2%}")
        g4.metric("🔁 Recall", f"{recall:.2%}")
        
        # Confusion Matrix
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📊 Confusion Matrix</div>", unsafe_allow_html=True)
        render_confusion_matrix(preds, true_labels)
    
    # Model Card
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📋 Model Card</div>", unsafe_allow_html=True)
    render_model_card("OneClassSVM", nu_val, kernel_val, gamma_val, len(feature_cols), len(train_df), "🛰️ Real OPS-SAT Data")
    
    # Charts
    st.markdown("<hr>", unsafe_allow_html=True)
    x_labels = test_df["segment"].tolist() if "segment" in test_df.columns else None
    render_charts(test_df, preds, scores, scaler, feature_cols, x_labels)
    
    # Inspector
    st.markdown("<hr>", unsafe_allow_html=True)
    render_inspector(test_df, preds, scores, normal_stats, feature_cols, OPSSAT_ROOT_CAUSE_LIBRARY, "Segment")
    
    # Limitations
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>⚠️ Limitations</div>", unsafe_allow_html=True)
    render_limitations()


# ═════════════════════════════════════════════════════════════════════════════
# MODE B — MISSION SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

else:
    st.markdown("<div style='text-align:center;margin-bottom:8px;'><span class='mode-badge-sim'>🔬 MISSION SIMULATION — All values are SYNTHETIC</span></div>", unsafe_allow_html=True)
    
    with st.spinner("🛰️ Generating synthetic telemetry stream…"):
        normal_df = generate_normal_data(n_frames)
        
        # Apply scenario if selected
        if selected_scenario != "Normal Operation":
            normal_df = apply_scenario(normal_df, selected_scenario)
        
        test_df = inject_anomalies(normal_df.copy(), anomaly_rate=anomaly_pct / 100)
        normal_stats = normal_df.describe().loc[["mean", "std"]]
    
    with st.spinner("🤖 Training OneClassSVM on synthetic nominal data…"):
        model, scaler = train_ocsvm(normal_df, SIM_FEATURES, nu=nu_val, kernel=kernel_val, gamma=gamma_val)
    
    with st.spinner("🔍 Scanning synthetic telemetry frames…"):
        preds, scores = run_predict(model, scaler, test_df, SIM_FEATURES)
    
    score_min = scores.min()
    score_max = scores.max()
    n_anomalies = int((preds == -1).sum())
    n_total = len(preds)
    
    risk_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for s_val, p in zip(scores, preds):
        if p == -1:
            risk_counts[risk_level(s_val, score_min, score_max)] += 1
    
    # Render Mission Status
    render_mission_status(risk_counts, n_anomalies)
    
    # Render Alert Banner
    render_alert_banner(n_anomalies, risk_counts)
    
    # Render KPI Row
    render_kpi_row(preds, scores, risk_counts)
    
    # Render Mission Intelligence Summary
    st.markdown("<hr>", unsafe_allow_html=True)
    summary_data = generate_mission_summary(preds, scores, risk_counts, n_anomalies, n_total)
    render_mission_summary(summary_data)
    
    # Render AI Mission Report
    st.markdown("<hr>", unsafe_allow_html=True)
    anomaly_indices = np.where(preds == -1)[0]
    all_causes = []
    if len(anomaly_indices) > 0:
        sorted_idx = sorted(anomaly_indices, key=lambda i: scores[i])[:10]
        for idx in sorted_idx:
            row = test_df.iloc[idx]
            causes = explain_anomaly(row, normal_stats, SIM_FEATURES, SIM_ROOT_CAUSE_LIBRARY)
            all_causes.extend(causes)
    render_ai_report(summary_data, all_causes[:3], risk_counts)
    
    # Scenario info
    if selected_scenario != "Normal Operation":
        st.markdown(
            f"<div class='dataset-info'>"
            f"<b>🎯 Active Scenario:</b> {selected_scenario}<br>"
            f"<b>📖 Description:</b> {SIMULATION_SCENARIOS[selected_scenario]['description']}"
            f"</div>",
            unsafe_allow_html=True
        )
    
    # Model Card
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📋 Model Card</div>", unsafe_allow_html=True)
    render_model_card("OneClassSVM", nu_val, kernel_val, gamma_val, len(SIM_FEATURES), len(normal_df), "🔬 Mission Simulation")
    
    # Charts
    st.markdown("<hr>", unsafe_allow_html=True)
    render_charts(test_df, preds, scores, scaler, SIM_FEATURES)
    
    # Inspector
    st.markdown("<hr>", unsafe_allow_html=True)
    render_inspector(test_df, preds, scores, normal_stats, SIM_FEATURES, SIM_ROOT_CAUSE_LIBRARY, "Frame")
    
    # Limitations
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>⚠️ Limitations</div>", unsafe_allow_html=True)
    render_limitations()


# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<div style='text-align:center;color:#57606a;font-size:0.78rem;padding:20px 0 8px 0;border-top:1px solid #1a2a6c;margin-top:16px;'>"
    "MIRA — Mission Intelligence &amp; Risk Analyzer &nbsp;|&nbsp; OneClassSVM &nbsp;|&nbsp; "
    "OPS-SAT dataset © ESA &nbsp;|&nbsp; Application built with <b style='color:#3b5de7;'>IBM Bob</b>"
    "</div>",
    unsafe_allow_html=True,
)
