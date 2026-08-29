import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="MIRA", page_icon="🛰️", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    return df

df = load_data()

st.title("🛰️ MIRA — Mission Intelligence & Root-cause Analyzer")
st.markdown("**AI-powered satellite anomaly detection & investigation system**")

st.sidebar.header("🔧 Configuration")
channel = st.sidebar.selectbox("Select Channel", df["channel"].unique())
nu_value = st.sidebar.slider("Anomaly Sensitivity (nu)", 0.10, 0.35, 0.22, 0.01)

features = [col for col in df.columns if col not in ["segment", "anomaly", "train", "channel"]]
channel_data = df[df["channel"] == channel]
train_data = channel_data[channel_data["train"] == 1]
test_data = channel_data[channel_data["train"] == 0]

X_train = train_data[features].values
X_test = test_data[features].values
y_test = test_data["anomaly"].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = OneClassSVM(nu=nu_value, kernel='rbf', gamma='scale')
model.fit(X_train_scaled)

y_pred = np.where(model.predict(X_test_scaled) == -1, 1, 0)

total = len(y_test)
correct = (y_pred == y_test).sum()
accuracy = correct / total * 100
anomalies = y_pred.sum()

col1, col2, col3 = st.columns(3)
col1.metric("Test Samples", total)
col2.metric("Predicted Anomalies", anomalies)
col3.metric("Accuracy", f"{accuracy:.2f}%")

st.subheader("📊 Anomaly Detection Results")
results = test_data[["segment", "anomaly", "train"]].copy()
results["predicted_anomaly"] = y_pred
results["status"] = results.apply(lambda row: "✅ Correct" if row["anomaly"] == row["predicted_anomaly"] else "⚠️ Mismatch", axis=1)
st.dataframe(results)

st.subheader("🔍 MIRA Investigation Report")
detected = results[results["predicted_anomaly"] == 1]

if len(detected) > 0:
    # 🔥 FIX: Use a unique key for selectbox
    detected_segments = detected["segment"].tolist()
    selected_segment = st.selectbox("Select Detected Anomaly", detected_segments, key="unique_anomaly_selectbox")
    
    anomaly_row = df[df["segment"] == selected_segment].iloc[0]

    st.markdown(f"**Segment:** {selected_segment} | **Channel:** {anomaly_row['channel']}")
    st.markdown(f"**True Label:** {'Anomaly' if anomaly_row['anomaly'] == 1 else 'Normal'}")
    st.markdown(f"**MIRA Prediction:** {'ANOMALY' if anomaly_row['anomaly'] == 1 else 'Normal'}")

    st.markdown("### 📈 Feature Values")
    feature_df = pd.DataFrame({
        "Feature": features,
        "Value": [anomaly_row[f] for f in features]
    })
    st.dataframe(feature_df)

    # AI Root-cause Explanation (Rule-based)
    st.markdown("### 🧠 Root-cause Explanation (MIRA)")

    explanation_parts = []

    if anomaly_row["smooth10_n_peaks"] > 2:
        explanation_parts.append("The signal shows multiple peaks in the smoothed data, suggesting sudden changes or noise.")
    if anomaly_row["kurtosis"] < -1.0:
        explanation_parts.append("Low kurtosis indicates a flat distribution, which may indicate a loss of signal variation.")
    if anomaly_row["diff_peaks"] > 20:
        explanation_parts.append("High differential peak count suggests rapid oscillations or instability.")
    if anomaly_row["var"] > 0.05:
        explanation_parts.append("Elevated variance suggests increased signal fluctuations.")
    if anomaly_row["len"] < 100:
        explanation_parts.append("Short segment duration may indicate an incomplete or interrupted signal.")

    if explanation_parts:
        st.markdown("**Possible causes:**")
        for part in explanation_parts:
            st.markdown(f"- {part}")
    else:
        st.markdown("No specific root cause identified. Further investigation required.")

    # Recommendation
    st.markdown("### 🛠️ Recommended Actions")
    st.markdown("1. **Check hardware status** — verify if the signal anomaly is due to sensor malfunction.")
    st.markdown("2. **Review recent commands** — determine if any recent telecommands affected the signal.")
    st.markdown("3. **Increase monitoring frequency** — monitor this channel more closely for subsequent anomalies.")

    st.markdown("### ⚠️ Mission Impact Assessment")
    if anomaly_row["anomaly"] == 1:
        st.error("**HIGH RISK:** This anomaly may indicate a critical failure. Immediate action is required.")
    else:
        st.warning("**MODERATE RISK:** This anomaly should be investigated. Potential for mission impact.")

else:
    st.info("No anomalies detected for this channel.")
