# 🛰️ MIRA — Mission Intelligence & Risk Analyzer

<div align="center">

![Version](https://img.shields.io/badge/Version-1.0.0-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.9+-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![Built with](https://img.shields.io/badge/Built%20with-IBM%20Bob-blueviolet)

**MIRA** is an AI-powered satellite anomaly detection and investigation system that transforms raw telemetry anomalies into actionable mission insights.

</div>

---

## 🚀 Features

- ✅ **Anomaly Detection**: Uses One-Class SVM to identify abnormal satellite telemetry patterns
- ✅ **Investigation Report**: Automatically explains the likely root cause of each anomaly
- ✅ **Root Cause Confidence**: Provides confidence scores for each identified root cause
- ✅ **Subsystem Classification**: Automatically classifies affected subsystems (Power, Thermal, Comms, etc.)
- ✅ **Recommendations**: Provides actionable steps for mission operators
- ✅ **Mission Impact**: Assesses the impact of anomalies on mission objectives
- ✅ **Channel Selection**: Analyze different satellite channels independently
- ✅ **Interactive UI**: Built with Streamlit for ease of use
- ✅ **Multiple Modes**: Supports both Real OPS-SAT data and Synthetic Simulation
- ✅ **Confusion Matrix**: Model performance evaluation with ground truth

---

## 🧠 How It Works

1. **Data Input**: Reads satellite telemetry from `dataset.csv` or generates synthetic data
2. **Preprocessing**: Standardizes the telemetry features using `StandardScaler`
3. **Model**: Uses One-Class SVM (nu=0.22, RBF kernel) to detect anomalies
4. **Risk Classification**: Categorizes anomalies into CRITICAL/HIGH/MEDIUM/LOW risk levels
5. **Root Cause Analysis**: Generates rule-based explanations for each detected anomaly
6. **Mission Recommendations**: Suggests next steps for mission operators

---

## 📊 Model Performance

> Evaluated on the ESA OPS-SAT-1 telemetry dataset.

| Metric | Score |
|--------|-------|
| **Accuracy** | 76.94% |
| **F1 Score** | 46.96% |
| **Model** | One-Class SVM with RBF kernel |
| **Nu** | 0.22 |
| **Training Data** | Normal telemetry only (train=1, anomaly=0) |

---

## 🖥️ UI Features

### 📊 Overview Tab
- Mission Status Banner (CRITICAL/HIGH/MEDIUM/LOW/NOMINAL)
- KPI Metrics (Segments, Normal, Anomalies, Risk Levels)
- Mission Intelligence Summary with Recommendations

### 🔬 Anomaly Inspector Tab
- Interactive selection of anomalies
- Root Cause Analysis (Top 3 features)
- Confidence Scores for each cause
- Subsystem Classification
- Mission Impact & Actions
- Deviation Radar Chart
- Feature Comparison Table

### 📈 Charts Tab
- Anomaly Score Timeline
- PCA Feature Space Visualization
- Full Telemetry Heatmap

### 📋 Model Info Tab
- Model Card with parameters
- Confusion Matrix
- Precision/Recall/F1 Metrics

### ⚠️ Limitations Tab
- Comprehensive list of model limitations

---

## 🛠️ Technologies Used

- **Python** 3.9+
- **Streamlit** 1.28+
- **Pandas** 2.0+
- **NumPy** 1.24+
- **Scikit-learn** 1.3+
- **Plotly** 5.18+

---

## 🤖 How IBM Bob Was Used

IBM Bob was used as the **primary AI development assistant** throughout the project. Specifically, Bob was utilized to:

- 🧠 **Generate and refine** Python code for data preprocessing, anomaly detection (One-Class SVM), and the Streamlit interface
- 🐛 **Debug and fix errors** during development (e.g., Streamlit installation issues, model training pipelines)
- 💡 **Suggest improvements** to the anomaly detection model and the root-cause explanation logic
- 📝 **Draft documentation** for the project, including this README
- 🏗️ **Design architecture** for the application with Mermaid diagrams
- 🎨 **Create UI/UX** with dark space theme and interactive visualizations

---

## 📂 Project Structure

```
MIRA-Satellite-Anomaly-Investigator/
├── app.py                 # Main Streamlit application
├── dataset.csv            # Satellite telemetry dataset
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 📦 Requirements

```
streamlit>=1.28
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
plotly>=5.18
```

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Satellite & Ground Segment"
        A[OPS-SAT-1<br/>Telemetry Sensors] -->|Raw Telemetry| B[Ground Station<br/>Data Reception]
    end

    subgraph "Data Processing Layer"
        B --> C[Data Loading<br/>dataset.csv]
        C --> D[Data Cleaning<br/>& Validation]
        D --> E[Feature Extraction<br/>Statistical Features]
        E --> F[StandardScaler<br/>Normalization]
        F --> G[Dataset Split<br/>Train/Test]
    end

    subgraph "ML Model Layer"
        G --> H[OneClassSVM<br/>Training]
        H --> I[Model Inference<br/>Prediction]
        I --> J[Anomaly Scores<br/>Decision Function]
    end

    subgraph "Analysis & Interpretation"
        J --> K[Risk Classification<br/>CRITICAL/HIGH/MEDIUM/LOW]
        K --> L[Root Cause Analysis<br/>Top 3 Features]
        L --> M[Confidence Scoring<br/>Z-Score Based]
        M --> N[Subsystem Classification<br/>Power/Thermal/Comms]
        N --> O[Mission Impact<br/>& Recommendations]
    end

    subgraph "Visualization & UI"
        O --> P[Streamlit Dashboard]
        P --> Q[KPI Metrics<br/>Anomaly Counts]
        P --> R[Interactive Charts<br/>Timeline/PCA/Radar]
        P --> S[Anomaly Inspector<br/>Detailed Reports]
        P --> T[Confusion Matrix<br/>& Model Card]
    end

    subgraph "Output & Reporting"
        S --> U[AI Mission Report]
        T --> V[Performance Metrics<br/>Precision/Recall/F1]
        U --> W[Actionable Insights<br/>Operator Guidance]
        V --> W
    end

    style A fill:#0a1f0a,stroke:#22c55e,color:#fff,stroke-width:2px
    style B fill:#0b1640,stroke:#3b5de7,color:#fff,stroke-width:2px
    style C fill:#0b1640,stroke:#3b5de7,color:#fff,stroke-width:2px
    style D fill:#0b1640,stroke:#3b5de7,color:#fff,stroke-width:2px
    style E fill:#0b1640,stroke:#3b5de7,color:#fff,stroke-width:2px
    style F fill:#0b1640,stroke:#3b5de7,color:#fff,stroke-width:2px
    style G fill:#0b1640,stroke:#3b5de7,color:#fff,stroke-width:2px
    style H fill:#1a1202,stroke:#f59e0b,color:#fff,stroke-width:2px
    style I fill:#1a1202,stroke:#f59e0b,color:#fff,stroke-width:2px
    style J fill:#1a1202,stroke:#f59e0b,color:#fff,stroke-width:2px
    style K fill:#080f2e,stroke:#7eb8f7,color:#fff,stroke-width:2px
    style L fill:#080f2e,stroke:#7eb8f7,color:#fff,stroke-width:2px
    style M fill:#080f2e,stroke:#7eb8f7,color:#fff,stroke-width:2px
    style N fill:#080f2e,stroke:#7eb8f7,color:#fff,stroke-width:2px
    style O fill:#080f2e,stroke:#7eb8f7,color:#fff,stroke-width:2px
    style P fill:#1a0808,stroke:#ef4444,color:#fff,stroke-width:2px
    style Q fill:#1a0808,stroke:#ef4444,color:#fff,stroke-width:2px
    style R fill:#1a0808,stroke:#ef4444,color:#fff,stroke-width:2px
    style S fill:#1a0808,stroke:#ef4444,color:#fff,stroke-width:2px
    style T fill:#1a0808,stroke:#ef4444,color:#fff,stroke-width:2px
    style U fill:#060d20,stroke:#7eb8f7,color:#fff,stroke-width:2px
    style V fill:#060d20,stroke:#7eb8f7,color:#fff,stroke-width:2px
    style W fill:#060d20,stroke:#7eb8f7,color:#fff,stroke-width:2px
```

---

## 🚀 How to Run

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/ShadenBawazir/MIRA-Satellite-Anomaly-Investigator.git
cd MIRA-Satellite-Anomaly-Investigator
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

---

## ⚠️ Limitations

- One-Class SVM only learns from normal data, cannot distinguish anomaly types
- Results depend on training dataset completeness and quality
- Normal seasonal variations may be flagged as anomalies
- Risk thresholds are heuristic and may need per-mission tuning

---
## 🎥 Demo

Demo video: [Watch MIRA Demo](https://youtu.be/SLFA2Bq9u98)

## 🚀 Live App

[Streamlit App](https://mira-satellite-anomaly-investigator-fmq5jcvybztm9vfxpdmrga.streamlit.app/)

---
## 📧 Contact

**Shaden Bawazir**

- GitHub: [ShadenBawazir](https://github.com/ShadenBawazir)
- Project: [MIRA-Satellite-Anomaly-Investigator](https://github.com/ShadenBawazir/MIRA-Satellite-Anomaly-Investigator)

---

© 2025 MIRA — Mission Intelligence & Risk Analyzer
