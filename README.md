# 🛰️ MIRA — Mission Intelligence & Root-cause Analyzer

MIRA is an AI-powered satellite anomaly detection and investigation system that transforms raw telemetry anomalies into actionable mission insights.

## 🚀 Features
- **Anomaly Detection**: Uses One-Class SVM to identify abnormal satellite telemetry patterns.
- **Investigation Report**: Automatically explains the likely root cause of each anomaly.
- **Recommendations**: Provides actionable steps for mission operators.
- **Channel Selection**: Analyze different satellite channels independently.
- **Interactive UI**: Built with Streamlit for ease of use.

## 🧠 How It Works
1. **Data Input**: Reads satellite telemetry from `dataset.csv`.
2. **Preprocessing**: Standardizes the telemetry features.
3. **Model**: Uses One-Class SVM (nu=0.22) to detect anomalies.
4. **Explanation**: Generates a rule-based root-cause explanation for each detected anomaly.
5. **Recommendations**: Suggests next steps for mission operators.

## 📊 Model Performance
- **Accuracy**: 76.94%
- **F1 Score**: 46.96%
- **Model**: One-Class SVM with RBF kernel

## 🛠️ Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn

## 🤖 How IBM Bob Was Used
IBM Bob was used as the primary AI development assistant throughout the project. Specifically, Bob was utilized to:
- Generate and refine Python code for data preprocessing, anomaly detection (One-Class SVM), and the Streamlit interface.
- Debug and fix errors during development (e.g., Streamlit installation issues, model training pipelines).
- Suggest improvements to the anomaly detection model and the root-cause explanation logic.
- Draft documentation for the project, including this README.

## 📂 Project Structure
MIRA-Satellite-Anomaly-Investigator/
├── app.py # Main Streamlit application
├── dataset.csv # Satellite telemetry dataset
├── requirements.txt # Python dependencies
└── README.md # Project documentation

# MIRA-Satellite-Anomaly-Investigator

MIRA is a satellite anomaly detection system powered by OneClassSVM.

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

    %% Styling
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
## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/ShadenBawazir/MIRA-Satellite-Anomaly-Investigator.git
Install dependencies:


pip install -r requirements.txt
Run the app:


streamlit run app.py
📧 Contact
Shaden Bawazir
GitHub
