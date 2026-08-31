graph TB
    subgraph "🛰️ Satellite & Ground Segment"
        A[OPS-SAT-1<br/>Telemetry Sensors] -->|Raw Telemetry| B[Ground Station<br/>Data Reception]
    end

    subgraph "📊 Data Processing Layer"
        B --> C[Data Loading<br/>dataset.csv]
        C --> D[Data Cleaning<br/>& Validation]
        D --> E[Feature Extraction<br/>Statistical Features]
        E --> F[StandardScaler<br/>Normalization]
        F --> G[Dataset Split<br/>Train/Test]
    end

    subgraph "🤖 ML Model Layer"
        G --> H[OneClassSVM<br/>Training]
        H --> I[Model Inference<br/>Prediction]
        I --> J[Anomaly Scores<br/>Decision Function]
    end

    subgraph "🎯 Analysis & Interpretation"
        J --> K[Risk Classification<br/>CRITICAL/HIGH/MEDIUM/LOW]
        K --> L[Root Cause Analysis<br/>Top 3 Features]
        L --> M[Confidence Scoring<br/>Z-Score Based]
        M --> N[Subsystem Classification<br/>Power/Thermal/Comms]
        N --> O[Mission Impact<br/>& Recommendations]
    end

    subgraph "📈 Visualization & UI"
        O --> P[Streamlit Dashboard]
        P --> Q[KPI Metrics<br/>Anomaly Counts]
        P --> R[Interactive Charts<br/>Timeline/PCA/Radar]
        P --> S[Anomaly Inspector<br/>Detailed Reports]
        P --> T[Confusion Matrix<br/>& Model Card]
    end

    subgraph "📦 Output & Reporting"
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
