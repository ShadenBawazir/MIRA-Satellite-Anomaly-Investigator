# IBM Bob Usage

IBM Bob was used as the primary AI development partner during the development of MIRA.

## How Bob was used

1. Project Architecture
2. Machine Learning Pipeline
3. Streamlit Application
4. Debugging
5. Generative AI Integration
6. Iterative Development

## Important Design Decision

MIRA separates anomaly detection from generative AI interpretation.

The machine learning model is responsible for detecting anomalous telemetry.
The generative AI layer is responsible for translating structured analysis results into an operator-facing mission brief.

This separation prevents the language model from being responsible for the underlying anomaly detection decision.
