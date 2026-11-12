# End-to-End Indian Urban Traffic Flow Prediction & Dynamic Route Optimization

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-1f77b4.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **MSc Informatics Capstone Project**  
> *An end-to-end spatiotemporal forecasting and decision-support system benchmarking regularized linear models, gradient-boosted trees, parametric time-series, sequential LSTMs, and Spatio-Temporal Graph Convolutional Networks (STGCN) on high-frequency Indian urban intersection telemetry.*

---

## 1. Executive Summary & Abstract

Urban traffic flow forecasting is a foundational pillar of Intelligent Transportation Systems (ITS), adaptive signal control, and dynamic route guidance. While traffic forecasting is well-studied on homogeneous, lane-disciplined Western freeways (e.g., California PeMS), Indian metropolitan corridors exhibit high vehicular heterogeneity, non-lane-based mixed flow, and sharp directional spillovers across signalized approaches.

This project builds a complete, end-to-end forecasting and routing system using the **Delhi Traffic Density Dataset (IIT Delhi)**—comprising 40 operational days of high-frequency (1Hz) camera telemetry across three intersecting arterial approaches. We formulate a multi-step multivariate forecasting pipeline predicting queue and stop density across **15, 30, 45, and 60-minute horizons ($t+1$ to $t+4$)** and integrate the predictions into a live **Highway Capacity Manual (HCM) Level-of-Service (LOS) classifier** and **Dijkstra shortest-path route optimizer**.

### Key Engineering & Research Highlights
* **Resolving the "Linear Trap" (Target Leakage Diagnosis):** Demonstrated how unregularized OLS collapses ($\text{MAE} = 0.0000$) due to severe collinearity between queue density and stopped-vehicle density during gridlock, resolving it via $L_2$ Tikhonov regularization (Ridge Regression, `1.65%` WMAPE).
* **Quantifying Parametric Multi-Step Drift:** Proved empirically that while traffic density is stationary (ADF $p = 2.41 \times 10^{-9}$), univariate $\text{SARIMA}(2,0,1)(1,0,1)_{60}$ degrades to `27.70%` WMAPE over long test horizons due to recursive mean-reversion.
* **Sequential Gating vs. Spatiotemporal Graph Topology:** Implemented a 2-layer PyTorch LSTM (`9.20%` WMAPE) and constructed a data-driven 3-node cross-correlation adjacency matrix ($w_{12}=0.6454, w_{13}=0.4405, w_{23}=0.4795$) for a custom **STGCN**, achieving `15.54%` joint network-wide WMAPE across all three approaches simultaneously.
* **Actionable Decision Support:** Packaged the multi-horizon inference engine with a NetworkX road graph utilizing dynamic Bureau of Public Roads (BPR) travel-time cost functions, exposed through an interactive **Streamlit** dashboard.

---

## 2. Empirical Benchmark Results

### 2.1 Master Cross-Stage Evaluation (15-Minute Horizon)
Evaluated strictly on the chronological 30% out-of-sample test holdout ($N = 702$ intervals):

| Stage | Model Paradigm | Architecture / Specification | Target Scope & Inputs | MAE | RMSE | WMAPE (%) | sMAPE (%) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Stage 1** | Regularized Linear | Ridge Regression ($L_2, \alpha=1.0$) | Approach 1 (25 tabular features) | **0.0085** | **0.0127** | **1.65%** | **2.01%** |
| **Stage 1** | Non-Linear Ensemble | Random Forest ($n=100$) | Approach 1 (25 tabular features) | 0.0163 | 0.0250 | 3.16% | 3.57% |
| **Stage 1** | Non-Linear Ensemble | XGBoost Regressor (GBDT) | Approach 1 (25 tabular features) | 0.0145 | 0.0231 | 2.81% | 3.14% |
| **Stage 2** | Parametric Time-Series | $\text{SARIMA}(2,0,1)(1,0,1)_{60}$ | Approach 1 (Univariate sequence) | 0.1429 | 0.1669 | 27.70% | 30.56% |
| **Stage 3** | Deep Sequential | 2-Layer LSTM ($H=64$) | Approach 1 (3D Window $T=4, F=25$) | 0.0474 | 0.0728 | 9.20% | 11.79% |
| **Stage 4** | Spatiotemporal Graph | STGCN (GCN + GRU) | Approach 1 (Node 0 Benchmark) | 0.0921 | 0.1191 | 15.50% | 20.85% |
| **Stage 4** | **Spatiotemporal Graph** | **STGCN (Joint 3-Node Graph)** | **All 3 Approaches Simultaneously** | **0.0851** | **0.1157** | **15.54%** | **22.68%** |

### 2.2 Multi-Horizon Forecast Degradation ($t+15\text{m}$ to $t+60\text{m}$)
Evaluating predictive stability across tactical and strategic ITS horizons:

| Forecast Horizon | Operational Use Case | XGBoost WMAPE | LSTM WMAPE | STGCN WMAPE | XGBoost RMSE | LSTM RMSE | STGCN RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$t+15\text{ min}$ ($H=1$)** | Tactical Signal Timing | **8.15%** | 11.79% | 15.50% | **0.0560** | 0.0851 | 0.1191 |
| **$t+30\text{ min}$ ($H=2$)** | Adaptive Rerouting | **9.75%** | 13.43% | 16.82% | **0.0689** | 0.0965 | 0.1265 |
| **$t+45\text{ min}$ ($H=3$)** | Corridor Coordination | **11.01%** | 15.31% | 18.15% | **0.0750** | 0.1049 | 0.1340 |
| **$t+60\text{ min}$ ($H=4$)** | Strategic Congestion Alerts | **12.03%** | 16.80% | 19.40% | **0.0839** | 0.1123 | 0.1412 |

---

## 3. System Architecture & Repository Structure

```mermaid
flowchart LR
    A["Raw 1Hz Camera Logs<br/>(6 Cameras, 3 Approaches)"] --> B["15-Min Resampling &<br/>Feature Engineering"]
    B --> C["4-Stage Model<br/>Training & Evaluation"]
    C --> D["Serialized Service Bundle<br/>(traffic_service_bundle.joblib)"]
    D --> E["TrafficIntelligenceService<br/>(LOS + Dijkstra Routing)"]
    E --> F["Interactive Streamlit UI<br/>(app.py)"]

traffic-flow-project/
├── app.py                            # Interactive Streamlit dashboard (Forecasting + Routing UI)
├── configs/
│   └── config.yaml                   # Centralized hyperparameters, paths, and split ratios
├── data/
│   ├── raw/                          # Raw 1-second IIT Delhi camera telemetry logs
│   ├── interim/                      # Resampled 15-minute multi-approach density parquet files
│   └── processed/                    # Feature matrix (delhi_features.parquet) & graph tensors (.npz)
├── checkpoints/
│   └── traffic_service_bundle.joblib # Production multi-horizon model & threshold artifact
├── reports/
│   └── figures/                      # Publication-grade metric bar charts & trajectory plots
├── src/
│   ├── data_loader.py                # 1Hz-to-15min aggregation & chronological splitting
│   ├── features.py                   # Cyclical Fourier encodings, autoregressive lags & ADF tests
│   ├── evaluate_multihorizon.py      # Multi-horizon (15m-60m) evaluation & degradation curves
│   ├── visualize_results.py          # Publication figure generator (300 DPI)
│   ├── export_pipeline.py            # Production model training and artifact bundler
│   ├── traffic_service.py            # Inference engine, LOS classifier & NetworkX Dijkstra router
│   └── models/
│       ├── baseline.py               # Stage 1: OLS, Ridge, Random Forest, XGBoost
│       ├── train_sarima.py           # Stage 2: Seasonal ARIMA state-space model
│       ├── train_lstm.py             # Stage 3: PyTorch 2-Layer Sequential LSTM
│       ├── build_graph.py            # 3-Node Pearson correlation adjacency matrix & 4D tensors
│       └── train_stgcn.py            # Stage 4: Spatio-Temporal Graph Convolutional Network
├── tests/
│   └── test_pipeline.py              # Unit tests for zero-safe metrics & leakage prevention
├── requirements.txt                  # Pinned environment dependencies
└── README.md                         # Project documentation


## 4. Methodology & Feature Engineering

Implemented in `src/features.py`, the preprocessing pipeline transforms raw, non-stationary camera observations into a stationary, zero-lookahead feature space.

### 4.1 Temporal Resampling & Stationarity Verification
1. **15-Minute Uniform Aggregation:** Raw 1-second (1Hz) camera observations from the 15-hour daily operating window (`07:00` to `21:45`, $60$ intervals/day) are aggregated into uniform 15-minute intervals ($N = 2,400$ total intervals; $2,340$ post-lag construction).
2. **Stationarity Verification:** Augmented Dickey-Fuller (ADF) testing on the primary target series confirmed strict stationarity ($\text{ADF Statistic} = -6.78, p = 2.41 \times 10^{-9}$), permitting direct autoregressive modeling without first-order differencing ($d = 0$).
3. **Leakage-Free Chronological Splitting:** A strict 70/30 temporal partition is enforced (Days 1–28 for training, Days 29–40 for out-of-sample testing) with zero random shuffling.

### 4.2 Continuous Cyclical Time Encodings
Discrete calendar timestamps introduce artificial discontinuities at day boundaries. Time is projected onto continuous 2D unit circles using trigonometric Fourier transformations:

$$x_{\sin} = \sin\left(\frac{2\pi \cdot t}{T}\right), \quad x_{\cos} = \cos\left(\frac{2\pi \cdot t}{T}\right)$$

* **Diurnal Operating Cycle:** $T = 60$ intervals (15-hour operating day)
* **Weekly Cycle:** $T = 7$ days

### 4.3 Multi-Horizon Autoregressive & Seasonal Lags
Captures short-term queue momentum alongside diurnal recurrence across all three intersection approaches:
* **Short-Term Inertia Lags:** $y_{t-1}$ (15m), $y_{t-2}$ (30m), $y_{t-3}$ (45m), and $y_{t-4}$ (60m).
* **Exact Diurnal Seasonal Lag:** $y_{t-60}$ (traffic density at the exact same 15-minute interval on the previous operating day).
* **Zero-Lookahead Rolling Statistics:** Rolling mean ($\mu$) and standard deviation ($\sigma$) computed strictly over shifted historical windows ($y_{t-1}, y_{t-2}, \dots$) to eliminate target leakage.

### 4.4 Spatiotemporal Graph Formulation (`src/models/train_stgcn.py`)
For Stage 4, the 3-approach intersection is modeled as a weighted spatial graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, A)$ with $N = 3$ nodes. The data-driven cross-correlation adjacency matrix ($w_{12} = 0.6454, w_{13} = 0.4405, w_{23} = 0.4795$) is normalized via the symmetric graph Laplacian $\hat{A} = \tilde{D}^{-1/2} A \tilde{D}^{-1/2}$ to propagate spatial congestion spillovers across approaches prior to temporal GRU gating:

$$Z^{(t)} = \text{ReLU}\left(\hat{A} X^{(t)} W_s\right)$$

## 5. Mathematical Evaluation Protocols

Evaluating urban traffic density requires metrics that remain numerically stable during low-volume or free-flow intervals. Standard MAPE explodes when actual values approach zero. Our evaluation suite in `src/evaluate.py` and `src/evaluate_multihorizon.py` computes:

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

3. **Weighted Mean Absolute Percentage Error (WMAPE):**
   $$\text{WMAPE} = \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{\sum_{i=1}^{n} |y_i| + \epsilon} \times 100\%$$

4. **Symmetric Mean Absolute Percentage Error (sMAPE):**
   $$\text{sMAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|) / 2 + \epsilon}$$

## 6. End-to-End Application: Congestion & Route Optimization

Implemented in `src/traffic_service.py` and exposed interactively in `app.py`:

* **Highway Capacity Manual (HCM) Level of Service (LOS) Mapping:** Converts forecasted proportional queue density $[0, 1]$ into actionable operational states:
  * **Free Flow (LOS A–B):** $\text{Density} < 0.20$
  * **Moderate Traffic (LOS C–D):** $0.20 \le \text{Density} < 0.45$
  * **Heavy Congestion (LOS E):** $0.45 \le \text{Density} < 0.65$
  * **Severe Gridlock (LOS F):** $\text{Density} \ge 0.65$
* **Dynamic Dijkstra Route Optimization:** Dynamically updates edge traversal speeds across the NetworkX road graph using a Bureau of Public Roads (BPR) degradation formulation:
  $$v_{\text{eff}} = \max\left(10.0, \, v_{\text{free}} \cdot (1 - 0.75 \cdot \hat{y}_{t+h})\right)$$
  The router computes the minimum travel-time trajectory and compares it against alternate arterial bypasses.

## 7. Installation & Quickstart

### Prerequisites
* Python 3.10+
* CUDA-compatible GPU (optional, used for PyTorch LSTM & STGCN training)

### 1. Environment Setup
```bash
# Clone the repository
git clone [https://github.com/your-username/indian-traffic-flow-prediction.git](https://github.com/your-username/indian-traffic-flow-prediction.git)
cd indian-traffic-flow-prediction

# Create and activate virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt


## Run 4-Stage Model Benchmarks
python src/models/baseline.py
python src/models/train_sarima.py
python src/models/train_lstm.py
python src/models/build_graph.py
python src/models/train_stgcn.py

# Run Multi-Horizon (15m-60m) Evaluation & Generate Figures
python src/evaluate_multihorizon.py
python src/visualize_results.py

# 1. Export trained production multi-horizon bundle to checkpoints/
python src/export_pipeline.py

# 2. Verify CLI inference & Dijkstra routing engine
python src/traffic_service.py

# 3. Launch the interactive Streamlit Dashboard
python -m streamlit run app.py



### Run Automated Test Suite
pytest tests/ -v


@mastersthesis{kartik_traffic_forecasting_2026,
  title={Indian Urban Traffic Flow Prediction using Machine Learning and Time-Series Forecasting},
  author={Kartik},
  school={Department of Informatics},
  year={2026},
  type={MSc Informatics Thesis}
}