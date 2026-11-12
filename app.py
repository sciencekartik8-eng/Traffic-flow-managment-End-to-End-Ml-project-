import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add project root to sys.path to load local src modules
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.traffic_service import TrafficIntelligenceService

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Urban Traffic Flow Intelligence | Delhi ITS",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RESOURCE CACHING & DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_service():
    """Cache model bundle and inference service."""
    return TrafficIntelligenceService()

@st.cache_data
def load_sample_dataset():
    """Load processed Delhi features for test simulation."""
    data_path = ROOT_DIR / "data" / "processed" / "delhi_features.parquet"
    if not data_path.exists():
        st.error(f"Missing dataset at {data_path}. Please verify feature pipeline.")
        return None
    df = pd.read_parquet(data_path)
    return df

service = load_service()
features_df = load_sample_dataset()

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/traffic-light.png", width=64)
st.sidebar.title("Delhi Traffic Intelligence")
st.sidebar.markdown("**MSc Informatics Capstone**")
st.sidebar.markdown("End-to-End Spatiotemporal Traffic Flow & Dynamic Routing System")
st.sidebar.divider()

if features_df is not None:
    st.sidebar.subheader("📅 Observation Timestamp")
    # Allow user to pick from the last 200 operational intervals
    available_indices = features_df.index[-200:]
    selected_time = st.sidebar.select_slider(
        "Select Time of Day (15-min Intervals):",
        options=available_indices,
        value=available_indices[-10],
        format_func=lambda x: x.strftime("%Y-%m-%d %H:%M")
    )
    selected_row = features_df.loc[[selected_time]]
else:
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("📍 Navigation Routing")
origin = st.sidebar.selectbox("Trip Origin:", options=["A (North Gateway)"], index=0)
dest = st.sidebar.selectbox("Trip Destination:", options=["D (Commercial Hub)"], index=0)
origin_code = origin.split()[0]
dest_code = dest.split()[0]

# -----------------------------------------------------------------------------
# HEADER & EXECUTIVE SUMMARY
# -----------------------------------------------------------------------------
st.title("🚦 Indian Urban Traffic Flow Prediction & Route Optimization")
st.markdown(f"**Target Location:** Delhi 3-Approach Intersection Network | **Current Timestamp:** `{selected_time.strftime('%A, %d %B %Y - %H:%M IST')}`")

# Run Inference Service
forecasts = service.predict_horizons(selected_row)
route_decision = service.optimize_route(origin_code, dest_code, forecasts[0].predicted_density)

# Top Metric Cards (Multi-Horizon Forecasts)
st.subheader("1. Multi-Horizon Congestion Forecasts")
col1, col2, col3, col4 = st.columns(4)
cols = [col1, col2, col3, col4]

for i, (col, f) in enumerate(zip(cols, forecasts)):
    with col:
        # Determine visual indicator
        color_map = {"green": "#28a745", "yellow": "#ffc107", "orange": "#fd7e14", "red": "#dc3545"}
        badge_color = color_map.get(f.status_color, "#6c757d")
        
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {badge_color};">
            <h4 style="margin:0; color:#333;">t+{f.horizon_minutes} min Horizon</h4>
            <h2 style="margin:5px 0; color:#111;">{f.predicted_density:.3f} <span style="font-size:1rem; color:#666;">density</span></h2>
            <span class="status-badge" style="background-color: {badge_color}; color: {'#000' if f.status_color=='yellow' else '#fff'};">
                {f.congestion_level.split('(')[0]}
            </span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN CONTENT AREA: TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🗺️ Dynamic Route Optimization", "📈 Multi-Horizon Trajectory", "📊 System Benchmark & Models"])

with tab1:
    st.subheader("Dynamic Dijkstra Path Cost Comparison")
    r_col1, r_col2 = st.columns([1.2, 1])

    with r_col1:
        # Build Interactive Road Graph Display using Plotly
        G = service.road_network
        pos = {
            "A": (0.0, 1.0),
            "B": (2.0, 1.8),
            "C": (2.0, 0.2),
            "D": (4.0, 1.0)
        }
        
        best_path_nodes = route_decision["recommended_path"].split(" -> ")
        best_edges = [(best_path_nodes[i], best_path_nodes[i+1]) for i in range(len(best_path_nodes)-1)]

        fig_graph = go.Figure()

        # Draw all network edges
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            is_best = (u, v) in best_edges
            line_color = "#28a745" if is_best else "#adb5bd"
            line_width = 5 if is_best else 2
            
            fig_graph.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=line_width, color=line_color),
                hoverinfo="text",
                text=f"{data['name']}<br>Length: {data['length_km']} km | Travel Time: {G[u][v].get('cost', 0):.1f}m",
                showlegend=False
            ))

        # Draw network nodes
        for node, (x, y) in pos.items():
            fig_graph.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(size=30, color="#1f77b4" if node not in [origin_code, dest_code] else "#dc3545"),
                text=[f"<b>{node}</b>"],
                textposition="middle center",
                textfont=dict(color="white", size=14),
                hoverinfo="text",
                hovertext=f"Junction Node: {node}",
                showlegend=False
            ))

        fig_graph.update_layout(
            title="Arterial Network Topology & Recommended Trajectory (Green)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=40, b=20),
            height=380,
            paper_bgcolor="#fafafa",
            plot_bgcolor="#fafafa"
        )
        st.plotly_chart(fig_graph, width='stretch')

    with r_col2:
        st.markdown("#### Real-Time Path Decision")
        st.success(f"**Recommended Route:** `{route_decision['recommended_path']}`")
        st.metric("Estimated Trip Travel Time", f"{route_decision['estimated_travel_time_min']} mins")

        st.markdown("#### Alternative Paths")
        for alt in route_decision["alternatives"]:
            diff = round(alt["travel_time_min"] - route_decision["estimated_travel_time_min"], 1)
            st.warning(f"**Alternative:** `{alt['path']}` — **{alt['travel_time_min']} mins** (+{diff} mins)")

        st.info(f"💡 **Bottleneck Impact:** Using predicted queue density `{forecasts[0].predicted_density:.3f}` on Approach 1 corridor.")

with tab2:
    st.subheader("Multi-Horizon Trajectory & Historical Momentum")
    
    # Extract last 10 historical actual points + 4 forecasted points
    hist_window = features_df.loc[:selected_time, "Approach1_Queue"].iloc[-12:]
    hist_times = hist_window.index
    hist_vals = hist_window.values
    
    future_times = [selected_time + pd.Timedelta(minutes=f.horizon_minutes) for f in forecasts]
    future_vals = [f.predicted_density for f in forecasts]

    fig_traj = go.Figure()
    
    # Ground truth history
    fig_traj.add_trace(go.Scatter(
        x=hist_times, y=hist_vals,
        mode="lines+markers",
        name="Historical Observed Density",
        line=dict(color="#111", width=2.5)
    ))

    # Multi-step forecast line
    fig_traj.add_trace(go.Scatter(
        x=[hist_times[-1]] + future_times,
        y=[hist_vals[-1]] + future_vals,
        mode="lines+markers",
        name="15m-60m Multi-Step Forecast",
        line=dict(color="#d62728", width=3, dash="dash"),
        marker=dict(size=8, symbol="diamond")
    ))

    # Add congestion threshold zones
    fig_traj.add_hrect(y0=0.0, y1=0.20, fillcolor="green", opacity=0.08, line_width=0, annotation_text="Free Flow")
    fig_traj.add_hrect(y0=0.20, y1=0.45, fillcolor="yellow", opacity=0.08, line_width=0, annotation_text="Moderate")
    fig_traj.add_hrect(y0=0.45, y1=0.65, fillcolor="orange", opacity=0.08, line_width=0, annotation_text="Heavy Congestion")
    fig_traj.add_hrect(y0=0.65, y1=1.0, fillcolor="red", opacity=0.08, line_width=0, annotation_text="Severe Gridlock")

    fig_traj.update_layout(
        title="Approach 1 Queue Density Momentum & 1-Hour Projections",
        xaxis_title="Time of Day",
        yaxis_title="Proportional Queue Density [0, 1]",
        yaxis_range=[0, 1.0],
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_traj, width="stretch")

with tab3:
    st.subheader("Master Experimental Benchmark (Delhi Dataset 40-Day Evaluation)")
    
    benchmark_df = pd.DataFrame([
        {"Stage": "Stage 1", "Model": "Ridge Regression (L2)", "Paradigm": "Regularized Linear", "Inputs": "25 Lags + Cyclical Time", "WMAPE (%)": "1.65%", "RMSE": "0.0127"},
        {"Stage": "Stage 1", "Model": "Random Forest", "Paradigm": "Non-Linear Ensemble", "Inputs": "25 Lags + Cyclical Time", "WMAPE (%)": "3.16%", "RMSE": "0.0250"},
        {"Stage": "Stage 1", "Model": "XGBoost Regressor", "Paradigm": "Gradient Boosted Trees", "Inputs": "25 Lags + Cyclical Time", "WMAPE (%)": "2.81%", "RMSE": "0.0231"},
        {"Stage": "Stage 2", "Model": "SARIMA (2,0,1)(1,0,1)60", "Paradigm": "Parametric Time-Series", "Inputs": "Univariate Approach 1", "WMAPE (%)": "27.70%", "RMSE": "0.1669"},
        {"Stage": "Stage 3", "Model": "2-Layer LSTM (H=64)", "Paradigm": "Sequential Deep Learning", "Inputs": "1h Sliding Windows (T=4)", "WMAPE (%)": "9.20%", "RMSE": "0.0728"},
        {"Stage": "Stage 4", "Model": "STGCN (Node 0 Benchmark)", "Paradigm": "Spatiotemporal Graph", "Inputs": "Graph Adjacency + Tensors", "WMAPE (%)": "15.50%", "RMSE": "0.1191"},
        {"Stage": "Stage 4", "Model": "STGCN (Network Average)", "Paradigm": "Spatiotemporal Graph", "Inputs": "Joint 3-Approach Optimization", "WMAPE (%)": "15.54%", "RMSE": "0.1157"}
    ])
    st.dataframe(benchmark_df, width="stretch", hide_index=True)

    st.markdown("""
    **Core Academic Findings:**
    * **Linear Trap**: Unregularized OLS suffered target leakage due to high collinearity between queue and stop density. Ridge resolved this with an empirical floor of 1.65% WMAPE.
    * **Parametric Drift**: SARIMA degraded to 27.70% WMAPE over extended test horizons due to recursive mean-regression without external feature gating.
    * **Deep Sequence & Topology**: LSTM achieved 9.20% WMAPE using temporal gates, while STGCN achieved joint multi-approach forecasting (15.54% Network WMAPE) via graph message passing.
    """)

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.divider()
st.caption("Delhi Traffic Flow Prediction Project | MSc Informatics | Streamlit Demonstrator Application")