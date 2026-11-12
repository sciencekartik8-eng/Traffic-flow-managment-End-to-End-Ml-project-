"""
Traffic Intelligence & Downstream Optimization Service.

Provides:
- Multi-step queue density forecasting (15, 30, 45, 60 mins).
- Highway Level of Service (LOS) congestion classification (LOS A to F).
- Dynamic congestion-aware route optimization using Dijkstra's algorithm.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import networkx as nx
import numpy as np
import pandas as pd

DEFAULT_CHECKPOINT_PATH = Path(r"checkpoints/traffic_service_bundle.joblib")


@dataclass
class ForecastResult:
    horizon_minutes: int
    predicted_density: float
    congestion_level: str
    status_color: str


class TrafficIntelligenceService:
    """Production service exposing predictive traffic telemetry and dynamic routing."""

    def __init__(self, bundle_path: Path = DEFAULT_CHECKPOINT_PATH) -> None:
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {bundle_path}. Run src/export_pipeline.py first."
            )
        self.bundle = joblib.load(bundle_path)
        self.models = self.bundle["models"]
        self.feature_cols = self.bundle["feature_cols"]
        self.thresholds = self.bundle["congestion_thresholds"]
        self.road_network = self._initialize_road_network()

    def estimate_congestion(self, density: float) -> Tuple[str, str]:
        """Translates continuous queue density [0, 1] to Highway Capacity Manual Level of Service (LOS)."""
        if density < self.thresholds["free_flow"]:
            return "Free Flow (LOS A-B)", "green"
        elif density < self.thresholds["moderate"]:
            return "Moderate Traffic (LOS C-D)", "yellow"
        elif density < self.thresholds["congested"]:
            return "Heavy Congestion (LOS E)", "orange"
        else:
            return "Severe Gridlock (LOS F)", "red"

    def predict_horizons(self, current_features: pd.DataFrame) -> List[ForecastResult]:
        """Generates multi-step predictions across all horizons and classifies congestion."""
        x_input = current_features[self.feature_cols]
        results: List[ForecastResult] = []
        for h_idx, (key, model) in enumerate(self.models.items(), start=1):
            pred_density = float(model.predict(x_input)[0])
            pred_density = max(0.0, min(1.0, pred_density))  # Physical bounds [0, 1]
            level, color = self.estimate_congestion(pred_density)
            results.append(
                ForecastResult(
                    horizon_minutes=h_idx * 15,
                    predicted_density=round(pred_density, 4),
                    congestion_level=level,
                    status_color=color,
                )
            )
        return results

    def _initialize_road_network(self) -> nx.DiGraph:
        """Constructs a directed graph representing arterial corridors and bypasses."""
        G = nx.DiGraph()
        G.add_edge("A", "C", name="Direct Arterial via Approach 1", length_km=4.2, base_speed_kmh=50.0)
        G.add_edge("C", "D", name="Approach 1 Exit to Commercial Hub", length_km=2.1, base_speed_kmh=45.0)
        G.add_edge("A", "B", name="Ring Bypass Sector 1", length_km=5.5, base_speed_kmh=65.0)
        G.add_edge("B", "D", name="Ring Bypass Sector 2 to Hub", length_km=4.0, base_speed_kmh=60.0)
        return G

    def optimize_route(self, origin: str, destination: str, predicted_density: float) -> Dict:
        """Calculates dynamic travel times based on predicted corridor bottlenecks and identifies optimal routes."""
        G = self.road_network.copy()
        for u, v, data in G.edges(data=True):
            length = data["length_km"]
            speed = data["base_speed_kmh"]
            if "Approach 1" in data["name"]:
                effective_speed = max(10.0, speed * (1.0 - 0.75 * predicted_density))
            else:
                effective_speed = speed * 0.90  # Normal bypass flow

            travel_time_min = (length / effective_speed) * 60.0
            data["cost"] = travel_time_min

        best_path = nx.shortest_path(G, source=origin, target=destination, weight="cost")
        best_cost = nx.shortest_path_length(G, source=origin, target=destination, weight="cost")

        all_paths = list(nx.all_simple_paths(G, source=origin, target=destination))
        alternate_paths = [p for p in all_paths if p != best_path]
        alt_summary = []
        for alt in alternate_paths:
            cost = sum(G[alt[i]][alt[i + 1]]["cost"] for i in range(len(alt) - 1))
            alt_summary.append({"path": " -> ".join(alt), "travel_time_min": round(cost, 2)})

        return {
            "origin": origin,
            "destination": destination,
            "recommended_path": " -> ".join(best_path),
            "estimated_travel_time_min": round(best_cost, 2),
            "alternatives": alt_summary,
            "approach1_density_used": round(predicted_density, 4),
        }
