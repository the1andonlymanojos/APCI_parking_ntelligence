import os
import json
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

# Import routing optimizer
from backend.optimizer import optimize_patrol_route

app = FastAPI(title="AI Parking Congestion Intelligence API")

# Load precalculated assets relative to backend directory (works 100% reliably in serverless environments)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "parking_ml_model.pkl")
CLUSTERS_PATH = os.path.join(BASE_DIR, "hotspot_clusters.json")
STATS_PATH = os.path.join(BASE_DIR, "overall_stats.json")
LOCATIONS_PATH = os.path.join(BASE_DIR, "locations_db.json")

# Load assets immediately on import (essential for stateless serverless environments like Vercel)
# Load Hotspots Database
if os.path.exists(CLUSTERS_PATH):
    with open(CLUSTERS_PATH, 'r') as f:
        hotspots_db = json.load(f)
else:
    hotspots_db = []
    print(f"Warning: {CLUSTERS_PATH} not found.")

# Load Overall Statistics
if os.path.exists(STATS_PATH):
    with open(STATS_PATH, 'r') as f:
        overall_stats = json.load(f)
else:
    overall_stats = {}
    print(f"Warning: {STATS_PATH} not found.")

# Load Locations Database
if os.path.exists(LOCATIONS_PATH):
    with open(LOCATIONS_PATH, 'r') as f:
        locations_db = json.load(f)
else:
    locations_db = {}
    print(f"Warning: {LOCATIONS_PATH} not found.")

# Load ML Model
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            ml_model = model_data['model']
            feature_names = model_data['features']
            cluster_centers = model_data['cluster_centers']
            label_encoder_classes = model_data.get('label_encoder_classes', [])
    except Exception as e:
        print(f"Error loading ML model: {e}")
        ml_model = None
        feature_names = []
        cluster_centers = []
        label_encoder_classes = []
else:
    ml_model = None
    feature_names = []
    cluster_centers = []
    label_encoder_classes = []
    print(f"Warning: {MODEL_PATH} not found. Predictions will use heuristics.")


VEHICLE_WEIGHTS = {
    'BUS': 2.0, 'TRUCK': 2.0, 'HEAVY GOODS VEHICLE': 2.0, 'MAXI-CAB': 1.8,
    'CAR': 1.0, 'SUV': 1.2, 'JEEP': 1.2, 'TEMPO': 1.5, 'THREE WHEELER': 0.8,
    'AUTO RICKSHAW': 0.8, 'TWO WHEELER': 0.3, 'SCOOTER': 0.3, 'MOTORCYCLE': 0.3,
    'CYCLE': 0.1
}

def get_vehicle_weight(v_type):
    if not isinstance(v_type, str):
        return 0.5
    v_type_upper = v_type.upper()
    for k, w in VEHICLE_WEIGHTS.items():
        if k in v_type_upper:
            return w
    return 0.5

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Ground truth Bengaluru POIs
METRO_STATIONS = [
    [12.9756, 77.5728], # Majestic
    [12.9754, 77.6067], # MG Road
    [12.9784, 77.6386], # Indiranagar
    [12.9307, 77.5824], # Jayanagar
    [13.0236, 77.5500], # Yeshwanthpur
    [12.9774, 77.6253], # Halasuru
    [12.9907, 77.6524]  # Baiyappanahalli
]

COMMERCIAL_HUBS = [
    [12.9345, 77.6192], # Koramangala
    [12.9105, 77.6450], # HSR Layout
    [12.9822, 77.6083], # Commercial Street
    [12.9740, 77.6074], # Brigade Road / Church Street
    [12.9984, 77.5704], # Malleshwaram 8th Cross
    [12.9950, 77.7290]  # Whitefield
]

# (Assets are loaded globally at import time above to support serverless lifecycles)

# Pydantic schemas for API inputs
class PredictionRequest(BaseModel):
    location: List[float]  # [latitude, longitude]
    hour: int
    day_of_week: int
    is_main_road: int
    is_intersection: int
    vehicle_type: str

class RoutingRequest(BaseModel):
    n_targets: Optional[int] = 15
    location: Optional[List[float]] = None

@app.get("/api/stats")
def get_stats():
    if not overall_stats:
        raise HTTPException(status_code=404, detail="Stats database not loaded yet.")
    return overall_stats

@app.get("/api/hotspots")
def get_hotspots():
    if not hotspots_db:
        raise HTTPException(status_code=404, detail="Hotspots database not loaded yet.")
    return hotspots_db

@app.get("/api/locations")
def get_locations():
    if not locations_db:
        raise HTTPException(status_code=404, detail="Locations database not loaded yet.")
    return locations_db

@app.post("/api/predict")
def predict_traffic_impact(req: PredictionRequest):
    if len(req.location) != 2:
        raise HTTPException(status_code=400, detail="Invalid location coordinates format.")
        
    lat, lon = req.location[0], req.location[1]
    
    # Compute vehicle weight
    v_weight = get_vehicle_weight(req.vehicle_type)
    
    # Calculate distance to nearest cluster and density
    nearest_dist = 999.0
    nearest_density = 0.0
    min_idx = -1
    
    if len(cluster_centers) > 0:
        dists = haversine_distance(lat, lon, cluster_centers[:, 1], cluster_centers[:, 2])
        min_idx = int(np.argmin(dists))
        nearest_dist = float(dists[min_idx])
        nearest_density = float(cluster_centers[min_idx, 4]) # Violation count

    # Calculate POI distances at inference time
    metro_dists = [haversine_distance(lat, lon, m[0], m[1]) for m in METRO_STATIONS]
    dist_to_metro = float(np.min(metro_dists))

    comm_dists = [haversine_distance(lat, lon, c[0], c[1]) for c in COMMERCIAL_HUBS]
    dist_to_commercial = float(np.min(comm_dists))

    # Run ML prediction if model is loaded
    if ml_model:
        # Encode vehicle_type using the saved classes list
        v_type = str(req.vehicle_type).upper()
        if label_encoder_classes:
            if v_type not in label_encoder_classes:
                # fallback to closest match or first
                matches = [c for c in label_encoder_classes if v_type in c or c in v_type]
                if matches:
                    v_type = matches[0]
                else:
                    v_type = label_encoder_classes[0]
            v_encoded = label_encoder_classes.index(v_type)
        else:
            v_encoded = 0
            
        feature_dict = {
            'latitude': lat,
            'longitude': lon,
            'hour': req.hour,
            'day_of_week': req.day_of_week,
            'vehicle_type_encoded': v_encoded,
            'nearest_hotspot_dist': nearest_dist,
            'nearest_hotspot_density': nearest_density,
            'dist_to_metro': dist_to_metro,
            'dist_to_commercial': dist_to_commercial
        }
        
        # Build features in same order
        X = np.array([[feature_dict[col] for col in feature_names]])
        pred_tis = float(ml_model.predict(X)[0])
    else:
        # Fallback to simple heuristic model if ML pickle not found
        # (0.35 * weight + 0.35 * mr_impact + 0.3 * int_impact) * peak
        v_weight = get_vehicle_weight(req.vehicle_type)
        mr_impact = 1.5 if req.is_main_road == 1 else 0.0
        int_impact = 1.5 if req.is_intersection == 1 else 0.0
        peak = 1.5 if (8 <= req.hour <= 11) or (17 <= req.hour <= 20) else 1.0
        score = (0.35 * v_weight + 0.35 * mr_impact + 0.3 * int_impact) * peak
        pred_tis = (score / 2.5125) * 100.0

    # Hotspot Score Estimation for new location:
    # Based on hourly multiplier and proximity to existing clusters
    
    # 1. Calculate hourly multiplier
    hourly_trends = overall_stats.get('hourly_trends', {})
    total_trends = sum(hourly_trends.values())
    avg_per_hour = total_trends / 24.0 if total_trends > 0 else 1.0
    hour_str = str(req.hour)
    multiplier = (hourly_trends.get(hour_str, 0) / avg_per_hour) if avg_per_hour > 0 else 1.0

    # 2. Determine hotspot score based on proximity
    max_score = float(cluster_centers[:, 3].max()) if len(cluster_centers) > 0 else 100.0
    
    if nearest_dist < 0.15 and len(cluster_centers) > 0 and min_idx != -1:
        # Match nearest cluster's raw score, normalized to [0, 100], and scale by hourly multiplier
        hs_score = float(cluster_centers[min_idx, 3])
        normalized_baseline = (hs_score / max_score) * 100.0
        estimated_hotspot_score = normalized_baseline * multiplier
    else:
        # For new custom locations: use decay proximity heuristic and scale by hourly multiplier
        norm_freq = np.log1p(nearest_density) / (np.log1p(cluster_centers[:, 4].max()) if len(cluster_centers) > 0 else 1.0)
        decay = np.exp(-nearest_dist / 0.5)  # decreases with distance, range ~500m
        raw_estimated = (
            0.4 * norm_freq * decay +
            0.3 * (req.is_main_road) +
            0.2 * (req.is_intersection) +
            0.1 * (0.2 if req.is_main_road and req.is_intersection else 0.0) # rough recurrence proxy
        ) * 100.0
        estimated_hotspot_score = raw_estimated * multiplier

    # Cap values
    pred_tis = float(np.clip(pred_tis, 0, 100))
    estimated_hotspot_score = float(np.clip(estimated_hotspot_score, 0, 100))
    
    return {
        "predicted_traffic_impact_score": round(pred_tis, 2),
        "estimated_hotspot_score": round(estimated_hotspot_score, 2),
        "nearest_hotspot_distance_km": round(nearest_dist, 3),
        "nearest_hotspot_density": int(nearest_density),
        "vehicle_weight_class": "Heavy" if v_weight >= 1.5 else ("Medium" if v_weight >= 0.8 else "Light")
    }

@app.post("/api/route")
def get_optimized_route(req: RoutingRequest):
    if not hotspots_db:
        raise HTTPException(status_code=404, detail="Hotspots database is empty.")
    
    ordered_route, total_distance = optimize_patrol_route(
        hotspots=hotspots_db,
        n_targets=req.n_targets,
        location=req.location
    )
    
    return {
        "ordered_route": ordered_route,
        "total_distance_km": round(total_distance, 2)
    }

# Mount frontend directory for static UI serving
frontend_path = os.path.abspath("frontend")
os.makedirs(frontend_path, exist_ok=True)
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
