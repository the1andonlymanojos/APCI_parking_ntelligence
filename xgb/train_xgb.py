import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
import xgboost as xgb

INPUT_PATH = "jan_to_may_police_violation_imputed.csv"
MODEL_DIR = "xgb"
MODEL_PATH = "xgb/xgb_model.pkl"
META_PATH = "xgb/xgb_meta.pkl"

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

def run_xgb_pipeline():
    print("Step 1: Loading and preprocessing dataset...")
    cols = ['id', 'latitude', 'longitude', 'location', 'vehicle_type', 'created_datetime', 'junction_name']
    df = pd.read_csv(INPUT_PATH, usecols=cols).dropna(subset=['latitude', 'longitude', 'created_datetime', 'vehicle_type'])
    
    # Filter coordinates
    df = df[(df['latitude'] >= 12.8) & (df['latitude'] <= 13.1) & 
            (df['longitude'] >= 77.4) & (df['longitude'] <= 77.8)]
            
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], utc=True, format='mixed')
    # Shift by 13.5 hours to convert from server-exported PST to local Bengaluru time (IST)
    df['local_datetime'] = df['created_datetime'] + pd.Timedelta(hours=13, minutes=30)
    df['hour'] = df['local_datetime'].dt.hour
    df['day_of_week'] = df['local_datetime'].dt.dayofweek
    
    unique_vtypes = df['vehicle_type'].unique()
    weight_map = {vt: get_vehicle_weight(vt) for vt in unique_vtypes}
    df['vehicle_weight'] = df['vehicle_type'].map(weight_map)
    
    df['is_main_road'] = df['location'].astype(str).str.upper().str.contains('MAIN').astype(int)
    df['is_intersection'] = (df['junction_name'].astype(str).str.upper() != 'NO JUNCTION').astype(int)

    df['is_peak_hour'] = np.where(((df['hour'] >= 8) & (df['hour'] <= 11)) | ((df['hour'] >= 17) & (df['hour'] <= 20)), 1.5, 1.0)
    
    df['traffic_impact_score'] = (
        (0.35 * df['vehicle_weight'] + 
         0.35 * 1.5 * df['is_main_road'] + 
         0.3 * 1.5 * df['is_intersection']) * 
        df['is_peak_hour']
    ) / 2.5125 * 100.0

    with open('backend/hotspot_clusters.json', 'r') as f:
        clusters = json.load(f)
    cluster_centers = np.array([[c['cluster_id'], c['location'][0], c['location'][1], c['hotspot_score'], c['violation_count']] for c in clusters])

    print("Calculating nearest hotspot features...")
    lats = np.radians(df['latitude'].values)[:, np.newaxis]
    lons = np.radians(df['longitude'].values)[:, np.newaxis]
    clats = np.radians(cluster_centers[:, 1])[np.newaxis, :]
    clons = np.radians(cluster_centers[:, 2])[np.newaxis, :]
    
    dlat = clats - lats
    dlon = clons - lons
    a = np.sin(dlat/2)**2 + np.cos(lats) * np.cos(clats) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    dists = 6371.0 * c
    
    min_idxs = np.argmin(dists, axis=1)
    df['nearest_hotspot_dist'] = dists[np.arange(len(min_idxs)), min_idxs]
    df['nearest_hotspot_density'] = cluster_centers[min_idxs, 4]

    # Ground truth Metro Stations and Commercial Hubs in Bengaluru
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

    print("Calculating distances to Metro and Commercial POIs...")
    
    # 1. Metro stations distance
    metros = np.array(METRO_STATIONS)
    mlats = np.radians(metros[:, 0])[np.newaxis, :]
    mlons = np.radians(metros[:, 1])[np.newaxis, :]
    dlat_m = mlats - lats
    dlon_m = mlons - lons
    a_m = np.sin(dlat_m/2)**2 + np.cos(lats) * np.cos(mlats) * np.sin(dlon_m/2)**2
    c_m = 2 * np.arcsin(np.sqrt(a_m))
    metro_dists = 6371.0 * c_m
    df['dist_to_metro'] = np.min(metro_dists, axis=1)

    # 2. Commercial hubs distance
    comms = np.array(COMMERCIAL_HUBS)
    clats = np.radians(comms[:, 0])[np.newaxis, :]
    clons = np.radians(comms[:, 1])[np.newaxis, :]
    dlat_c = clats - lats
    dlon_c = clons - lons
    a_c = np.sin(dlat_c/2)**2 + np.cos(lats) * np.cos(clats) * np.sin(dlon_c/2)**2
    c_c = 2 * np.arcsin(np.sqrt(a_c))
    comm_dists = 6371.0 * c_c
    df['dist_to_commercial'] = np.min(comm_dists, axis=1)

    le = LabelEncoder()
    df['vehicle_type_encoded'] = le.fit_transform(df['vehicle_type'].astype(str).str.upper())

    features = [
        'latitude', 'longitude', 'hour', 'day_of_week', 'vehicle_type_encoded', 
        'nearest_hotspot_dist', 'nearest_hotspot_density', 'dist_to_metro', 'dist_to_commercial'
    ]
    
    # ------------------ RANDOM SPLIT EVALUATION ------------------
    print("\n--- Running Random Split Evaluation ---")
    X = df[features].values
    y = df['traffic_impact_score'].values
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    xgb_rand = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.7,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1
    )
    xgb_rand.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    rand_r2_train = r2_score(y_train, xgb_rand.predict(X_train))
    rand_r2_val = r2_score(y_val, xgb_rand.predict(X_val))
    rand_r2_test = r2_score(y_test, xgb_rand.predict(X_test))
    
    print(f"XGBoost Random Split R^2 Results:")
    print(f" - Train R^2: {rand_r2_train:.4f}")
    print(f" - Val R^2:   {rand_r2_val:.4f}")
    print(f" - Test R^2:  {rand_r2_test:.4f}")

    # ------------------ CHRONOLOGICAL SPLIT EVALUATION ------------------
    print("\n--- Running Chronological Split Evaluation ---")
    df_sorted = df.sort_values(by='created_datetime').reset_index(drop=True)
    X_sorted = df_sorted[features].values
    y_sorted = df_sorted['traffic_impact_score'].values
    
    n_samples = len(df_sorted)
    train_end = int(0.70 * n_samples)
    val_end = int(0.85 * n_samples)
    
    X_c_train, y_c_train = X_sorted[:train_end], y_sorted[:train_end]
    X_c_val, y_c_val = X_sorted[train_end:val_end], y_sorted[train_end:val_end]
    X_c_test, y_c_test = X_sorted[val_end:], y_sorted[val_end:]
    
    xgb_chrono = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.7,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1
    )
    xgb_chrono.fit(X_c_train, y_c_train, eval_set=[(X_c_val, y_c_val)], verbose=False)
    
    chrono_r2_train = r2_score(y_c_train, xgb_chrono.predict(X_c_train))
    chrono_r2_val = r2_score(y_c_val, xgb_chrono.predict(X_c_val))
    chrono_r2_test = r2_score(y_c_test, xgb_chrono.predict(X_c_test))
    
    print(f"XGBoost Chronological Split R^2 Results:")
    print(f" - Train R^2: {chrono_r2_train:.4f}")
    print(f" - Val R^2:   {chrono_r2_val:.4f}")
    print(f" - Test R^2:  {chrono_r2_test:.4f}")

    # Save directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save the chronological model as primary
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(xgb_chrono, f)
    print(f"\nSaved XGBoost model to '{MODEL_PATH}'")
    
    # Save metadata
    meta = {
        'features': features,
        'label_encoder': le,
        'random_split_metrics': {
            'train_r2': rand_r2_train,
            'val_r2': rand_r2_val,
            'test_r2': rand_r2_test
        },
        'chrono_split_metrics': {
            'train_r2': chrono_r2_train,
            'val_r2': chrono_r2_val,
            'test_r2': chrono_r2_test
        }
    }
    with open(META_PATH, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved metadata to '{META_PATH}'")

    # Save backend-compatible pickle model (without scikit-learn dependency)
    backend_model_data = {
        'model': xgb_chrono,
        'features': features,
        'label_encoder_classes': list(le.classes_),
        'cluster_centers': cluster_centers
    }
    with open('backend/parking_ml_model.pkl', 'wb') as f:
        pickle.dump(backend_model_data, f)
    print("Saved active model to 'backend/parking_ml_model.pkl'")

if __name__ == "__main__":
    run_xgb_pipeline()
