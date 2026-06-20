import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    # Earth radius in km
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def optimize_patrol_route(hotspots, n_targets=15, location=None):
    """
    Selects the top N hotspots based on Hotspot Score and computes
    an optimized patrol path using the Nearest Neighbor TSP heuristic.
    """
    if not hotspots:
        return [], 0.0
        
    # Sort hotspots by score descending and take top N
    targets = sorted(hotspots, key=lambda x: x['hotspot_score'], reverse=True)[:n_targets]
    
    # Initialize route list and tracking
    route = []
    unvisited = targets.copy()
    
    # Choose start point: either user coordinates or the highest priority hotspot
    if location and len(location) == 2:
        current_lat, current_lon = location
        # Add a dummy start location for reference
        start_point = {
            'cluster_id': -99,
            'location': [current_lat, current_lon],
            'hotspot_score': 0.0,
            'violation_count': 0,
            'police_station': 'Start Location'
        }
    else:
        # Start at the highest score hotspot
        first = unvisited.pop(0)
        route.append(first)
        current_lat, current_lon = first['location'][0], first['location'][1]
        start_point = None

    total_dist = 0.0
    
    while unvisited:
        # Find closest unvisited hotspot
        closest_idx = -1
        min_dist = float('inf')
        
        for i, hs in enumerate(unvisited):
            d = haversine_distance(current_lat, current_lon, hs['location'][0], hs['location'][1])
            if d < min_dist:
                min_dist = d
                closest_idx = i
                
        if closest_idx != -1:
            closest_hs = unvisited.pop(closest_idx)
            route.append(closest_hs)
            total_dist += min_dist
            current_lat, current_lon = closest_hs['location'][0], closest_hs['location'][1]
            
    # If we had a custom starting point, prepend it to the route
    if start_point:
        route.insert(0, start_point)
        # Compute distance from start point to the first hotspot in route
        if len(route) > 1:
            d_start = haversine_distance(start_point['location'][0], start_point['location'][1], route[1]['location'][0], route[1]['location'][1])
            total_dist += d_start

    return route, total_dist

