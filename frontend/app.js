// APCI System Client-Side Application Controller

// Global state variables
let hotspotMap = null;
let routerMap = null;
let hotspotLayerGroup = null;
let routerLayerGroup = null;
let routerPolyline = null;
let hotspotsData = [];
let overallStats = {};
let locationsData = {};
// Real hourly multipliers computed from actual violation data (replaces hardcoded guesses)
let hourlyMultipliers = {};
let hotspotCircles = [];
let correlationChartInstance = null;


// Colors for hotspot scores
function getScoreColor(score) {
    if (score >= 80) return '#ff1744'; // Red
    if (score >= 60) return '#ff9100'; // Orange
    if (score >= 40) return '#ffea00'; // Yellow
    return '#00e676'; // Green
}

// Helper to get closest location name — uses junction names or police station zone as fallback
function getClosestLocationName(lat, lon) {
    let closestName = null;
    let minD = Infinity;

    // 1. Try to match a named junction
    if (locationsData && locationsData.junctions) {
        locationsData.junctions.forEach(item => {
            if (item.location && item.location.length === 2) {
                const d = Math.sqrt(Math.pow(item.location[0] - lat, 2) + Math.pow(item.location[1] - lon, 2));
                if (d < minD) {
                    minD = d;
                    closestName = item.name.replace('Junction: ', '');
                }
            }
        });
    }

    // 2. Fallback: use police station zone from nearest hotspot cluster (covers all 865 clusters)
    if ((minD > 0.005 || !closestName) && hotspotsData && hotspotsData.length > 0) {
        let nearestHotspot = null;
        let minHD = Infinity;
        hotspotsData.forEach(hs => {
            const d = Math.sqrt(Math.pow(hs.location[0] - lat, 2) + Math.pow(hs.location[1] - lon, 2));
            if (d < minHD) { minHD = d; nearestHotspot = hs; }
        });
        if (nearestHotspot && minHD < 0.05) {
            closestName = `${nearestHotspot.police_station} Zone (C#${nearestHotspot.cluster_id})`;
        }
    }

    return closestName || `${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E`;
}

// Haversine distance calculator on the client side
function getHaversineDistance(lat1, lon1, lat2, lon2) {
    const R = 6371.0; // km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
}

document.addEventListener('DOMContentLoaded', () => {
    initTabNavigation();
    fetchStats();
    fetchHotspots();
    fetchLocations();
    initPredictionForm();
    initRoutingForm();
});

// 1. Tab Navigation Handler
function initTabNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanels = document.querySelectorAll('.tab-panel');
    const tabTitle = document.getElementById('current-tab-title');
    const tabSubtitle = document.getElementById('current-tab-subtitle');

    const tabMetadata = {
        dashboard: {
            title: "Operational Dashboard",
            subtitle: "Real-time illegal parking hotspot analysis and traffic impact metrics."
        },
        hotspots: {
            title: "Hotspots Density Map",
            subtitle: "Interactive spatial map displaying clustered parking violations in Bengaluru."
        },
        predictor: {
            title: "Congestion Impact Predictor",
            subtitle: "AI-driven simulation predicting localized traffic impact for hypothetical parking situations."
        },
        routing: {
            title: "Patrol Route Planner",
            subtitle: "Enforcement team route optimizer using Nearest Neighbor TSP algorithms."
        }
    };

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const tabId = item.getAttribute('data-tab');

            // Set active navigation tab
            navItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');

            // Display active tab panel
            tabPanels.forEach(panel => {
                panel.classList.remove('active');
                if (panel.id === `tab-${tabId}`) {
                    panel.classList.add('active');
                }
            });

            // Update Header Text
            if (tabMetadata[tabId]) {
                tabTitle.textContent = tabMetadata[tabId].title;
                tabSubtitle.textContent = tabMetadata[tabId].subtitle;
            }

            // Lazy initialize maps to avoid rendering glitches
            if (tabId === 'hotspots') {
                setTimeout(initHotspotsMap, 100);
            } else if (tabId === 'routing') {
                setTimeout(initRouterMap, 100);
            }
        });
    });
}

// 2. Fetch and Draw Statistics Charts
async function fetchStats() {
    try {
        const response = await fetch('/api/stats');
        if (!response.ok) throw new Error("Stats not loaded");
        
        overallStats = await response.json();
        
        // Update Sidebar and Header metrics
        document.getElementById('sidebar-total-violations').textContent = overallStats.total_violations.toLocaleString();
        
        // Calculate average stats dynamically
        document.getElementById('avg-impact-score').textContent = `${overallStats.avg_traffic_impact.toFixed(1)} / 100`;
        document.getElementById('avg-recurrence').textContent = `${(overallStats.avg_recurrence * 100).toFixed(1)}%`;
        document.getElementById('main-road-pct').textContent = `${overallStats.main_road_pct.toFixed(1)}%`;

        // Update operational overview insights total violations
        const insightTotal = document.getElementById('insight-total-violations');
        if (insightTotal) insightTotal.textContent = overallStats.total_violations.toLocaleString();

        // Gap #12: populate the dataset record count badge
        const recordsBadge = document.getElementById('total-records-badge');
        if (recordsBadge) recordsBadge.textContent = overallStats.total_violations.toLocaleString();

        // Compute REAL hourly multipliers from actual violation distribution
        // avg violations/hour = total / 24; multiplier = hourly_count / avg
        const hourlyTrends = overallStats.hourly_trends || {};
        const totalViol = Object.values(hourlyTrends).reduce((a, b) => a + b, 0);
        const avgViolPerHour = totalViol / 24;
        for (let h = 0; h < 24; h++) {
            hourlyMultipliers[h] = avgViolPerHour > 0
                ? (hourlyTrends[String(h)] || 0) / avgViolPerHour
                : 1.0;
        }

        // Render charts & leaderboard
        renderCharts();
        populateJurisdictionsLeaderboard(overallStats.police_station_distribution, overallStats.total_violations);

        // If hotspots map is initialized, update forecasting with real multipliers
        if (hotspotMap) {
            const sliderEl = document.getElementById('map-timeline-slider');
            const selectedH = sliderEl ? parseFloat(sliderEl.value) : new Date().getHours();
            runHotspotForecasting(selectedH);
        }
    } catch (err) {
        console.error("Error loading statistics:", err);
    }
}

function populateJurisdictionsLeaderboard(distribution, total) {
    const tbody = document.getElementById('jurisdiction-tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    if (!distribution || Object.keys(distribution).length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="text-center" style="color: var(--text-secondary);">No jurisdiction data found.</td></tr>';
        return;
    }
    
    // Sort jurisdictions by count descending
    const sorted = Object.entries(distribution)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8); // top 8 jurisdictions
        
    const maxCount = sorted[0][1];
        
    sorted.forEach(([station, count], index) => {
        const rank = index + 1;
        const pct = (count / total) * 100;
        const fillWidth = (count / maxCount) * 100;
        
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="rank-num">${rank}</td>
            <td class="station-name">${station}</td>
            <td class="violation-count text-right">${count.toLocaleString()}</td>
            <td class="text-right">
                <div class="share-bar-container">
                    <span class="share-percentage">${pct.toFixed(1)}%</span>
                    <div class="share-bar-bg">
                        <div class="share-bar-fill" style="width: ${fillWidth}%;"></div>
                    </div>
                </div>
            </td>
        `;
        tbody.appendChild(tr);
    });
}


function renderCharts() {
    // A. Hourly Distribution Chart
    const hourlyCtx = document.getElementById('hourlyChart').getContext('2d');
    const hourlyData = overallStats.hourly_trends || {};
    const hours = Array.from({length: 24}, (_, i) => `${i}:00`);
    const hourlyCounts = Array.from({length: 24}, (_, i) => hourlyData[i] || 0);
    new Chart(hourlyCtx, {
        type: 'line',
        data: {
            labels: hours,
            datasets: [{ label: 'Violations', data: hourlyCounts, borderColor: '#00e5ff', backgroundColor: 'rgba(0, 229, 255, 0.05)', borderWidth: 2, fill: false, tension: 0.2, pointBackgroundColor: '#00e5ff', pointRadius: 3 }]
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#8b92b6', font: { size: 10 } } }, y: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#8b92b6', font: { size: 10 } } } } }
    });

    // B. Vehicle Type Chart
    const vehicleCtx = document.getElementById('vehicleChart').getContext('2d');
    const vehicleData = overallStats.vehicle_distribution || {};
    const sortedVehicles = Object.entries(vehicleData).sort((a, b) => b[1] - a[1]).slice(0, 7);
    new Chart(vehicleCtx, {
        type: 'bar',
        data: { labels: sortedVehicles.map(v => v[0]), datasets: [{ data: sortedVehicles.map(v => v[1]), backgroundColor: 'rgba(124, 77, 255, 0.8)', borderWidth: 0, borderRadius: 4 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false }, ticks: { color: '#8b92b6', font: { size: 10 } } }, y: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#8b92b6', font: { size: 10 } } } } }
    });

    // C. Day-of-Week Pattern Chart (weekdays vs weekends highlighted)
    const dowCtxEl = document.getElementById('dowChart');
    if (dowCtxEl) {
        const dowData = overallStats.day_of_week_trends || {};
        const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
        const dowCounts = days.map((_, i) => dowData[String(i)] || 0);
        const dowColors = days.map((_, i) => i >= 5 ? 'rgba(255, 179, 0, 0.85)' : 'rgba(0, 229, 255, 0.7)');
        new Chart(dowCtxEl.getContext('2d'), {
            type: 'bar',
            data: { labels: days, datasets: [{ data: dowCounts, backgroundColor: dowColors, borderWidth: 0, borderRadius: 4 }] },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => `${ctx.parsed.y.toLocaleString()} violations` } } },
                scales: { x: { grid: { display: false }, ticks: { color: '#8b92b6', font: { size: 11 } } }, y: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#8b92b6', font: { size: 10 } } } }
            }
        });
    }

    // D. Violation Types Mini Horizontal Bar (top 5 categories)
    const vtCtxEl = document.getElementById('violationTypesChart');
    if (vtCtxEl) {
        const vtData = overallStats.violation_types || {};
        const vtSorted = Object.entries(vtData).sort((a, b) => b[1] - a[1]).slice(0, 5);
        if (vtSorted.length > 0) {
            // Strip JSON array brackets e.g. '["WRONG PARKING"]' → 'WRONG PARKING'
            const cleanLabel = (k) => {
                const cleaned = k.replace(/^\["|"\]$/g, '').replace(/","/, ' + ').slice(0, 18);
                return cleaned.length < k.length ? cleaned + (cleaned.length >= 16 ? '' : '') : cleaned;
            };
            new Chart(vtCtxEl.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: vtSorted.map(v => cleanLabel(v[0])),
                    datasets: [{ data: vtSorted.map(v => v[1]), backgroundColor: 'rgba(255, 145, 0, 0.75)', borderWidth: 0, borderRadius: 3 }]
                },
                options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#8b92b6', font: { size: 9 } } }, y: { grid: { display: false }, ticks: { color: '#8b92b6', font: { size: 9 }, maxRotation: 0 } } } }
            });
        }
    }
}

// 3. Fetch Hotspots and Map Plotting
async function fetchHotspots() {
    try {
        const response = await fetch('/api/hotspots');
        if (!response.ok) throw new Error("Hotspots not loaded");
        
        hotspotsData = await response.json();
        
        // Update total hotspots counters
        document.getElementById('sidebar-total-hotspots').textContent = hotspotsData.length.toString();
        
        // Build sidebar hotspots list in UI
        populateHotspotsList();

        // Initialize forecasting slider & initial state alerts
        initTimelineForecast();
        runHotspotForecasting(new Date().getHours()); // use current hour as direct 0-23 selector

        // Draw scatter plot correlation chart
        renderCorrelationChart();
    } catch (err) {
        console.error("Error loading hotspots:", err);
    }
}

// HOURLY_TRAFFIC_WEIGHT is now replaced by hourlyMultipliers computed from real data in fetchStats()
// Kept here only as a zero-fallback before stats load
const HOURLY_TRAFFIC_WEIGHT_FALLBACK = {
    0: 1.76, 1: 1.40, 2: 2.01, 3: 2.09, 4: 2.38, 5: 2.77,
    6: 2.17, 7: 1.16, 8: 0.67, 9: 0.25, 10: 0.04, 11: 0.05,
    12: 0.02, 13: 0.005, 14: 0.001, 15: 0.005, 16: 0.03, 17: 0.07,
    18: 0.16, 19: 0.83, 20: 0.94, 21: 1.58, 22: 1.82, 23: 1.81
};

function formatTimeFromValue(x) {
    const totalMinutes = Math.round(x * 60);
    const h24 = Math.floor(totalMinutes / 60) % 24;
    const m = totalMinutes % 60;
    const period = h24 < 12 ? 'AM' : 'PM';
    const h12 = h24 % 12 === 0 ? 12 : h24 % 12;
    const minStr = m.toString().padStart(2, '0');
    return `${h12.toString().padStart(2, '0')}:${minStr} ${period}`;
}

function getInterpolatedMultiplier(x) {
    const h1 = Math.floor(x) % 24;
    const h2 = (h1 + 1) % 24;
    const t = x - Math.floor(x);
    
    const m1 = Object.keys(hourlyMultipliers).length > 0
        ? (hourlyMultipliers[h1] ?? 1.0)
        : (HOURLY_TRAFFIC_WEIGHT_FALLBACK[h1] ?? 1.0);
        
    const m2 = Object.keys(hourlyMultipliers).length > 0
        ? (hourlyMultipliers[h2] ?? 1.0)
        : (HOURLY_TRAFFIC_WEIGHT_FALLBACK[h2] ?? 1.0);
        
    return (1 - t) * m1 + t * m2;
}

function formatExpectedViolations(hsCount, multiplier, isHourly = true) {
    // 152 unique days in the dataset, 24 hours in a day
    const val = isHourly 
        ? (hsCount / (152 * 24)) * multiplier
        : (hsCount / 152) * multiplier;
    if (val < 0.05) return "Near zero";
    if (val < 1.0) return `~${val.toFixed(1)} cases`;
    return `~${Math.round(val)} cases`;
}

function initTimelineForecast() {
    const slider = document.getElementById('map-timeline-slider');
    const label = document.getElementById('simulation-hour-label');
    if (!slider || !label) return;

    // Set slider to exact current hour decimal (e.g. 1.5 for 1:30 AM)
    const now = new Date();
    const currentHourDecimal = parseFloat((now.getHours() + now.getMinutes() / 60).toFixed(1));
    slider.value = currentHourDecimal;
    
    const updateLabelAndForecast = (val) => {
        const selectedHour = parseFloat(val);
        const nowHDecimal = now.getHours() + now.getMinutes() / 60;
        // Label "Now" if within 10 minutes (0.17 hours)
        const isNow = Math.abs(selectedHour - nowHDecimal) < 0.17;
        label.textContent = isNow
            ? `${formatTimeFromValue(selectedHour)} ← Now`
            : formatTimeFromValue(selectedHour);
        runHotspotForecasting(selectedHour);
    };

    // Initial run
    updateLabelAndForecast(slider.value);

    slider.addEventListener('input', () => {
        updateLabelAndForecast(slider.value);
    });
}

// runHotspotForecasting: selectedHour is the DIRECT decimal hour of day (0–23.9)
// Multiplier is continuously interpolated from dynamic stats or fallbacks
function runHotspotForecasting(selectedHour) {
    const peakMultiplier = getInterpolatedMultiplier(selectedHour);
    
    const activeAlerts = [];
    
    // Fallback: build hotspot circles if they aren't initialized yet
    if (hotspotCircles.length === 0 && hotspotsData.length > 0) {
        plotHotspotMarkers();
    }
    
    if (hotspotCircles.length > 0) {
        // Dynamically find maximum hotspot score in dataset to normalize baseline to [0, 100]
        const maxScore = Math.max(...hotspotsData.map(h => h.hotspot_score));
        
        hotspotCircles.forEach(item => {
            const hs = item.hotspot;
            const circle = item.marker;
            
            // Normalize baseline score to a [0, 100] scale
            const normalizedBaseScore = maxScore > 0 ? (hs.hotspot_score / maxScore) * 100 : hs.hotspot_score;
            
            // Forecast TIS = normalized base score * peakMultiplier
            const forecastTIS = Math.min(100, Math.max(0, normalizedBaseScore * peakMultiplier));
            
            // Radius scales dynamically with expected violations at this hour
            const maxFreq = Math.max(...hotspotsData.map(h => h.violation_count));
            const minRadius = 15;
            const maxRadius = 45;
            const ratio = Math.log1p(hs.violation_count) / Math.log1p(maxFreq);
            const sizeScale = Math.max(0.2, Math.sqrt(peakMultiplier));
            const radius = (minRadius + ratio * (maxRadius - minRadius)) * sizeScale;
            
            // Color scales with simulated forecast TIS
            const color = getScoreColor(forecastTIS);
            const name = getClosestLocationName(hs.location[0], hs.location[1]);
            const expectedThisHourStr = formatExpectedViolations(hs.violation_count, peakMultiplier, true);
            const dailyAvgTotalStr = formatExpectedViolations(hs.violation_count, 1.0, false);
            
            // Update circle style and size in-place (causes smooth transition if CSS is set)
            circle.setRadius(radius);
            circle.setStyle({
                fillColor: color,
                color: color
            });
            
            // Add dispatch alert if predicted TIS >= 70
            if (forecastTIS >= 70) {
                activeAlerts.push({
                    name: name,
                    score: forecastTIS,
                    location: hs.location,
                    station: hs.police_station
                });
            }
            
            // Detailed popup showing both the dynamic forecast and daily average statistics
            const popupContent = `
                <div class="popup-details">
                    <div class="popup-title" style="margin-bottom: 2px;">${name}</div>
                    <div style="font-size: 10px; color: var(--text-secondary); margin-bottom: 8px;">Cluster #${hs.cluster_id}</div>
                    
                    <div class="popup-row">
                        <span class="popup-label">Forecast TIS:</span>
                        <span class="popup-val" style="color: ${color}; font-weight: 700;">${Math.round(forecastTIS)} / 100</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Expected (This Hour):</span>
                        <span class="popup-val" style="font-weight: 600; color: var(--color-secondary);">${expectedThisHourStr}</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Daily Avg Total:</span>
                        <span class="popup-val" style="font-weight: 500;">${dailyAvgTotalStr}</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Enforcement Status:</span>
                        <span class="popup-val" style="color: ${forecastTIS >= 70 ? 'var(--color-danger)' : 'var(--color-success)'}; font-weight: 600;">
                            ${forecastTIS >= 70 ? 'Immediate Dispatch' : 'Routine Monitoring'}
                        </span>
                    </div>
                    <hr style="border-color: rgba(255,255,255,0.1); margin: 6px 0;">
                    <div class="popup-row">
                        <span class="popup-label">Police Jurisdiction:</span>
                        <span class="popup-val">${hs.police_station}</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Recurrence Rate:</span>
                        <span class="popup-val">${Math.round(hs.recurrence_rate * 100)}%</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Main Road Parked:</span>
                        <span class="popup-val">${Math.round(hs.main_road_ratio * 100)}%</span>
                    </div>
                    <div class="popup-row">
                        <span class="popup-label">Crossing Parked:</span>
                        <span class="popup-val">${Math.round(hs.intersection_ratio * 100)}%</span>
                    </div>
                    <button onclick="simulateHere(${hs.location[0]}, ${hs.location[1]});" style="margin-top: 8px; width: 100%; padding: 6px; font-size: 10px; background-color: var(--color-primary); border: none; border-radius: 4px; color: #fff; cursor: pointer; font-family: inherit;">
                        <i class='fa-solid fa-wand-magic-sparkles' style='margin-right:4px'></i>Simulate Here
                    </button>
                </div>
            `;
            circle.bindPopup(popupContent);
        });
    }
    
    // Render alerts
    renderDispatchAlerts(activeAlerts);
}

function renderDispatchAlerts(alerts) {
    const container = document.getElementById('dispatch-alerts-list');
    if (!container) return;
    
    container.innerHTML = '';
    
    if (alerts.length === 0) {
        container.innerHTML = `
            <div class="no-alerts-msg">
                <i class="fa-solid fa-circle-check"></i>
                No active chokepoints predicted.
            </div>
        `;
        return;
    }
    
    // Sort alerts by score descending, show top 4
    const topAlerts = alerts
        .sort((a, b) => b.score - a.score)
        .slice(0, 4);
        
    topAlerts.forEach(alert => {
        const card = document.createElement('div');
        card.className = 'proactive-alert-card';
        card.style.cursor = 'pointer';
        card.innerHTML = `
            <div class="alert-card-title">
                <span>🚨 Impact Spike Alert</span>
                <span>TIS: ${Math.round(alert.score)}</span>
            </div>
            <div class="alert-card-desc">
                Enforce immediately at <strong>${alert.name.replace('Near ', '')}</strong> (${alert.station} jurisdiction). Predicted lane choke risk is high.
            </div>
        `;
        
        card.addEventListener('click', () => {
            if (hotspotMap) {
                hotspotMap.setView(alert.location, 16);
                hotspotLayerGroup.eachLayer(layer => {
                    if (layer.getLatLng().lat === alert.location[0] && layer.getLatLng().lng === alert.location[1]) {
                        layer.openPopup();
                    }
                });
            }
        });
        
        container.appendChild(card);
    });
}

function renderCorrelationChart() {
    const correlationCtxEl = document.getElementById('correlationChart');
    if (!correlationCtxEl) return;

    // Destroy previous Chart instance if it exists to avoid canvas reuse errors
    if (correlationChartInstance) {
        correlationChartInstance.destroy();
        correlationChartInstance = null;
    }

    const scatterData = hotspotsData.map(hs => ({ x: hs.violation_count, y: hs.hotspot_score }));

    // Compute linear regression trendline (violations → impact)
    let trendData = [];
    if (scatterData.length > 1) {
        const n = scatterData.length;
        const sumX = scatterData.reduce((a, b) => a + b.x, 0);
        const sumY = scatterData.reduce((a, b) => a + b.y, 0);
        const sumXY = scatterData.reduce((a, b) => a + b.x * b.y, 0);
        const sumX2 = scatterData.reduce((a, b) => a + b.x * b.x, 0);
        const denom = (n * sumX2 - sumX * sumX);
        if (denom !== 0) {
            const slope = (n * sumXY - sumX * sumY) / denom;
            const intercept = (sumY - slope * sumX) / n;
            const minX = Math.min(...scatterData.map(d => d.x));
            const maxX = Math.max(...scatterData.map(d => d.x));
            trendData = [{ x: minX, y: slope * minX + intercept }, { x: maxX, y: slope * maxX + intercept }];
        }
    }

    correlationChartInstance = new Chart(correlationCtxEl.getContext('2d'), {
        type: 'scatter',
        data: {
            datasets: [
                { label: 'Clusters', data: scatterData, backgroundColor: 'rgba(0,229,255,0.45)', borderColor: 'rgba(0,229,255,0.7)', pointRadius: 3, pointHoverRadius: 5 },
                { label: 'Trend', data: trendData, type: 'line', borderColor: 'rgba(255,145,0,0.85)', borderWidth: 2, pointRadius: 0, fill: false, tension: 0, showLine: true }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: ctx => ctx.dataset.label === 'Trend' ? '' : `Vol: ${ctx.parsed.x} | Score: ${ctx.parsed.y.toFixed(1)}` } }
            },
            scales: {
                x: { title: { display: true, text: 'Violation Volume', color: '#8b92b6', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#8b92b6', font: { size: 9 } } },
                y: { title: { display: true, text: 'Hotspot Score', color: '#8b92b6', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#8b92b6', font: { size: 9 } } }
            }
        }
    });
}

function populateHotspotsList() {
    const listWrapper = document.getElementById('hotspots-list');
    listWrapper.innerHTML = '';

    if (hotspotsData.length === 0) {
        listWrapper.innerHTML = '<div class="loading-placeholder">No hotspots detected.</div>';
        return;
    }

    // Top 20 hotspots sorted by Hotspot Score
    const topHotspots = hotspotsData.slice(0, 30);

    topHotspots.forEach(hs => {
        const item = document.createElement('div');
        item.className = 'hotspot-item';
        const name = getClosestLocationName(hs.location[0], hs.location[1]);
        item.innerHTML = `
            <div class="hotspot-item-header">
                <span class="hotspot-id" style="font-weight: 700; color: #fff;">${name}</span>
                <span class="hotspot-pill-score" style="background-color: ${getScoreColor(hs.hotspot_score)}">
                    Score: ${Math.round(hs.hotspot_score)}
                </span>
            </div>
            <div class="hotspot-item-body">
                <span style="font-size: 10px; color: var(--text-secondary); margin-bottom: 4px; display: block;">Cluster #${hs.cluster_id} • Division: ${hs.police_station}</span>
                <span><i class="fa-solid fa-car-burst"></i> ${hs.violation_count} Violations</span>
                <span><i class="fa-solid fa-rotate"></i> ${Math.round(hs.recurrence_rate * 100)}% Recurr.</span>
                <span><i class="fa-solid fa-road"></i> ${Math.round(hs.main_road_ratio * 100)}% Main Rd</span>
                <span><i class="fa-solid fa-circle-nodes"></i> ${Math.round(hs.intersection_ratio * 100)}% Crossing</span>
            </div>
        `;

        item.addEventListener('click', () => {
            // Center map on the selected hotspot
            if (hotspotMap) {
                hotspotMap.setView(hs.location, 16);
                
                // Find and trigger click on corresponding map marker popup
                hotspotLayerGroup.eachLayer(layer => {
                    if (layer.getLatLng().lat === hs.location[0] && layer.getLatLng().lng === hs.location[1]) {
                        layer.openPopup();
                    }
                });
            }
        });

        listWrapper.appendChild(item);
    });
}

// Map instances initialization
function initHotspotsMap() {
    if (hotspotMap) {
        hotspotMap.invalidateSize();
        return;
    }

    // Default center to Bengaluru
    const defaultCenter = [12.935, 77.62]; // Koramangala area
    hotspotMap = L.map('hotspot-map').setView(defaultCenter, 13);

    // Dark Map Tiles (CartoDB Dark Matter)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(hotspotMap);

    hotspotLayerGroup = L.layerGroup().addTo(hotspotMap);

    // Click map to select coordinates for Predictor
    hotspotMap.on('click', (e) => {
        setPredictorCoordinates(e.latlng.lat, e.latlng.lng);
    });

    plotHotspotMarkers();
    addMapLegend(hotspotMap);
    // Re-run forecast at the current hour now that the map layer exists
    const currentH = new Date().getHours();
    const sliderEl = document.getElementById('map-timeline-slider');
    const selectedH = sliderEl ? parseFloat(sliderEl.value) : currentH;
    runHotspotForecasting(selectedH);
}

// Leaflet legend control showing score → color mapping
function addMapLegend(map) {
    if (!map) return;
    const legend = L.control({ position: 'bottomleft' });
    legend.onAdd = function () {
        const div = L.DomUtil.create('div', 'map-legend');
        div.innerHTML = `
            <div class="legend-title"><i class="fa-solid fa-layer-group"></i> Hotspot Score</div>
            <div class="legend-item"><span class="legend-dot" style="background:#ff1744"></span>Critical &nbsp;≥ 80</div>
            <div class="legend-item"><span class="legend-dot" style="background:#ff9100"></span>High &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;60 – 79</div>
            <div class="legend-item"><span class="legend-dot" style="background:#ffea00"></span>Medium &nbsp;40 – 59</div>
            <div class="legend-item"><span class="legend-dot" style="background:#00e676"></span>Low &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt; 40</div>
            <div class="legend-note">Dot size = violation volume</div>
        `;
        return div;
    };
    legend.addTo(map);
}

function plotHotspotMarkers() {
    if (!hotspotsData || hotspotsData.length === 0) return;
    if (!hotspotLayerGroup) return; // Map layer group not initialized yet

    hotspotLayerGroup.clearLayers();
    hotspotCircles = [];

    // Map min/max values for radius scaling
    const maxFreq = Math.max(...hotspotsData.map(h => h.violation_count));
    const minRadius = 15;
    const maxRadius = 45;

    hotspotsData.forEach(hs => {
        // Logarithmic scale for radius
        const ratio = Math.log1p(hs.violation_count) / Math.log1p(maxFreq);
        const radius = minRadius + ratio * (maxRadius - minRadius);
        const color = getScoreColor(hs.hotspot_score);

        const circle = L.circleMarker(hs.location, {
            radius: radius,
            fillColor: color,
            fillOpacity: 0.45,
            color: color,
            weight: 2,
            opacity: 0.95
        });

        // Add to map layer
        circle.addTo(hotspotLayerGroup);
        
        hotspotCircles.push({
            marker: circle,
            hotspot: hs
        });
    });
}

function setPredictorCoordinates(lat, lon) {
    document.getElementById('pred-lat').value = lat.toFixed(7);
    document.getElementById('pred-lon').value = lon.toFixed(7);
    
    const locationName = getClosestLocationName(lat, lon);
    
    // Add custom coordinate option to select dropdown
    const select = document.getElementById('pred-location-select');
    if (select) {
        let customOpt = document.getElementById('custom-map-option');
        if (!customOpt) {
            customOpt = document.createElement('option');
            customOpt.id = 'custom-map-option';
            select.appendChild(customOpt);
        }
        customOpt.value = 'custom';
        customOpt.textContent = locationName;
        select.value = 'custom';
    }

    // Set defaults for manual selection: search if coordinates are close to any hotspot
    if (hotspotsData && hotspotsData.length > 0) {
        const closeHotspot = hotspotsData.find(hs => {
            const d = Math.sqrt(Math.pow(hs.location[0] - lat, 2) + Math.pow(hs.location[1] - lon, 2));
            return d < 0.0015; // roughly 150m
        });
        if (closeHotspot) {
            document.getElementById('pred-main-road').checked = closeHotspot.main_road_ratio > 0.4;
            document.getElementById('pred-intersection').checked = closeHotspot.intersection_ratio > 0.4;
        } else {
            document.getElementById('pred-main-road').checked = false;
            document.getElementById('pred-intersection').checked = false;
        }
    }
    
    // Provide visually appealing notify indicator
    const feedbackMsg = document.createElement('div');
    feedbackMsg.style.cssText = "position: fixed; bottom: 20px; right: 20px; background-color: var(--color-primary); color: #fff; padding: 12px 24px; border-radius: 30px; z-index: 10000; font-weight: 600; box-shadow: 0 4px 15px rgba(0,0,0,0.4); animation: fadeOut 3s forwards;";
    feedbackMsg.innerHTML = `<i class="fa-solid fa-location-crosshairs"></i> Location set: ${locationName}`;
    document.body.appendChild(feedbackMsg);

    setTimeout(() => feedbackMsg.remove(), 3000);
}

// simulateHere: sets predictor coords AND navigates to the Predictor tab
function simulateHere(lat, lon) {
    setPredictorCoordinates(lat, lon);
    const predictorNav = document.querySelector('[data-tab="predictor"]');
    if (predictorNav) predictorNav.click();
}

// exportPatrolRoute: download patrol route as a .txt file
function exportPatrolRoute(route, distanceKm) {
    const now = new Date();
    const dateStr = now.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });
    const timeStr = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
    const lines = [
        'APCI PATROL ROUTE — BENGALURU TRAFFIC POLICE',
        `Generated: ${dateStr} at ${timeStr}`,
        `Total Route Distance: ${distanceKm} km | Patrol Stops: ${route.filter(s => s.cluster_id !== -99).length}`,
        '================================================================', ''
    ];
    route.forEach((hs, i) => {
        if (hs.cluster_id === -99) {
            lines.push(`[ORIGIN]  ${getClosestLocationName(hs.location[0], hs.location[1])}`);
        } else {
            lines.push(`Stop ${String(i).padStart(2, '0')}  Cluster #${hs.cluster_id} | Score: ${Math.round(hs.hotspot_score)}/100 | ${hs.violation_count} cases`);
            lines.push(`        Zone: ${hs.police_station}`);
            lines.push(`        GPS:  ${hs.location[0].toFixed(6)}, ${hs.location[1].toFixed(6)}`);
        }
        lines.push('');
    });
    lines.push('================================================================');
    lines.push('AI-Driven Parking Congestion Intelligence (APCI) | Bengaluru');
    const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `apci_patrol_${now.toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// 4. ML Congestion Impact Predictor Form
function initPredictionForm() {
    const form = document.getElementById('prediction-form');
    const resultPanel = document.getElementById('prediction-result');
    const select = document.getElementById('pred-location-select');

    if (select) {
        select.addEventListener('change', () => {
            const val = select.value;
            if (!val || val === 'custom') return;

            let loc = null;
            if (val.startsWith('junction_')) {
                const idx = parseInt(val.split('_')[1]);
                loc = locationsData.junctions[idx];
            } else if (val.startsWith('hotspot_')) {
                const idx = parseInt(val.split('_')[1]);
                loc = locationsData.hotspots[idx];
            }

            if (loc) {
                document.getElementById('pred-lat').value = loc.location[0];
                document.getElementById('pred-lon').value = loc.location[1];
                document.getElementById('pred-main-road').checked = loc.is_main_road === 1;
                document.getElementById('pred-intersection').checked = loc.is_intersection === 1;
            }
        });
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // --- Gap #11 fix: validate lat/lon before sending ---
        const predLat = parseFloat(document.getElementById('pred-lat').value);
        const predLon = parseFloat(document.getElementById('pred-lon').value);
        if (isNaN(predLat) || isNaN(predLon)) {
            resultPanel.innerHTML = `
                <div class="result-placeholder">
                    <i class="fa-solid fa-location-crosshairs placeholder-icon" style="color: var(--color-warning); font-size: 36px;"></i>
                    <p style="color: var(--color-warning); font-weight: 600; margin: 0;">No Location Selected</p>
                    <span class="tip-text">Choose a junction from the dropdown above, or go to the Hotspots Map, click a cluster circle, and press <strong>Simulate Here</strong>.</span>
                </div>
            `;
            return;
        }

        // Collect form data
        const requestData = {
            location: [predLat, predLon],
            hour: parseInt(document.getElementById('pred-hour').value),
            day_of_week: parseInt(document.getElementById('pred-day').value),
            is_main_road: document.getElementById('pred-main-road').checked ? 1 : 0,
            is_intersection: document.getElementById('pred-intersection').checked ? 1 : 0,
            vehicle_type: document.getElementById('pred-vehicle').value
        };

        // Render Loading State
        resultPanel.innerHTML = `
            <div class="loading-placeholder">
                <i class="fa-solid fa-circle-notch fa-spin placeholder-icon"></i>
                <p>Running XGBoost A+B Regressor model...</p>
            </div>
        `;

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) throw new Error("Prediction API error");
            const res = await response.json();

            // Determine severity badge style
            let badgeClass = 'pred-severity-low';
            let recommendation = "Congestion risk is low. Routine patrol monitoring is adequate.";
            const score = res.predicted_traffic_impact_score;

            if (score >= 70) {
                badgeClass = 'pred-severity-high';
                recommendation = "CRITICAL RISK: Illegal parking here will cause severe queue spillover. Deploy immediate enforcement / towing patrol.";
            } else if (score >= 40) {
                badgeClass = 'pred-severity-medium';
                recommendation = "MODERATE RISK: Noticeable lane capacity reduction. Schedule parking enforcement patrol within the hour.";
            }

            resultPanel.innerHTML = `
                <div class="prediction-score-display">
                    <div class="prediction-radial-gauge" style="border-color: ${getScoreColor(score)}">
                        <span class="score-num">${Math.round(score)}</span>
                        <span class="score-lbl">Impact Score</span>
                    </div>
                    <div class="prediction-badge ${badgeClass}">
                        ${score >= 70 ? 'High Congestion Risk' : (score >= 40 ? 'Medium Congestion Risk' : 'Low Congestion Risk')}
                    </div>
                </div>

                <div class="results-metrics-grid">
                    <div class="metric-box">
                        <span class="label">Hotspot Score</span>
                        <span class="val" style="color: ${getScoreColor(res.estimated_hotspot_score)}">${Math.round(res.estimated_hotspot_score)}/100</span>
                    </div>
                    <div class="metric-box">
                        <span class="label">Nearest Hotspot</span>
                        <span class="val">${res.nearest_hotspot_distance_km} km</span>
                    </div>
                    <div class="metric-box">
                        <span class="label">Nearby Violation Density</span>
                        <span class="val">${res.nearest_hotspot_density} cases</span>
                    </div>
                    <div class="metric-box">
                        <span class="label">Vehicle Class Weight</span>
                        <span class="val">${res.vehicle_weight_class}</span>
                    </div>
                </div>

                <div style="background-color: rgba(255,255,255,0.02); border: 1px solid var(--glass-border); border-radius: 8px; padding: 16px; margin-top: 20px;">
                    <h4 style="font-family: 'Space Grotesk', sans-serif; font-size: 13px; font-weight: 600; color: var(--color-secondary); margin-bottom: 6px;"><i class="fa-solid fa-bullseye"></i> Tactical Directive</h4>
                    <p style="font-size: 12px; color: var(--text-secondary); line-height: 1.5;">${recommendation}</p>
                </div>
            `;
        } catch (err) {
            console.error("Error fetching prediction:", err);
            resultPanel.innerHTML = `
                <div class="result-placeholder">
                    <i class="fa-solid fa-triangle-exclamation placeholder-icon text-danger"></i>
                    <p>Failed to evaluate prediction. Please check inputs and try again.</p>
                </div>
            `;
        }
    });
}

// 5. Patrol Router Map and Controls
function initRouterMap() {
    if (routerMap) {
        routerMap.invalidateSize();
        return;
    }

    // Default center to Bengaluru
    const defaultCenter = [12.935, 77.62];
    routerMap = L.map('router-map').setView(defaultCenter, 13);

    // Dark Map Tiles
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(routerMap);

    routerLayerGroup = L.layerGroup().addTo(routerMap);

    // Click map to set start location
    routerMap.on('click', (e) => {
        const lat = e.latlng.lat;
        const lon = e.latlng.lng;
        document.getElementById('route-start-lat').value = lat.toFixed(7);
        document.getElementById('route-start-lon').value = lon.toFixed(7);
        
        // Update select option to show map selection name
        const select = document.getElementById('route-start-select');
        if (select) {
            let customOpt = document.getElementById('custom-route-option');
            if (!customOpt) {
                customOpt = document.createElement('option');
                customOpt.id = 'custom-route-option';
                select.appendChild(customOpt);
            }
            const locationName = getClosestLocationName(lat, lon);
            customOpt.value = 'custom';
            customOpt.textContent = locationName;
            select.value = 'custom';
        }
        
        // Visual indicator on router map
        plotStartMarker(lat, lon);
    });
}

function plotStartMarker(lat, lon) {
    // Clear any previous starting points
    routerLayerGroup.eachLayer(layer => {
        if (layer.options && layer.options.isStartMarker) {
            routerLayerGroup.removeLayer(layer);
        }
    });

    const locationName = getClosestLocationName(lat, lon);
    const startMarker = L.marker([lat, lon], {
        isStartMarker: true,
        icon: L.divIcon({
            className: 'custom-div-icon',
            html: `<div class="marker-pin-label" style="background-color: var(--color-secondary);"><i class="fa-solid fa-house-chimney"></i></div>`,
            iconSize: [28, 28],
            iconAnchor: [14, 14]
        })
    }).addTo(routerLayerGroup);
    
    startMarker.bindPopup(`<b>Enforcement Origin</b><br>${locationName}`);
}

function initRoutingForm() {
    const btn = document.getElementById('btn-generate-route');
    const summaryPanel = document.getElementById('route-summary-panel');
    const stepsList = document.getElementById('patrol-steps-list');
    const routeSelect = document.getElementById('route-start-select');

    if (routeSelect) {
        routeSelect.addEventListener('change', () => {
            const val = routeSelect.value;
            if (!val) {
                document.getElementById('route-start-lat').value = '';
                document.getElementById('route-start-lon').value = '';
                // remove start marker if any
                routerLayerGroup.eachLayer(layer => {
                    if (layer.options && layer.options.isStartMarker) {
                        routerLayerGroup.removeLayer(layer);
                    }
                });
                return;
            }
            if (val === 'custom') return;

            let loc = null;
            if (val.startsWith('junction_')) {
                const idx = parseInt(val.split('_')[1]);
                loc = locationsData.junctions[idx];
            } else if (val.startsWith('hotspot_')) {
                const idx = parseInt(val.split('_')[1]);
                loc = locationsData.hotspots[idx];
            }

            if (loc) {
                document.getElementById('route-start-lat').value = loc.location[0];
                document.getElementById('route-start-lon').value = loc.location[1];
                plotStartMarker(loc.location[0], loc.location[1]);
            }
        });
    }

    btn.addEventListener('click', async () => {
        const nTargets = parseInt(document.getElementById('route-targets').value);
        const startLatVal = document.getElementById('route-start-lat').value;
        const startLonVal = document.getElementById('route-start-lon').value;
        
        let startCoords = null;
        if (startLatVal && startLonVal) {
            startCoords = [parseFloat(startLatVal), parseFloat(startLonVal)];
        }

        // Show loading state
        btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Generating Path...`;
        btn.disabled = true;

        try {
            const response = await fetch('/api/route', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    n_targets: nTargets,
                    location: startCoords
                })
            });

            if (!response.ok) throw new Error("Routing API error");
            const res = await response.json();

            // Clear previous routing lines/markers from map
            routerLayerGroup.clearLayers();
            if (routerPolyline) {
                routerMap.removeLayer(routerPolyline);
            }

            // Gap #10: draw faint background hotspot severity circles for context
            if (hotspotsData && hotspotsData.length > 0) {
                const maxFreqBg = Math.max(...hotspotsData.map(h => h.violation_count));
                hotspotsData.forEach(hs => {
                    const bgColor = getScoreColor(hs.hotspot_score);
                    const bgRatio = Math.log1p(hs.violation_count) / Math.log1p(maxFreqBg);
                    L.circleMarker(hs.location, {
                        radius: 5 + bgRatio * 16,
                        fillColor: bgColor, fillOpacity: 0.10,
                        color: bgColor, weight: 0.5, opacity: 0.20
                    }).addTo(routerLayerGroup);
                });
            }

            const route = res.ordered_route;
            const distance = res.total_distance_km;

            // Plot route markers and build sidebar list
            stepsList.innerHTML = '';
            const latlngs = [];

            // Add a visual route breadcrumb header showing start, intermediate stops, and destination
            const breadcrumbEl = document.createElement('div');
            breadcrumbEl.style.cssText = "padding: 12px; background: rgba(255, 255, 255, 0.03); border: 1px dashed var(--glass-border); border-radius: 8px; margin-bottom: 15px; font-size: 11px; color: var(--text-secondary); line-height: 1.5; box-shadow: 0 4px 10px rgba(0,0,0,0.2);";
            
            const startNode = route[0];
            const endNode = route[route.length - 1];
            const originLabel = startNode.cluster_id === -99 ? 'Custom Start' : getClosestLocationName(startNode.location[0], startNode.location[1]);
            const destLabel = getClosestLocationName(endNode.location[0], endNode.location[1]);
            
            breadcrumbEl.innerHTML = `
                <div style="font-weight: 700; color: var(--color-secondary); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; font-size: 10px;"><i class="fa-solid fa-compass"></i> Planned Patrol Path</div>
                <div style="margin-bottom: 4px;"><b>Origin:</b> ${originLabel.replace('Junction: ', '').replace('Near ', '')}</div>
                <div style="margin-bottom: 6px;"><b>Destination:</b> ${destLabel.replace('Junction: ', '').replace('Near ', '')}</div>
                <div style="display: flex; align-items: center; gap: 4px; overflow-x: auto; white-space: nowrap; padding-top: 4px; border-top: 1px solid rgba(255,255,255,0.05);">
                    <span style="background: var(--color-secondary); color: #fff; padding: 2px 5px; border-radius: 3px; font-size: 9px; font-weight: 600;">Start</span>
                    <span style="color: var(--text-secondary); font-size: 9px;">&rarr;</span>
                    ${route.slice(1, -1).map((_, i) => `<span style="background: rgba(255,255,255,0.08); padding: 2px 5px; border-radius: 3px; font-size: 9px; color: #fff;">S${i+1}</span>`).join('<span style="color: var(--text-secondary); font-size: 9px;">&rarr;</span>')}
                    ${route.length > 1 ? `<span style="color: var(--text-secondary); font-size: 9px;">&rarr;</span><span style="background: var(--color-primary); color: #fff; padding: 2px 5px; border-radius: 3px; font-size: 9px; font-weight: 600;">End</span>` : ''}
                </div>
            `;
            stepsList.appendChild(breadcrumbEl);

            route.forEach((hs, idx) => {
                const lat = hs.location[0];
                const lon = hs.location[1];
                latlngs.push([lat, lon]);

                // 1. Plot on map
                const color = hs.cluster_id === -99 ? 'var(--color-secondary)' : getScoreColor(hs.hotspot_score);
                const markerHtml = hs.cluster_id === -99 
                    ? `<div class="marker-pin-label" style="background-color: var(--color-secondary);"><i class="fa-solid fa-location-dot"></i></div>`
                    : `<div class="marker-pin-label" style="background-color: ${color};">${idx}</div>`;

                const marker = L.marker([lat, lon], {
                    icon: L.divIcon({
                        className: 'custom-div-icon',
                        html: markerHtml,
                        iconSize: [28, 28],
                        iconAnchor: [14, 14]
                    })
                }).addTo(routerLayerGroup);

                const name = getClosestLocationName(lat, lon);
                const popupText = hs.cluster_id === -99
                    ? `<b>Origin</b><br>${name}`
                    : `<b>Stop #${idx}: ${name}</b><br><span style="font-size: 10px; color: var(--text-secondary);">Cluster #${hs.cluster_id}</span><br>Score: ${Math.round(hs.hotspot_score)}<br>Violations: ${hs.violation_count}`;
                marker.bindPopup(popupText);

                // Calculate leg details from previous stop
                let legDetails = '';
                if (idx > 0) {
                    const prevLat = route[idx-1].location[0];
                    const prevLon = route[idx-1].location[1];
                    const legDist = getHaversineDistance(prevLat, prevLon, lat, lon);
                    const prevName = route[idx-1].cluster_id === -99 ? 'Patrol Origin' : getClosestLocationName(prevLat, prevLon);
                    legDetails = `
                        <div style="margin-top: 6px; padding: 6px 8px; background: rgba(0, 229, 255, 0.03); border: 1px solid rgba(0, 229, 255, 0.1); border-radius: 4px; font-size: 10px; color: #00e5ff; font-weight: 500;">
                            <i class="fa-solid fa-arrows-left-right"></i> <b>Leg ${idx}:</b> ${prevName.replace('Near ', '').replace('Junction: ', '')} &rarr; ${name.replace('Near ', '').replace('Junction: ', '')} <b>(${legDist.toFixed(2)} km)</b>
                        </div>
                    `;
                }

                // 2. Add to sidebar list
                const step = document.createElement('div');
                step.className = 'patrol-step';
                step.innerHTML = `
                    <div class="step-num" style="background-color: ${color}">${hs.cluster_id === -99 ? 'O' : idx}</div>
                    <div class="patrol-step-info" style="width: 100%;">
                        <span class="patrol-step-name" style="font-weight: 600;">${hs.cluster_id === -99 ? 'Patrol Origin' : name}</span>
                        <span class="patrol-step-score">${hs.cluster_id === -99 ? 'Start Location' : 'Cluster #' + hs.cluster_id + ' | Score: ' + Math.round(hs.hotspot_score) + ' | ' + hs.violation_count + ' cases'}</span>
                        <span class="patrol-step-score" style="color: var(--text-secondary); font-size: 9px;"><i class="fa-solid fa-map-pin"></i> Division: ${hs.police_station}</span>
                        ${legDetails}
                    </div>
                `;
                
                step.addEventListener('click', () => {
                    routerMap.setView([lat, lon], 16);
                    marker.openPopup();
                });
                
                stepsList.appendChild(step);
            });

            // Draw connecting routing lines
            routerPolyline = L.polyline(latlngs, {
                color: 'var(--color-secondary)',
                weight: 3,
                dashArray: '5, 10',
                opacity: 0.8
            }).addTo(routerMap);

            // Fit map view to route bounds
            if (latlngs.length > 0) {
                routerMap.fitBounds(routerPolyline.getBounds(), { padding: [50, 50] });
            }

            // Update metrics panel
            document.getElementById('route-total-distance').textContent = `${distance} km`;
            document.getElementById('route-stops-count').textContent = route.length - (startCoords ? 1 : 0);

            // Show panel and configure export button
            summaryPanel.style.display = 'block';

            // Gap #16: reveal export button and bind the route data
            const exportBtn = document.getElementById('btn-export-route');
            if (exportBtn) {
                exportBtn.style.display = 'flex';
                exportBtn.onclick = () => exportPatrolRoute(route, distance);
            }

        } catch (err) {
            console.error("Error generating patrol route:", err);
        } finally {
            btn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Generate Route`;
            btn.disabled = false;
        }
    });
}

// 6. Fetch Locations List for selector dropdown
async function fetchLocations() {
    try {
        const response = await fetch('/api/locations');
        if (!response.ok) throw new Error("Locations not loaded");
        
        locationsData = await response.json();
        populateAllDropdowns();
    } catch (err) {
        console.error("Error loading locations:", err);
        const select = document.getElementById('pred-location-select');
        if (select) {
            select.innerHTML = '<option value="">-- Failed to load locations --</option>';
        }
    }
}

function populateAllDropdowns() {
    const predSelect = document.getElementById('pred-location-select');
    const routeSelect = document.getElementById('route-start-select');
    
    const fillDropdown = (select) => {
        if (!select) return;
        select.innerHTML = select.id === 'route-start-select' 
            ? '<option value="">-- Start at Highest-Impact Hotspot --</option>' 
            : '<option value="">-- Choose a Location --</option>';

        // Add Junctions Group
        if (locationsData.junctions && locationsData.junctions.length > 0) {
            const group = document.createElement('optgroup');
            group.label = "Traffic Junctions";
            locationsData.junctions.forEach((item, idx) => {
                const opt = document.createElement('option');
                opt.value = `junction_${idx}`;
                opt.textContent = item.name.replace('Junction: ', '');
                group.appendChild(opt);
            });
            select.appendChild(group);
        }

        // Add Hotspots Group
        if (locationsData.hotspots && locationsData.hotspots.length > 0) {
            const group = document.createElement('optgroup');
            group.label = "Illegal Parking Hotspots";
            locationsData.hotspots.forEach((item, idx) => {
                const opt = document.createElement('option');
                opt.value = `hotspot_${idx}`;
                opt.textContent = item.name;
                group.appendChild(opt);
            });
            select.appendChild(group);
        }
    };

    fillDropdown(predSelect);
    fillDropdown(routeSelect);
}
