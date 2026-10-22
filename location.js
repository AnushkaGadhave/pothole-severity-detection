/**
 * ==============================================================================
 * CRACKSIGHT — GEOSPATIAL LOCATION & INTERACTIVE MAPPING ENGINE (location.js)
 * Place Search · OpenStreetMap Leaflet · Reverse Geocoding · Spatial Dashboard
 * ==============================================================================
 */

import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// ----------------------------------------------------------------------------
// 1. Constants & Caching
// ----------------------------------------------------------------------------
const NOMINATIM_SEARCH_ENDPOINT = 'https://nominatim.openstreetmap.org/search';
const NOMINATIM_REVERSE_ENDPOINT = 'https://nominatim.openstreetmap.org/reverse';
const DEFAULT_TILE_LAYER = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
const TILE_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors';

// Default initial map centers (Fallback coordinates: Mumbai Municipal Road Network / Central)
const DEFAULT_COORDS = [19.0760, 72.8777];
const DEFAULT_ZOOM = 13;

// In-memory caches to prevent redundant network calls
const searchCache = new Map();
const reverseGeocodeCache = new Map();

// Singletons for map instances
let analysisMap = null;
let analysisMarker = null;
let dashboardMap = null;
let dashboardMarkersLayer = null;
let resultsMiniMap = null;
let resultsMiniMarker = null;

// Search debouncing and abort controllers
let searchDebounceTimer = null;
let searchAbortController = null;
let reverseDebounceTimer = null;
let reverseAbortController = null;

// Cache of current filtered records on dashboard for CSV export
let lastFilteredGeotaggedRecords = [];

// Safe Toast Notification helper
function toast(msg, type = 'info') {
  if (typeof window !== 'undefined' && typeof window.showToast === 'function') {
    window.showToast(msg, type);
  } else {
    console.log(`[Toast ${type}]`, msg);
  }
}

// ----------------------------------------------------------------------------
// 2. Custom SVG Severity Marker Icons
// ----------------------------------------------------------------------------
export function createPotholeMarkerIcon(tier = 'Medium', status = 'Reported', isSelected = false) {
  const colors = {
    High: '#B42368',   // Berry Pink
    Medium: '#F59E0B', // Amber
    Low: '#10B981',    // Emerald Teal
    None: '#6B7280'    // Slate
  };

  const pinColor = colors[tier] || colors.Medium;
  const size = isSelected ? 38 : 30;
  const pinHeight = Math.round(size * 1.28);

  const ringGlow = isSelected
    ? `<circle cx="16" cy="15" r="14" fill="none" stroke="${pinColor}" stroke-width="2.5" stroke-dasharray="3 3"/>`
    : '';

  // Small indicator dot for repair status
  const statusColors = {
    'Reported': '#EF4444',
    'Under Review': '#F59E0B',
    'Repair Scheduled': '#3B82F6',
    'Repaired': '#10B981',
    'Reopened': '#8B5CF6'
  };
  const statusDotColor = statusColors[status] || '#EF4444';

  const html = `
    <div class="cracksight-marker-pin ${isSelected ? 'is-selected' : ''}" style="width:${size}px; height:${pinHeight}px;">
      <svg viewBox="0 0 32 40" width="${size}" height="${pinHeight}" style="filter: drop-shadow(0 3px 6px rgba(0,0,0,0.3));">
        ${ringGlow}
        <path d="M16 2 C9 2 3 8 3 15 C3 24 16 38 16 38 C16 38 29 24 29 15 C29 8 23 2 16 2 Z" fill="${pinColor}" stroke="#FFFFFF" stroke-width="2"/>
        <circle cx="16" cy="15" r="5" fill="#FFFFFF"/>
        <circle cx="16" cy="15" r="3" fill="${statusDotColor}"/>
      </svg>
    </div>
  `;

  return L.divIcon({
    className: 'cracksight-custom-div-icon',
    html: html,
    iconSize: [size, pinHeight],
    iconAnchor: [size / 2, pinHeight],
    popupAnchor: [0, -pinHeight + 6]
  });
}

// ----------------------------------------------------------------------------
// 3. Geocoding & Reverse Geocoding Services
// ----------------------------------------------------------------------------

/**
 * Search places and roads using Nominatim OpenStreetMap API.
 * Rate-limited and debounced; handles network errors gracefully.
 * Never sends image contents; sends only search query text.
 */
export async function searchPlaces(query) {
  const cleanQ = (query || '').trim();
  if (cleanQ.length < 2) return [];

  if (searchCache.has(cleanQ.toLowerCase())) {
    return searchCache.get(cleanQ.toLowerCase());
  }

  if (searchAbortController) {
    searchAbortController.abort();
  }
  searchAbortController = new AbortController();

  try {
    const url = `${NOMINATIM_SEARCH_ENDPOINT}?format=json&q=${encodeURIComponent(cleanQ)}&addressdetails=1&limit=6`;
    const res = await fetch(url, {
      signal: searchAbortController.signal,
      headers: {
        'Accept': 'application/json'
      }
    });

    if (!res.ok) throw new Error(`Geocoding HTTP ${res.status}`);
    const rawItems = await res.json();

    const formatted = rawItems.map(item => {
      const addr = item.address || {};
      const road = addr.road || addr.street || addr.pedestrian || addr.suburb || '';
      const locality = addr.city || addr.town || addr.village || addr.county || '';
      const state = addr.state || '';
      const country = addr.country || '';

      const mainTitle = road ? (locality ? `${road}, ${locality}` : road) : item.display_name.split(',')[0];
      const subtitle = [locality, state, country].filter(Boolean).join(', ');

      return {
        id: item.place_id,
        displayName: item.display_name,
        mainTitle: mainTitle || item.display_name,
        subtitle: subtitle || item.display_name,
        lat: parseFloat(item.lat),
        lng: parseFloat(item.lon)
      };
    });

    searchCache.set(cleanQ.toLowerCase(), formatted);
    return formatted;
  } catch (err) {
    if (err.name === 'AbortError') return [];
    console.warn('Geocoding search service exception:', err);
    return [];
  }
}

/**
 * Reverse geocode numeric coordinates to a readable street/locality address.
 * Never sends image data; sends only numeric coordinates.
 */
export async function reverseGeocode(lat, lng) {
  if (typeof lat !== 'number' || typeof lng !== 'number' || isNaN(lat) || isNaN(lng)) {
    return null;
  }
  if (lat < -90 || lat > 90 || lng < -180 || lng > 180) {
    return null;
  }

  const cacheKey = `${lat.toFixed(4)},${lng.toFixed(4)}`;
  if (reverseGeocodeCache.has(cacheKey)) {
    return reverseGeocodeCache.get(cacheKey);
  }

  if (reverseAbortController) {
    reverseAbortController.abort();
  }
  reverseAbortController = new AbortController();

  try {
    const url = `${NOMINATIM_REVERSE_ENDPOINT}?format=json&lat=${lat}&lon=${lng}&zoom=18&addressdetails=1`;
    const res = await fetch(url, {
      signal: reverseAbortController.signal,
      headers: {
        'Accept': 'application/json'
      }
    });

    if (!res.ok) throw new Error(`Reverse geocoding HTTP ${res.status}`);
    const data = await res.json();

    const addr = data.address || {};
    const road = addr.road || addr.street || addr.pedestrian || addr.suburb || '';
    const locality = addr.city || addr.town || addr.village || addr.county || '';
    const state = addr.state || '';
    const country = addr.country || '';

    let shortAddress = road ? (locality ? `${road}, ${locality}` : road) : (data.display_name ? data.display_name.split(',').slice(0, 3).join(',') : '');
    if (!shortAddress) {
      shortAddress = `${lat.toFixed(5)}°, ${lng.toFixed(5)}°`;
    }

    const result = {
      displayName: data.display_name || shortAddress,
      shortAddress: shortAddress,
      fullAddress: data.display_name || shortAddress,
      lat: lat,
      lng: lng
    };

    reverseGeocodeCache.set(cacheKey, result);
    return result;
  } catch (err) {
    if (err.name === 'AbortError') return null;
    console.warn('Reverse geocoding service exception:', err);
    return {
      displayName: `${lat.toFixed(5)}°, ${lng.toFixed(5)}°`,
      shortAddress: `${lat.toFixed(5)}°, ${lng.toFixed(5)}°`,
      fullAddress: `Coordinates: ${lat.toFixed(6)}, ${lng.toFixed(6)}`,
      lat: lat,
      lng: lng
    };
  }
}

// ----------------------------------------------------------------------------
// 4. Analysis Location Picker Component
// ----------------------------------------------------------------------------

export function initAnalysisLocationMap(AppState, onLocationChanged) {
  const container = document.getElementById('analysisLocationMap');
  const searchInput = document.getElementById('locationSearchInput');
  const suggestionsList = document.getElementById('locationSuggestionsList');
  const searchSpinner = document.getElementById('locationSearchSpinner');
  const myLocationBtn = document.getElementById('useMyLocationBtn');
  const clearBtn = document.getElementById('clearLocationBtn');
  const removeBtn = document.getElementById('removeLocationTagBtn');

  if (!container) return;

  // 1. Initialize Leaflet Map if not already created
  if (!analysisMap) {
    analysisMap = L.map(container, {
      zoomControl: true,
      attributionControl: true
    }).setView(DEFAULT_COORDS, DEFAULT_ZOOM);

    L.tileLayer(DEFAULT_TILE_LAYER, {
      attribution: TILE_ATTRIBUTION,
      maxZoom: 19
    }).addTo(analysisMap);

    // Click anywhere on the map to pick coordinates
    analysisMap.on('click', async (e) => {
      const lat = Number(e.latlng.lat.toFixed(6));
      const lng = Number(e.latlng.lng.toFixed(6));
      setAnalysisLocation(lat, lng, `${lat.toFixed(5)}°, ${lng.toFixed(5)}°`, 'Map Click', AppState, onLocationChanged, true);
    });

    // Ensure Leaflet recalculates viewport dimensions across wide layout
    setTimeout(() => {
      if (analysisMap) analysisMap.invalidateSize();
    }, 100);
    setTimeout(() => {
      if (analysisMap) analysisMap.invalidateSize();
    }, 300);

    window.addEventListener('resize', () => {
      invalidateAnalysisMapSize();
    });
  } else {
    invalidateAnalysisMapSize();
  }

  // 2. Setup Autocomplete Place Search Input
  if (searchInput && suggestionsList) {
    let activeSuggestionIndex = -1;

    searchInput.addEventListener('input', (e) => {
      const q = e.target.value.trim();
      clearTimeout(searchDebounceTimer);
      activeSuggestionIndex = -1;

      if (q.length < 2) {
        suggestionsList.style.display = 'none';
        suggestionsList.innerHTML = '';
        if (searchSpinner) searchSpinner.style.display = 'none';
        return;
      }

      if (searchSpinner) searchSpinner.style.display = 'block';

      searchDebounceTimer = setTimeout(async () => {
        const results = await searchPlaces(q);
        if (searchSpinner) searchSpinner.style.display = 'none';

        if (results.length === 0) {
          suggestionsList.innerHTML = `
            <li class="location-suggestion-empty" role="option">
              No matching roads or landmarks found
            </li>
          `;
          suggestionsList.style.display = 'block';
          return;
        }

        suggestionsList.innerHTML = results.map((item, idx) => `
          <li class="location-suggestion-item" data-index="${idx}" role="option" tabindex="0">
            <svg class="suggestion-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
              <circle cx="12" cy="10" r="3"></circle>
            </svg>
            <div class="suggestion-text-wrap">
              <span class="suggestion-title">${escapeHtml(item.mainTitle)}</span>
              <span class="suggestion-sub">${escapeHtml(item.subtitle)}</span>
            </div>
          </li>
        `).join('');

        suggestionsList.style.display = 'block';

        // Bind clicks on suggestions
        suggestionsList.querySelectorAll('.location-suggestion-item').forEach(el => {
          el.addEventListener('click', () => {
            const idx = parseInt(el.getAttribute('data-index'), 10);
            const chosen = results[idx];
            if (chosen) {
              searchInput.value = chosen.mainTitle;
              suggestionsList.style.display = 'none';
              setAnalysisLocation(chosen.lat, chosen.lng, chosen.displayName, 'Search', AppState, onLocationChanged, false);
            }
          });
        });
      }, 350);
    });

    // Keyboard navigation inside dropdown
    searchInput.addEventListener('keydown', (e) => {
      const items = suggestionsList.querySelectorAll('.location-suggestion-item');
      if (suggestionsList.style.display === 'none' || items.length === 0) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        activeSuggestionIndex = (activeSuggestionIndex + 1) % items.length;
        updateActiveSuggestion(items, activeSuggestionIndex);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        activeSuggestionIndex = (activeSuggestionIndex - 1 + items.length) % items.length;
        updateActiveSuggestion(items, activeSuggestionIndex);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (activeSuggestionIndex >= 0 && items[activeSuggestionIndex]) {
          items[activeSuggestionIndex].click();
        }
      } else if (e.key === 'Escape') {
        suggestionsList.style.display = 'none';
      }
    });

    // Close dropdown on outside click
    document.addEventListener('click', (e) => {
      if (!searchInput.contains(e.target) && !suggestionsList.contains(e.target)) {
        suggestionsList.style.display = 'none';
      }
    });
  }

  // 3. "Use My Location" Geolocation Action
  if (myLocationBtn) {
    myLocationBtn.addEventListener('click', () => {
      if (!navigator.geolocation) {
        toast('Geolocation is not supported by your browser.', 'error');
        return;
      }

      myLocationBtn.disabled = true;
      const originalText = myLocationBtn.innerHTML;
      myLocationBtn.innerHTML = `
        <span class="status-dot-pulse" style="width:6px; height:6px;"></span>
        <span>Locating...</span>
      `;

      navigator.geolocation.getCurrentPosition(
        async (pos) => {
          myLocationBtn.disabled = false;
          myLocationBtn.innerHTML = originalText;
          const lat = Number(pos.coords.latitude.toFixed(6));
          const lng = Number(pos.coords.longitude.toFixed(6));
          setAnalysisLocation(lat, lng, 'My GPS Location', 'GPS', AppState, onLocationChanged, true);
          toast(`Acquired GPS coordinates: ${lat}, ${lng}`);
        },
        (err) => {
          myLocationBtn.disabled = false;
          myLocationBtn.innerHTML = originalText;
          console.warn('Browser geolocation denied or unavailable:', err);
          let msg = 'Location permission denied. You can pick a point on the map or search an address.';
          if (err.code === 2) msg = 'Location unavailable from device sensors.';
          if (err.code === 3) msg = 'Geolocation request timed out.';
          toast(msg, 'error');
        },
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
      );
    });
  }

  // 4. Clear Location Action
  const clearHandler = () => {
    clearAnalysisLocation(AppState, onLocationChanged);
  };
  if (clearBtn) clearBtn.addEventListener('click', clearHandler);
  if (removeBtn) removeBtn.addEventListener('click', clearHandler);
}

function updateActiveSuggestion(items, activeIdx) {
  items.forEach((item, idx) => {
    if (idx === activeIdx) {
      item.classList.add('is-active');
      item.scrollIntoView({ block: 'nearest' });
    } else {
      item.classList.remove('is-active');
    }
  });
}

/**
 * Set and synchronize selected location in the analysis workflow.
 */
export async function setAnalysisLocation(lat, lng, displayLabel, source, AppState, onLocationChanged, shouldReverseGeocode = false) {
  if (typeof lat !== 'number' || typeof lng !== 'number' || isNaN(lat) || isNaN(lng)) return;
  if (lat < -90 || lat > 90 || lng < -180 || lng > 180) return;

  // Place or move draggable marker
  if (analysisMap) {
    if (analysisMarker) {
      analysisMarker.setLatLng([lat, lng]);
    } else {
      analysisMarker = L.marker([lat, lng], {
        draggable: true,
        icon: createPotholeMarkerIcon('High', 'Reported', true)
      }).addTo(analysisMap);

      // Handle dragging marker to fine-tune position
      analysisMarker.on('dragend', async (e) => {
        const newPos = e.target.getLatLng();
        const nLat = Number(newPos.lat.toFixed(6));
        const nLng = Number(newPos.lng.toFixed(6));
        setAnalysisLocation(nLat, nLng, `${nLat}, ${nLng}`, 'Manual Pin', AppState, onLocationChanged, true);
      });
    }

    analysisMap.setView([lat, lng], Math.max(analysisMap.getZoom(), 15), { animate: true });
  }

  // Update AppState
  AppState.selectedLocation = {
    lat: lat,
    lng: lng,
    address: displayLabel || `${lat.toFixed(5)}°, ${lng.toFixed(5)}°`,
    source: source || 'map'
  };

  updateLocationSummaryUI(AppState.selectedLocation);

  // If reverse geocoding is requested (e.g., from map click, drag, or GPS)
  if (shouldReverseGeocode) {
    const summaryAddr = document.getElementById('selectedAddressText');
    if (summaryAddr) summaryAddr.textContent = 'Resolving street address...';

    const geoData = await reverseGeocode(lat, lng);
    if (geoData && geoData.shortAddress) {
      AppState.selectedLocation.address = geoData.shortAddress;
      AppState.selectedLocation.fullAddress = geoData.fullAddress;
      const sInput = document.getElementById('locationSearchInput');
      if (sInput) {
        sInput.value = geoData.shortAddress;
      }
      updateLocationSummaryUI(AppState.selectedLocation);
    }
  }

  if (typeof onLocationChanged === 'function') {
    onLocationChanged(AppState.selectedLocation);
  }
}

/**
 * Clear the selected location from analysis.
 */
export function clearAnalysisLocation(AppState, onLocationChanged) {
  AppState.selectedLocation = null;

  if (analysisMarker && analysisMap) {
    analysisMap.removeLayer(analysisMarker);
    analysisMarker = null;
  }

  const sInput = document.getElementById('locationSearchInput');
  if (sInput) sInput.value = '';

  const summary = document.getElementById('selectedLocationSummary');
  if (summary) summary.style.display = 'none';

  const clearBtn = document.getElementById('clearLocationBtn');
  if (clearBtn) clearBtn.style.display = 'none';

  if (typeof onLocationChanged === 'function') {
    onLocationChanged(null);
  }
}

function updateLocationSummaryUI(loc) {
  const summaryBox = document.getElementById('selectedLocationSummary');
  const addrText = document.getElementById('selectedAddressText');
  const latVal = document.getElementById('selectedLatVal');
  const lngVal = document.getElementById('selectedLngVal');
  const sourceTag = document.getElementById('selectedSourceTag');
  const clearBtn = document.getElementById('clearLocationBtn');

  if (!loc) {
    if (summaryBox) summaryBox.style.display = 'none';
    if (clearBtn) clearBtn.style.display = 'none';
    return;
  }

  if (summaryBox) summaryBox.style.display = 'block';
  if (clearBtn) clearBtn.style.display = 'inline-flex';
  if (addrText) addrText.textContent = loc.address;
  if (latVal) latVal.textContent = loc.lat.toFixed(6);
  if (lngVal) lngVal.textContent = loc.lng.toFixed(6);
  if (sourceTag) sourceTag.textContent = (loc.source || 'TAGGED').toUpperCase();
}

/**
 * Triggers Leaflet to recalculate its viewport size when view changes or window resizes.
 */
export function invalidateAnalysisMapSize() {
  if (analysisMap) {
    try {
      analysisMap.invalidateSize();
    } catch (_) {}
    setTimeout(() => {
      if (analysisMap) {
        try {
          analysisMap.invalidateSize();
        } catch (_) {}
      }
    }, 150);
    setTimeout(() => {
      if (analysisMap) {
        try {
          analysisMap.invalidateSize();
        } catch (_) {}
      }
    }, 350);
  }
}

// ----------------------------------------------------------------------------
// 5. Results Studio Location Summary & Mini-Map
// ----------------------------------------------------------------------------

export function renderResultsLocationSummary(activeAnalysis, onOpenInMap) {
  const card = document.getElementById('resultsLocationCard');
  const noLocNotice = document.getElementById('resultsNoLocationNotice');
  const addrDisplay = document.getElementById('resultsAddressDisplay');
  const coordsDisplay = document.getElementById('resultsCoordsDisplay');
  const sourceTag = document.getElementById('resultsLocationSourceTag');
  const viewOnMapBtn = document.getElementById('resultsViewOnMapBtn');
  const miniMapContainer = document.getElementById('resultsMiniMap');

  if (!activeAnalysis || !activeAnalysis.location || typeof activeAnalysis.location.lat !== 'number' || typeof activeAnalysis.location.lng !== 'number') {
    if (card) card.style.display = 'none';
    if (noLocNotice) noLocNotice.style.display = 'flex';
    return;
  }

  const loc = activeAnalysis.location;
  if (card) card.style.display = 'block';
  if (noLocNotice) noLocNotice.style.display = 'none';

  if (addrDisplay) addrDisplay.textContent = loc.address || 'Inspection Location';
  if (coordsDisplay) coordsDisplay.textContent = `${loc.lat.toFixed(5)}° N, ${loc.lng.toFixed(5)}° E`;
  if (sourceTag) sourceTag.textContent = loc.source ? `Tagged via ${loc.source}` : 'Geotagged';

  if (viewOnMapBtn) {
    viewOnMapBtn.onclick = () => {
      if (typeof onOpenInMap === 'function') {
        onOpenInMap(activeAnalysis.id);
      }
    };
  }

  // Mini Map inside Results Studio
  if (miniMapContainer) {
    if (!resultsMiniMap) {
      resultsMiniMap = L.map(miniMapContainer, {
        zoomControl: false,
        attributionControl: false,
        dragging: false,
        scrollWheelZoom: false,
        doubleClickZoom: false,
        touchZoom: false
      }).setView([loc.lat, loc.lng], 15);

      L.tileLayer(DEFAULT_TILE_LAYER, { maxZoom: 19 }).addTo(resultsMiniMap);
      resultsMiniMarker = L.marker([loc.lat, loc.lng], {
        icon: createPotholeMarkerIcon(activeAnalysis.tier || 'High', activeAnalysis.status || 'Reported')
      }).addTo(resultsMiniMap);
    } else {
      resultsMiniMap.setView([loc.lat, loc.lng], 15);
      if (resultsMiniMarker) {
        resultsMiniMarker.setLatLng([loc.lat, loc.lng]);
        resultsMiniMarker.setIcon(createPotholeMarkerIcon(activeAnalysis.tier || 'High', activeAnalysis.status || 'Reported'));
      }
      setTimeout(() => resultsMiniMap.invalidateSize(), 150);
    }
  }
}

// ----------------------------------------------------------------------------
// 6. Dedicated Geospatial Inspection Dashboard
// ----------------------------------------------------------------------------

export function initGeospatialDashboard(AppState, callbacks = {}) {
  const mapContainer = document.getElementById('dashboardGeospatialMap');
  const searchInput = document.getElementById('mapSearchInput');
  const severityFilter = document.getElementById('mapSeverityFilter');
  const statusFilter = document.getElementById('mapStatusFilter');
  const dateFilter = document.getElementById('mapDateFilter');
  const exportCsvBtn = document.getElementById('mapExportCsvBtn');

  if (!mapContainer) return;

  // Initialize main dashboard map if needed
  if (!dashboardMap) {
    dashboardMap = L.map(mapContainer, {
      zoomControl: true,
      attributionControl: true
    }).setView(DEFAULT_COORDS, DEFAULT_ZOOM);

    L.tileLayer(DEFAULT_TILE_LAYER, {
      attribution: TILE_ATTRIBUTION,
      maxZoom: 19
    }).addTo(dashboardMap);

    dashboardMarkersLayer = L.layerGroup().addTo(dashboardMap);
  }

  // Search and Filter Events
  if (searchInput) {
    searchInput.addEventListener('input', () => renderDashboardMap(AppState, callbacks));
  }
  if (severityFilter) {
    severityFilter.addEventListener('change', () => renderDashboardMap(AppState, callbacks));
  }
  if (statusFilter) {
    statusFilter.addEventListener('change', () => renderDashboardMap(AppState, callbacks));
  }
  if (dateFilter) {
    dateFilter.addEventListener('change', () => renderDashboardMap(AppState, callbacks));
  }
  if (exportCsvBtn) {
    exportCsvBtn.addEventListener('click', () => {
      exportFilteredGeotaggedCsv(lastFilteredGeotaggedRecords);
    });
  }

  renderDashboardMap(AppState, callbacks);
}

/**
 * Filter records and render markers + sidebar list in the Geospatial Dashboard.
 */
export function renderDashboardMap(AppState, callbacks = {}) {
  if (!dashboardMap || !dashboardMarkersLayer) return;

  const searchInput = document.getElementById('mapSearchInput');
  const severityFilter = document.getElementById('mapSeverityFilter');
  const statusFilter = document.getElementById('mapStatusFilter');
  const dateFilter = document.getElementById('mapDateFilter');
  const countBadge = document.getElementById('mapGeotaggedCountBadge');
  const emptyOverlay = document.getElementById('mapEmptyStateOverlay');
  const recordsListContainer = document.getElementById('mapRecordsList');

  const query = (searchInput ? searchInput.value : '').trim().toLowerCase();
  const severity = severityFilter ? severityFilter.value : 'all';
  const status = statusFilter ? statusFilter.value : 'all';
  const dateRange = dateFilter ? dateFilter.value : 'all';

  dashboardMarkersLayer.clearLayers();

  // Find all genuine records with valid coordinates
  const geotaggedRecords = AppState.history.filter(item => {
    return (
      item &&
      item.location &&
      typeof item.location.lat === 'number' &&
      typeof item.location.lng === 'number' &&
      !isNaN(item.location.lat) &&
      !isNaN(item.location.lng) &&
      item.location.lat >= -90 &&
      item.location.lat <= 90 &&
      item.location.lng >= -180 &&
      item.location.lng <= 180
    );
  });

  const totalGeotagged = geotaggedRecords.length;
  const totalAnalyses = AppState.history.length;

  // Filter according to user controls
  const filtered = geotaggedRecords.filter(item => {
    // Search
    if (query) {
      const matchId = (item.id || '').toLowerCase().includes(query);
      const matchAddr = (item.location?.address || '').toLowerCase().includes(query);
      const matchFile = (item.filename || '').toLowerCase().includes(query);
      if (!matchId && !matchAddr && !matchFile) return false;
    }
    // Severity
    if (severity !== 'all' && (item.tier || '').toLowerCase() !== severity.toLowerCase()) {
      return false;
    }
    // Status
    const itemStatus = item.status || 'Reported';
    if (status !== 'all' && itemStatus.toLowerCase() !== status.toLowerCase()) {
      return false;
    }
    // Date Range
    if (dateRange !== 'all') {
      const itemDate = new Date(item.timestamp || 0);
      const now = new Date();
      if (dateRange === 'today') {
        if (itemDate.toDateString() !== now.toDateString()) return false;
      } else if (dateRange === '7days') {
        const diffDays = (now - itemDate) / (1000 * 60 * 60 * 24);
        if (diffDays > 7) return false;
      } else if (dateRange === '30days') {
        const diffDays = (now - itemDate) / (1000 * 60 * 60 * 24);
        if (diffDays > 30) return false;
      }
    }
    return true;
  });

  lastFilteredGeotaggedRecords = filtered;

  // Update Count Badge
  if (countBadge) {
    countBadge.textContent = `${filtered.length} Geotagged Site${filtered.length === 1 ? '' : 's'} (${totalGeotagged} of ${totalAnalyses} records)`;
  }

  // Update Dashboard Overview Stat if element exists
  const statGeotagged = document.getElementById('statGeotaggedCount');
  if (statGeotagged) {
    statGeotagged.textContent = totalGeotagged.toLocaleString();
  }

  // Empty State Handling
  if (filtered.length === 0) {
    if (emptyOverlay) emptyOverlay.style.display = 'flex';
    if (recordsListContainer) {
      const unpinnedCount = totalAnalyses - totalGeotagged;
      recordsListContainer.innerHTML = `
        ${unpinnedCount > 0 ? `
          <div class="map-unpinned-notice">
            ℹ️ ${unpinnedCount} inspection${unpinnedCount === 1 ? '' : 's'} recorded without GPS coordinates (accessible in History).
          </div>
        ` : ''}
        <div style="text-align: center; padding: 30px 16px; color: var(--color-text-muted); font-size: 0.8125rem;">
          No geotagged inspections match current filters.
        </div>
      `;
    }
    return;
  }

  if (emptyOverlay) emptyOverlay.style.display = 'none';

  const markerBounds = [];

  // Plot each geotagged inspection on map
  filtered.forEach(item => {
    const lat = item.location.lat;
    const lng = item.location.lng;
    const tier = item.tier || 'Medium';
    const repStatus = item.status || 'Reported';
    markerBounds.push([lat, lng]);

    const marker = L.marker([lat, lng], {
      icon: createPotholeMarkerIcon(tier, repStatus)
    });

    const popupHtml = `
      <div class="cracksight-map-popup">
        <div class="popup-top">
          <span class="tier-badge ${tier.toLowerCase()}">${tier} Severity</span>
          <span class="popup-status-pill status-${repStatus.toLowerCase().replace(/\s+/g, '-')}">${escapeHtml(repStatus)}</span>
        </div>
        ${item.source_img ? `
          <div class="popup-thumb-wrap">
            <img src="${item.source_img}" alt="Inspection Thumbnail" class="popup-thumb-img" />
          </div>
        ` : ''}
        <div class="popup-address-text">${escapeHtml(item.location.address || 'Tagged Location')}</div>
        <div class="popup-meta-row">
          <span>Area: <strong>${(item.mask_area_pixels || 0).toLocaleString()} px</strong></span>
          <span>${(item.timestamp || '').slice(0, 10)}</span>
        </div>
        <div class="popup-status-action">
          <label style="font-size:0.6875rem; font-weight:700;">UPDATE REPAIR STATUS:</label>
          <select class="popup-status-select" data-id="${item.id}">
            <option value="Reported" ${repStatus === 'Reported' ? 'selected' : ''}>Reported</option>
            <option value="Under Review" ${repStatus === 'Under Review' ? 'selected' : ''}>Under Review</option>
            <option value="Repair Scheduled" ${repStatus === 'Repair Scheduled' ? 'selected' : ''}>Repair Scheduled</option>
            <option value="Repaired" ${repStatus === 'Repaired' ? 'selected' : ''}>Repaired</option>
            <option value="Reopened" ${repStatus === 'Reopened' ? 'selected' : ''}>Reopened</option>
          </select>
        </div>
        <button class="btn btn-primary btn-xs popup-open-btn" data-id="${item.id}" style="width:100%; justify-content:center; margin-top:8px;">
          Inspect in Studio →
        </button>
      </div>
    `;

    marker.bindPopup(popupHtml, { maxWidth: 280, className: 'cracksight-custom-popup' });

    // Handle popup events (status change & inspect button)
    marker.on('popupopen', (e) => {
      const popupEl = e.popup.getElement();
      if (!popupEl) return;

      const sel = popupEl.querySelector('.popup-status-select');
      if (sel) {
        sel.addEventListener('change', (ev) => {
          const newStatus = ev.target.value;
          item.status = newStatus;
          if (typeof callbacks.onStatusChanged === 'function') {
            callbacks.onStatusChanged(item.id, newStatus);
          }
          marker.setIcon(createPotholeMarkerIcon(tier, newStatus));

          // Update popup pill
          const pill = popupEl.querySelector('.popup-status-pill');
          if (pill) {
            pill.className = `popup-status-pill status-${newStatus.toLowerCase().replace(/\s+/g, '-')}`;
            pill.textContent = newStatus;
          }

          // Update corresponding sidebar card pill if present
          const cardPill = document.querySelector(`.map-record-item-card[data-id="${item.id}"] .map-status-pill`);
          if (cardPill) {
            cardPill.className = `map-status-pill status-${newStatus.toLowerCase().replace(/\s+/g, '-')}`;
            cardPill.textContent = newStatus;
          }
        });
      }

      const openBtn = popupEl.querySelector('.popup-open-btn');
      if (openBtn) {
        openBtn.addEventListener('click', () => {
          if (typeof callbacks.onOpenInspection === 'function') {
            callbacks.onOpenInspection(item.id);
          }
        });
      }
    });

    dashboardMarkersLayer.addLayer(marker);
    item._marker = marker; // Keep reference to focus on marker from sidebar card
  });

  // Fit bounds if markers exist
  if (markerBounds.length > 1) {
    dashboardMap.fitBounds(markerBounds, { padding: [40, 40], maxZoom: 16 });
  } else if (markerBounds.length === 1) {
    dashboardMap.setView(markerBounds[0], 15);
  }

  // Render left sidebar list of cards
  if (recordsListContainer) {
    const unpinnedCount = totalAnalyses - totalGeotagged;
    const unpinnedNotice = unpinnedCount > 0 ? `
      <div class="map-unpinned-notice">
        ℹ️ ${unpinnedCount} inspection${unpinnedCount === 1 ? '' : 's'} recorded without GPS coordinates (accessible in History).
      </div>
    ` : '';

    recordsListContainer.innerHTML = unpinnedNotice + filtered.map(item => {
      const tier = item.tier || 'Medium';
      const repStatus = item.status || 'Reported';
      return `
        <div class="map-record-item-card" data-id="${item.id}">
          <div class="map-card-top">
            <span class="tier-badge ${tier.toLowerCase()}">${tier}</span>
            <span class="map-status-pill status-${repStatus.toLowerCase().replace(/\s+/g, '-')}">${escapeHtml(repStatus)}</span>
          </div>
          <div class="map-card-address">${escapeHtml(item.location.address || 'Road Location')}</div>
          <div class="map-card-coords">${item.location.lat.toFixed(4)}° N, ${item.location.lng.toFixed(4)}° E</div>
          <div class="map-card-footer">
            <span>${(item.timestamp || '').slice(0, 10)}</span>
            <span style="font-weight:600;">${(item.mask_area_pixels || 0).toLocaleString()} px</span>
          </div>
        </div>
      `;
    }).join('');

    // Bind card clicks to highlight and zoom to marker
    recordsListContainer.querySelectorAll('.map-record-item-card').forEach(card => {
      card.addEventListener('click', () => {
        const id = card.getAttribute('data-id');
        const match = filtered.find(m => m.id === id);
        if (match && match._marker && dashboardMap) {
          recordsListContainer.querySelectorAll('.map-record-item-card').forEach(c => c.classList.remove('is-active'));
          card.classList.add('is-active');
          dashboardMap.flyTo([match.location.lat, match.location.lng], 16, { animate: true });
          match._marker.openPopup();
        }
      });
    });
  }
}

/**
 * Focus and zoom to a specific inspection record on the Geospatial Dashboard.
 */
export function focusInspectionOnMap(id, AppState) {
  if (!dashboardMap) return;

  const target = AppState.history.find(h => h.id === id);
  if (!target || !target.location) return;

  setTimeout(() => {
    dashboardMap.invalidateSize();
    dashboardMap.flyTo([target.location.lat, target.location.lng], 16, { animate: true });

    // Find marker and open popup
    if (dashboardMarkersLayer) {
      dashboardMarkersLayer.eachLayer(layer => {
        const pos = layer.getLatLng();
        if (Math.abs(pos.lat - target.location.lat) < 0.0001 && Math.abs(pos.lng - target.location.lng) < 0.0001) {
          layer.openPopup();
        }
      });
    }
  }, 200);
}

export function invalidateDashboardMapSize() {
  if (dashboardMap) {
    setTimeout(() => {
      dashboardMap.invalidateSize();
    }, 150);
  }
}

/**
 * Export filtered geotagged inspection records as RFC-4180 CSV file.
 */
export function exportFilteredGeotaggedCsv(records) {
  if (!records || records.length === 0) {
    toast('No geotagged records available to export.', 'info');
    return;
  }

  const headers = [
    'Inspection_ID',
    'Filename',
    'Timestamp',
    'Latitude',
    'Longitude',
    'Address',
    'Severity_Tier',
    'Repair_Status',
    'Mask_Area_Pixels',
    'Coverage_Pct',
    'Circularity',
    'RPHI_Score'
  ];

  const rows = records.map(r => [
    `"${(r.id || '').replace(/"/g, '""')}"`,
    `"${(r.filename || '').replace(/"/g, '""')}"`,
    `"${(r.timestamp || '').replace(/"/g, '""')}"`,
    r.location?.lat ?? '',
    r.location?.lng ?? '',
    `"${(r.location?.address || '').replace(/"/g, '""')}"`,
    `"${r.tier || 'Medium'}"`,
    `"${r.status || 'Reported'}"`,
    r.mask_area_pixels ?? 0,
    r.coverage_pct ?? 0,
    r.circularity ?? 0,
    r.rphi_score ?? ''
  ]);

  const csvContent = [headers.join(','), ...rows.map(row => row.join(','))].join('\r\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', `cracksight_geotagged_inspections_${new Date().toISOString().slice(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
  toast(`Exported ${records.length} geotagged records to CSV.`);
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
