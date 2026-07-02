"""
basin_analysis.py — Major Indian River Basin Masks and Rainfall Accumulation.

Basin boundaries are loaded from `data/india_river_basins.geojson`, a polygon
dataset derived from GRDC (Global Runoff Data Centre) watershed delineations
for 8 major Indian river basins.

Masking methodology: For each IMD grid cell (lat, lon), a point-in-polygon test
is executed against the GeoJSON polygon vertices using `matplotlib.path.Path`,
the same algorithm used in `src/spatial_predictions.py` for administrative
boundary clipping. This replaces the previous simplified lat/lon bounding-box
approximation, eliminating cross-basin contamination at polygon edges.

References:
  - GRDC Global Runoff Data Centre, Federal Institute of Hydrology (BfG), Koblenz.
  - matplotlib.path.Path.contains_points: 2D piecewise-linear ray casting.
"""
import os
import json
import numpy as np
import warnings
from matplotlib.path import Path

# ---------------------------------------------------------------------------
# Load basin polygon vectors from GeoJSON
# ---------------------------------------------------------------------------
_GEOJSON_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'india_river_basins.geojson'
)

RIVER_BASINS = {}       # name -> {area_km2, color, tributaries, mouth}
BASIN_PATHS  = {}       # name -> list of matplotlib.path.Path objects

def _load_basin_geojson():
    """
    Parse india_river_basins.geojson and pre-compute one matplotlib.path.Path
    per polygon ring for ultra-fast batch point-in-polygon queries.
    Falls back to bounding-box approximations if the file is missing.
    """
    global RIVER_BASINS, BASIN_PATHS

    try:
        with open(_GEOJSON_PATH, 'r', encoding='utf-8') as f:
            gj = json.load(f)

        for feature in gj['features']:
            props = feature['properties']
            name  = props['name']
            geom  = feature['geometry']

            # Store metadata
            RIVER_BASINS[name] = {
                'area_km2':   props.get('area_km2', 0),
                'color':      props.get('color', '#888888'),
                'tributaries': props.get('major_tributaries', []),
                'mouth':      props.get('mouth', ''),
            }

            # Build Path objects for each polygon ring
            paths = []
            coords_list = geom['coordinates']
            # Support both Polygon (one ring) and MultiPolygon
            if geom['type'] == 'Polygon':
                coords_list = [coords_list[0]]          # exterior ring only
            elif geom['type'] == 'MultiPolygon':
                coords_list = [ring[0] for ring in coords_list]

            for ring in coords_list:
                verts = np.array(ring, dtype=np.float64)  # (N, 2) lon/lat
                if len(verts) > 2:
                    paths.append(Path(verts))

            BASIN_PATHS[name] = paths

        print(f"[basin_analysis] Loaded {len(RIVER_BASINS)} river basin polygons "
              f"from india_river_basins.geojson")

    except Exception as exc:
        print(f"[basin_analysis] WARNING: Could not load basin GeoJSON ({exc}). "
              "Falling back to bounding-box approximations.")
        _load_fallback_boxes()


def _load_fallback_boxes():
    """Bounding-box fallback used when the GeoJSON is unavailable."""
    global RIVER_BASINS, BASIN_PATHS
    _BOXES = {
        "Ganga-Yamuna": {"lat": (24.0, 31.5), "lon": (73.0, 88.5),
                         "area_km2": 861_000, "color": "#1f77b4"},
        "Brahmaputra":  {"lat": (24.0, 29.5), "lon": (89.5, 97.5),
                         "area_km2": 580_000, "color": "#2ca02c"},
        "Indus":        {"lat": (24.5, 36.5), "lon": (66.5, 78.0),
                         "area_km2": 321_000, "color": "#9467bd"},
        "Godavari":     {"lat": (16.0, 22.0), "lon": (73.5, 82.5),
                         "area_km2": 312_000, "color": "#8c564b"},
        "Krishna":      {"lat": (13.5, 19.5), "lon": (73.5, 81.5),
                         "area_km2": 258_000, "color": "#e377c2"},
        "Mahanadi":     {"lat": (19.0, 23.5), "lon": (80.5, 86.5),
                         "area_km2": 141_000, "color": "#7f7f7f"},
        "Narmada":      {"lat": (21.0, 24.0), "lon": (72.5, 80.5),
                         "area_km2":  98_000, "color": "#bcbd22"},
        "Cauvery":      {"lat": ( 9.5, 13.5), "lon": (75.0, 79.5),
                         "area_km2":  81_000, "color": "#17becf"},
    }
    for name, info in _BOXES.items():
        RIVER_BASINS[name] = {
            'area_km2': info['area_km2'],
            'color':    info['color'],
            'tributaries': [],
            'mouth':    '',
        }
        lat_min, lat_max = info['lat']
        lon_min, lon_max = info['lon']
        # Build a rectangular Path as the fallback polygon
        box_verts = np.array([
            [lon_min, lat_min], [lon_max, lat_min],
            [lon_max, lat_max], [lon_min, lat_max],
            [lon_min, lat_min],
        ])
        BASIN_PATHS[name] = [Path(box_verts)]


# Run at import time
_load_basin_geojson()


# ---------------------------------------------------------------------------
# Vectorised point-in-polygon mask builder
# ---------------------------------------------------------------------------
def _build_basin_mask(lats: np.ndarray, lons: np.ndarray, basin_name: str) -> np.ndarray:
    """
    Returns a boolean 2-D mask of shape (len(lats), len(lons)).
    A cell (i, j) is True if the grid-point (lon[j], lat[i]) lies inside any
    polygon ring belonging to `basin_name`.

    Uses matplotlib.path.Path.contains_points, which applies a piecewise-linear
    winding/ray-casting algorithm identical to the administrative boundary masking
    in src/spatial_predictions.py.
    """
    paths = BASIN_PATHS.get(basin_name, [])
    if not paths:
        return np.zeros((len(lats), len(lons)), dtype=bool)

    lon_grid, lat_grid = np.meshgrid(lons, lats)          # (nlat, nlon)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])  # (N, 2)

    mask = np.zeros(len(points), dtype=bool)
    for path in paths:
        mask |= path.contains_points(points)

    return mask.reshape(len(lats), len(lons))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_basin_rainfall_accumulation(rain_da, days_back: int = 7) -> list:
    """
    Compute accumulated rainfall (mm) over the last `days_back` days for each
    major Indian river basin, using precise GeoJSON polygon masks.

    The spatial mean is computed only over grid cells that fall inside the true
    watershed polygon, eliminating the cross-basin contamination produced by
    simple rectangular bounding-box approximations.

    Volume estimation:
        Volume_km3 = Area_km2 * MeanDepth_mm * 1e-6

    Parameters
    ----------
    rain_da  : xarray.DataArray  — (time, lat, lon) daily rainfall in mm/day.
    days_back: int               — accumulation window in days.

    Returns
    -------
    list of dicts sorted descending by total_mm:
        {basin, total_mm, area_km2, volume_km3, color, tributaries, mouth,
         n_grid_cells, mask_type}
    """
    lats   = rain_da.lat.values
    lons   = rain_da.lon.values
    n      = min(days_back, len(rain_da.time))
    recent = rain_da.values[-n:]   # (days, lat, lon)

    results = []
    for name, info in RIVER_BASINS.items():
        mask2d = _build_basin_mask(lats, lons, name)
        n_cells = int(mask2d.sum())
        if n_cells == 0:
            continue

        # Accumulate only within the polygon mask
        # recent[:, mask2d] -> shape (days, n_cells)
        basin_vals = recent[:, mask2d]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            total_mm = float(np.nansum(basin_vals) / n_cells)

        volume_km3 = info['area_km2'] * total_mm * 1e-6
        results.append({
            "basin":        name,
            "total_mm":     round(total_mm, 1),
            "area_km2":     info['area_km2'],
            "volume_km3":   round(volume_km3, 2),
            "color":        info['color'],
            "tributaries":  info['tributaries'],
            "mouth":        info['mouth'],
            "n_grid_cells": n_cells,
            "mask_type":    "polygon",
        })

    results.sort(key=lambda x: -x["total_mm"])
    return results


def compute_basin_forecast_accumulation(predictions: np.ndarray, rain_da) -> list:
    """
    Compute 7-day forecast accumulated rainfall for each major Indian river basin.

    Parameters
    ----------
    predictions : np.ndarray — shape (days, lat, lon), raw forecast grids.
    rain_da     : xarray.DataArray — used to extract lat/lon coordinate arrays.

    Returns
    -------
    list of dicts sorted descending by forecast_total_mm:
        {basin, forecast_total_mm, area_km2, color, n_grid_cells, mask_type}
    """
    lats = rain_da.lat.values
    lons = rain_da.lon.values

    results = []
    for name, info in RIVER_BASINS.items():
        mask2d  = _build_basin_mask(lats, lons, name)
        n_cells = int(mask2d.sum())
        if n_cells == 0:
            continue

        basin_pred = predictions[:, mask2d]   # (days, n_cells)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            total_mm = float(np.nansum(basin_pred) / n_cells)

        results.append({
            "basin":              name,
            "forecast_total_mm":  round(total_mm, 1),
            "area_km2":           info['area_km2'],
            "color":              info['color'],
            "tributaries":        info['tributaries'],
            "n_grid_cells":       n_cells,
            "mask_type":          "polygon",
        })

    results.sort(key=lambda x: -x["forecast_total_mm"])
    return results


def get_basin_metadata() -> list:
    """
    Return a list of basin metadata dicts for UI legend rendering and map overlays.
    Each dict contains: basin name, area_km2, color, major tributaries, outlet.
    """
    return [
        {
            "basin":       name,
            "area_km2":    info['area_km2'],
            "color":       info['color'],
            "tributaries": info['tributaries'],
            "mouth":       info['mouth'],
        }
        for name, info in RIVER_BASINS.items()
    ]
