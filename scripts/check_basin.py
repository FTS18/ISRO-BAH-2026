import sys
sys.path.insert(0, 'src')
import basin_analysis as ba

print(f"Basins loaded: {len(ba.RIVER_BASINS)}")
for name, info in ba.RIVER_BASINS.items():
    n_paths = len(ba.BASIN_PATHS.get(name, []))
    print(f"  {name:<20s}  area={info['area_km2']:>8,} km2  polygon_rings={n_paths}  color={info['color']}")

# Spot check point-in-polygon for a known Ganga grid cell
import numpy as np
lats = np.arange(24.0, 28.0, 0.25)
lons = np.arange(78.0, 82.0, 0.25)
mask = ba._build_basin_mask(lats, lons, "Ganga-Yamuna")
print(f"\nGanga-Yamuna mask over lat=[24-28) lon=[78-82):")
print(f"  Grid shape : {mask.shape}")
print(f"  Cells inside polygon : {mask.sum()}")

print("\n[OK] All imports and polygon masks compiled successfully")
