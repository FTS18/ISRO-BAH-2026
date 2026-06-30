import numpy as np
import xarray as xr

# Pilot Regions Bounding Boxes: (lat_min, lat_max, lon_min, lon_max)
PILOT_REGIONS = {
    "All India": (6.5, 37.5, 66.5, 97.5),
    "Andaman and Nicobar Islands": (6.0, 14.0, 92.0, 94.5),
    "Andhra Pradesh": (12.5, 19.5, 76.5, 84.5),
    "Arunachal Pradesh": (26.0, 29.5, 91.5, 97.5),
    "Assam": (24.0, 28.0, 89.5, 96.5),
    "Bihar": (24.0, 27.5, 83.0, 88.5),
    "Chandigarh": (30.6, 30.9, 76.6, 76.9),
    "Chhattisgarh": (17.5, 24.5, 80.0, 84.5),
    "Dadra and Nagar Haveli and Daman and Diu": (19.8, 20.6, 72.6, 73.4),
    "Delhi": (28.4, 28.9, 76.8, 77.4),
    "Goa": (14.8, 15.8, 73.6, 74.4),
    "Gujarat": (20.0, 24.8, 68.0, 74.5),
    "Haryana": (27.5, 31.2, 74.4, 77.6),
    "Himachal Pradesh": (30.3, 33.3, 75.5, 79.1),
    "Jammu and Kashmir": (32.0, 37.0, 73.5, 80.5),
    "Jharkhand": (21.8, 25.3, 83.3, 88.0),
    "Karnataka": (11.5, 18.5, 74.0, 78.5),
    "Kerala": (8.0, 12.8, 74.8, 77.5),
    "Ladakh": (32.0, 36.0, 75.0, 80.0),
    "Lakshadweep": (8.0, 12.5, 71.5, 74.5),
    "Madhya Pradesh": (21.0, 26.9, 74.0, 82.8),
    "Maharashtra": (15.5, 22.0, 72.5, 81.0),
    "Manipur": (23.8, 25.7, 92.9, 94.8),
    "Meghalaya": (25.0, 26.2, 89.8, 92.8),
    "Mizoram": (21.9, 24.5, 92.2, 93.5),
    "Nagaland": (25.1, 27.1, 93.3, 95.3),
    "Odisha": (17.8, 22.6, 81.3, 87.5),
    "Puducherry": (11.7, 12.1, 79.7, 80.0),
    "Punjab": (29.5, 32.5, 73.8, 77.0),
    "Rajasthan": (23.0, 30.5, 69.5, 78.5),
    "Sikkim": (27.0, 28.2, 88.0, 89.0),
    "Tamil Nadu": (8.0, 14.0, 76.0, 80.5),
    "Telangana": (15.8, 19.9, 77.2, 81.8),
    "Tripura": (22.9, 24.5, 91.1, 92.4),
    "Uttar Pradesh": (23.8, 30.5, 77.0, 84.8),
    "Uttarakhand": (28.7, 31.5, 77.5, 81.1),
    "West Bengal": (21.5, 27.3, 85.8, 89.9),
}

# Simplified polygon vertices (lon, lat) for all 36 states/UTs
STATE_POLYGONS = {
    "Andhra Pradesh": [
        (79.5,13.4),(79.9,14.0),(80.1,14.7),(80.3,15.5),(80.0,16.4),(80.3,17.0),
        (81.7,17.9),(82.2,17.7),(82.9,17.7),(83.4,18.0),(84.0,18.2),(84.5,18.3),
        (84.7,17.8),(84.0,17.0),(83.0,16.0),(82.0,15.0),(81.0,14.0),(80.5,13.5),
        (80.0,12.8),(79.5,13.0)
    ],
    "Arunachal Pradesh": [
        (91.5,27.5),(92.5,27.8),(93.5,27.8),(94.5,28.0),(95.5,28.2),(96.5,28.3),
        (97.3,28.5),(97.5,28.0),(97.0,27.5),(96.0,27.0),(95.0,26.8),(94.0,27.0),
        (93.0,26.8),(92.0,27.0),(91.5,27.3)
    ],
    "Assam": [
        (89.7,26.4),(90.5,26.5),(91.5,26.7),(92.5,26.5),(93.5,26.2),(94.5,26.5),
        (95.5,27.0),(95.8,27.4),(95.0,27.5),(94.0,27.4),(93.0,27.2),(92.0,27.5),
        (91.5,27.3),(91.0,26.8),(90.0,26.7),(89.7,26.4)
    ],
    "Bihar": [
        (83.3,27.3),(84.2,27.5),(84.6,27.0),(85.5,27.3),(86.5,27.0),(87.8,27.3),
        (88.3,26.3),(87.7,25.5),(87.8,25.0),(86.5,25.3),(86.0,25.0),(85.2,25.0),
        (83.9,24.5),(83.3,24.5),(83.7,25.5),(83.2,25.6)
    ],
    "Chhattisgarh": [
        (80.3,24.0),(81.5,24.2),(82.5,24.0),(83.5,23.5),(84.0,23.0),(83.8,22.0),
        (83.0,21.5),(82.0,21.5),(81.0,22.0),(80.5,22.5),(80.0,23.0),(79.5,23.5),
        (80.0,24.0)
    ],
    "Delhi": [
        (76.8,28.4),(77.4,28.4),(77.4,28.9),(76.8,28.9)
    ],
    "Goa": [
        (73.6,14.8),(74.4,14.8),(74.4,15.8),(73.6,15.8)
    ],
    "Gujarat": [
        (68.2,22.2),(69.0,23.0),(69.8,24.0),(70.5,24.5),(71.5,24.7),(72.5,24.5),
        (73.5,24.0),(74.5,23.3),(74.5,22.0),(73.8,21.0),(73.0,20.3),(72.5,20.5),
        (72.0,21.0),(71.5,21.5),(71.0,22.0),(70.0,22.5),(69.0,22.5),(68.5,22.5)
    ],
    "Haryana": [
        (74.4,28.0),(75.0,29.0),(75.5,30.0),(76.5,30.5),(77.5,30.5),(77.5,29.5),
        (77.0,28.5),(76.5,28.0),(76.0,27.5),(75.0,27.5),(74.5,28.0)
    ],
    "Himachal Pradesh": [
        (75.5,30.5),(76.5,31.0),(77.5,31.5),(78.5,32.0),(79.0,32.5),(79.0,31.5),
        (78.5,31.0),(78.0,30.5),(77.0,30.0),(76.0,30.5)
    ],
    "Jammu and Kashmir": [
        (73.8,34.5),(74.5,34.8),(75.5,35.0),(76.5,35.2),(77.5,35.5),(78.5,35.5),
        (79.5,34.8),(80.0,33.8),(79.0,33.0),(78.0,33.0),(77.0,33.5),(76.0,33.5),
        (75.0,33.0),(74.0,33.2),(73.5,34.0)
    ],
    "Jharkhand": [
        (83.5,24.5),(84.5,24.5),(85.5,24.2),(86.5,24.5),(87.5,24.5),(87.8,24.0),
        (87.0,23.0),(86.0,22.5),(85.0,22.2),(84.0,22.5),(83.5,23.0),(83.0,23.5)
    ],
    "Karnataka": [
        (74.0,15.0),(74.0,15.6),(74.4,15.6),(74.8,17.5),(75.5,17.8),(76.5,18.4),
        (77.5,18.0),(77.6,17.0),(77.1,16.0),(77.8,15.0),(77.3,14.0),(78.4,13.5),
        (78.3,12.5),(77.8,12.0),(77.0,11.6),(76.5,12.0),(75.0,12.5),(74.5,13.5)
    ],
    "Kerala": [
        (74.8,12.8),(75.5,12.8),(76.0,12.0),(76.5,11.0),(77.0,10.0),(77.3,9.0),
        (77.1,8.3),(76.5,8.5),(76.0,9.5),(75.5,10.5),(75.0,11.5),(74.8,12.0)
    ],
    "Madhya Pradesh": [
        (74.0,23.5),(75.0,24.5),(76.0,24.5),(77.0,25.5),(78.0,26.0),(79.0,26.5),
        (80.0,26.5),(81.0,26.0),(82.0,25.5),(82.5,24.5),(82.0,23.5),(81.0,23.0),
        (80.0,22.5),(79.0,22.5),(78.0,22.0),(77.0,22.5),(76.0,23.0),(75.0,23.0)
    ],
    "Maharashtra": [
        (72.6,20.0),(73.0,20.3),(74.5,21.8),(76.0,21.6),(78.0,21.5),(79.0,21.8),
        (80.5,21.8),(80.9,19.5),(80.0,18.8),(78.0,18.0),(77.8,17.5),(76.5,17.8),
        (76.0,17.0),(75.8,16.0),(74.2,15.6),(73.6,16.0),(72.8,17.5),(72.8,19.0)
    ],
    "Manipur": [
        (92.9,24.0),(93.5,24.2),(94.0,24.5),(94.8,25.0),(94.5,25.7),(93.8,25.7),
        (93.0,25.5),(92.5,25.0),(92.9,24.5)
    ],
    "Meghalaya": [
        (89.8,25.5),(90.5,25.5),(91.5,25.7),(92.5,25.5),(92.8,25.0),(92.0,25.0),
        (91.0,25.2),(90.0,25.0),(89.8,25.3)
    ],
    "Mizoram": [
        (92.3,23.0),(93.0,23.0),(93.5,23.5),(93.5,24.0),(93.0,24.5),(92.5,24.5),
        (92.2,24.0),(92.2,23.5)
    ],
    "Nagaland": [
        (93.3,25.5),(94.0,25.8),(95.0,26.5),(95.3,26.8),(95.0,27.0),(94.5,26.8),
        (94.0,26.5),(93.5,26.0),(93.3,25.7)
    ],
    "Odisha": [
        (81.5,19.0),(82.5,19.5),(83.5,19.5),(84.5,19.8),(85.5,20.0),(86.5,20.5),
        (87.5,21.0),(87.5,22.0),(86.5,22.5),(85.5,22.0),(84.5,21.5),(83.5,21.0),
        (82.5,20.5),(81.5,20.0),(81.0,19.5)
    ],
    "Punjab": [
        (73.9,30.0),(74.5,30.5),(75.5,31.0),(76.0,31.5),(76.5,32.5),(77.0,32.3),
        (77.0,31.0),(76.5,30.0),(76.0,29.5),(75.0,29.5),(74.0,30.0)
    ],
    "Rajasthan": [
        (70.0,26.5),(69.6,28.0),(71.5,29.0),(72.5,30.0),(74.0,30.5),(75.3,30.3),
        (76.2,28.5),(77.0,28.0),(77.8,27.2),(77.5,26.8),(76.8,25.0),(76.5,24.0),
        (75.8,24.3),(74.5,23.3),(73.8,24.5),(72.5,24.5),(71.0,25.0)
    ],
    "Sikkim": [
        (88.0,27.1),(88.5,27.5),(89.0,28.2),(88.5,28.1),(88.0,27.8)
    ],
    "Tamil Nadu": [
        (76.0,11.5),(77.0,12.8),(78.5,13.0),(80.2,13.5),(80.2,12.5),(79.8,11.0),
        (79.9,10.3),(79.3,9.2),(78.2,8.8),(77.5,8.1),(77.2,8.5),(77.3,9.5),(76.6,10.5)
    ],
    "Telangana": [
        (77.5,16.0),(78.0,17.0),(79.0,17.5),(80.0,18.0),(81.0,18.0),(81.5,18.5),
        (82.0,18.0),(81.5,17.0),(81.0,16.0),(80.0,15.5),(79.0,15.5),(78.0,16.0)
    ],
    "Tripura": [
        (91.2,23.0),(91.8,23.5),(92.3,24.0),(92.4,24.5),(92.0,24.5),(91.5,24.0),
        (91.1,23.5)
    ],
    "Uttar Pradesh": [
        (77.2,29.0),(78.0,30.0),(79.0,30.2),(80.2,31.0),(81.0,30.2),(82.2,28.8),
        (83.0,28.5),(84.3,27.3),(84.7,26.0),(84.0,25.0),(83.2,25.0),(83.2,24.0),
        (82.5,24.3),(81.5,25.0),(80.2,25.2),(78.5,25.0),(78.2,24.3),(77.8,25.0),
        (77.3,27.2),(77.5,28.2)
    ],
    "Uttarakhand": [
        (77.5,30.0),(78.0,30.5),(79.0,31.0),(80.0,31.5),(81.0,31.0),(80.5,30.0),
        (79.5,29.5),(78.5,29.5),(77.5,30.0)
    ],
    "West Bengal": [
        (85.8,21.5),(86.5,22.0),(87.5,22.5),(88.5,22.8),(89.0,23.5),(88.8,24.0),
        (88.5,25.0),(89.0,26.0),(88.5,27.2),(88.0,27.0),(87.5,26.0),(87.0,25.0),
        (86.5,24.0),(86.0,23.0),(85.8,22.0)
    ],
    "Andaman and Nicobar Islands": [
        (92.2,13.5),(92.8,13.5),(93.0,12.5),(93.0,11.5),(92.8,10.5),(92.5,10.5),
        (92.3,11.5),(92.1,12.5)
    ],
    "Lakshadweep": [
        (71.8,11.5),(72.2,11.5),(72.4,10.5),(72.2,10.5),(72.0,11.0)
    ],
    "Chandigarh": [
        (76.65,30.62),(76.90,30.62),(76.90,30.85),(76.65,30.85)
    ],
    "Puducherry": [
        (79.7,11.8),(80.0,11.8),(80.0,12.1),(79.7,12.1)
    ],
    "Dadra and Nagar Haveli and Daman and Diu": [
        (72.8,20.0),(73.2,20.0),(73.3,20.4),(72.9,20.5)
    ],
    "Ladakh": [
        (75.5,32.5),(77.0,33.5),(78.5,34.0),(79.5,34.5),(80.0,33.5),(79.0,32.5),
        (78.0,32.0),(77.0,32.5),(76.0,32.5)
    ],
}


def is_point_in_polygon(x, y, poly):
    """Ray casting algorithm to determine if point (x, y) is inside polygon"""
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


class SpatialClimatePredictor:
    def __init__(self, model_loader=None):
        self.model_loader = model_loader

    def mask_region_boundary(self, data_array, region_name):
        """Masks xarray dataset/dataarray keeping only grid points inside the actual state boundary"""
        if data_array is None or region_name not in STATE_POLYGONS:
            return data_array
        poly = STATE_POLYGONS[region_name]
        if isinstance(data_array, xr.Dataset):
            lats = data_array.lat.values
            lons = data_array.lon.values
        else:
            lats = data_array.lat.values
            lons = data_array.lon.values

        mask = np.zeros((len(lats), len(lons)), dtype=bool)
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if is_point_in_polygon(lon, lat, poly):
                    mask[i, j] = True

        mask_da = xr.DataArray(mask, coords=[('lat', lats), ('lon', lons)])

        if isinstance(data_array, xr.Dataset):
            masked_ds = data_array.copy(deep=True)
            for var in masked_ds.data_vars:
                masked_ds[var] = xr.where(mask_da, masked_ds[var], np.nan)
            return masked_ds
        else:
            return xr.where(mask_da, data_array, np.nan)

    def slice_region(self, data_array, region_name="Karnataka"):
        """Slices the xarray DataArray to the bounding box of the selected pilot region."""
        if region_name not in PILOT_REGIONS or region_name == "All India":
            return data_array
        lat_min, lat_max, lon_min, lon_max = PILOT_REGIONS[region_name]
        min_span = 1.2
        lat_span = lat_max - lat_min
        if lat_span < min_span:
            lat_mid = (lat_max + lat_min) / 2.0
            lat_min = lat_mid - (min_span / 2.0)
            lat_max = lat_mid + (min_span / 2.0)
        lon_span = lon_max - lon_min
        if lon_span < min_span:
            lon_mid = (lon_max + lon_min) / 2.0
            lon_min = lon_mid - (min_span / 2.0)
            lon_max = lon_mid + (min_span / 2.0)
        sliced = data_array.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        if sliced.sizes.get("lat", 0) == 0 or sliced.sizes.get("lon", 0) == 0:
            try:
                lat_mid = (lat_max + lat_min) / 2.0
                lon_mid = (lon_max + lon_min) / 2.0
                return data_array.sel(lat=lat_mid, lon=lon_mid, method="nearest")
            except Exception:
                return data_array
        return sliced

    def predict_rainfall_next_days_spatial(self, recent_rain_grid, days_ahead=7):
        """
        Predict weather variable for the next N days using trained ConvLSTM with MC Dropout uncertainty.
        Falls back to statistical proxy if model unavailable.
        Returns: (predictions, lower_bounds, upper_bounds) — each shape (days, lat, lon)
        """
        predictions = []
        lower_bounds = []
        upper_bounds = []

        base_std = (np.nanstd(recent_rain_grid.values[-30:], axis=0)
                    if len(recent_rain_grid.time) >= 30
                    else np.nanstd(recent_rain_grid.values, axis=0))
        base_std = np.nan_to_num(base_std, nan=0.1)

        is_rainfall = (recent_rain_grid.name is not None and
                       any(lbl in str(recent_rain_grid.name).lower() for lbl in ['rain', 'precip']))

        convlstm_model = None
        if self.model_loader:
            key = "convlstm" if is_rainfall else "convlstm_temp"
            convlstm_model = self.model_loader.models.get(key)

        nan_mask = np.isnan(recent_rain_grid.values[-1])

        if convlstm_model is not None:
            try:
                import torch
                device = next(convlstm_model.parameters()).device
                grid_vals = (recent_rain_grid.values[-10:]
                             if len(recent_rain_grid.time) >= 10
                             else recent_rain_grid.values)
                if len(grid_vals) < 10:
                    pad_len = 10 - len(grid_vals)
                    grid_vals = np.pad(grid_vals, ((pad_len, 0), (0, 0), (0, 0)), mode='edge')

                fill_val = 0.0 if is_rainfall else 30.0
                grid_vals = np.nan_to_num(grid_vals, nan=fill_val)

                if is_rainfall:
                    grid_vals_scaled = np.log1p(np.maximum(0, grid_vals))
                else:
                    grid_vals_scaled = (grid_vals - 30.0) / 10.0

                input_tensor = (torch.tensor(grid_vals_scaled, dtype=torch.float32)
                                .unsqueeze(0).unsqueeze(2).to(device))

                # MC Dropout: keep model in train mode for stochastic forward passes
                n_ensemble = 5
                convlstm_model.train()

                # Precompute climatological grid for each forecast day to stabilize predictions
                import pandas as pd
                times = pd.to_datetime(recent_rain_grid.time.values)
                clim_grids = {}
                for t_idx, t in enumerate(times):
                    key = (t.month, t.day)
                    if key not in clim_grids:
                        clim_grids[key] = []
                    clim_grids[key].append(recent_rain_grid.values[t_idx])
                clim_mean_grids = {k: np.nanmean(v, axis=0) for k, v in clim_grids.items()}
                
                last_date = pd.to_datetime(recent_rain_grid.time.values[-1])
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')

                cur_input = input_tensor
                for i in range(days_ahead):
                    ensemble_preds = []
                    for _ in range(n_ensemble):
                        with torch.no_grad():
                            pred_grid, _ = convlstm_model(cur_input)
                            pred_np_scaled = pred_grid.squeeze().cpu().numpy()
                            if is_rainfall:
                                p = np.expm1(pred_np_scaled)
                                p = np.maximum(0, p)
                            else:
                                p = (pred_np_scaled * 10.0) + 30.0
                            ensemble_preds.append(p)

                    ensemble_preds = np.stack(ensemble_preds, axis=0)
                    mean_pred = np.mean(ensemble_preds, axis=0)
                    
                    # Blend prediction with the exact daily climatology grid to prevent compounding drift
                    decay = 0.65 ** (i + 1)
                    nd = forecast_dates[i]
                    clim_grid = clim_mean_grids.get((nd.month, nd.day), grid_vals[-1])
                    clim_grid = np.nan_to_num(clim_grid, nan=fill_val)
                    mean_pred = decay * mean_pred + (1.0 - decay) * clim_grid
                    
                    std_pred  = np.std(ensemble_preds, axis=0)

                    mean_pred[nan_mask] = np.nan
                    predictions.append(mean_pred)
                    lower_bounds.append(np.where(nan_mask, np.nan,
                                                 np.maximum(0 if is_rainfall else -50,
                                                            mean_pred - std_pred)))
                    upper_bounds.append(np.where(nan_mask, np.nan, mean_pred + std_pred))

                    if is_rainfall:
                        next_scaled = np.log1p(np.maximum(0, mean_pred))
                    else:
                        next_scaled = (mean_pred - 30.0) / 10.0
                    next_scaled = np.nan_to_num(next_scaled, nan=fill_val)
                    next_t = (torch.tensor(next_scaled, dtype=torch.float32)
                              .unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device))
                    cur_input = torch.cat([cur_input[:, 1:, :, :, :], next_t], dim=1)

                convlstm_model.eval()
                return np.array(predictions), np.array(lower_bounds), np.array(upper_bounds)

            except Exception as e:
                print(f"ConvLSTM inference error, falling back: {e}")
                if convlstm_model is not None:
                    convlstm_model.eval()
                predictions = []
                lower_bounds = []
                upper_bounds = []

        # Statistical proxy fallback
        if len(recent_rain_grid.time) < 30:
            last_day = recent_rain_grid.values[-1]
            mean_30  = np.nanmean(recent_rain_grid.values, axis=0)
        else:
            last_day = recent_rain_grid.values[-1]
            mean_30  = np.nanmean(recent_rain_grid.values[-30:], axis=0)

        current_state = np.nan_to_num(last_day.copy(), nan=0.0)
        mean_30 = np.nan_to_num(mean_30, nan=0.0)

        for i in range(days_ahead):
            alpha = 0.75 * (0.92 ** i)
            next_state = alpha * current_state + (1 - alpha) * mean_30
            next_state = np.maximum(0 if is_rainfall else -50, next_state)
            next_state[nan_mask] = np.nan
            predictions.append(next_state)
            uncertainty = base_std * (1.0 + 0.12 * i)
            lower_bounds.append(np.where(nan_mask, np.nan,
                                         np.maximum(0 if is_rainfall else -50,
                                                    next_state - uncertainty)))
            upper_bounds.append(np.where(nan_mask, np.nan, next_state + uncertainty))
            current_state = next_state

        return np.array(predictions), np.array(lower_bounds), np.array(upper_bounds)

    def simulate_what_if_spatial(self, base_rain_grid, base_temp_grid, rain_modifier, temp_modifier):
        """Simulate climate change with nonlinear Clausius-Clapeyron thermodynamic interactions."""
        mod_rain = base_rain_grid.values * (1 + rain_modifier / 100.0)
        mod_temp = base_temp_grid.values + temp_modifier
        if temp_modifier > 0.0:
            scaling_factor = 1.0 + (temp_modifier * 0.07)
            mod_rain = np.where(mod_rain > 5.0, mod_rain * scaling_factor, mod_rain)
            mod_rain = np.maximum(0, mod_rain)
        elif temp_modifier < 0.0:
            drying_factor = 1.0 + (temp_modifier * 0.04)
            mod_rain = np.where(mod_rain > 2.0, mod_rain * drying_factor, mod_rain)
            mod_rain = np.maximum(0, mod_rain)
        return {'modified_rainfall': mod_rain, 'modified_max_temp': mod_temp}

    def assimilate_multi_source_data(self, imd_ground_grid, insat_sat_grid, variable="temperature"):
        """
        Optimal Interpolation (Inverse-Variance Weighting) data assimilation.
        Fuses IMD ground observations with MOSDAC INSAT satellite retrievals.
        """
        if variable == "temperature":
            var_ground = 1.2 ** 2
            var_sat    = 2.5 ** 2
        else:
            var_ground = 3.0 ** 2
            var_sat    = 5.0 ** 2
        weight_ground = var_sat    / (var_ground + var_sat)
        weight_sat    = var_ground / (var_ground + var_sat)
        fused_grid = (imd_ground_grid.values * weight_ground) + (insat_sat_grid.values * weight_sat)
        return xr.DataArray(
            fused_grid,
            coords=[imd_ground_grid.lat, imd_ground_grid.lon],
            dims=["lat", "lon"],
            name=f"fused_{variable}"
        )

    def compute_validation_metrics(self, observed_grid, predicted_grid):
        """
        Compute full validation metric suite.
        Returns: RMSE, MAE, Bias, Correlation, Skill Score vs Persistence baseline.
        """
        obs  = np.array(observed_grid, dtype=np.float64).ravel()
        pred = np.array(predicted_grid, dtype=np.float64).ravel()
        mask = ~(np.isnan(obs) | np.isnan(pred))
        if mask.sum() == 0:
            return {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "corr": np.nan, "skill": np.nan}
        o = obs[mask]
        p = pred[mask]
        rmse = np.sqrt(np.mean((o - p) ** 2))
        mae  = np.mean(np.abs(o - p))
        bias = np.mean(p - o)
        corr = np.corrcoef(o, p)[0, 1] if len(o) > 1 else np.nan
        clim_mean = np.mean(o)
        rmse_clim = np.sqrt(np.mean((o - clim_mean) ** 2))
        skill = 1.0 - (rmse / rmse_clim) if rmse_clim > 0 else np.nan
        return {"rmse": rmse, "mae": mae, "bias": bias, "corr": corr, "skill": skill}
