import pydeck as pdk
import pandas as pd
import numpy as np
import xarray as xr

def plot_pydeck_3d_grid(da: xr.DataArray, title: str, color_scale_name: str = "viridis") -> pdk.Deck:
    """
    Renders high-resolution gridded climate observations as 3D elevation columns using PyDeck.
    """
    df = da.to_dataframe(da.name or 'value').reset_index().dropna()
    val_col = da.name or 'value'
    
    # Scale values for column height (elevation)
    val_min = df[val_col].min()
    val_max = df[val_col].max()
    val_range = val_max - val_min + 1e-5
    
    df['elevation'] = ((df[val_col] - val_min) / val_range) * 150000.0
    
    # Normalize for color channels
    norm_val = (df[val_col] - val_min) / val_range
    
    if color_scale_name == "Blues":
        df['r'] = 30
        df['g'] = (norm_val * 150 + 50).astype(int)
        df['b'] = (norm_val * 200 + 55).astype(int)
    elif color_scale_name == "YlOrRd":
        df['r'] = (norm_val * 200 + 55).astype(int)
        df['g'] = (norm_val * 100 + 30).astype(int)
        df['b'] = 30
    else:  # default viridis
        df['r'] = (norm_val * 150 + 20).astype(int)
        df['g'] = (norm_val * 180 + 30).astype(int)
        df['b'] = (norm_val * 100 + 50).astype(int)
        
    df['a'] = 210  # opacity
    
    mean_lat = float(df['lat'].mean()) if not df.empty else 20.0
    mean_lon = float(df['lon'].mean()) if not df.empty else 77.0
    
    layer = pdk.Layer(
        "ColumnLayer",
        df,
        get_position="[lon, lat]",
        get_elevation="elevation",
        elevation_scale=1.0,
        elevation_range=[0, 150000],
        radius=15000,
        get_fill_color="[r, g, b, a]",
        pickable=True,
        auto_highlight=True,
    )
    
    view_state = pdk.ViewState(
        latitude=mean_lat,
        longitude=mean_lon,
        zoom=4.5,
        pitch=45,
        bearing=15
    )
    
    tooltip_text = "Coordinates: {lat}°N, {lon}°E\nValue: {" + val_col + ":.2f}"
    
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": tooltip_text},
        map_style="mapbox://styles/mapbox/dark-v10"
    )
