# AI-Powered Digital Twin of India's Climate
## ISRO Hackathon Submission — Problem Statement 5

---

## Executive Summary

This project implements an AI-powered digital twin of India's climate, designed to assimilate heterogeneous space-based remote sensing datasets and ground-based meteorological observations. The system provides a highly interactive virtual reanalysis dashboard that enables users to monitor real-time weather grids, generate 7-day spatiotemporal forecasts, simulate custom climate anomalies, and assess agricultural and water resources risk indicators.

The core technology relies on a PyTorch Spatio-Temporal Convolutional LSTM (ConvLSTM) neural network combined with a NOAA Climate Prediction Center (CPC) Analog Year Ensemble. This hybrid framework ensures both short-term dynamic accuracy and long-term seasonal stability, avoiding the compounding drift errors common in standalone autoregressive models.

---

## Objectives Met

### 1. Design Scalable Framework
* Created a modular Python architecture located inside the `src/` directory.
* Built dynamic slicing utilities capable of cropping NetCDF4 datasets to any regional pilot boundary (e.g. Karnataka, Uttar Pradesh, Odisha) or evaluating national All-India grids.
* Configured automated pipeline scripts that handle binary decoding, coordinate mapping, and NetCDF file compilation.

### 2. Demonstrate Proof of Concept (PoC)
* Validated the PyTorch ConvLSTM engine on 2022-2023 holdout data, achieving a holdout validation RMSE of 8.45 mm for gridded rainfall and 1.25°C for maximum temperature.
* Implemented a probabilistic validation suite that computes the Brier Score (BS) and Brier Skill Score (BSS) for heavy precipitation anomalies.
* Built interactive calibration/reliability diagrams to measure forecast uncertainty reliability.

### 3. Interactive Geospatial Dashboard
* Fuses multiple observations into a single dashboard: gridded precipitation, maximum and minimum temperatures, satellite-derived Land Surface Temperature (LST), Sea Surface Temperature (SST), and simulated soil moisture indexes.
* Integrates an interactive 4D animated playback visualization to map forecast progressions over a 7-day lead time.
* Embeds real-time satellite imagery layers via NASA GIBS and JAXA AMSR2 WMS feeds.

### 4. What-If Scenario Simulator
* Enables users to scale observed rainfall (+/- 100%) and shift temperatures (+/- 5°C).
* Renders a comparing visualizer that isolates simulated scenario values, baseline observed values, and scenario anomalies (Simulated - Baseline) with custom diverging colorscales.
* Supports grid data exports to CSV files.

---

## Technical Specifications

### Data Ingestion and Formatting
* **Precipitation**: IMD daily binary observations (`0.25° × 0.25°` resolution).
* **Temperatures**: IMD daily binary maximum and minimum observations (`1.0° × 1.0°` resolution).
* **Satellites**: MOSDAC INSAT-3D/3DR geostationary products, NASA GPM IMERG, Terra MODIS LST, and JAXA AMSR2 soil moisture composites.
* **Storage**: CF-compliant NetCDF4 (`.nc`) files.

### Model Architecture
* **Spatiotemporal ConvLSTM**: PyTorch-based neural network operating on 5D tensors `[Batch, Time, Channels, Height, Width]`, using convolutions in place of matrix multiplications inside LSTM cells to retain geographical topology.
* **NOAA CPC Analog Ensemble**: Identifies the 3 most spatially-similar historical years using Pearson correlation over a 30-day window, extracting subsequent trajectories to form an ensemble.
* **Dynamic Seasonal Blending**: Blends ConvLSTM neural anomalies, analog ensemble trajectories, and historical climatological averages using an exponential decay schedule over the forecast horizon.
* **Mean Bias Correction (MBC)**: Post-processing grid adjustment to align prediction averages with seasonal climatology.

### Technology Stack
* **ML Framework**: PyTorch
* **Data Handling**: Xarray, NetCDF4, Pandas, NumPy, Scikit-learn
* **Frontend Console**: Streamlit
* **Geospatial Mapping**: Plotly Express (`Scattermap`, `Density_map`), PyDeck
* **Platform**: Python 3.12+

---

## Deliverables and Files Included

* **`app/streamlit_app.py`** - Streamlit console frontend script.
* **`checkpoints/climate_twin_convlstm_final.pth`** - PyTorch rainfall model weights.
* **`checkpoints/climate_twin_convlstm_temp.pth`** - PyTorch temperature model weights.
* **`src/spatial_predictions.py`** - 5-layer prediction pipeline, analog correlation, and boundary masking.
* **`src/climate_indices.py`** - Indices calculators (Monsoon, SPI, CWSI, FFG).
* **`src/teleconnections.py`** - Fetcher for ENSO/IOD/MJO indices from NOAA CPC.
* **`src/basin_analysis.py`** - River basin rainfall accumulation engine.
* **`src/climate_alerts.py`** - Extreme threshold checking and alert broadcasting.
* **`src/climate_copilot.py`** - Conversational agricultural advisor RAG engine.
* **`src/models/pytorch_convlstm.py`** - ConvLSTM neural network layer definitions.
* **`src/model_loader.py`** - Model state-dict checkpoint loader.
* **`scripts/train_convlstm.py`** - Model training pipeline.
* **`scripts/download_and_decode_all_real.py`** - Data ingestion orchestrator.

---

## Replication and Setup

1. **Initialize Environment**:
   ```bash
   python -m venv venv312
   source venv312/bin/activate
   pip install -r requirements.txt
   ```
2. **Execute Application**:
   ```bash
   streamlit run app/streamlit_app.py
   ```
3. **Trigger Pipelines (Optional)**:
   ```bash
   python scripts/download_and_decode_all_real.py
   python scripts/train_convlstm.py
   ```

---

## Innovation and Novelty Highlights

1. **Strict Boundary Clipping**: Implements local matplotlib path checks to mask gridded datasets to exact state boundaries, preventing visual data spillage.
2. **4D animated Playback**: Dynamic timeline controls mapping 2D grids smoothly across forecast steps.
3. **Probabilistic Skill Assessment**: Holds out 2022-2023 observations to compute Brier Skill Scores and Plotly calibration reliability curves.
4. **Anomalous Difference Mapping**: isolatable scenario anomaly layer to directly view absolute changes.
5. **Decoupled Architecture**: Cleaned out TensorFlow and legacy 1D predictors to optimize runtime performance.