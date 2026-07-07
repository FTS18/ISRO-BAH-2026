import numpy as np
import xarray as xr

class MultiModelEnsemble:
    """
    Ensembles multiple forecasting techniques (ConvLSTM + NWP climatological baseline)
    and applies Kalman Filter-style weighted bias corrections.
    """
    @staticmethod
    def forecast_ensemble(baseline_grid: xr.DataArray, convlstm_preds: list, days_ahead: int = 7) -> dict:
        """
        Combines ConvLSTM forecasts with a simulated Numerical Weather Prediction (NWP) model
        weighted by historical climatology.
        Returns ensemble mean, upper (95%), and lower (5%) confidence bounds.
        """
        ensemble_preds = []
        nwp_preds = []
        
        # Calculate standard deviation of historical variations
        std_val = float(baseline_grid.std()) if float(baseline_grid.std()) > 0 else 1.0
        mean_val = float(baseline_grid.mean())
        
        for idx, cl_pred in enumerate(convlstm_preds):
            # Simulated NWP Model: starts from baseline and reverts towards climatological mean with minor noise
            reversion_factor = 0.15 * (idx + 1)
            nwp_pred = cl_pred * (1 - reversion_factor) + mean_val * reversion_factor
            # Add spatial noise perturbation
            noise = np.random.normal(0, std_val * 0.05, cl_pred.shape)
            nwp_pred = np.maximum(0, nwp_pred + noise)
            nwp_preds.append(nwp_pred)
            
            # Kalman-style weighted ensembling: ConvLSTM is weighted higher early on, NWP dominates later
            weight_cl = max(0.3, 0.8 - 0.08 * idx)
            weight_nwp = 1.0 - weight_cl
            
            ens_mean = (cl_pred * weight_cl) + (nwp_pred * weight_nwp)
            ensemble_preds.append(ens_mean)
            
        # Compute confidence bounds
        lower_bounds = []
        upper_bounds = []
        for idx, ens in enumerate(ensemble_preds):
            # Uncertainty grows with forecast horizon
            uncertainty = std_val * (0.1 + 0.04 * idx)
            lower_bounds.append(np.maximum(0, ens - 1.96 * uncertainty))
            upper_bounds.append(ens + 1.96 * uncertainty)
            
        return {
            "ensemble_mean": ensemble_preds,
            "nwp_forecast": nwp_preds,
            "lower_bound": lower_bounds,
            "upper_bound": upper_bounds
        }
