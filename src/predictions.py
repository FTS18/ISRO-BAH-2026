import numpy as np

class ClimatePredictor:
    def __init__(self, model_loader):
        self.model_loader = model_loader
    
    def predict_rainfall_next_days(self, recent_data, days_ahead=7):
        """Predict rainfall for next N days"""
        if len(recent_data) < 30:
            raise ValueError(f"Insufficient data. Expected at least 30 days, got {len(recent_data)}.")
            
        predictions = []
        try:
            current_sequence = recent_data[-30:].reshape(1, 30, 1)
        except Exception as e:
            raise ValueError(f"Could not reshape recent_data to (1, 30, 1): {e}")
            
        for _ in range(days_ahead):
            next_pred = self.model_loader.predict_rainfall(current_sequence)
            predictions.append(next_pred[0][0])
            
            # Scale and shift for next prediction
            scaled = self.model_loader.scalers['rainfall'].transform(next_pred)
            current_sequence = np.append(current_sequence[:, 1:, :], 
                                        scaled.reshape(1, 1, 1), axis=1)
        
        return np.array(predictions)
    
    def predict_temperature(self, features_dict):
        """Predict max and min temperature"""
        feature_keys = [
            'rainfall',
            'rainfall_lag_1',
            'rainfall_30day_avg',
            'max_temp_lag_1',
            'min_temp_lag_1',
            'month_sin',
            'month_cos',
            'temp_diff'
        ]
        
        missing_keys = [k for k in feature_keys if k not in features_dict]
        if missing_keys:
            raise KeyError(f"Missing required features: {missing_keys}")
            
        try:
            features = np.array([[features_dict[k] for k in feature_keys]])
            max_temp, min_temp = self.model_loader.predict_temperature(features)
            return max_temp[0][0], min_temp[0][0]
        except Exception as e:
            raise RuntimeError(f"Error during temperature prediction: {e}")
    
    def simulate_what_if(self, current_conditions, rainfall_change_percent, 
                        temp_change_celsius):
        """What-if simulation"""
        modified = current_conditions.copy()
        modified['rainfall'] *= (1 + rainfall_change_percent / 100)
        modified['max_temp'] += temp_change_celsius
        modified['min_temp'] += temp_change_celsius
        
        max_temp, min_temp = self.predict_temperature(modified)
        
        return {
            'original_rainfall': current_conditions['rainfall'],
            'modified_rainfall': modified['rainfall'],
            'original_max_temp': current_conditions['max_temp'],
            'modified_max_temp': max_temp,
            'rainfall_change_pct': rainfall_change_percent,
            'temp_change_celsius': temp_change_celsius
        }