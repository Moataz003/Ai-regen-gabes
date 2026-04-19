"""
agents/lucas_agent.py ─── LUCAS Spectral Model for Heavy Metal Prediction
"""
import joblib
import pandas as pd
import numpy as np
import os
import sys
import warnings

# Ignore sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import MODEL_DIR

class LucasPredictor:
    def __init__(self):
        self.model = None
        self.loaded = False
        self.load_model()

    def load_model(self):
        try:
            # Import and register the SpectralPreprocessor class for pickle
            from utils.preprocessing import SpectralPreprocessor
            
            # Register in ALL places pickle might look for the class
            sys.modules['__main__'].SpectralPreprocessor = SpectralPreprocessor
            sys.modules['utils.preprocessing'].SpectralPreprocessor = SpectralPreprocessor
            if 'preprocessing' not in sys.modules:
                sys.modules['preprocessing'] = sys.modules['utils.preprocessing']
                sys.modules['preprocessing'].SpectralPreprocessor = SpectralPreprocessor
            
            # Load the model
            path = os.path.join(MODEL_DIR, "gabes_regenerate_ai_model_LUCAS_only.pkl")
            self.model = joblib.load(path)
            self.loaded = True
            print("🌍 LUCAS Model Loaded Successfully")
            
        except ImportError as e:
            print(f"⚠️ Import Error: {e}")
            print("💡 Ensure utils/preprocessing.py exists with class SpectralPreprocessor")
            self.loaded = False
        except Exception as e:
            print(f"⚠️ LUCAS Model Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self.loaded = False

    def predict_heavy_metals(self, lat: float, lon: float, ph: float, ec: float, dist: float, spectrum: list):
        if not self.loaded:
            print("⚠️ Model not loaded — returning fallback")
            return 2.5, 50.0

        try:
            # 1. Validate & prepare spectrum
            wavelengths = np.arange(350, 2501, 0.5)  # 4301 values
            expected_len = len(wavelengths)
            
            if len(spectrum) != expected_len:
                print(f"⚠️ Spectrum length: got {len(spectrum)}, expected {expected_len}")
                if len(spectrum) < 100:
                    raise ValueError(f"Spectrum too short ({len(spectrum)}). Need 4301 values (350-2500nm @ 0.5nm).")
                # Interpolate if we have enough points
                if len(spectrum) < expected_len:
                    from scipy.interpolate import interp1d
                    orig_wl = np.linspace(350, 2500, len(spectrum))
                    f = interp1d(orig_wl, spectrum, kind='linear', fill_value="extrapolate")
                    spectrum = f(wavelengths).tolist()
                    print(f"✓ Spectrum interpolated to {expected_len} values")
                else:
                    spectrum = spectrum[:expected_len]
                    print(f"✓ Spectrum truncated to {expected_len} values")
            
            # 2. Build DataFrame with EXACT column names
            input_data = pd.DataFrame({
                f'spec.{w}': [spectrum[i]] for i, w in enumerate(wavelengths)
            })
            
            # 3. Preprocess spectral data
            spec_cols = [f'spec.{w}' for w in wavelengths]
            if 'preprocessor' not in self.model:
                raise KeyError("Model dict missing 'preprocessor' key")
            processed_spectral = self.model['preprocessor'].transform(input_data, spec_cols)
            
            # 4. Calculate actual distance from factory (not slider value)
            from math import radians, sin, cos, asin, sqrt
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371.0
                dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
                a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
                return R * 2 * asin(sqrt(a))
            
            actual_dist = haversine(lat, lon, 33.883, 10.100)  # SIAPE coords
            print(f"📍 Using calculated distance: {actual_dist:.2f} km")
            
            # 5. Scale extra features
            extra_df = pd.DataFrame({
                'ph_init': [ph],
                'ec_init': [ec],
                'dist_usine_km': [actual_dist]
            })
            if 'scaler_extra' not in self.model:
                raise KeyError("Model dict missing 'scaler_extra' key")
            scaled_extra = self.model['scaler_extra'].transform(extra_df)
            
            # 6. Concatenate & predict
            processed_data = np.hstack([processed_spectral, scaled_extra])
            if 'model' not in self.model:
                raise KeyError("Model dict missing 'model' key")
            prediction = self.model['model'].predict(processed_data)
            
            # 7. Extract outputs safely
            if prediction.ndim == 2:
                cd_pred = float(prediction[0][0])
                pb_pred = float(prediction[0][1]) if prediction.shape[1] > 1 else 50.0
            else:
                cd_pred = float(prediction[0])
                pb_pred = float(prediction[1]) if len(prediction) > 1 else 50.0
            
            print(f"✅ Prediction: Cd={cd_pred:.3f} mg/kg, Pb={pb_pred:.1f} mg/kg")
            return max(0, cd_pred), max(0, pb_pred)

        except Exception as e:
            import traceback
            print(f"❌ Prediction FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 2.5, 50.0  # Fallback


# ✅ CRITICAL: Module-level singleton cache (was missing!)
_lucas_agent = None

def get_lucas_agent():
    global _lucas_agent
    if _lucas_agent is None:
        _lucas_agent = LucasPredictor()
    return _lucas_agent