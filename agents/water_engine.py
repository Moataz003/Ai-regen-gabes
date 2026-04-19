# agents/water_engine.py
import joblib
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import MODEL_DIR

class WaterPredictionEngine:
    def __init__(self):
        self.model = None
        self.model_columns = None
        self.crop_factors = None
        self.loaded = False
        self._load_models()

    def _load_models(self):
        """Load the three pickle files safely."""
        try:
            # Ensure correct path handling
            base_path = os.path.dirname(os.path.dirname(__file__))
            models_path = os.path.join(base_path, "agents", "models")

            self.model = joblib.load(os.path.join(models_path, "water_model.pkl"))
            self.model_columns = joblib.load(os.path.join(models_path, "model_columns.pkl"))
            self.crop_factors = joblib.load(os.path.join(models_path, "crop_factor.pkl"))
            self.loaded = True
            print("💧 Water ML Engine Loaded Successfully.")
        except FileNotFoundError as e:
            print(f"⚠️ Water Model files missing: {e}")
            self.loaded = False
        except Exception as e:
            print(f"⚠️ Error loading water models: {e}")
            self.loaded = False

    def predict(self, soil_ph: float, temperature: float, crop_type: str, season: str):
        """
        Runs the full ML pipeline:
        1. Apply Crop Factor logic.
        2. Create DataFrame.
        3. One-Hot Encode.
        4. Align columns with training data.
        5. Predict.
        """
        if not self.loaded:
            return None, "Models not loaded."

        try:
            # 1. Apply Crop Factor
            # Default to 1.0 if crop not found in the dictionary
            factor = self.crop_factors.get(crop_type, 1.0)

            # 2. Create Input Dictionary
            input_data = {
                'Soil_pH': [soil_ph],
                'Temperature_C': [temperature],
                'Crop_Type': [crop_type],
                'Season': [season],
                'Crop_Factor': [factor] # This is the engineered feature
            }
            
            df = pd.DataFrame(input_data)

            # 3. One-Hot Encode (must match training)
            # Use get_dummies, then align columns
            df_encoded = pd.get_dummies(df, columns=['Crop_Type', 'Season'])

            # 4. Align Columns
            # Add missing columns (fill with 0), Remove extra columns
            df_aligned = df_encoded.reindex(columns=self.model_columns, fill_value=0)

            # 5. Predict
            prediction = self.model.predict(df_aligned)[0]

            return prediction, "Success"

        except Exception as e:
            return None, str(e)

# Singleton pattern
_water_engine = None

def get_water_engine():
    global _water_engine
    if _water_engine is None:
        _water_engine = WaterPredictionEngine()
    return _water_engine