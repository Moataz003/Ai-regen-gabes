# agents/train_prescription_agent.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error
import joblib
import os

def train_models():
    print("📂 Loading dataset...")
    df = pd.read_csv('data/gabes_soil_dataset.csv')

    # Define Features (OBSERVED sensor data) and Targets
    features = ['cd_initial_mgkg', 'salinity_dS_m', 'ph']
    X = df[features]
    
    # --- 1. Stress Profile Classifier ---
    y_profile = df['stress_profile']
    le_profile = LabelEncoder()
    y_profile_encoded = le_profile.fit_transform(y_profile)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_profile_encoded, test_size=0.2, random_state=42)
    
    print("\n=== STRESS PROFILE CLASSIFIER ===")
    profile_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,             # Shallow trees prevent memorization
        learning_rate=0.1,
        reg_alpha=1.0,           # L1 Regularization
        reg_lambda=1.0,          # L2 Regularization
        subsample=0.8,           # Randomly sample 80% of data per tree
        colsample_bytree=0.8,    # Randomly sample 80% of features per tree
        objective='multi:softprob',
        num_class=len(le_profile.classes_),
        eval_metric='mlogloss',
        random_state=42
    )
    profile_model.fit(X_train, y_train)
    y_pred_profile = profile_model.predict(X_test)
    print(classification_report(y_test, y_pred_profile, target_names=le_profile.classes_))

    # --- 2. Zone Color Classifier ---
    y_color = df['zone_color']
    le_color = LabelEncoder()
    y_color_encoded = le_color.fit_transform(y_color)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_color_encoded, test_size=0.2, random_state=42)
    
    print("=== ZONE COLOR CLASSIFIER ===")
    color_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        reg_alpha=1.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )
    color_model.fit(X_train, y_train)
    y_pred_color = color_model.predict(X_test)
    print(classification_report(y_test, y_pred_color, target_names=le_color.classes_))

    # --- 3. Months to Safe Crop Regressor ---
    y_months = df['months_to_safe']
    X_train, X_test, y_train, y_test = train_test_split(X, y_months, test_size=0.2, random_state=42)
    
    print("=== MONTHS TO SAFE CROP ===")
    months_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        reg_alpha=1.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    months_model.fit(X_train, y_train)
    y_pred_months = months_model.predict(X_test)
    mae_months = mean_absolute_error(y_test, y_pred_months)
    print(f"MAE = {mae_months:.1f} months")

    # --- 4. Water Flush Regressor ---
    y_water = df['water_flush_m3']
    X_train, X_test, y_train, y_test = train_test_split(X, y_water, test_size=0.2, random_state=42)
    
    print("=== WATER FLUSH ===")
    water_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        reg_alpha=1.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    water_model.fit(X_train, y_train)
    y_pred_water = water_model.predict(X_test)
    mae_water = mean_absolute_error(y_test, y_pred_water)
    print(f"MAE = {mae_water:.1f} m³/cycle")

    # --- Save Models ---
    model_dir = 'agents/models'
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(profile_model, os.path.join(model_dir, 'profile_model.joblib'))
    joblib.dump(le_profile, os.path.join(model_dir, 'profile_encoder.joblib'))
    joblib.dump(color_model, os.path.join(model_dir, 'color_model.joblib'))
    joblib.dump(le_color, os.path.join(model_dir, 'color_encoder.joblib'))
    joblib.dump(months_model, os.path.join(model_dir, 'months_model.joblib'))
    joblib.dump(water_model, os.path.join(model_dir, 'water_model.joblib'))
    
    print(f"\n✅ All models saved to {model_dir}/")

if __name__ == "__main__":
    train_models()