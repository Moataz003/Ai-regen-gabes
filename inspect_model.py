import joblib
import os
from config import MODEL_DIR

path = os.path.join(MODEL_DIR, 'gabes_regenerate_ai_model_LUCAS_only.pkl')
print('model path:', path)
model = joblib.load(path)
print('keys:', list(model.keys()))
pre = model['preprocessor']
print('preprocessor type:', type(pre))
print('selected_spec_cols len:', len(pre.selected_spec_cols))
print('selected_spec_cols first,last:', pre.selected_spec_cols[0], pre.selected_spec_cols[-1])
print('pca components shape:', pre.pca.components_.shape)
print('scaler_extra feature_names:', getattr(model['scaler_extra'], 'feature_names_in_', None))
print('scaler_extra mean:', getattr(model['scaler_extra'], 'mean_', None))
print('model estimator type:', type(model['model']))
