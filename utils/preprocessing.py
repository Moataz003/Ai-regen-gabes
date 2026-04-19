# utils/preprocessing.py
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

class SpectralPreprocessor:
    """
    Pipeline de preprocessing des spectres NIR.
    """
    
    def __init__(self, wavelength_min=400, wavelength_max=2450,
                 sg_window=15, sg_polyorder=2, sg_deriv=1,
                 n_pca_components=100):
        self.wavelength_min = wavelength_min
        self.wavelength_max = wavelength_max
        self.sg_window = sg_window
        self.sg_polyorder = sg_polyorder
        self.sg_deriv = sg_deriv
        self.n_pca_components = n_pca_components
        self.pca = None
        self.selected_spec_cols = None
        self.wavelengths = None
    
    def _get_wavelength_from_col(self, col_name):
        """Extrait la longueur d'onde depuis le nom de colonne."""
        return float(str(col_name).replace('spec.', '').replace('X', ''))
    
    def select_spectral_range(self, df, spec_cols):
        """Filtre les colonnes spectrales sur la plage utile."""
        # Handle case where df might be a dict or list (from UI input)
        # We assume for the app, we might not use this method directly if input is scalar
        
        # Logic preserved from original:
        filtered = [c for c in spec_cols 
                    if self.wavelength_min <= self._get_wavelength_from_col(c) <= self.wavelength_max]
        self.selected_spec_cols = filtered
        self.wavelengths = np.array([self._get_wavelength_from_col(c) for c in filtered])
        return df[filtered].values
    
    def snv(self, X):
        """Standard Normal Variate : centre et réduit chaque spectre."""
        X_mean = X.mean(axis=1, keepdims=True)
        X_std  = X.std(axis=1, keepdims=True)
        X_std[X_std == 0] = 1e-8
        return (X - X_mean) / X_std
    
    def savitzky_golay(self, X):
        """Dérivée de Savitzky-Golay appliquée à chaque spectre."""
        return savgol_filter(X, window_length=self.sg_window,
                             polyorder=self.sg_polyorder,
                             deriv=self.sg_deriv, axis=1)
    
    def fit_transform(self, df, spec_cols):
        """Preprocessing complet + fit PCA."""
        X = self.select_spectral_range(df, spec_cols)
        X = self.snv(X)
        X = self.savitzky_golay(X)
        
        n_comp = min(self.n_pca_components, X.shape[0] - 1, X.shape[1])
        self.pca = PCA(n_components=n_comp, random_state=42)
        X_pca = self.pca.fit_transform(X)
        
        return X_pca
    
    def transform(self, df, spec_cols):
        """Preprocessing seul (sans fit PCA)."""
        available = [c for c in self.selected_spec_cols if c in df.columns]
        X = df[available].values
        X = self.snv(X)
        X = self.savitzky_golay(X)
        return self.pca.transform(X)