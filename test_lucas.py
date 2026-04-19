# test_lucas.py
import sys, os
sys.path.append('.')

# Register class BEFORE loading model
from utils.preprocessing import SpectralPreprocessor
sys.modules['__main__'].SpectralPreprocessor = SpectralPreprocessor
sys.modules['utils.preprocessing'].SpectralPreprocessor = SpectralPreprocessor

from agents.lucas_agent import get_lucas_agent
import numpy as np

agent = get_lucas_agent()
if agent.loaded:
    print("✅ Model loaded")
    # Dummy full spectrum (4301 values)
    dummy = [0.1 + 0.01*np.sin(i*0.01) for i in range(4301)]
    cd, pb = agent.predict_heavy_metals(33.87, 10.10, 5.0, 8.0, 1.0, dummy)
    print(f"🔴 High-input test: Cd={cd:.3f}, Pb={pb:.1f}")
else:
    print("❌ Model failed to load")