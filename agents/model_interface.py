# agents/model_interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any

# Import our constants from Step 1
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import CD_SAFE_THRESHOLD, INFILTRATION_FACTORS

class RemediationModel(ABC):
    """
    Abstract Base Class for all Remediation Models.
    This ensures all future models (ML, Deep Learning, Rules) 
    have the same structure and outputs.
    """
    
    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes soil features, returns a prescription.
        Must be implemented by child classes.
        """
        pass

    # --- Scientific Helper Methods ---

    def _calculate_water_flush(self, ec: float, soil_type: str) -> int:
        """
        Calculates water needed to flush salt below root zone.
        Logic: Base volume depends on Salinity (EC).
        Soil Texture Adjustment: Clay needs more water than Sand.
        """
        # Base Formula: (EC * 4.2) + 5
        base_water = ec * 4.2 + 5
        
        # Get soil factor from config, default to 1.0
        soil_factor = INFILTRATION_FACTORS.get(soil_type, 1.0)
        
        # Final volume
        total_volume = int(base_water * soil_factor)
        return max(5, total_volume) # Minimum 5 m3

    def _calculate_timeline(self, cd: float, ph: float, plant_mix: str) -> int:
        """
        Calculates months needed to clean soil.
        Logic: Total Cd / Monthly Removal Rate.
        Adjustment: pH affects metal availability.
        """
        if cd <= CD_SAFE_THRESHOLD: 
            return 0
        
        # Base rate depends on plant type
        rate = 0.2 # Default low rate
        if "Noccaea" in plant_mix: rate = 0.35
        elif "Sedum" in plant_mix: rate = 0.40
        elif "Atriplex" in plant_mix: rate = 0.25 # Good for salt, moderate for metal
        
        # pH Modifier:
        # Acidic soils (pH < 7) release metals faster -> faster cleanup
        # Alkaline soils (pH > 8) lock metals -> slower cleanup
        ph_modifier = 1.0
        if ph < 6.5:
            ph_modifier = 1.2 
        elif ph > 8.0:
            ph_modifier = 0.8
            
        effective_rate = rate * ph_modifier
        
        months = int(cd / effective_rate)
        return max(3, months) # Minimum 3 months cycle