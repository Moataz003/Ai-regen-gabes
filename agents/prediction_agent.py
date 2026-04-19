# agents/prediction_agent.py
import sys
import os
import json
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *
from agents.model_interface import RemediationModel

# --- GLOBAL CACHE ---
_CACHED_AGENT = None

# --- 1. RULE-BASED FALLBACK ---
class ScientificRuleModel(RemediationModel):
    def __init__(self):
        self.loaded = True

    def predict(self, features: dict) -> dict:
        ec = features.get('ec', 0)
        cd = features.get('cd', 0)
        pb = features.get('pb', 0)
        ph = features.get('ph', 7.0)
        soil_type = features.get('soil_type', "Loam")
        
        high_salt = ec >= 6.0
        high_metal = cd >= CD_EU_LIMIT  # 3.0
        moderate_metal = cd >= CD_SAFE_THRESHOLD  # 1.0
        high_pb = pb >= 50.0
        
        if high_metal or high_pb:
            profile = "ScenarioC_MetalDominant"; color = "RED"
            plant = "50% Sedum alfredii + 50% Noccaea caerulescens"
            microbe = "Metal-Resistant Bacillus + EDTA"
            action = "⚠️ Toxic Zone."
        elif moderate_metal:
            profile = "ScenarioB_DualStress"; color = "ORANGE"
            plant = "70% Atriplex halimus + 30% Noccaea caerulescens"
            microbe = "Mycorrhizal Fungi + Metal-Resistant Bacillus"
            action = "⚠️ Moderate risk zone."
        elif high_salt:
            profile = "ScenarioA_SalinityDominant"; color = "ORANGE"
            plant = "100% Atriplex halimus"
            microbe = "Halotolerant PGPR + Date Palm Biochar"
            action = "✅ Safe for sheep fodder."
        else:
            profile = "ScenarioD_LowStress"; color = "GREEN"
            plant = "Pomegranate / Olive"
            microbe = "Azospirillum brasilense"
            action = "✅ SAFE. Issue Passport."

        water = self._calculate_water_flush(ec, soil_type)
        months = self._calculate_timeline(cd, ph, plant)
        
        return {
            "stress_profile": profile, "zone_color": color, "scenario_name": profile.replace("_", " "),
            "plant_mix": plant, "microbe_mix": microbe, "safe_for_fodder": color != "RED",
            "action_note": action, "months_to_safe": months, "water_flush_m3": water,
            "confidence_pct": 95.0, "input": features
        }

# --- 2. OPENROUTER LLM AGENT ---
class LLMAgent(RemediationModel):
    def __init__(self):
        self.loaded = True
        self.client = OpenAI(
            base_url=LLM_BASE_URL, 
            api_key=LLM_API_KEY,
            default_headers={
                "HTTP-Referer": "https://gabes-regenerate-ai.com",
                "X-Title": "Gabes Regenerate AI"
            }
        )
        self.model = LLM_MODEL_NAME
    
    def predict(self, features: dict) -> dict:
        # FIXED: Added pb=features.get('pb', 0)
        user_prompt = PRESCRIPTION_USER_PROMPT.format(
            ec=features['ec'], 
            ph=features['ph'], 
            cd=features['cd'], 
            pb=features.get('pb', 0), # <--- FIX HERE
            soil_type=features.get('soil_type', 'Unknown')
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PRESCRIPTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            if "```json" in content: content = content.split("```json")[1].split("```")[0]
            
            result = json.loads(content)
            water = self._calculate_water_flush(features['ec'], features.get('soil_type', 'Loam'))

            return {
                "stress_profile": "LLM_Analysis", "scenario_name": "AI Dynamic Analysis",
                "zone_color": result.get("zone_color", "ORANGE"),
                "plant_mix": result.get("plant_mix", "Error"),
                "microbe_mix": result.get("microbe_mix", "Error"),
                "safe_for_fodder": result.get("zone_color") in ["GREEN", "ORANGE"],
                "action_note": result.get("reasoning", "Analysis complete."),
                "months_to_safe": int(result.get("months_to_safe", 12)),
                "water_flush_m3": water, "confidence_pct": 95.0, "input": features
            }
        except Exception as e:
            print(f"OpenRouter Error: {e}")
            return None

# --- 3. FACTORY ---
def get_agent():
    global _CACHED_AGENT
    if _CACHED_AGENT is not None:
        return _CACHED_AGENT

    try:
        agent = LLMAgent()
        print(f"🤖 Connected to OpenRouter ({LLM_MODEL_NAME})...")
        _CACHED_AGENT = agent
        return agent
    except Exception as e:
        print(f"⚠️ Connection failed: {e}. Using Rules.")
        return ScientificRuleModel()