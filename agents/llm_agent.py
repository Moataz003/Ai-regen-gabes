# agents/llm_agent.py
import os
import json
import sys
from openai import OpenAI

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import PRESCRIPTION_SYSTEM_PROMPT, PRESCRIPTION_USER_PROMPT, GROQ_API_KEY

class LLMAgent:
    def __init__(self):
        self.model_name = "llama3-8b-8192" # Fast, cheap, smart enough for this
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY", "") # Reads from .env or system vars
        )
    
    def get_context(self):
        """
        This simulates RAG retrieval. 
        In a full RAG setup, you would query a Vector DB (FAISS).
        For now, we paste the core knowledge directly to ensure accuracy.
        """
        return """
        SCENARIO A - HIGH SALINITY / LOW METAL TOXICITY ZONES:
        Trigger: EC > 6 dS/m but Cd < 3 mg/kg.
        Prescription: 100% Atriplex halimus.
        Microbes: Halotolerant PGPR (Pseudomonas putida) + Biochar.
        Safety: Safe for fodder.
        Color: ORANGE.

        SCENARIO B - HIGH HEAVY METALS + HIGH SALINITY:
        Trigger: Cd > 3 mg/kg AND EC > 6 dS/m.
        Prescription: 70% Atriplex halimus + 30% Noccaea caerulescens.
        Microbes: Mycorrhizal Fungi + Metal-Resistant Bacillus.
        Safety: NOT safe for fodder. Toxic biomass.
        Color: RED.

        SCENARIO C - HIGH METALS / LOW SALINITY:
        Trigger: Cd > 3 mg/kg but EC < 4 dS/m.
        Prescription: 50% Sedum alfredii + 50% Noccaea caerulescens.
        Microbes: Metal-Resistant Bacillus + EDTA chelation.
        Safety: NOT safe for fodder.
        Color: RED.

        SCENARIO D - SAFE ZONE:
        Trigger: Cd < 1 mg/kg AND EC < 4 dS/m.
        Prescription: Pomegranate / Olive.
        Microbes: Azospirillum brasilense.
        Safety: Safe.
        Color: GREEN.
        """

    def predict(self, features: dict) -> dict:
        # 1. Format the User Prompt
        user_prompt = PRESCRIPTION_USER_PROMPT.format(
            ec=features['ec'],
            ph=features['ph'],
            cd=features['cd'],
            zn=features.get('zn', 0),
            pb=features.get('pb', 0),
            dist_km=features.get('dist_km', 0),
            context=self.get_context()
        )

        try:
            # 2. Call the LLM (Groq/Llama3)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": PRESCRIPTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, # Low temp for consistency
                response_format={"type": "json_object"} # Forces JSON output
            )

            # 3. Parse the JSON string
            content = response.choices[0].message.content
            result = json.loads(content)

            # 4. Standardize Output to match the UI expectations
            return {
                "stress_profile": result.get("scenario_name", "Unknown"),
                "scenario_name": result.get("scenario_name", "Unknown"),
                "zone_color": result.get("zone_color", "ORANGE"),
                "plant_mix": result.get("plant_mix", "Error in generation"),
                "microbe_mix": result.get("microbe_mix", "Error in generation"),
                "months_to_safe": int(result.get("months_to_safe", 12)),
                "reasoning": result.get("reasoning", ""),
                "safe_for_fodder": result.get("zone_color") == "GREEN" or result.get("zone_color") == "ORANGE",
                "confidence_pct": 90.0, # Placeholder
                "input": features
            }

        except Exception as e:
            # Fallback if LLM fails
            print(f"LLM Error: {e}")
            return {
                "stress_profile": "Error",
                "scenario_name": "LLM Connection Error",
                "zone_color": "ORANGE",
                "plant_mix": "N/A",
                "microbe_mix": "N/A",
                "months_to_safe": 0,
                "reasoning": f"Could not connect to LLM: {str(e)}",
                "safe_for_fodder": False,
                "confidence_pct": 0,
                "input": features
            }