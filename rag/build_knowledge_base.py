"""
rag/build_knowledge_base.py
───────────────────────────
Builds a local TF-IDF vector store from scientific knowledge about Gabes soil
remediation. No external API needed for retrieval — uses scikit-learn TF-IDF.

The knowledge base powers the AI Chat Assistant (Gemini generates answers
using retrieved context chunks).

Run: python rag/build_knowledge_base.py
"""

import os, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── EXPANDED KNOWLEDGE BASE — Gabes-specific documents ────────────────────────
KNOWLEDGE_DOCS = [

    """GABES CONTAMINATION PROFILE:
The Gabes governorate in Tunisia hosts the GCT phosphate processing complex.
Since 1972, the facility has discharged over 70 million tonnes of phosphogypsum
into the Gulf of Gabes. Key contaminants in surrounding agricultural zones:
Cadmium (Cd): up to 14 mg/kg near the factory (EU limit: 3 mg/kg).
Zinc (Zn): up to 500 mg/kg. Lead (Pb): up to 150 mg/kg.
EC ranges: 2-18 dS/m. pH ranges: 5.5-8.8.
Distance from the GCT factory is the strongest predictor of contamination severity.
Within 1 km: Cd > 10, EC > 12. 1-3 km: Cd 5-10, EC 6-12.
3-8 km: Cd 2-5, EC 3-8. >8 km: Cd < 2, EC < 5.""",

    """SCENARIO A — SALINITY DOMINANT ZONE (EC > 6, Cd < 3):
Most common in the 3-8 km band.
PLANT PRESCRIPTION: 100% Atriplex halimus (Mediterranean saltbush).
Tolerates EC up to 20 dS/m. Biomass: 3-8 tonnes/ha/year. SAFE for sheep/goat fodder.
Market price: 50-80 TND per quintal.
MICROBE PRESCRIPTION: Halotolerant PGPR (Pseudomonas putida HS1, Azospirillum brasilense).
Application: 10^8 CFU/mL, 50 mL per m2. Combined with date palm biochar at 5T/ha.
EC reduction: 1-2 dS/m per season. Income during cleanup: 2,500-4,000 TND/ha/year.
Zone color: ORANGE.""",

    """SCENARIO B — DUAL STRESS ZONE (Cd > 3 AND EC > 6):
The most challenging scenario. Found in the 0-3 km band.
PLANT PRESCRIPTION: 70% Atriplex halimus + 30% Noccaea caerulescens.
Noccaea accumulates up to 1000 mg/kg Cd in shoots. CRITICAL: biomass must NEVER be
used as fodder. Track all harvested material to Ghannouch biochar facility.
MICROBE PRESCRIPTION: Glomus mosseae (mycorrhizal) + Rhizophagus irregularis +
Metal-Resistant Bacillus subtilis MR1.
CADMIUM REMOVAL RATE: 0.35 mg/kg/month.
Zone starting at Cd 10 mg/kg -> safe in ~26 months.
Zone color: RED.""",

    """SCENARIO C — HEAVY METAL DOMINANT (Cd > 3, EC < 4):
Lower salinity allows more aggressive hyperaccumulators.
PLANT PRESCRIPTION: 50% Sedum alfredii + 50% Noccaea caerulescens.
Sedum BCF for Cd: 34-67 (Liao et al. 2004). Fastest removal: 0.4 mg/kg/month.
MICROBE PRESCRIPTION: Bacillus subtilis MR1 + EDTA chelation (500 mg/kg at week 6).
WARNING: EDTA must not be applied within 100m of water body.
Zone color: RED. Toxic biomass to certified facility.""",

    """SCENARIO D — SAFE ZONE (Cd < 1, EC < 4, pH 6.5-8.0):
Ready for cash crops. No phytoremediation needed.
RECOMMENDED: Pomegranate (8T/ha, 1.35 TND/kg premium), Olive, Citrus, Henna.
ENHANCEMENT: Azospirillum brasilense + Glomus mosseae.
Issue Soil Passport immediately. Green QR = premium export markets.
Gabes regenerative label: +25-40% price premium.""",

    """WATER MANAGEMENT FOR SALINE SOILS:
Standard irrigation kills microbial colonies. Use the leaching fraction formula:
Water volume (m3/cycle) = EC (dS/m) x 4.2 + 5
EC 4: 21.8 m3. EC 8: 38.6 m3. EC 12: 55.4 m3. EC 16: 72.2 m3.
Mycorrhizal colonies die if root-zone EC exceeds 15 dS/m.
Schedule: Month 1-3 flush every 10 days. Month 4-6 every 12-14 days.
Drip irrigation at 15cm depth preferred. Flood irrigation wastes 40% to evaporation.""",

    """ATRIPLEX HALIMUS — COMPLETE GUIDE:
Native North African halophyte. Salinity tolerance: EC up to 20 dS/m.
Cd uptake: roots accumulate 217-607 mg/kg (phytostabilizer). Shoots: <5 mg/kg (safe fodder).
Biomass: 3-8 T/ha/year. Crude protein: 14-18%. Market: 50-80 TND/quintal.
Seed spacing: 1m x 1.5m (6600 plants/ha). Plant October-November.
First harvest: 8 months. Cut above 20cm. Roots must remain intact for bioremediation.""",

    """NOCCAEA CAERULESCENS — CADMIUM HYPERACCUMULATOR:
Shoot Cd: up to 1000 mg/kg. Root Cd: up to 2400 mg/kg. BCF: 100-300.
Temperature tolerance: 5-35C. Summer heat >38C reduces yield 30%.
Companion planting with Atriplex provides shade reducing heat stress 15-20%.
Biomass: 2-3 T/ha/year. Cd removal: 2-5 kg Cd/ha/year.
Disposal: hazardous waste. High-temperature pyrolysis >500C at Ghannouch facility.""",

    """SEDUM ALFREDII — HIGH-RATE CD EXTRACTOR:
Highest Cd extraction rate: 0.40 mg/kg/month. BCF: 34-67.
3-4 harvests/year. Spacing: 0.3m x 0.3m. Height at harvest: 15-25cm.
NOT salt-tolerant (fails at EC > 4). Only for Scenario C zones.
Propagation: vegetative cuttings. Irrigation: 25 mm/week.""",

    """PGPR APPLICATION PROTOCOLS:
Pseudomonas putida HS1: BEST for saline soils (EC 4-15). Produces IAA. Fixes P.
Azospirillum brasilense Sp245: Fixes 20-40 kg N/ha/year. For safe zones + Scenario A.
Bacillus subtilis MR1: Metal-resistant. Produces metallothioneins. For Scenarios B/C.
Storage: 4C, shelf life 6 months. Do not mix with chemical fertilizers.
Apply early morning or evening to avoid UV damage.""",

    """MYCORRHIZAL FUNGI:
Glomus mosseae: Best in saline soils (to EC 18). Reduces Cd shoot accumulation 35-55%.
Rhizophagus irregularis: Broadest host range and metal resistance.
Application: 50g/m2 granular or 200 mL/m2 liquid, mixed into 5-10cm depth before planting.
NO fungicide for 6 weeks. NO copper sprays for 4 weeks.
EDTA and AMF are incompatible in same zone.""",

    """CLEAN-TO-CROP COUNTDOWN:
Formula: Months = Cd_initial / removal_rate.
Scenario A: 0.50 mg/kg/month. Scenario B: 0.35. Scenario C: 0.40. Scenario D: 0 months.
SAFE when THREE consecutive quarterly tests show Cd < 1 mg/kg.
Example: Cd 4 mg/kg Scenario A = 8 months. Cd 14 mg/kg Scenario B = 40 months.
Monitoring: Month 0 baseline, Month 6 re-test, then quarterly.""",

    """SOIL PASSPORT AND QR TRACEABILITY:
Tamper-evident certificate per 10x10m micro-zone documenting:
BEFORE: Initial readings + GPS. INTERVENTION: Bio-cocktail applied.
PROCESS: Water flush logs, re-tests. UNLOCK: Three tests Cd < 1 mg/kg.
GREEN QR: Safe for food. Unlocks Tunis organic (+25%), EU export (+40%).
ORANGE QR: Remediation ongoing, fodder buyers verify by scanning.
RED QR: Toxic biomass tracking to Ghannouch facility.""",

    """ECONOMIC ANALYSIS — ROI:
Year 1 costs per hectare: Seeds 200-800 TND, PGPR 150-300, Mycorrhizal 300-500,
Biochar 300-600, Irrigation infrastructure 800-1500, Soil testing 200.
Total Year 1: 2,150-4,100 TND/ha.
Atriplex fodder income: 5T/year x 65 TND/quintal = 3,250 TND/year.
Post-certification pomegranate: 8T x 1.35 TND/kg = 10,800 TND/year.
Break-even: < 1 year from certification for Scenario A.""",

    """GABES CLIMATE:
Semi-arid Mediterranean. Rainfall: 170-220 mm (Nov-Mar). Mean temp: 20C.
Summer: 35-42C. Winter: 10-18C. Planting season: Oct-Nov.
NW winds carry phosphogypsum dust. Fields with windbreaks show 30% lower Cd deposition.
Summer soil at 10cm: 45-52C — can kill mycorrhizal spores if exposed.
Maintain Atriplex canopy or 5cm mulch layer.""",

    """EU AND FAO THRESHOLDS:
EU Cd soil limit: 3 mg/kg (some apply 1 mg/kg). Pb: 300 mg/kg. Zn: 300 mg/kg.
FAO/WHO: Cd in vegetables 0.1 mg/kg FW, leafy greens 0.2 mg/kg.
Tunisian NT 106.02: Cd 3 mg/kg (NOT enforced in Gabes 2026).
For EU export: soil Cd < 1 mg/kg, irrigation water EC < 0.7 dS/m.
Soil Passport provides GlobalGAP chain-of-custody evidence.""",

    """DATE PALM BIOCHAR:
pH 9.2-9.8. CEC: 45-60 cmol/kg. Surface area: 180-250 m2/g.
Cd sorption: 15-25 mg Cd/g. C content: 65-75%.
Application at 5T/ha reduces Cd phytoavailability 20-35%.
Reduces water need 15-20%. Production cost: 50-100 TND/tonne.
Synergistic with PGPR: extends colony survival from 60 to 180+ days.""",

    """PHYTOEXTRACTION vs PHYTOSTABILIZATION:
Phytoextraction: Plants accumulate metals in harvestable shoots. Noccaea, Sedum.
Permanently reduces soil metal concentration. Requires hazardous waste disposal.
Phytostabilization: Plants lock metals in roots/rhizosphere. Atriplex.
Reduces mobility and bioavailability. Works in 1-3 seasons.
Gabes strategy: Combined approach. Phytoextraction for Cd. Phytostabilization for Pb.""",

    """PHOSPHOGYPSUM CHEMISTRY:
Phosphate rock + H2SO4 -> phosphoric acid + phosphogypsum waste.
Tunisian phosphate rock: 30-80 mg Cd/kg. Phosphogypsum dump: 140,000-200,000 tonnes Cd.
Dispersion: wind transport, rainwater leachate, coastal discharge, truck dust.
At pH >7: Cd forms CdCO3 (less mobile). At pH <6.5: Cd2+ highly mobile.
Maintain pH 7.0-7.5 with lime during remediation.""",

    """ARABIC AND FRENCH GLOSSARY:
Arabic: talawwuth al-turba (soil contamination), al-cadmium, moluhat al-turba (salinity),
istislah al-turba (remediation), shahat al-turba (soil passport),
mazra amen (safe zone), mintaqa khatra (danger zone).
French: Depollution des sols, Conductivite electrique (CE),
Espece hyperaccumulatrice, Champignons mycorhiziens, Passeport du sol.""",

]


def split_chunks(text, chunk_size=400, overlap=60):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def build_knowledge_base():
    """Build TF-IDF based knowledge store (no API key needed)."""
    print(f"Building RAG knowledge base with {len(KNOWLEDGE_DOCS)} documents...")

    all_chunks = []
    for doc in KNOWLEDGE_DOCS:
        chunks = split_chunks(doc.strip(), chunk_size=80, overlap=15)
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    # Build TF-IDF index
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(all_chunks)

    # Save index
    os.makedirs("rag/faiss_index", exist_ok=True)
    data = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "chunks": all_chunks,
    }
    with open("rag/faiss_index/tfidf_index.pkl", "wb") as f:
        pickle.dump(data, f)

    print(f"TF-IDF index saved to rag/faiss_index/tfidf_index.pkl ({len(all_chunks)} chunks)")
    return data


def search(query: str, top_k: int = 4):
    """Search the knowledge base with a query string."""
    index_path = "rag/faiss_index/tfidf_index.pkl"
    if not os.path.exists(index_path):
        return []
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    query_vec = data["vectorizer"].transform([query])
    scores = cosine_similarity(query_vec, data["tfidf_matrix"]).flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    return [data["chunks"][i] for i in top_indices if scores[i] > 0.05]


if __name__ == "__main__":
    build_knowledge_base()
