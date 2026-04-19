# 🌱 Gabès Regenerate AI — Hackathon Guide

> **SoilRevive & GreenShield Precision Remediation Engine**

---

## ⚡ 3-Command Quickstart (Copy-Paste)

```bash
# 1. Clone / enter project folder
cd gabes_regenerate

# 2. One-command setup + launch
bash setup_and_run.sh

# 3. Open browser
# → http://localhost:8501
```

Or manually:
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python data/generate_dataset.py          # generate 2000-row dataset
python agents/train_prescription_agent.py # train 4 ML models
streamlit run app.py
```

---

## 📁 Project Structure

```
gabes_regenerate/
├── app.py                          ← Main Streamlit app (all 4 phases)
├── requirements.txt
├── setup_and_run.sh                ← One-click setup script
├── .env.example                    ← Copy to .env, add API keys
│
├── data/
│   ├── generate_dataset.py         ← Synthetic dataset generator (run first)
│   └── gabes_soil_dataset.csv      ← Generated: 2000 micro-zone records
│
├── agents/
│   ├── train_prescription_agent.py ← ML training (RF classifier + regressors)
│   ├── prediction_agent.py         ← Inference engine used by Streamlit
│   └── models/                     ← Saved .pkl model files (auto-created)
│       ├── clf_stress.pkl          ← Stress profile classifier
│       ├── clf_color.pkl           ← Zone color classifier
│       ├── reg_months.pkl          ← Months-to-safe regressor
│       └── reg_water.pkl           ← Water flush volume regressor
│
├── rag/
│   ├── build_knowledge_base.py     ← FAISS vector store builder
│   └── faiss_index/                ← Auto-created after running build
│
└── utils/
    └── passport_generator.py       ← QR code + PDF Soil Passport generator
```

---

## 🛠️ Tools & Stack

| Layer | Tool | Why |
|---|---|---|
| **Frontend** | Streamlit | Rapid UI, great for demos |
| **Maps** | Folium + streamlit-folium | Interactive zone maps |
| **Charts** | Plotly | Gauges, radar, scatter |
| **ML (Phase 1&2)** | Scikit-learn RandomForest | Fast, interpretable, no GPU needed |
| **RAG (Chat)** | LangChain + FAISS | Local vector store, no latency |
| **LLM** | Claude Haiku (Anthropic) | Fast, cheap AI assistant fallback |
| **Embeddings** | OpenAI (optional) | Better RAG quality |
| **QR Codes** | qrcode library | Soil Passport QR generation |
| **PDF** | ReportLab | Professional Soil Passport PDF |
| **Dataset** | Synthetic (literature-grounded) | El Zrelli 2015, Galfati 2011 |

---

## ⏰ 24-Hour Hackathon Plan

### Hour 0–2: Setup & Data ✅
- [ ] `bash setup_and_run.sh` — installs everything & launches app
- [ ] Edit `.env` with your Anthropic API key
- [ ] Verify app loads at http://localhost:8501
- [ ] Explore the dataset: `data/gabes_soil_dataset.csv`

### Hour 2–5: Phase 1 & 2 — Diagnostics + Prescription
- [ ] Test Phase 1 input form with real Gabès soil values
- [ ] Verify the ML classifier gives correct prescriptions
- [ ] Tune the rule-based fallback scenarios in `agents/prediction_agent.py`
- [ ] Add your team's soil test readings as test cases

### Hour 5–8: Phase 3 — Timeline & Water Engine
- [ ] Verify cadmium depletion chart trajectory is scientifically accurate
- [ ] Test ROI calculator with different farm sizes
- [ ] Add more irrigation scenarios (drip vs. flood)

### Hour 8–12: Phase 4 — Soil Passport
- [ ] Test QR code generation for each zone color
- [ ] Download and verify PDF passport layout
- [ ] Add farmer photo upload feature (optional enhancement)

### Hour 12–17: RAG + AI Assistant
- [ ] Build FAISS index: `python rag/build_knowledge_base.py`
- [ ] Test AI chat with real agronomic questions
- [ ] Add more knowledge documents to `rag/build_knowledge_base.py`
- [ ] Test Arabic and French queries

### Hour 17–20: Dashboard & Polish
- [ ] Load real uploaded soil data from CSV
- [ ] Polish the map visualization
- [ ] Add multi-zone batch input (CSV upload feature)

### Hour 20–23: Demo Preparation
- [ ] Prepare a demo flow: enter 3 different micro-zones (green/orange/red)
- [ ] Screenshot key results
- [ ] Write 2-minute pitch script
- [ ] Test on mobile browser

### Hour 23–24: Final Checks
- [ ] `requirements.txt` complete?
- [ ] All features working without API keys? (fallback mode)
- [ ] README accurate?
- [ ] Backup: push to GitHub

---

## 🔑 API Keys

| Key | Used For | Required? |
|---|---|---|
| `ANTHROPIC_API_KEY` | AI chat assistant fallback | **Recommended** |
| `OPENAI_API_KEY` | RAG embeddings (FAISS) | Optional |

**Without any keys:** The app works fully — ML models, all 4 phases, QR codes, PDF passports. Only the AI chat needs a key.

---

## 📊 Dataset Sources

The synthetic dataset is grounded in real published measurements:

- **El Zrelli et al. 2015** — Heavy metal contamination in Gulf of Gabès coastal sediments. *Marine Pollution Bulletin* 101(2):922-929. Cd peak: 14 mg/kg near factory.
- **Galfati et al. 2011** — Trace metals near phosphate treatment industry, Tunisia. Provides spatial distribution data.
- **Wali et al. 2013** — Phosphate industry (SIAPE) soil contamination, Sfax. EC and pH ranges.
- **Manousaki & Kalogerakis 2009** — Atriplex halimus Cd/Pb uptake rates. Provides removal rate coefficients.
- **Nedjimi & Daoud 2009** — Cd accumulation in Atriplex halimus. Root: 606 mg/kg, shoot: 217 mg/kg.
- **Untold Mag 2026** — 70M tonnes phosphogypsum discharged since 1972, 130,000 protest participants.

---

## 🌿 Scientific Basis for Prescriptions

| Scenario | Trigger | Plant | Microbe | Removal |
|---|---|---|---|---|
| **A** Salinity Dominant | EC>6, Cd<3 | Atriplex halimus 100% | Halotolerant PGPR + Biochar | 0.5 mg/kg/mo |
| **B** Dual Stress | EC>6, Cd>3 | Atriplex 70% + Noccaea 30% | Mycorrhizae + Bacillus MR1 | 0.35 mg/kg/mo |
| **C** Metal Dominant | EC<4, Cd>3 | Sedum 50% + Noccaea 50% | Bacillus + EDTA | 0.4 mg/kg/mo |
| **D** Safe | EC<4, Cd<1 | Pomegranate/Olive | Azospirillum | 0 (already safe) |

---

## 🚀 Quick Enhancement Ideas (if you have time)

1. **CSV batch upload** — let farmers upload a CSV of their zones
2. **Arabic UI** — add `locale` support for right-to-left text
3. **Satellite imagery** — plug in Sentinel-2 NDVI bands to estimate salinity
4. **Blockchain traceability** — replace QR with NFT-style on-chain passport
5. **Mobile camera** — colorimetric strip photo → AI reads the color → auto-fills Cd value

---

## 🏆 Why This Wins

- ✅ **Ultra-low cost inputs** — only a $5 test strip needed
- ✅ **First in Tunisia** — no existing tool does this for Gabès specifically  
- ✅ **Immediate ROI** — Atriplex fodder income during the cleanup period
- ✅ **Erases the Gabès stigma** — verifiable Soil Passport = premium market access
- ✅ **Full stack** — ML + RAG + LLM + maps + PDF + QR in one Streamlit app
