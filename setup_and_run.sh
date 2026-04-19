#!/bin/bash
# ═══════════════════════════════════════════════════════
#  Gabès Regenerate AI — Full Setup & Launch Script
#  Run: bash setup_and_run.sh
# ═══════════════════════════════════════════════════════

echo ""
echo "🌱 ═══════════════════════════════════════════════"
echo "   Gabès Regenerate AI — Hackathon Setup"
echo "   ═══════════════════════════════════════════════"
echo ""

# 1. Virtual environment
echo "📦 Step 1/5: Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
echo "📦 Step 2/5: Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ Dependencies installed"

# 3. Copy env file
if [ ! -f .env ]; then
  cp .env.example .env
  echo "⚠️  Created .env file — EDIT IT and add your API keys!"
  echo "   nano .env"
fi

# 4. Generate dataset
echo "🔬 Step 3/5: Generating synthetic Gabès soil dataset..."
python data/generate_dataset.py
echo "✅ Dataset generated: data/gabes_soil_dataset.csv"

# 5. Train ML models
echo "🤖 Step 4/5: Training ML models..."
python agents/train_prescription_agent.py
echo "✅ Models saved to agents/models/"

# 6. Build RAG index (only if OPENAI_API_KEY set)
echo "📚 Step 5/5: Checking RAG setup..."
if grep -q "your_openai_key_here" .env; then
  echo "⚠️  Skipping FAISS index build (no OpenAI key). AI chat will use Anthropic Claude fallback."
  echo "   Set OPENAI_API_KEY in .env and run: python rag/build_knowledge_base.py"
else
  source .env 2>/dev/null || true
  python rag/build_knowledge_base.py
  echo "✅ RAG knowledge base built"
fi

echo ""
echo "🚀 ═══════════════════════════════════════════════"
echo "   LAUNCHING Streamlit App..."
echo "   Open: http://localhost:8501"
echo "   ═══════════════════════════════════════════════"
echo ""

streamlit run app.py --server.port=8501