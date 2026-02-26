#!/bin/bash
# ============================================================================
# ClinicalRAG — Setup and Run Script
# ============================================================================
# This script sets up the complete ClinicalRAG project from scratch.
# Run this after cloning the repo or downloading the project files.
#
# Prerequisites:
#   - Python 3.10+
#   - Ollama installed (https://ollama.ai)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ============================================================================

set -e  # Exit on any error

echo "============================================"
echo "🏥 ClinicalRAG — Project Setup"
echo "============================================"
echo ""

# Step 1: Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "   ✅ Virtual environment created and activated"

# Step 2: Install dependencies
echo ""
echo "📥 Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "   ✅ Dependencies installed"

# Step 3: Pull Llama 3 model via Ollama
echo ""
echo "🤖 Checking Ollama and pulling Llama 3..."
if command -v ollama &> /dev/null; then
    echo "   Ollama found. Pulling llama3 model..."
    ollama pull llama3
    echo "   ✅ Llama 3 model ready"
else
    echo "   ⚠️  Ollama not found. Please install from https://ollama.ai"
    echo "   After installing, run: ollama pull llama3"
fi

# Step 4: Generate synthetic clinical data
echo ""
echo "📝 Generating synthetic clinical notes..."
python -m src.data_generation.generate_clinical_notes
echo "   ✅ Clinical notes generated"

# Step 5: Run the ingestion pipeline
echo ""
echo "🔄 Running document ingestion pipeline..."
python -m src.ingestion.ingest_documents
echo "   ✅ Documents ingested into vector store"

# Step 6: Verify setup
echo ""
echo "============================================"
echo "✅ Setup complete! To launch the app:"
echo ""
echo "   source venv/bin/activate"
echo "   streamlit run streamlit_app/app.py"
echo ""
echo "Or to test the RAG chain from CLI:"
echo ""
echo "   python -m src.retrieval.rag_chain"
echo "============================================"
