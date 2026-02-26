# ClinicalRAG - Windows Setup Script (PowerShell)
# Prerequisites: Python 3.10+, Ollama (https://ollama.ai)
# Usage: powershell -ExecutionPolicy Bypass -File .\setup.ps1

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ClinicalRAG - Project Setup (Windows)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python not found. Install from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Step 2: Create virtual environment
Write-Host "" 
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Failed to create virtual environment." -ForegroundColor Red
    exit 1
}

# Step 3: Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "  Virtual environment activated." -ForegroundColor Green

# Step 4: Install dependencies
Write-Host ""
Write-Host "Installing dependencies (this may take 3-5 minutes)..." -ForegroundColor Yellow
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Some dependencies failed to install." -ForegroundColor Red
    exit 1
}
Write-Host "  Dependencies installed." -ForegroundColor Green

# Step 5: Check Ollama
Write-Host ""
Write-Host "Checking Ollama..." -ForegroundColor Yellow
try {
    $ollamaCheck = ollama --version 2>&1
    Write-Host "  Ollama found. Pulling llama3 model..." -ForegroundColor Green
    ollama pull llama3
    Write-Host "  Llama 3 model ready." -ForegroundColor Green
} catch {
    Write-Host "  WARNING: Ollama not found." -ForegroundColor Yellow
    Write-Host "  Install from https://ollama.ai then run: ollama pull llama3" -ForegroundColor Yellow
}

# Step 6: Generate synthetic clinical data
Write-Host ""
Write-Host "Generating synthetic clinical notes..." -ForegroundColor Yellow
python -m src.data_generation.generate_clinical_notes
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Data generation failed." -ForegroundColor Red
    exit 1
}
Write-Host "  Clinical notes generated." -ForegroundColor Green

# Step 7: Run ingestion pipeline
Write-Host ""
Write-Host "Running document ingestion pipeline..." -ForegroundColor Yellow
python -m src.ingestion.ingest_documents
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Ingestion failed." -ForegroundColor Red
    exit 1
}
Write-Host "  Documents ingested into vector store." -ForegroundColor Green

# Done
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To launch the app:" -ForegroundColor White
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "  streamlit run streamlit_app\app.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "To test from command line:" -ForegroundColor White
Write-Host "  python -m src.retrieval.rag_chain" -ForegroundColor Yellow
Write-Host ""
