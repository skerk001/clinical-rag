"""
Reset and reingest the clinical notes corpus.

Use this script when:
- The vector store has duplicate entries from multiple ingestion runs
- You want to start fresh with clean data
- You've changed chunking parameters and want to re-embed

Usage: python -m src.utils.reset_and_ingest
"""

import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def reset_vector_store(vectordb_dir: str = "data/vectordb"):
    """Delete the existing vector store to start fresh."""
    vectordb_path = Path(vectordb_dir)
    if vectordb_path.exists():
        shutil.rmtree(vectordb_path)
        print(f"🗑️  Deleted existing vector store at {vectordb_dir}")
    else:
        print(f"ℹ️  No existing vector store found at {vectordb_dir}")


def reset_raw_data(raw_dir: str = "data/raw"):
    """Delete existing generated data."""
    raw_path = Path(raw_dir)
    if raw_path.exists():
        shutil.rmtree(raw_path)
        print(f"🗑️  Deleted existing raw data at {raw_dir}")
    raw_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("=" * 60)
    print("🔄 ClinicalRAG — Reset and Reingest")
    print("=" * 60)

    # Step 1: Clean slate
    reset_raw_data()
    reset_vector_store()

    # Step 2: Generate fresh data
    print("\n📝 Generating fresh clinical notes...")
    from src.data_generation.generate_clinical_notes import generate_clinical_notes
    generate_clinical_notes()

    # Step 3: Reingest
    print("\n🔄 Running ingestion pipeline...")
    from src.ingestion.ingest_documents import run_ingestion_pipeline
    run_ingestion_pipeline()

    print("\n✅ Reset complete! You can now run the app.")
