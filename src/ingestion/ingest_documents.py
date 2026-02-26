"""
Document Ingestion Pipeline for ClinicalRAG

This module handles the complete ingestion pipeline:
1. Load clinical documents from JSON corpus
2. Preprocess and clean text (preserving clinical structure)
3. Chunk documents using configurable strategies
4. Generate embeddings using sentence-transformers
5. Store in ChromaDB vector database with metadata

The chunking strategy is particularly important for clinical notes because
the document structure carries semantic meaning. A "MEDICATIONS AT DISCHARGE"
section should ideally stay together rather than being split across chunks.
We use RecursiveCharacterTextSplitter with clinical-aware separators to
respect these boundaries as much as possible.

Author: Samir Kerkar
"""

import json
import time
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import config


# ============================================================================
# CLINICAL-AWARE TEXT PREPROCESSING
# ============================================================================

def preprocess_clinical_text(text: str) -> str:
    """
    Clean and normalize clinical text while preserving medically meaningful structure.
    
    Unlike general NLP preprocessing (which aggressively strips formatting),
    clinical note preprocessing must preserve:
    - Section headers (HISTORY OF PRESENT ILLNESS, ASSESSMENT AND PLAN, etc.)
    - Medication dosing notation (e.g., "40mg PO BID")  
    - Lab value formatting (e.g., "Cr 1.7 mg/dL")
    - Bullet/numbered lists in plans (they carry treatment priorities)
    
    We intentionally do NOT lowercase text because drug names, abbreviations
    (CHF, COPD, BID, etc.), and section headers are semantically meaningful
    in their original case.
    """
    # Remove excessive whitespace while preserving intentional line breaks
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:  # Keep non-empty lines
            cleaned_lines.append(stripped)
        elif cleaned_lines and cleaned_lines[-1] != '':
            cleaned_lines.append('')  # Preserve single blank lines (section separators)
    
    text = '\n'.join(cleaned_lines)
    
    # Normalize common clinical abbreviations that might confuse retrieval
    # (but keep the abbreviations — clinicians search using them)
    text = text.replace('w/', 'with ')
    text = text.replace('s/p', 'status post')
    text = text.replace('h/o', 'history of')
    
    return text.strip()


# ============================================================================
# DOCUMENT LOADING
# ============================================================================

def load_clinical_documents(corpus_path: str = None) -> list[Document]:
    """
    Load clinical documents from the generated JSON corpus and convert
    them to LangChain Document objects with rich metadata.
    
    The metadata is crucial for filtered retrieval — e.g., a query about
    "COPD medications" can be routed to documents where 
    metadata.primary_condition == "COPD_exacerbation" AND 
    metadata.document_type IN ("discharge_summary", "medication_reconciliation").
    """
    if corpus_path is None:
        corpus_path = Path(config.data.raw_data_dir) / "clinical_notes_corpus.json"
    else:
        corpus_path = Path(corpus_path)
    
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Clinical corpus not found at {corpus_path}. "
            f"Run 'python -m src.data_generation.generate_clinical_notes' first."
        )
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        raw_documents = json.load(f)
    
    langchain_docs = []
    for doc in raw_documents:
        # Preprocess the clinical text
        cleaned_content = preprocess_clinical_text(doc['content'])
        
        # Build rich metadata for filtered retrieval
        metadata = {
            "doc_id": doc["id"],
            "document_type": doc["type"],
            "condition": doc["condition"],
            "patient_name": doc.get("patient_name", ""),
            "mrn": doc.get("mrn", ""),
            "date": doc.get("date", ""),
            # Include nested metadata fields at top level for ChromaDB compatibility
            # (ChromaDB doesn't support nested dicts in metadata)
            **{k: v for k, v in doc.get("metadata", {}).items() 
               if isinstance(v, (str, int, float, bool))},
        }
        
        langchain_docs.append(Document(
            page_content=cleaned_content,
            metadata=metadata,
        ))
    
    print(f"📄 Loaded {len(langchain_docs)} clinical documents")
    return langchain_docs


# ============================================================================
# CHUNKING PIPELINE
# ============================================================================

def chunk_documents(
    documents: list[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[Document]:
    """
    Split clinical documents into chunks for embedding and retrieval.
    
    The chunking strategy is one of the most impactful parameters in a RAG
    system. For clinical notes specifically, we face a tension:
    
    - SMALLER chunks (256-512 chars): More precise retrieval, but risk losing
      context. A chunk containing "Started IV furosemide 40mg BID" loses meaning
      without knowing the patient has CHF.
      
    - LARGER chunks (1024-2048 chars): Preserve more clinical context, but
      retrieval becomes less precise and we waste context window space with
      irrelevant text.
    
    We default to 1000 chars with 200 overlap — a balanced starting point.
    The evaluation phase will systematically compare 256, 512, and 1024.
    
    The RecursiveCharacterTextSplitter tries separators in order:
    1. "\n\n" — section boundaries (HISTORY, ASSESSMENT, MEDICATIONS)
    2. "\n" — within-section paragraph breaks  
    3. ". " — sentence boundaries
    4. " " — word boundaries (last resort)
    
    This means a section like "MEDICATIONS AT DISCHARGE:" will stay intact
    if it fits within chunk_size, which is exactly what we want.
    """
    chunk_size = chunk_size or config.chunking.chunk_size
    chunk_overlap = chunk_overlap or config.chunking.chunk_overlap
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=config.chunking.separators,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Enrich each chunk's metadata with chunk-level info
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size_chars"] = len(chunk.page_content)
    
    print(f"✂️  Split {len(documents)} documents into {len(chunks)} chunks")
    print(f"   Chunk size: {chunk_size} chars, Overlap: {chunk_overlap} chars")
    print(f"   Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    
    # Deduplicate chunks with identical text content.
    # This happens when multiple documents use the same template and produce
    # chunks with identical text (e.g., medication lists that only differ
    # in patient name, which gets chunked away). Duplicate chunks cause
    # retrieval to waste slots returning the same content multiple times.
    seen_texts = set()
    unique_chunks = []
    for chunk in chunks:
        text_key = chunk.page_content[:500]
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            unique_chunks.append(chunk)
    
    duplicates_removed = len(chunks) - len(unique_chunks)
    if duplicates_removed > 0:
        print(f"   🔄 Removed {duplicates_removed} duplicate chunks")
    print(f"   Final unique chunks: {len(unique_chunks)}")
    
    return unique_chunks


# ============================================================================
# EMBEDDING AND VECTOR STORE
# ============================================================================

def create_vector_store(
    chunks: list[Document],
    embedding_model: str = None,
    persist_directory: str = None,
    collection_name: str = "clinical_notes",
) -> Chroma:
    """
    Generate embeddings for all chunks and store them in ChromaDB.
    
    We use HuggingFaceEmbeddings (sentence-transformers) for local,
    free embedding generation. The default model (all-MiniLM-L6-v2) 
    produces 384-dimensional vectors and is fast enough for interactive use.
    
    ChromaDB is chosen because:
    - Persistent storage (survives restarts)
    - Metadata filtering (filter by condition, document type, etc.)
    - Both similarity and MMR search built in
    - Easy to integrate with LangChain
    - No external server needed (runs in-process)
    """
    embedding_model = embedding_model or config.embedding.model_name
    persist_directory = persist_directory or config.data.vectordb_dir
    
    print(f"🧠 Loading embedding model: {embedding_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": config.embedding.device},
        encode_kwargs={"normalize_embeddings": True},  # L2 normalize for cosine similarity
    )
    
    print(f"📊 Generating embeddings for {len(chunks)} chunks...")
    start_time = time.time()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    
    elapsed = time.time() - start_time
    print(f"✅ Vector store created in {elapsed:.1f}s")
    print(f"   Model: {embedding_model}")
    print(f"   Chunks indexed: {len(chunks)}")
    print(f"   Persisted to: {persist_directory}")
    
    return vectorstore


def load_vector_store(
    embedding_model: str = None,
    persist_directory: str = None,
    collection_name: str = "clinical_notes",
) -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.
    
    This is used when the vector store has already been created (e.g.,
    when launching the Streamlit app or running evaluation). Avoids
    re-embedding the entire corpus on every startup.
    """
    embedding_model = embedding_model or config.embedding.model_name
    persist_directory = persist_directory or config.data.vectordb_dir
    
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": config.embedding.device},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    
    # Verify it loaded correctly
    count = vectorstore._collection.count()
    print(f"📂 Loaded vector store with {count} chunks from {persist_directory}")
    
    return vectorstore


# ============================================================================
# FULL INGESTION PIPELINE
# ============================================================================

def run_ingestion_pipeline(
    corpus_path: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    embedding_model: str = None,
    persist_directory: str = None,
) -> Chroma:
    """
    Execute the complete ingestion pipeline end-to-end:
    Load → Preprocess → Chunk → Embed → Store
    
    This is the main entry point for building the vector store.
    Can be called with custom parameters for evaluation experiments
    (e.g., testing different chunk sizes or embedding models).
    
    Returns the populated ChromaDB vector store ready for retrieval.
    """
    print("=" * 60)
    print("🏥 ClinicalRAG — Document Ingestion Pipeline")
    print("=" * 60)
    
    # Step 1: Load documents
    documents = load_clinical_documents(corpus_path)
    
    # Step 2: Chunk documents
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    
    # Step 3: Create vector store with embeddings
    vectorstore = create_vector_store(chunks, embedding_model, persist_directory)
    
    print("\n" + "=" * 60)
    print("✅ Ingestion pipeline complete!")
    print("=" * 60)
    
    return vectorstore


if __name__ == "__main__":
    vectorstore = run_ingestion_pipeline()
