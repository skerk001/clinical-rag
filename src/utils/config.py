"""
Configuration for ClinicalRAG project.

Centralized config management using dataclasses for type safety
and easy modification of key parameters during experimentation.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data generation and storage paths."""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    vectordb_dir: str = "data/vectordb"


@dataclass
class ChunkingConfig:
    """
    Configuration for document chunking strategy.
    
    These parameters significantly impact RAG quality and will be
    systematically compared during evaluation (Phase 4).
    
    chunk_size: Number of characters per chunk. Smaller chunks = more precise
                retrieval but less context. Larger chunks = more context but 
                may include irrelevant information.
    chunk_overlap: Characters of overlap between consecutive chunks. Prevents
                   information loss at chunk boundaries — critical for clinical
                   notes where a diagnosis in one sentence may be explained
                   in the next.
    """
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = field(default_factory=lambda: [
        "\n\n",      # Double newline — section boundaries in clinical notes
        "\n",        # Single newline — paragraph/list item boundaries
        ". ",        # Sentence boundaries
        " ",         # Word boundaries (last resort)
    ])


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding model selection.
    
    We start with all-MiniLM-L6-v2 (general purpose, fast) and will compare
    against PubMedBERT (domain-specific, potentially better for clinical text)
    during evaluation.
    """
    model_name: str = "all-MiniLM-L6-v2"
    # Alternative for clinical domain: "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
    device: str = "cpu"


@dataclass
class RetrievalConfig:
    """
    Configuration for retrieval strategy.
    
    search_type options:
      - "similarity": Pure cosine similarity — returns k most similar chunks
      - "mmr": Maximal Marginal Relevance — balances relevance with diversity
               to avoid returning redundant chunks. Lambda controls the
               balance (0 = max diversity, 1 = max relevance).
    """
    search_type: str = "mmr"
    k: int = 5                    # Number of chunks to retrieve
    mmr_lambda: float = 0.7       # Only used when search_type = "mmr" (0.7 = favor relevance with some diversity)
    score_threshold: float = 0.3  # Minimum similarity score to include
    fetch_k: int = 20             # Number of candidates to fetch before MMR reranking


@dataclass
class LLMConfig:
    """
    Configuration for the local LLM via Ollama.
    
    We use Llama 3 8B as the default — it's the best balance of quality
    and speed for local inference. The temperature is set low (0.1) because
    clinical question answering demands factual precision over creativity.
    """
    model_name: str = "llama3"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 0.9


@dataclass 
class AppConfig:
    """Top-level application configuration aggregating all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # Project metadata
    project_name: str = "ClinicalRAG"
    version: str = "0.1.0"


# Default configuration instance — import this throughout the project
config = AppConfig()
