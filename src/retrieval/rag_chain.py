"""
RAG Retrieval Chain for ClinicalRAG

This is the core "brain" of the system — it connects all the pieces:
1. Takes a user's clinical question
2. Searches the vector database for relevant document chunks
3. Assembles those chunks as context for the LLM
4. Sends the question + context to Llama 3 (via Ollama)
5. Returns a grounded answer with source citations

The key design philosophy: EVERY claim in the response must be traceable
back to a specific document chunk. This is non-negotiable in healthcare AI
because hallucinated clinical information could be dangerous.

Think of this file as the "librarian's brain" from our analogy:
- It knows HOW to search the shelves (retrieval strategy)
- It knows HOW to read the relevant pages (context assembly)  
- It knows HOW to compose a clear answer (prompt engineering)
- It knows HOW to cite its sources (citation extraction)
- It knows WHEN to say "I don't know" (hallucination guardrails)

Author: Samir Kerkar
"""

import time
from pathlib import Path
from typing import Optional

from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import config
from src.ingestion.ingest_documents import load_vector_store


# ============================================================================
# CLINICAL PROMPT TEMPLATE
# ============================================================================
# This is one of the most important pieces of the entire project.
# The prompt template tells Llama 3 exactly HOW to behave when answering
# clinical questions. A poorly written prompt = hallucinated medical info.
# A well-written prompt = grounded, cited, trustworthy answers.
#
# Key design decisions in this prompt:
#
# 1. ROLE FRAMING: We tell the LLM it's a "clinical information retrieval 
#    assistant" — not a doctor, not giving medical advice. This frames the
#    task as information lookup rather than medical decision-making.
#
# 2. STRICT GROUNDING: "Base your answer ONLY on the provided context."
#    This is the #1 most important instruction. Without it, Llama 3 will
#    happily make up clinical details from its training data — which may
#    be outdated or wrong.
#
# 3. CITATION FORMAT: We tell it exactly how to cite sources using [Source N]
#    notation. This makes it easy to parse citations programmatically later.
#
# 4. "I DON'T KNOW" INSTRUCTION: Explicitly telling the LLM it's okay to
#    say "the provided documents don't contain this information" prevents
#    the model from guessing when it doesn't have relevant context.
#
# 5. STRUCTURED OUTPUT: Requesting a specific format (Answer, Sources, 
#    Confidence) makes the output predictable and parseable.
# ============================================================================

CLINICAL_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a clinical information retrieval assistant. Your job is to answer clinical questions using ONLY the information found in the provided medical documents below. You must be accurate, precise, and always cite your sources.

CRITICAL RULES:
1. Base your answer ONLY on the provided context documents. Do NOT use any external knowledge or make assumptions beyond what the documents state.
2. Cite every factual claim using [Source N] notation, where N corresponds to the source number listed below.
3. If the provided documents do not contain enough information to answer the question, clearly state: "The provided clinical documents do not contain sufficient information to answer this question."
4. Never fabricate, infer, or speculate about clinical information not explicitly stated in the sources.
5. When discussing medications, always include the dose, route, and frequency exactly as stated in the source documents.
6. If different sources contain conflicting information, acknowledge the discrepancy and cite both sources.

CONTEXT DOCUMENTS:
{context}

CLINICAL QUESTION: {question}

Provide your response in the following format:

ANSWER:
[Your detailed answer here, with [Source N] citations for every factual claim]

SOURCES USED:
[List each source you cited with a brief description of what information it provided]

CONFIDENCE LEVEL:
[HIGH — if the answer is directly and completely supported by the context documents]
[MEDIUM — if the answer is partially supported but some aspects required interpretation]
[LOW — if the documents provide only tangential information relevant to the question]
[INSUFFICIENT — if the documents do not contain relevant information]
"""
)


# ============================================================================
# A simpler prompt for cases where we want more conversational responses
# (e.g., for the Streamlit chat interface where the structured format
# might feel too rigid)
# ============================================================================

CONVERSATIONAL_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a clinical information retrieval assistant helping healthcare professionals find information in medical records. Answer the following question using ONLY the information in the provided documents.

Rules:
- Only use information from the provided context. Do not use external knowledge.
- Cite sources as [Source N] after each claim.
- If the documents don't contain the answer, say so honestly.
- For medications, always include dose, route, and frequency as documented.

Context Documents:
{context}

Question: {question}

Answer (cite sources with [Source N]):"""
)


# ============================================================================
# CONTEXT FORMATTER
# ============================================================================
# When we retrieve chunks from ChromaDB, they come as LangChain Document
# objects with raw text and metadata. Before sending them to the LLM,
# we need to format them into a numbered, readable format that the LLM
# can reference with [Source N] citations.
#
# We also include metadata (document type, condition, date) because this
# helps the LLM understand what kind of document each chunk came from.
# A medication list carries different weight than a progress note.
# ============================================================================

def format_retrieved_context(documents: list[Document]) -> str:
    """
    Format retrieved document chunks into a numbered, LLM-readable context string.
    
    Each source gets a header with its metadata (type, condition, date) and
    a sequential number that the LLM will use for [Source N] citations.
    
    Example output:
        [Source 1] (discharge_summary | CHF | 03/15/2025)
        Patient was started on IV furosemide 40mg BID with strict I&O monitoring...
        
        [Source 2] (medication_reconciliation | CHF | 03/17/2025)
        Furosemide: INCREASED from 40mg PO daily → 60mg PO daily...
    """
    formatted_parts = []
    
    for i, doc in enumerate(documents, 1):
        # Extract useful metadata for the source header
        doc_type = doc.metadata.get("document_type", "unknown")
        condition = doc.metadata.get("condition", "unknown")
        date = doc.metadata.get("date", "unknown")
        
        # Build a clear, informative source header
        header = f"[Source {i}] ({doc_type} | {condition} | {date})"
        
        # Combine header with the actual content
        formatted_parts.append(f"{header}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(formatted_parts)


# ============================================================================
# RESPONSE PARSER
# ============================================================================
# After the LLM generates a response, we need to parse it into structured
# components: the answer text, the sources used, and the confidence level.
# This structured parsing enables the Streamlit frontend to display each
# component differently (e.g., sources in a collapsible section, confidence
# as a colored badge).
# ============================================================================

def parse_rag_response(raw_response: str) -> dict:
    """
    Parse the LLM's structured response into components.
    
    Expected format from the LLM:
        ANSWER:
        [answer text with citations]
        
        SOURCES USED:
        [source list]
        
        CONFIDENCE LEVEL:
        [HIGH/MEDIUM/LOW/INSUFFICIENT]
    
    Returns a dictionary with keys: answer, sources, confidence, raw_response.
    If parsing fails (LLM didn't follow format), returns the raw response
    as the answer with confidence "UNKNOWN".
    """
    result = {
        "answer": "",
        "sources": "",
        "confidence": "UNKNOWN",
        "raw_response": raw_response.strip(),
    }
    
    # Try to extract the ANSWER section
    # We look for "ANSWER:" and then grab everything until "SOURCES USED:"
    text = raw_response.strip()
    
    # Split by section headers (case-insensitive matching)
    answer_start = -1
    sources_start = -1
    confidence_start = -1
    
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line_upper = line.strip().upper()
        if line_upper.startswith("ANSWER"):
            answer_start = i + 1
        elif line_upper.startswith("SOURCES USED") or line_upper.startswith("SOURCES:"):
            sources_start = i + 1
        elif line_upper.startswith("CONFIDENCE"):
            confidence_start = i + 1
    
    # Extract each section based on found boundaries
    if answer_start >= 0:
        end = sources_start - 1 if sources_start > 0 else (confidence_start - 1 if confidence_start > 0 else len(lines))
        result["answer"] = '\n'.join(lines[answer_start:end]).strip()
    
    if sources_start >= 0:
        end = confidence_start - 1 if confidence_start > 0 else len(lines)
        result["sources"] = '\n'.join(lines[sources_start:end]).strip()
    
    if confidence_start >= 0:
        confidence_text = '\n'.join(lines[confidence_start:]).strip().upper()
        # Extract just the confidence level keyword
        for level in ["HIGH", "MEDIUM", "LOW", "INSUFFICIENT"]:
            if level in confidence_text:
                result["confidence"] = level
                break
    
    # Fallback: if we couldn't parse the structured format, use raw response
    if not result["answer"]:
        result["answer"] = raw_response.strip()
    
    return result


# ============================================================================
# CITATION EXTRACTOR
# ============================================================================
# This pulls out which [Source N] references the LLM actually used in its
# answer, and maps them back to the original retrieved documents. This lets
# us show the user exactly which document chunks informed the answer.
# ============================================================================

def extract_citations(answer_text: str, retrieved_docs: list[Document]) -> list[dict]:
    """
    Extract [Source N] citations from the answer and map them to documents.
    
    Returns a list of citation objects, each containing:
    - source_number: The N from [Source N]
    - document_type: What kind of document (discharge_summary, etc.)
    - condition: The clinical condition
    - date: When the document was created
    - excerpt: First 200 chars of the chunk for preview
    - full_text: The complete chunk text
    
    This powers the "Sources" panel in the Streamlit UI where users can
    click to see exactly what the LLM based its answer on.
    """
    import re
    
    # Find all [Source N] references in the answer
    citation_pattern = r'\[Source\s+(\d+)\]'
    found_numbers = set(int(n) for n in re.findall(citation_pattern, answer_text))
    
    citations = []
    for num in sorted(found_numbers):
        # Source numbers are 1-indexed, document list is 0-indexed
        doc_index = num - 1
        if 0 <= doc_index < len(retrieved_docs):
            doc = retrieved_docs[doc_index]
            citations.append({
                "source_number": num,
                "document_type": doc.metadata.get("document_type", "unknown"),
                "condition": doc.metadata.get("condition", "unknown"),
                "date": doc.metadata.get("date", "unknown"),
                "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "full_text": doc.page_content,
            })
    
    return citations


# ============================================================================
# HALLUCINATION GUARDRAIL
# ============================================================================
# This is a simple but effective guardrail that checks whether the LLM's
# answer contains claims that appear to be grounded in the provided context.
#
# How it works:
# 1. Extract key medical terms from the LLM's answer
# 2. Check how many of those terms also appear in the retrieved context
# 3. If too few terms are grounded, flag the response as potentially
#    containing hallucinated information
#
# This is a heuristic approach — not perfect, but catches the most common
# failure mode: the LLM making up specific drug names, dosages, or lab
# values that aren't in the source documents.
#
# A more sophisticated approach (for Phase 4) would use a separate LLM
# call to verify each claim, but that doubles latency and cost.
# ============================================================================

def check_groundedness(answer: str, context: str) -> dict:
    """
    Simple hallucination check: verify that key clinical terms in the answer
    also appear in the provided context.
    
    Returns a dict with:
    - is_grounded: bool — overall assessment
    - grounding_score: float — 0.0 to 1.0, proportion of answer terms found in context
    - flagged_terms: list — terms found in answer but NOT in context (potential hallucinations)
    - explanation: str — human-readable explanation of the check
    """
    import re
    
    # Clinical terms we specifically check for grounding:
    # Drug names, dosages, lab values, and specific clinical findings
    # These are the categories most dangerous to hallucinate
    
    # Extract potential drug dosages (e.g., "40mg", "500mg PO BID")
    dosage_pattern = r'\d+\s*(?:mg|mcg|mEq|units?|mL|g)\b'
    answer_dosages = set(re.findall(dosage_pattern, answer, re.IGNORECASE))
    context_dosages = set(re.findall(dosage_pattern, context, re.IGNORECASE))
    
    # Extract potential lab values (e.g., "creatinine 1.7", "BNP 1847")
    lab_pattern = r'(?:creatinine|BNP|potassium|sodium|WBC|hemoglobin|HbA1c|lactate|pH|pCO2|eGFR|BUN)\s*(?:of\s+|:\s*|was\s+|is\s+)?[\d.]+'
    answer_labs = set(re.findall(lab_pattern, answer, re.IGNORECASE))
    context_labs = set(re.findall(lab_pattern, context, re.IGNORECASE))
    
    # Check grounding for dosages
    ungrounded_dosages = answer_dosages - context_dosages
    grounded_dosages = answer_dosages & context_dosages
    
    # Check grounding for lab values
    ungrounded_labs = answer_labs - context_labs
    grounded_labs = answer_labs & context_labs
    
    # Calculate overall grounding score
    total_checkable = len(answer_dosages) + len(answer_labs)
    total_grounded = len(grounded_dosages) + len(grounded_labs)
    
    if total_checkable == 0:
        # No specific values to check — give benefit of the doubt
        grounding_score = 1.0
        is_grounded = True
    else:
        grounding_score = total_grounded / total_checkable
        is_grounded = grounding_score >= 0.7  # 70% threshold
    
    flagged_terms = list(ungrounded_dosages | ungrounded_labs)
    
    # Build explanation
    if is_grounded:
        explanation = f"Response appears well-grounded. {total_grounded}/{total_checkable} clinical values verified in source documents."
    else:
        explanation = (
            f"⚠️ Potential hallucination detected. Only {total_grounded}/{total_checkable} "
            f"clinical values were found in source documents. "
            f"Unverified terms: {', '.join(flagged_terms[:5])}"
        )
    
    return {
        "is_grounded": is_grounded,
        "grounding_score": grounding_score,
        "flagged_terms": flagged_terms,
        "explanation": explanation,
        "details": {
            "dosages_checked": len(answer_dosages),
            "dosages_grounded": len(grounded_dosages),
            "labs_checked": len(answer_labs),
            "labs_grounded": len(grounded_labs),
        }
    }


# ============================================================================
# CLINICAL RAG CHAIN — THE MAIN ORCHESTRATOR
# ============================================================================
# This class ties everything together. It's the "conductor" of the orchestra:
#   - The vector store is the sheet music library
#   - The retriever is the musician who finds the right pages
#   - The LLM is the performer who plays the music
#   - The prompt template is the conductor's instructions
#   - The guardrails are the quality control checks
#
# Usage:
#   chain = ClinicalRAGChain()       # Initialize once
#   result = chain.query("What medications are used for CHF?")  # Ask questions
# ============================================================================

class ClinicalRAGChain:
    """
    End-to-end RAG chain for clinical question answering.
    
    This class orchestrates the complete pipeline:
    Query → Retrieve → Format Context → Generate Answer → Parse → Verify
    
    It's designed to be initialized once (loading the vector store and LLM)
    and then called repeatedly with different questions. This avoids the
    overhead of reloading models for each query.
    """
    
    def __init__(
        self,
        vectorstore: Optional[Chroma] = None,
        model_name: str = None,
        search_type: str = None,
        search_k: int = None,
        prompt_style: str = "structured",  # "structured" or "conversational"
    ):
        """
        Initialize the RAG chain by loading the vector store and LLM.
        
        Args:
            vectorstore: Pre-loaded ChromaDB instance. If None, loads from disk.
            model_name: Ollama model name (default: from config, usually "llama3")
            search_type: "similarity" or "mmr" (default: from config)
            search_k: Number of chunks to retrieve (default: from config)
            prompt_style: "structured" for formal output, "conversational" for chat
            streaming: Whether to stream LLM output token-by-token
        """
        # Load or use provided vector store
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            print("📂 Loading vector store from disk...")
            self.vectorstore = load_vector_store()
        
        # Configure retrieval strategy
        self.search_type = search_type or config.retrieval.search_type
        self.search_k = search_k or config.retrieval.k
        
        # Set up the retriever from the vector store
        # This creates a LangChain Retriever object that wraps ChromaDB's search
        retriever_kwargs = {"k": self.search_k}
        if self.search_type == "mmr":
            retriever_kwargs["lambda_mult"] = config.retrieval.mmr_lambda
            retriever_kwargs["fetch_k"] = config.retrieval.fetch_k
        
        self.retriever = self.vectorstore.as_retriever(
            search_type=self.search_type,
            search_kwargs=retriever_kwargs,
        )
        
        # Initialize the LLM (Llama 3 via Ollama)
        model_name = model_name or config.llm.model_name
        print(f"🤖 Connecting to Ollama ({model_name})...")
        
        self.llm = OllamaLLM(
            model=model_name,
            base_url=config.llm.base_url,
            temperature=config.llm.temperature,
            num_predict=config.llm.max_tokens,
        )
        
        # Select prompt template
        if prompt_style == "conversational":
            self.prompt = CONVERSATIONAL_RAG_PROMPT
        else:
            self.prompt = CLINICAL_RAG_PROMPT
        
        print(f"✅ RAG chain initialized")
        print(f"   Search: {self.search_type} (k={self.search_k})")
        print(f"   LLM: {model_name}")
        print(f"   Prompt: {prompt_style}")
    
    def retrieve(self, question: str) -> list[Document]:
        """
        Step 1: Search the vector database for chunks relevant to the question.
        
        This converts the question into a vector (using the same embedding model
        that was used during ingestion) and finds the k most similar chunks.
        
        With "similarity" search, it returns the k closest vectors by cosine distance.
        With "mmr" search, it balances relevance with diversity to avoid returning
        5 chunks that all say the same thing.
        """
        docs = self.retriever.invoke(question)
        return docs
    
    def generate(self, question: str, context_docs: list[Document]) -> str:
        """
        Step 2: Send the question + retrieved context to Llama 3 for generation.
        
        The formatted context includes source numbers that the LLM uses for
        [Source N] citations. The prompt template instructs the LLM to ONLY
        use information from the provided context — this is the core of
        preventing hallucinations.
        """
        # Format the retrieved documents into numbered context
        formatted_context = format_retrieved_context(context_docs)
        
        # Fill in the prompt template
        full_prompt = self.prompt.format(
            context=formatted_context,
            question=question,
        )
        
        # Send to Llama 3 and get the response
        raw_response = self.llm.invoke(full_prompt)
        
        return raw_response
    
    def query(self, question: str) -> dict:
        """
        The main entry point — run the complete RAG pipeline for a question.
        
        This orchestrates the full flow:
        1. Retrieve relevant chunks from the vector database
        2. Format them as context with source numbers
        3. Send to Llama 3 with the clinical prompt
        4. Parse the structured response
        5. Extract citations and map to source documents
        6. Run hallucination check
        7. Return everything in a clean result dictionary
        
        Returns a dict containing:
        - question: The original question
        - answer: The LLM's answer text
        - sources: Source description from the LLM
        - confidence: HIGH/MEDIUM/LOW/INSUFFICIENT
        - citations: List of cited documents with excerpts
        - groundedness: Hallucination check results
        - retrieved_docs: The raw retrieved document chunks
        - latency: How long the query took (seconds)
        - raw_response: The unprocessed LLM output
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant context
        retrieved_docs = self.retrieve(question)
        retrieval_time = time.time() - start_time
        
        # Step 2: Generate answer
        raw_response = self.generate(question, retrieved_docs)
        generation_time = time.time() - start_time - retrieval_time
        
        # Step 3: Parse the structured response
        parsed = parse_rag_response(raw_response)
        
        # Step 4: Extract and map citations
        citations = extract_citations(parsed["answer"], retrieved_docs)
        
        # Step 5: Run hallucination check
        context_text = format_retrieved_context(retrieved_docs)
        groundedness = check_groundedness(parsed["answer"], context_text)
        
        total_time = time.time() - start_time
        
        return {
            "question": question,
            "answer": parsed["answer"],
            "sources": parsed["sources"],
            "confidence": parsed["confidence"],
            "citations": citations,
            "groundedness": groundedness,
            "retrieved_docs": retrieved_docs,
            "latency": {
                "retrieval_seconds": round(retrieval_time, 3),
                "generation_seconds": round(generation_time, 3),
                "total_seconds": round(total_time, 3),
            },
            "raw_response": raw_response,
        }
    
    def query_with_filter(
        self, 
        question: str, 
        condition: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> dict:
        """
        Query with metadata filtering — search only specific types of documents.
        
        This is powerful for targeted clinical queries. For example:
        - "What medications for CHF?" → filter to condition="CHF" 
        - "What did the radiology report show?" → filter to document_type="radiology_report"
        
        This reduces noise in retrieval by excluding irrelevant document types,
        which improves both answer quality and reduces hallucination risk.
        """
        # Build the metadata filter for ChromaDB
        where_filter = {}
        if condition:
            where_filter["condition"] = condition
        if document_type:
            where_filter["document_type"] = document_type
        
        # If we have filters, create a filtered retriever
        if where_filter:
            # For multiple filters, ChromaDB uses $and operator
            if len(where_filter) > 1:
                chroma_filter = {
                    "$and": [
                        {k: {"$eq": v}} for k, v in where_filter.items()
                    ]
                }
            else:
                key, value = next(iter(where_filter.items()))
                chroma_filter = {key: {"$eq": value}}
            
            filtered_docs = self.vectorstore.similarity_search(
                question,
                k=self.search_k,
                filter=chroma_filter,
            )
        else:
            filtered_docs = self.retrieve(question)
        
        # Generate answer using retrieved docs
        start_time = time.time()
        raw_response = self.generate(question, filtered_docs)
        parsed = parse_rag_response(raw_response)
        citations = extract_citations(parsed["answer"], filtered_docs)
        context_text = format_retrieved_context(filtered_docs)
        groundedness = check_groundedness(parsed["answer"], context_text)
        total_time = time.time() - start_time
        
        return {
            "question": question,
            "answer": parsed["answer"],
            "sources": parsed["sources"],
            "confidence": parsed["confidence"],
            "citations": citations,
            "groundedness": groundedness,
            "retrieved_docs": filtered_docs,
            "filters_applied": where_filter,
            "latency": {"total_seconds": round(total_time, 3)},
            "raw_response": raw_response,
        }


# ============================================================================
# PRETTY PRINTER FOR CLI USAGE
# ============================================================================
# This formats the query results nicely for terminal output. Not strictly
# necessary, but makes development and testing much more pleasant.
# ============================================================================

def print_result(result: dict) -> None:
    """Pretty-print a query result to the terminal."""
    print("\n" + "=" * 70)
    print(f"❓ QUESTION: {result['question']}")
    print("=" * 70)
    
    print(f"\n📝 ANSWER:")
    print(result["answer"])
    
    if result.get("sources"):
        print(f"\n📚 SOURCES:")
        print(result["sources"])
    
    print(f"\n📊 CONFIDENCE: {result['confidence']}")
    
    # Groundedness check
    g = result["groundedness"]
    if g["is_grounded"]:
        print(f"✅ GROUNDED: {g['explanation']}")
    else:
        print(f"⚠️  GROUNDEDNESS WARNING: {g['explanation']}")
    
    # Citations
    if result["citations"]:
        print(f"\n🔗 CITATIONS ({len(result['citations'])} sources referenced):")
        for c in result["citations"]:
            print(f"   [Source {c['source_number']}] {c['document_type']} | {c['condition']} | {c['date']}")
            print(f"   Preview: {c['excerpt'][:100]}...")
    
    # Performance
    latency = result["latency"]
    print(f"\n⏱️  LATENCY: {latency['total_seconds']}s total", end="")
    if "retrieval_seconds" in latency:
        print(f" (retrieval: {latency['retrieval_seconds']}s, generation: {latency['generation_seconds']}s)")
    else:
        print()
    
    print("=" * 70)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================
# Run this file directly to test the RAG chain interactively from terminal.
# This is useful during development — you can ask questions and see the
# full pipeline in action without launching the Streamlit UI.
#
# Usage: python -m src.retrieval.rag_chain
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🏥 ClinicalRAG — Interactive Clinical Question Answering")
    print("=" * 70)
    print("\nInitializing RAG chain (this may take a moment)...\n")
    
    try:
        chain = ClinicalRAGChain(prompt_style="structured")
    except Exception as e:
        print(f"\n❌ Error initializing RAG chain: {e}")
        print("\nMake sure:")
        print("  1. You've run the ingestion pipeline first:")
        print("     python -m src.ingestion.ingest_documents")
        print("  2. Ollama is running with Llama 3:")
        print("     ollama serve")
        print("     ollama pull llama3")
        sys.exit(1)
    
    # Sample clinical questions for testing
    sample_questions = [
        "What medications are typically prescribed at discharge for a patient with acute decompensated heart failure?",
        "What are the key discharge instructions for COPD patients after an acute exacerbation?",
        "How is sepsis managed in the initial 24 hours based on these clinical documents?",
        "What are the risks of using NSAIDs in patients with chronic kidney disease?",
        "What is the recommended follow-up schedule after discharge for a stroke patient?",
    ]
    
    print("📋 Sample questions you can try:")
    for i, q in enumerate(sample_questions, 1):
        print(f"   {i}. {q}")
    
    print(f"\nType a question (or a number 1-{len(sample_questions)} for a sample), or 'quit' to exit.\n")
    
    while True:
        user_input = input("🔍 Your question: ").strip()
        
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("\nGoodbye! 👋")
            break
        
        # Check if user entered a sample question number
        if user_input.isdigit() and 1 <= int(user_input) <= len(sample_questions):
            question = sample_questions[int(user_input) - 1]
            print(f"   → Using sample question: {question}")
        else:
            question = user_input
        
        if not question:
            continue
        
        print("\n⏳ Searching and generating answer...")
        result = chain.query(question)
        print_result(result)
        print()
