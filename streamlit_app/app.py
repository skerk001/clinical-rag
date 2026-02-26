"""
ClinicalRAG — Streamlit Frontend

This is the user-facing interface for the RAG system. It provides:
- A chat-style interface for asking clinical questions
- Real-time display of retrieved sources with expandable details
- Confidence and groundedness indicators
- Metadata filtering (by condition and document type)
- Performance metrics (retrieval and generation latency)

The design prioritizes clinical trustworthiness: sources are always visible,
confidence is always displayed, and groundedness warnings are prominent.

Launch with: streamlit run streamlit_app/app.py

Author: Samir Kerkar
"""

import streamlit as st
import sys
from pathlib import Path
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="ClinicalRAG — Clinical Question Answering",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a clean clinical interface
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    
    /* Confidence badge colors */
    .confidence-high { 
        background-color: #d1fae5; color: #065f46; 
        padding: 4px 12px; border-radius: 12px; font-weight: 600;
    }
    .confidence-medium { 
        background-color: #fef3c7; color: #92400e; 
        padding: 4px 12px; border-radius: 12px; font-weight: 600;
    }
    .confidence-low { 
        background-color: #fee2e2; color: #991b1b; 
        padding: 4px 12px; border-radius: 12px; font-weight: 600;
    }
    .confidence-insufficient { 
        background-color: #e5e7eb; color: #374151; 
        padding: 4px 12px; border-radius: 12px; font-weight: 600;
    }
    
    /* Source card styling */
    .source-card {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    
    /* Groundedness warning */
    .grounded-ok { color: #065f46; }
    .grounded-warn { color: #991b1b; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR — Configuration and Filters
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Search settings
    st.markdown("### Retrieval Settings")
    search_type = st.selectbox(
        "Search Strategy",
        options=["similarity", "mmr"],
        index=0,
        help="'similarity' returns the most similar chunks. 'mmr' (Maximal Marginal Relevance) balances relevance with diversity to avoid redundant results."
    )
    
    search_k = st.slider(
        "Number of sources to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="How many document chunks to retrieve and provide as context to the LLM. More sources = more context but slower generation."
    )
    
    # Document filters
    st.markdown("### 🔍 Document Filters")
    st.caption("Optionally filter to specific conditions or document types")
    
    condition_filter = st.selectbox(
        "Filter by Condition",
        options=["All Conditions", "CHF", "COPD_exacerbation", "Pneumonia", "Sepsis", "Type2_diabetes", "AKI", "Stroke"],
        index=0,
    )
    
    doctype_filter = st.selectbox(
        "Filter by Document Type",
        options=["All Types", "discharge_summary", "progress_note", "medication_reconciliation", "radiology_report", "lab_interpretation"],
        index=0,
    )
    
    # Prompt style
    st.markdown("### 💬 Response Style")
    prompt_style = st.radio(
        "Output format",
        options=["structured", "conversational"],
        index=0,
        help="'structured' gives formal output with labeled sections. 'conversational' is more natural for chat."
    )
    
    st.markdown("---")
    st.markdown("### 📊 About This System")
    st.markdown("""
    **ClinicalRAG** is a retrieval-augmented generation system for clinical question answering.
    
    It answers questions by searching a corpus of clinical documents (discharge summaries, progress notes, 
    medication reconciliation notes, radiology reports, and lab interpretations) and generating grounded 
    responses with source citations.
    
    **Tech Stack:** LangChain · ChromaDB · Llama 3 (via Ollama) · Streamlit
    
    **⚠️ Disclaimer:** This system uses synthetic clinical data for demonstration purposes. 
    It is not intended for actual clinical decision-making.
    """)


# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

st.markdown('<p class="main-header">🏥 ClinicalRAG</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Clinical Question Answering with Source Citations and Hallucination Guardrails</p>', unsafe_allow_html=True)


# ============================================================================
# INITIALIZE THE RAG CHAIN (cached so it only loads once)
# ============================================================================

@st.cache_resource
def initialize_rag_chain(search_type, search_k, prompt_style):
    """
    Initialize the RAG chain. This is cached by Streamlit so the vector store
    and LLM connection are only loaded once, not on every page refresh.
    """
    from src.retrieval.rag_chain import ClinicalRAGChain
    
    try:
        chain = ClinicalRAGChain(
            search_type=search_type,
            search_k=search_k,
            prompt_style=prompt_style,
        )
        return chain, None
    except Exception as e:
        return None, str(e)


# Try to initialize
chain, init_error = initialize_rag_chain(search_type, search_k, prompt_style)

if init_error:
    st.error(f"""
    **Failed to initialize the RAG chain.** 
    
    Error: `{init_error}`
    
    Please make sure:
    1. You've run the ingestion pipeline: `python -m src.ingestion.ingest_documents`
    2. Ollama is running: `ollama serve`
    3. Llama 3 is pulled: `ollama pull llama3`
    """)
    st.stop()


# ============================================================================
# CHAT INTERFACE
# ============================================================================

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If this is an assistant message with extra data, show expandable details
        if message["role"] == "assistant" and "result_data" in message:
            result = message["result_data"]
            _display_result_details(result) if callable(globals().get('_display_result_details')) else None

# Sample questions as quick-start buttons
if not st.session_state.messages:
    st.markdown("### 💡 Try a sample question:")
    
    sample_cols = st.columns(2)
    sample_questions = [
        "What medications are prescribed at discharge for CHF patients?",
        "What are the discharge instructions for COPD exacerbation?",
        "How is sepsis initially managed based on these records?",
        "What are the risks of NSAIDs in kidney disease patients?",
        "What follow-up is recommended after a stroke?",
        "What drug interactions should be monitored in CHF patients?",
    ]
    
    for i, q in enumerate(sample_questions):
        col = sample_cols[i % 2]
        if col.button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()


# ============================================================================
# HANDLE USER INPUT
# ============================================================================

# Check for pending question from sample buttons
pending = st.session_state.pop("pending_question", None)

# Chat input
user_question = st.chat_input("Ask a clinical question...")

# Use either the pending question or the typed question
question = pending or user_question

if question:
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching clinical documents and generating answer..."):
            # Apply filters if set
            condition = condition_filter if condition_filter != "All Conditions" else None
            doctype = doctype_filter if doctype_filter != "All Types" else None
            
            if condition or doctype:
                result = chain.query_with_filter(question, condition=condition, document_type=doctype)
            else:
                result = chain.query(question)
        
        # Display the answer
        st.markdown(result["answer"])
        
        # Confidence badge
        confidence = result["confidence"]
        confidence_class = {
            "HIGH": "confidence-high",
            "MEDIUM": "confidence-medium", 
            "LOW": "confidence-low",
            "INSUFFICIENT": "confidence-insufficient",
        }.get(confidence, "confidence-low")
        
        st.markdown(f'<span class="{confidence_class}">Confidence: {confidence}</span>', unsafe_allow_html=True)
        
        # Groundedness check
        g = result["groundedness"]
        if g["is_grounded"]:
            st.success(f"✅ {g['explanation']}")
        else:
            st.warning(f"⚠️ {g['explanation']}")
        
        # Sources in expandable section
        with st.expander(f"📚 View Sources ({len(result['citations'])} cited)", expanded=False):
            if result["citations"]:
                for c in result["citations"]:
                    st.markdown(f"""
                    **[Source {c['source_number']}]** `{c['document_type']}` | `{c['condition']}` | `{c['date']}`
                    
                    > {c['excerpt']}
                    """)
                    st.markdown("---")
            else:
                st.info("No specific sources were cited in the response.")
        
        # Performance metrics in expandable section
        with st.expander("⏱️ Performance Metrics", expanded=False):
            latency = result["latency"]
            metric_cols = st.columns(3)
            metric_cols[0].metric("Total Latency", f"{latency['total_seconds']}s")
            if "retrieval_seconds" in latency:
                metric_cols[1].metric("Retrieval", f"{latency['retrieval_seconds']}s")
                metric_cols[2].metric("Generation", f"{latency['generation_seconds']}s")
            
            # Groundedness details
            g_details = g.get("details", {})
            if g_details:
                st.markdown(f"""
                **Groundedness Check Details:**
                - Dosages checked: {g_details.get('dosages_checked', 0)} (grounded: {g_details.get('dosages_grounded', 0)})
                - Lab values checked: {g_details.get('labs_checked', 0)} (grounded: {g_details.get('labs_grounded', 0)})
                - Overall score: {g['grounding_score']:.0%}
                """)
            
            if result.get("filters_applied"):
                st.markdown(f"**Filters applied:** {result['filters_applied']}")
        
        # All retrieved chunks (for debugging/transparency)
        with st.expander("🔎 All Retrieved Chunks (raw)", expanded=False):
            for i, doc in enumerate(result["retrieved_docs"], 1):
                st.markdown(f"**Chunk {i}** — `{doc.metadata.get('document_type', '?')}` | `{doc.metadata.get('condition', '?')}`")
                st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                st.markdown("---")
    
    # Save to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": result["answer"],
        "result_data": {
            "confidence": result["confidence"],
            "citations": result["citations"],
            "groundedness": result["groundedness"],
            "latency": result["latency"],
        }
    })
