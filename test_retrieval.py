"""Quick diagnostic to check what the vector store retrieves for test queries."""

from src.ingestion.ingest_documents import load_vector_store

vs = load_vector_store()

test_questions = [
    "What medications are prescribed at discharge for CHF patients?",
    "What are the discharge instructions for COPD exacerbation?",
    "How is sepsis managed in the initial 24 hours?",
]

for question in test_questions:
    print("=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)
    
    # Test MMR retrieval (what the app now uses)
    print("\n  --- MMR Retrieval (k=5, fetch_k=20, lambda=0.7) ---")
    results = vs.max_marginal_relevance_search(question, k=5, fetch_k=20, lambda_mult=0.7)
    for i, doc in enumerate(results, 1):
        condition = doc.metadata.get("condition", "?")
        doc_type = doc.metadata.get("document_type", "?")
        print(f"  {i}. Condition: {condition} | Type: {doc_type}")
        print(f"     Preview: {doc.page_content[:120]}...")
        print()
    print()
