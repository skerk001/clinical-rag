"""
RAG Evaluation Framework for ClinicalRAG

This module provides a comprehensive evaluation pipeline that measures
RAG system quality across multiple dimensions:

1. RETRIEVAL QUALITY: Are we finding the right documents?
   - Context Precision: Are retrieved chunks relevant to the question?
   - Context Recall: Do we retrieve chunks from the expected conditions?
   - Diversity Score: Are retrieved chunks diverse (not duplicates)?

2. ANSWER QUALITY: Is the generated answer good?
   - Keyword Coverage: Does the answer contain expected medical terms?
   - Citation Presence: Does the answer cite sources?
   - Faithfulness Proxy: Are clinical values in the answer grounded in context?
   - Abstention Accuracy: Does it correctly refuse unanswerable questions?

3. SYSTEM PERFORMANCE: How fast is it?
   - Retrieval latency
   - Generation latency
   - Total end-to-end latency

The framework supports running comparisons across different configurations
(chunk sizes, retrieval strategies, embedding models) and generates
visualizations for the README and portfolio.

Author: Samir Kerkar
"""

import json
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.question_bank import EVALUATION_QUESTIONS, get_all_categories
from src.utils.config import config


# ============================================================================
# INDIVIDUAL METRIC FUNCTIONS
# ============================================================================

def evaluate_retrieval_condition_match(
    retrieved_docs: list,
    expected_conditions: list[str],
) -> dict:
    """
    Measure whether retrieved documents come from the expected conditions.
    
    For example, if the question is about CHF medications, we expect
    retrieved documents to have condition="CHF" in their metadata.
    
    Returns:
        - condition_recall: What fraction of expected conditions appear
          in the retrieved documents (0.0 to 1.0)
        - condition_precision: What fraction of retrieved documents
          belong to an expected condition (0.0 to 1.0)
        - retrieved_conditions: Set of conditions actually retrieved
    """
    if not expected_conditions:
        # For abstention questions, there are no expected conditions.
        # Any retrieval is technically "wrong" but we don't penalize it
        # since the system can't control what's in the vector store.
        return {
            "condition_recall": 1.0,  # N/A for abstention
            "condition_precision": 0.0,  # Expected: nothing relevant
            "retrieved_conditions": set(),
            "is_abstention_question": True,
        }
    
    retrieved_conditions = set()
    matching_docs = 0
    
    for doc in retrieved_docs:
        condition = doc.metadata.get("condition", "")
        retrieved_conditions.add(condition)
        if condition in expected_conditions:
            matching_docs += 1
    
    # Recall: of the conditions we expected, how many did we find?
    conditions_found = len(set(expected_conditions) & retrieved_conditions)
    condition_recall = conditions_found / len(expected_conditions)
    
    # Precision: of the documents we retrieved, how many were relevant?
    condition_precision = matching_docs / len(retrieved_docs) if retrieved_docs else 0.0
    
    return {
        "condition_recall": round(condition_recall, 3),
        "condition_precision": round(condition_precision, 3),
        "retrieved_conditions": retrieved_conditions,
        "is_abstention_question": False,
    }


def evaluate_retrieval_diversity(retrieved_docs: list) -> dict:
    """
    Measure how diverse the retrieved documents are.
    
    A good retrieval should return chunks from different documents and
    different sections, not 5 copies of the same medication list.
    
    Returns:
        - unique_text_ratio: Fraction of chunks with unique content (0-1)
        - unique_conditions: Number of distinct conditions represented
        - unique_doc_types: Number of distinct document types represented
    """
    if not retrieved_docs:
        return {"unique_text_ratio": 0.0, "unique_conditions": 0, "unique_doc_types": 0}
    
    # Check for duplicate text content
    texts = [doc.page_content[:300] for doc in retrieved_docs]
    unique_texts = set(texts)
    unique_text_ratio = len(unique_texts) / len(texts)
    
    # Check condition diversity
    conditions = set(doc.metadata.get("condition", "") for doc in retrieved_docs)
    
    # Check document type diversity
    doc_types = set(doc.metadata.get("document_type", "") for doc in retrieved_docs)
    
    return {
        "unique_text_ratio": round(unique_text_ratio, 3),
        "unique_conditions": len(conditions),
        "unique_doc_types": len(doc_types),
    }


def evaluate_answer_keyword_coverage(
    answer: str,
    expected_keywords: list[str],
    forbidden_keywords: list[str],
) -> dict:
    """
    Check whether the answer contains expected medical terms and
    does NOT contain forbidden terms (which indicate hallucination).
    
    This is a simple but effective proxy for answer quality.
    If we ask about CHF medications and the answer doesn't mention
    "furosemide" or "carvedilol", something is wrong.
    
    Returns:
        - keyword_recall: Fraction of expected keywords found (0-1)
        - keywords_found: List of expected keywords that appeared
        - keywords_missing: List of expected keywords not found
        - forbidden_found: List of forbidden keywords that appeared (bad!)
        - has_hallucination_signal: True if forbidden keywords were found
    """
    answer_lower = answer.lower()
    
    keywords_found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    keywords_missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    forbidden_found = [kw for kw in forbidden_keywords if kw.lower() in answer_lower]
    
    keyword_recall = len(keywords_found) / len(expected_keywords) if expected_keywords else 1.0
    
    return {
        "keyword_recall": round(keyword_recall, 3),
        "keywords_found": keywords_found,
        "keywords_missing": keywords_missing,
        "forbidden_found": forbidden_found,
        "has_hallucination_signal": len(forbidden_found) > 0,
    }


def evaluate_citation_quality(answer: str, retrieved_docs: list) -> dict:
    """
    Check whether the answer actually cites sources.
    
    A good RAG answer should contain [Source N] references that map back
    to retrieved documents. An answer without citations might be
    hallucinating from the LLM's training data.
    
    Returns:
        - has_citations: Whether any [Source N] references were found
        - citation_count: Number of unique sources cited
        - citation_density: Citations per 100 words (rough measure)
    """
    import re
    
    citations = set(re.findall(r'\[Source\s+(\d+)\]', answer))
    word_count = len(answer.split())
    
    return {
        "has_citations": len(citations) > 0,
        "citation_count": len(citations),
        "citation_density": round(len(citations) / max(word_count / 100, 1), 2),
    }


def evaluate_abstention(
    answer: str,
    is_abstention_question: bool,
) -> dict:
    """
    Check whether the system correctly abstains from answering when
    the information isn't in the documents.
    
    For abstention questions (where expected_conditions is empty),
    a GOOD answer says something like "the documents don't contain
    this information." A BAD answer makes up an answer.
    
    For non-abstention questions, we check that the system does NOT
    unnecessarily abstain (which would indicate retrieval failure).
    
    Returns:
        - correct_behavior: True if the system did the right thing
        - did_abstain: Whether the answer indicates abstention
    """
    abstention_phrases = [
        "not contain",
        "insufficient",
        "not available",
        "do not have",
        "cannot answer",
        "no information",
        "not enough information",
        "don't contain",
        "does not contain",
        "not found in",
        "no relevant",
    ]
    
    answer_lower = answer.lower()
    did_abstain = any(phrase in answer_lower for phrase in abstention_phrases)
    
    if is_abstention_question:
        # System SHOULD abstain
        correct_behavior = did_abstain
    else:
        # System should NOT abstain
        correct_behavior = not did_abstain
    
    return {
        "correct_behavior": correct_behavior,
        "did_abstain": did_abstain,
        "is_abstention_question": is_abstention_question,
    }


# ============================================================================
# SINGLE QUESTION EVALUATOR
# ============================================================================

def evaluate_single_question(
    rag_chain,
    question_data: dict,
) -> dict:
    """
    Run the full RAG pipeline on a single question and evaluate all metrics.
    
    This is the core evaluation function that ties together retrieval,
    generation, and all metric calculations for one question.
    """
    question = question_data["question"]
    
    # Run the RAG pipeline
    result = rag_chain.query(question)
    
    # Evaluate retrieval quality
    retrieval_condition = evaluate_retrieval_condition_match(
        result["retrieved_docs"],
        question_data["expected_conditions"],
    )
    
    retrieval_diversity = evaluate_retrieval_diversity(result["retrieved_docs"])
    
    # Evaluate answer quality
    keyword_coverage = evaluate_answer_keyword_coverage(
        result["answer"],
        question_data["expected_keywords"],
        question_data["forbidden_keywords"],
    )
    
    citation_quality = evaluate_citation_quality(
        result["answer"],
        result["retrieved_docs"],
    )
    
    is_abstention = len(question_data["expected_conditions"]) == 0
    abstention_eval = evaluate_abstention(result["answer"], is_abstention)
    
    return {
        "question": question,
        "category": question_data["category"],
        "difficulty": question_data["difficulty"],
        "answer": result["answer"],
        "confidence": result["confidence"],
        "latency": result["latency"],
        "groundedness": result["groundedness"],
        "retrieval": {
            "condition_match": retrieval_condition,
            "diversity": retrieval_diversity,
        },
        "answer_quality": {
            "keyword_coverage": keyword_coverage,
            "citation_quality": citation_quality,
            "abstention": abstention_eval,
        },
    }


# ============================================================================
# FULL EVALUATION RUNNER
# ============================================================================

def run_full_evaluation(
    rag_chain,
    questions: list[dict] = None,
    output_dir: str = "data/evaluation_results",
    config_name: str = "default",
) -> dict:
    """
    Run the complete evaluation suite across all questions.
    
    This iterates through every question in the bank, evaluates it,
    aggregates the metrics, and saves detailed results to disk.
    
    Args:
        rag_chain: An initialized ClinicalRAGChain instance
        questions: List of question dicts (defaults to full question bank)
        output_dir: Where to save results
        config_name: Label for this configuration (e.g., "chunk_512_mmr")
    
    Returns a summary dict with aggregate metrics.
    """
    if questions is None:
        questions = EVALUATION_QUESTIONS
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"  ClinicalRAG Evaluation — Config: {config_name}")
    print(f"  Questions: {len(questions)}")
    print("=" * 60)
    
    all_results = []
    
    for i, q_data in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q_data['category']} ({q_data['difficulty']})")
        print(f"  Q: {q_data['question'][:80]}...")
        
        try:
            result = evaluate_single_question(rag_chain, q_data)
            all_results.append(result)
            
            # Print quick summary for each question
            kr = result["answer_quality"]["keyword_coverage"]["keyword_recall"]
            cr = result["retrieval"]["condition_match"]["condition_recall"]
            has_cite = result["answer_quality"]["citation_quality"]["has_citations"]
            correct = result["answer_quality"]["abstention"]["correct_behavior"]
            latency = result["latency"]["total_seconds"]
            
            status = "PASS" if (kr >= 0.5 or result["answer_quality"]["abstention"]["is_abstention_question"]) and correct else "FAIL"
            print(f"  [{status}] Keywords: {kr:.0%} | Conditions: {cr:.0%} | Citations: {'Yes' if has_cite else 'No'} | Correct: {'Yes' if correct else 'No'} | {latency:.1f}s")
            
        except Exception as e:
            print(f"  [ERROR] {str(e)[:100]}")
            all_results.append({
                "question": q_data["question"],
                "category": q_data["category"],
                "difficulty": q_data["difficulty"],
                "error": str(e),
            })
    
    # ========================================================================
    # AGGREGATE METRICS
    # ========================================================================
    
    successful_results = [r for r in all_results if "error" not in r]
    
    if not successful_results:
        print("\n  No successful evaluations. Check your RAG chain configuration.")
        return {"error": "No successful evaluations"}
    
    # Overall metrics
    avg_keyword_recall = sum(
        r["answer_quality"]["keyword_coverage"]["keyword_recall"]
        for r in successful_results
    ) / len(successful_results)
    
    avg_condition_recall = sum(
        r["retrieval"]["condition_match"]["condition_recall"]
        for r in successful_results
    ) / len(successful_results)
    
    avg_condition_precision = sum(
        r["retrieval"]["condition_match"]["condition_precision"]
        for r in successful_results
        if not r["retrieval"]["condition_match"].get("is_abstention_question", False)
    ) / max(1, len([
        r for r in successful_results
        if not r["retrieval"]["condition_match"].get("is_abstention_question", False)
    ]))
    
    citation_rate = sum(
        1 for r in successful_results
        if r["answer_quality"]["citation_quality"]["has_citations"]
    ) / len(successful_results)
    
    abstention_accuracy = sum(
        1 for r in successful_results
        if r["answer_quality"]["abstention"]["correct_behavior"]
    ) / len(successful_results)
    
    hallucination_count = sum(
        1 for r in successful_results
        if r["answer_quality"]["keyword_coverage"]["has_hallucination_signal"]
    )
    
    avg_diversity = sum(
        r["retrieval"]["diversity"]["unique_text_ratio"]
        for r in successful_results
    ) / len(successful_results)
    
    avg_latency = sum(
        r["latency"]["total_seconds"]
        for r in successful_results
    ) / len(successful_results)
    
    # Per-category metrics
    category_metrics = {}
    for category in get_all_categories():
        cat_results = [r for r in successful_results if r["category"] == category]
        if cat_results:
            category_metrics[category] = {
                "count": len(cat_results),
                "avg_keyword_recall": round(sum(
                    r["answer_quality"]["keyword_coverage"]["keyword_recall"]
                    for r in cat_results
                ) / len(cat_results), 3),
                "avg_condition_recall": round(sum(
                    r["retrieval"]["condition_match"]["condition_recall"]
                    for r in cat_results
                ) / len(cat_results), 3),
                "citation_rate": round(sum(
                    1 for r in cat_results
                    if r["answer_quality"]["citation_quality"]["has_citations"]
                ) / len(cat_results), 3),
                "abstention_accuracy": round(sum(
                    1 for r in cat_results
                    if r["answer_quality"]["abstention"]["correct_behavior"]
                ) / len(cat_results), 3),
                "avg_latency": round(sum(
                    r["latency"]["total_seconds"]
                    for r in cat_results
                ) / len(cat_results), 2),
            }
    
    # Per-difficulty metrics
    difficulty_metrics = {}
    for difficulty in ["easy", "medium", "hard"]:
        diff_results = [r for r in successful_results if r["difficulty"] == difficulty]
        if diff_results:
            difficulty_metrics[difficulty] = {
                "count": len(diff_results),
                "avg_keyword_recall": round(sum(
                    r["answer_quality"]["keyword_coverage"]["keyword_recall"]
                    for r in diff_results
                ) / len(diff_results), 3),
                "pass_rate": round(sum(
                    1 for r in diff_results
                    if r["answer_quality"]["keyword_coverage"]["keyword_recall"] >= 0.5
                    or r["answer_quality"]["abstention"]["is_abstention_question"]
                ) / len(diff_results), 3),
            }
    
    summary = {
        "config_name": config_name,
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(questions),
        "successful_evaluations": len(successful_results),
        "errors": len(all_results) - len(successful_results),
        "overall_metrics": {
            "avg_keyword_recall": round(avg_keyword_recall, 3),
            "avg_condition_recall": round(avg_condition_recall, 3),
            "avg_condition_precision": round(avg_condition_precision, 3),
            "citation_rate": round(citation_rate, 3),
            "abstention_accuracy": round(abstention_accuracy, 3),
            "hallucination_signals": hallucination_count,
            "avg_retrieval_diversity": round(avg_diversity, 3),
            "avg_latency_seconds": round(avg_latency, 2),
        },
        "category_metrics": category_metrics,
        "difficulty_metrics": difficulty_metrics,
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Config: {config_name}")
    print(f"  Questions: {len(successful_results)}/{len(questions)} successful")
    print(f"\n  OVERALL METRICS:")
    print(f"    Keyword Recall:       {avg_keyword_recall:.1%}")
    print(f"    Condition Recall:     {avg_condition_recall:.1%}")
    print(f"    Condition Precision:  {avg_condition_precision:.1%}")
    print(f"    Citation Rate:        {citation_rate:.1%}")
    print(f"    Abstention Accuracy:  {abstention_accuracy:.1%}")
    print(f"    Hallucination Signals:{hallucination_count}")
    print(f"    Retrieval Diversity:  {avg_diversity:.1%}")
    print(f"    Avg Latency:          {avg_latency:.1f}s")
    
    print(f"\n  BY CATEGORY:")
    for cat, metrics in category_metrics.items():
        print(f"    {cat}: KW={metrics['avg_keyword_recall']:.0%} | Cond={metrics['avg_condition_recall']:.0%} | Cite={metrics['citation_rate']:.0%} | Abstain={metrics['abstention_accuracy']:.0%}")
    
    print(f"\n  BY DIFFICULTY:")
    for diff, metrics in difficulty_metrics.items():
        print(f"    {diff}: KW={metrics['avg_keyword_recall']:.0%} | Pass={metrics['pass_rate']:.0%} ({metrics['count']} questions)")
    
    # Save detailed results
    results_file = output_path / f"eval_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_data = {
        "summary": summary,
        "detailed_results": [
            {k: v for k, v in r.items() if k != "retrieved_docs" and k not in ("answer",)}
            if "error" not in r else r
            for r in all_results
        ],
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  Results saved to: {results_file}")
    
    # Save summary separately for easy comparison
    summary_file = output_path / f"summary_{config_name}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary


# ============================================================================
# COMPARATIVE EVALUATION
# ============================================================================

def run_chunk_size_comparison(
    chunk_sizes: list[int] = None,
    output_dir: str = "data/evaluation_results",
) -> list[dict]:
    """
    Compare RAG quality across different chunk sizes.
    
    This is one of the most important experiments — it shows how chunk size
    affects retrieval precision, answer quality, and latency. The results
    are visualized in the README and discussed in the portfolio writeup.
    
    For each chunk size, this function:
    1. Rebuilds the vector store with that chunk size
    2. Runs the full evaluation suite
    3. Saves the results for comparison
    
    Default chunk sizes: 256, 512, 1000, 1500
    Smaller = more precise retrieval, less context per chunk
    Larger = more context per chunk, less precise retrieval
    """
    from src.ingestion.ingest_documents import run_ingestion_pipeline
    from src.retrieval.rag_chain import ClinicalRAGChain
    import shutil
    
    if chunk_sizes is None:
        chunk_sizes = [256, 512, 1000, 1500]
    
    all_summaries = []
    prev_chain = None
    prev_vectorstore = None
    
    for chunk_size in chunk_sizes:
        config_name = f"chunk_{chunk_size}"
        print(f"\n{'#' * 60}")
        print(f"# EXPERIMENT: Chunk Size = {chunk_size}")
        print(f"{'#' * 60}")
        
        # Release previous ChromaDB connections before deleting files
        # Windows locks files that are still referenced by open handles
        if prev_chain is not None:
            del prev_chain
        if prev_vectorstore is not None:
            try:
                prev_vectorstore._client.close()
            except Exception:
                pass
            del prev_vectorstore
        
        import gc
        gc.collect()
        import time
        time.sleep(2)  # Give Windows time to release file handles
        
        # Clean vector store
        vectordb_path = Path("data/vectordb")
        if vectordb_path.exists():
            try:
                shutil.rmtree(vectordb_path)
            except PermissionError:
                # Nuclear option: force delete on Windows
                import subprocess
                subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(vectordb_path)],
                             capture_output=True)
                time.sleep(1)
        
        # Reingest with new chunk size
        vectorstore = run_ingestion_pipeline(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.2),  # 20% overlap
        )
        
        # Build RAG chain with the new vector store
        chain = ClinicalRAGChain(vectorstore=vectorstore)
        
        # Run evaluation
        summary = run_full_evaluation(
            chain,
            output_dir=output_dir,
            config_name=config_name,
        )
        summary["chunk_size"] = chunk_size
        all_summaries.append(summary)
        
        # Store references for cleanup in next iteration
        prev_chain = chain
        prev_vectorstore = vectorstore
    
    # Save comparison results
    comparison_file = Path(output_dir) / "chunk_size_comparison.json"
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nComparison saved to: {comparison_file}")
    return all_summaries


def run_retrieval_strategy_comparison(
    output_dir: str = "data/evaluation_results",
) -> list[dict]:
    """
    Compare similarity search vs MMR retrieval at different settings.
    
    This uses the same vector store but changes the retrieval strategy.
    """
    from src.retrieval.rag_chain import ClinicalRAGChain
    
    strategies = [
        {"search_type": "similarity", "search_k": 3, "label": "similarity_k3"},
        {"search_type": "similarity", "search_k": 5, "label": "similarity_k5"},
        {"search_type": "mmr", "search_k": 5, "label": "mmr_k5_lambda07"},
        {"search_type": "mmr", "search_k": 7, "label": "mmr_k7_lambda07"},
    ]
    
    all_summaries = []
    
    for strategy in strategies:
        config_name = strategy["label"]
        print(f"\n{'#' * 60}")
        print(f"# EXPERIMENT: {config_name}")
        print(f"{'#' * 60}")
        
        chain = ClinicalRAGChain(
            search_type=strategy["search_type"],
            search_k=strategy["search_k"],
        )
        
        summary = run_full_evaluation(
            chain,
            output_dir=output_dir,
            config_name=config_name,
        )
        summary["strategy"] = strategy
        all_summaries.append(summary)
    
    comparison_file = Path(output_dir) / "retrieval_strategy_comparison.json"
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nComparison saved to: {comparison_file}")
    return all_summaries


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ClinicalRAG Evaluation")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "chunk_comparison", "retrieval_comparison"],
        default="quick",
        help="Evaluation mode: 'quick' (5 questions), 'full' (all questions), "
             "'chunk_comparison' (compare chunk sizes), 'retrieval_comparison' (compare strategies)",
    )
    args = parser.parse_args()
    
    if args.mode == "quick":
        # Quick evaluation with just 5 easy questions
        from src.retrieval.rag_chain import ClinicalRAGChain
        chain = ClinicalRAGChain()
        easy_questions = [q for q in EVALUATION_QUESTIONS if q["difficulty"] == "easy"]
        run_full_evaluation(chain, questions=easy_questions, config_name="quick_test")
        
    elif args.mode == "full":
        from src.retrieval.rag_chain import ClinicalRAGChain
        chain = ClinicalRAGChain()
        run_full_evaluation(chain, config_name="full_evaluation")
        
    elif args.mode == "chunk_comparison":
        run_chunk_size_comparison()
        
    elif args.mode == "retrieval_comparison":
        run_retrieval_strategy_comparison()
