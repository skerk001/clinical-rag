"""
Clinical Question Bank for RAG Evaluation

This module provides a curated set of test questions with expected answers,
expected source conditions, and difficulty levels. These questions are designed
to test different aspects of RAG system quality:

1. FACTUAL RETRIEVAL: Can the system find the right documents?
   "What medications are prescribed for CHF?" → should find CHF discharge summaries

2. CROSS-DOCUMENT REASONING: Can it synthesize from multiple chunks?
   "Compare CHF and COPD discharge medications" → needs chunks from both conditions

3. SPECIFICITY: Does it return precise answers, not vague summaries?
   "What is the dose of furosemide for CHF?" → should cite "60mg PO daily"

4. ABSTENTION: Does it correctly refuse to answer when info isn't available?
   "What is the average BMI of patients?" → should say "not in documents"

5. SAFETY: Does it avoid hallucinating dangerous clinical information?
   "Can I take ibuprofen with my kidney medications?" → should reference NSAID risks

Author: Samir Kerkar
"""


# Each question has:
#   - question: The actual clinical question to ask
#   - expected_conditions: Which conditions SHOULD appear in retrieved documents
#   - expected_keywords: Key terms that SHOULD appear in a correct answer
#   - forbidden_keywords: Terms that should NOT appear (indicate hallucination)
#   - category: What aspect of RAG quality this tests
#   - difficulty: How hard this is for the system (easy/medium/hard)
#   - notes: Why this question is useful for evaluation

EVALUATION_QUESTIONS = [
    # =========================================================================
    # CATEGORY 1: Direct Factual Retrieval (Easy)
    # These test basic retrieval — can the system find the right condition?
    # =========================================================================
    {
        "question": "What medications are prescribed at discharge for patients with heart failure?",
        "expected_conditions": ["CHF"],
        "expected_keywords": ["furosemide", "carvedilol", "lisinopril", "spironolactone", "empagliflozin"],
        "forbidden_keywords": [],
        "category": "factual_retrieval",
        "difficulty": "easy",
        "notes": "Core retrieval test — CHF is the most common condition in our corpus",
    },
    {
        "question": "What antibiotics are used to treat community-acquired pneumonia?",
        "expected_conditions": ["Pneumonia"],
        "expected_keywords": ["ceftriaxone", "azithromycin", "amoxicillin"],
        "forbidden_keywords": [],
        "category": "factual_retrieval",
        "difficulty": "easy",
        "notes": "Tests retrieval of pneumonia-specific treatment protocols",
    },
    {
        "question": "How is acute COPD exacerbation treated in the hospital?",
        "expected_conditions": ["COPD_exacerbation"],
        "expected_keywords": ["albuterol", "ipratropium", "methylprednisolone", "prednisone", "azithromycin"],
        "forbidden_keywords": [],
        "category": "factual_retrieval",
        "difficulty": "easy",
        "notes": "Tests COPD treatment retrieval with multiple medication classes",
    },
    {
        "question": "What is the initial management of sepsis?",
        "expected_conditions": ["Sepsis"],
        "expected_keywords": ["fluid", "antibiotic", "culture", "lactate"],
        "forbidden_keywords": [],
        "category": "factual_retrieval",
        "difficulty": "easy",
        "notes": "Tests sepsis protocol retrieval — time-sensitive clinical info",
    },
    {
        "question": "What medications are given for diabetic ketoacidosis?",
        "expected_conditions": ["Type2_diabetes"],
        "expected_keywords": ["insulin", "fluid", "potassium"],
        "forbidden_keywords": [],
        "category": "factual_retrieval",
        "difficulty": "easy",
        "notes": "Tests DKA management retrieval",
    },

    # =========================================================================
    # CATEGORY 2: Specific Detail Retrieval (Medium)
    # These test whether the system can find precise dosages and details,
    # not just general treatment categories.
    # =========================================================================
    {
        "question": "What is the discharge dose of furosemide for heart failure patients and why was it changed?",
        "expected_conditions": ["CHF"],
        "expected_keywords": ["60mg", "increased", "40mg"],
        "forbidden_keywords": [],
        "category": "specific_detail",
        "difficulty": "medium",
        "notes": "Tests precise dosage retrieval and clinical reasoning",
    },
    {
        "question": "What new medications were added during the CHF hospitalization and what is the evidence for each?",
        "expected_conditions": ["CHF"],
        "expected_keywords": ["spironolactone", "empagliflozin", "SGLT2"],
        "forbidden_keywords": [],
        "category": "specific_detail",
        "difficulty": "medium",
        "notes": "Tests multi-medication detail extraction with evidence-based reasoning",
    },
    {
        "question": "What were the patient's lab values on admission for acute kidney injury?",
        "expected_conditions": ["AKI"],
        "expected_keywords": ["creatinine", "potassium", "5.8", "5.9"],
        "forbidden_keywords": [],
        "category": "specific_detail",
        "difficulty": "medium",
        "notes": "Tests numeric lab value retrieval accuracy",
    },
    {
        "question": "What caused the acute kidney injury and what medications were discontinued?",
        "expected_conditions": ["AKI"],
        "expected_keywords": ["NSAID", "ibuprofen", "triple whammy", "lisinopril"],
        "forbidden_keywords": [],
        "category": "specific_detail",
        "difficulty": "medium",
        "notes": "Tests causal reasoning retrieval — the 'triple whammy' mechanism",
    },
    {
        "question": "What is the prednisone taper protocol for COPD exacerbation?",
        "expected_conditions": ["COPD_exacerbation"],
        "expected_keywords": ["40mg", "5-day", "REDUCE"],
        "forbidden_keywords": [],
        "category": "specific_detail",
        "difficulty": "medium",
        "notes": "Tests specific protocol retrieval with evidence citation",
    },

    # =========================================================================
    # CATEGORY 3: Discharge Instructions (Medium)
    # These test retrieval of patient education and follow-up plans,
    # which are clinically critical and often asked about.
    # =========================================================================
    {
        "question": "What discharge instructions are given to heart failure patients about daily monitoring?",
        "expected_conditions": ["CHF"],
        "expected_keywords": ["weight", "sodium", "2 lbs", "5 lbs", "2000mg"],
        "forbidden_keywords": [],
        "category": "discharge_instructions",
        "difficulty": "medium",
        "notes": "Tests retrieval of patient education content",
    },
    {
        "question": "What follow-up appointments are recommended after a stroke?",
        "expected_conditions": ["Stroke"],
        "expected_keywords": ["neurology", "cardiology", "MRI", "rehabilitation"],
        "forbidden_keywords": [],
        "category": "discharge_instructions",
        "difficulty": "medium",
        "notes": "Tests multi-specialty follow-up plan retrieval",
    },
    {
        "question": "What are the warning signs that a COPD patient should return to the emergency department?",
        "expected_conditions": ["COPD_exacerbation"],
        "expected_keywords": ["dyspnea", "fever", "hemoptysis"],
        "forbidden_keywords": [],
        "category": "discharge_instructions",
        "difficulty": "medium",
        "notes": "Tests safety-critical return-to-ED criteria retrieval",
    },

    # =========================================================================
    # CATEGORY 4: Drug Interactions and Safety (Medium-Hard)
    # These test whether the system can identify safety-critical information
    # about medication interactions and contraindications.
    # =========================================================================
    {
        "question": "What drug interactions should be monitored when prescribing spironolactone with an ACE inhibitor?",
        "expected_conditions": ["CHF"],
        "expected_keywords": ["hyperkalemia", "potassium", "monitoring"],
        "forbidden_keywords": [],
        "category": "drug_safety",
        "difficulty": "medium",
        "notes": "Tests drug interaction safety retrieval from med rec notes",
    },
    {
        "question": "Why should NSAIDs be avoided in patients with chronic kidney disease?",
        "expected_conditions": ["AKI"],
        "expected_keywords": ["NSAID", "kidney", "vasoconstriction", "afferent"],
        "forbidden_keywords": [],
        "category": "drug_safety",
        "difficulty": "medium",
        "notes": "Tests mechanism-of-harm retrieval for medication safety",
    },
    {
        "question": "What precautions should be taken with empagliflozin during acute illness?",
        "expected_conditions": ["CHF", "Type2_diabetes"],
        "expected_keywords": ["hold", "euglycemic", "DKA", "dehydration"],
        "forbidden_keywords": [],
        "category": "drug_safety",
        "difficulty": "hard",
        "notes": "Tests retrieval of SGLT2 inhibitor sick-day rules across conditions",
    },

    # =========================================================================
    # CATEGORY 5: Abstention — Should Answer "I Don't Know" (Hard)
    # These test the hallucination guardrail. The system should recognize
    # that the information isn't in the documents and say so.
    # =========================================================================
    {
        "question": "What is the recommended dosing of remdesivir for COVID-19?",
        "expected_conditions": [],
        "expected_keywords": ["not contain", "insufficient", "not available"],
        "forbidden_keywords": ["200mg", "remdesivir", "5 days"],
        "category": "abstention",
        "difficulty": "hard",
        "notes": "COVID isn't in our corpus — system should abstain, not hallucinate",
    },
    {
        "question": "What surgical procedures are recommended for heart failure patients?",
        "expected_conditions": [],
        "expected_keywords": ["not contain", "insufficient", "not available"],
        "forbidden_keywords": ["LVAD", "transplant", "surgery"],
        "category": "abstention",
        "difficulty": "hard",
        "notes": "Surgical info isn't in our notes — tests abstention for adjacent topics",
    },
    {
        "question": "What is the average length of stay for pneumonia patients in this dataset?",
        "expected_conditions": [],
        "expected_keywords": ["not contain", "insufficient", "not available"],
        "forbidden_keywords": [],
        "category": "abstention",
        "difficulty": "hard",
        "notes": "Aggregate statistics aren't in individual notes — tests abstention",
    },

    # =========================================================================
    # CATEGORY 6: Cross-Condition Reasoning (Hard)
    # These require the system to pull information from multiple conditions
    # and synthesize a coherent answer.
    # =========================================================================
    {
        "question": "Which conditions in these records require anticoagulation therapy and what agents are used?",
        "expected_conditions": ["Stroke", "CHF"],
        "expected_keywords": ["apixaban", "atrial fibrillation"],
        "forbidden_keywords": [],
        "category": "cross_condition",
        "difficulty": "hard",
        "notes": "Requires pulling anticoagulation info from stroke and AF-related docs",
    },
    {
        "question": "What role do SGLT2 inhibitors play across different conditions in these documents?",
        "expected_conditions": ["CHF", "Type2_diabetes"],
        "expected_keywords": ["empagliflozin", "heart failure", "diabetes", "cardiorenal"],
        "forbidden_keywords": [],
        "category": "cross_condition",
        "difficulty": "hard",
        "notes": "Tests cross-condition synthesis — empagliflozin appears in CHF and DKA notes",
    },
]


def get_questions_by_category(category: str) -> list[dict]:
    """Return all questions matching a specific category."""
    return [q for q in EVALUATION_QUESTIONS if q["category"] == category]


def get_questions_by_difficulty(difficulty: str) -> list[dict]:
    """Return all questions matching a specific difficulty level."""
    return [q for q in EVALUATION_QUESTIONS if q["difficulty"] == difficulty]


def get_all_categories() -> list[str]:
    """Return all unique question categories."""
    return sorted(set(q["category"] for q in EVALUATION_QUESTIONS))
