"""
evaluate.py — Test RAG retrieval quality against specific facts from indexed papers.

Run from the project root after ingestion to verify the knowledge base:
    python tools/rag_search/evaluate.py

What it checks
--------------
Each test case is built from a specific, verifiable fact extracted directly from
the indexed research papers — not generic topics, but concrete numbers, method
names, and conclusions that must appear in the retrieved chunks.

  PASS  — at least one expected keyword found in the retrieved content
  FAIL  — no expected keyword found, or wrong "not found" behaviour

Re-run this script after:
  - Adding new documents to the knowledge base
  - Changing _SCORE_THRESHOLD in rag_search_tool.py
  - Changing chunk size or overlap in ingest.py

A drop in score signals a regression in retrieval quality.
"""

import sys
from pathlib import Path

# Add project root so imports resolve when run from any directory
sys.path.insert(0, str(Path(__file__).parents[2]))

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Test cases — built from specific facts extracted from each indexed paper
# ---------------------------------------------------------------------------

TEST_CASES = [

    # ------------------------------------------------------------------
    # Paper: Hip contact forces and gait patterns from routine activities
    # Authors: Bergmann et al. (2001)
    # ------------------------------------------------------------------
    {
        "question":          "What was the average peak hip contact force during normal walking for the typical patient NPA in the Bergmann 2001 study?",
        "expected_keywords": ["238", "body weight", "NPA", "walking"],
        "fact":              "Peak hip contact force during walking was 238% BW for the typical patient NPA.",
        "should_find":       True,
    },
    {
        "question":          "How did hip contact forces during stair climbing compare between going upstairs and downstairs according to Bergmann et al. 2001?",
        "expected_keywords": ["251", "260", "upstairs", "downstairs"],
        "fact":              "Peak force was 251% BW upstairs and 260% BW downstairs.",
        "should_find":       True,
    },
    {
        "question":          "By how much was the inward torsional moment larger when going upstairs compared to normal walking in Bergmann 2001?",
        "expected_keywords": ["23", "torsional", "upstairs", "walking"],
        "fact":              "Inward torsional moment was 23% larger going upstairs than during level walking.",
        "should_find":       True,
    },

    # ------------------------------------------------------------------
    # Paper: Machine Learning techniques for optimization of joint replacements
    # Authors: Cilla et al. (2017)
    # ------------------------------------------------------------------
    {
        "question":          "How many finite element models were generated in the Cilla 2017 parametric study of the Nanos short-stem hip implant?",
        "expected_keywords": ["256", "Nanos", "parametric"],
        "fact":              "256 FE models were generated based on the Nanos short stem by Smith & Nephew.",
        "should_find":       True,
    },
    {
        "question":          "What correlation coefficient did the machine learning methods achieve in predicting stress shielding in the Cilla 2017 study?",
        "expected_keywords": ["0.9998", "RSQ", "correlation"],
        "fact":              "Both ANN and SVM achieved RSQ = 0.9998.",
        "should_find":       True,
    },
    {
        "question":          "What were the optimal implant geometry parameters found by the SVM and pattern-search algorithm in the Cilla 2017 study?",
        "expected_keywords": ["90 mm", "36", "SVM", "stress shielding"],
        "fact":              "Optimal parameters: L = 90 mm, D = 36%, R1 = 4 mm, R2 = 1.5 mm.",
        "should_find":       True,
    },

    # ------------------------------------------------------------------
    # Paper: Physiologically based boundary conditions in FE modelling
    # Authors: Speirs et al. (2007)
    # ------------------------------------------------------------------
    {
        "question":          "What was the femoral head deflection in the physiological boundary condition case E compared to non-physiological cases in Speirs 2007?",
        "expected_keywords": ["2 mm", "case E", "physiological", "deflection"],
        "fact":              "Only Case E produced deflections below 2 mm; Cases A–D produced 8–19 mm.",
        "should_find":       True,
    },
    {
        "question":          "How much higher were reaction forces in non-physiological boundary condition models compared to the physiological case in Speirs 2007?",
        "expected_keywords": ["10,000", "reaction forces", "non-physiological"],
        "fact":              "Reaction forces in non-physiological cases were more than 10,000% higher.",
        "should_find":       True,
    },

    # ------------------------------------------------------------------
    # Paper: Realistic loads for testing hip implants
    # Authors: Bergmann et al. (2010)
    # ------------------------------------------------------------------
    {
        "question":          "What was the peak hip contact force recorded during stumbling in the Bergmann 2010 realistic loads study?",
        "expected_keywords": ["11,000", "stumbling", "peak"],
        "fact":              "Peak contact force during stumbling reached 11,000 N.",
        "should_find":       True,
    },
    {
        "question":          "How many years of implant use do 10 million loading cycles represent for active patients according to Bergmann 2010?",
        "expected_keywords": ["3.9", "years", "active", "10 million"],
        "fact":              "10 million cycles = 3.9 years for active patients, 7.3 years for normal patients.",
        "should_find":       True,
    },

    # ------------------------------------------------------------------
    # Paper: Standardized loads acting in hip implants
    # Authors: Bergmann et al. (2016)
    # ------------------------------------------------------------------
    {
        "question":          "What sinusoidal force does ISO 7206-4 specify for hip stem endurance testing?",
        "expected_keywords": ["2300 N", "ISO", "stem", "endurance"],
        "fact":              "ISO 7206-4 specifies 2300 N for stem endurance testing.",
        "should_find":       True,
    },
    {
        "question":          "What was the peak hip contact force during jogging at 7 km/h for the HIGH100 scenario in Bergmann 2016?",
        "expected_keywords": ["4839", "jogging", "HIGH100"],
        "fact":              "During jogging, HIGH100 peak force was 4839 N — the largest of all activities.",
        "should_find":       True,
    },

    # ------------------------------------------------------------------
    # Paper: Structural analysis of endoprosthesis with graded lattice structures
    # Authors: Sufiiarov et al. (2021)
    # ------------------------------------------------------------------
    {
        "question":          "What Young's modulus was assigned to Ti-6Al-4V in the Sufiiarov lattice endoprosthesis study?",
        "expected_keywords": ["113 GPa", "Ti-6Al-4V", "Young's modulus"],
        "fact":              "Ti-6Al-4V was assigned Young's modulus of 113 GPa and Poisson's ratio of 0.36.",
        "should_find":       True,
    },
    {
        "question":          "What target Young's modulus range was used for the cortical bone region in the Sufiiarov graded lattice endoprosthesis?",
        "expected_keywords": ["14", "28 GPa", "cortical", "lattice"],
        "fact":              "Cortical bone region: 14–28 GPa; trabecular bone region: 0.1–4 GPa.",
        "should_find":       True,
    },

    # ------------------------------------------------------------------
    # Paper: The difference-based equivalent static load model (DiESL)
    # Authors: Triller et al. (2021)
    # ------------------------------------------------------------------
    {
        "question":          "What commercial solvers were used to implement the DiESL method in the Triller 2021 study?",
        "expected_keywords": ["LS-DYNA", "OptiStruct", "DiESL"],
        "fact":              "LS-DYNA for nonlinear analysis and Altair OptiStruct for optimization.",
        "should_find":       True,
    },
    {
        "question":          "What impactor mass and speed were used in the three-point bending test in the DiESL verification study by Triller 2021?",
        "expected_keywords": ["1.256 kg", "7.5 m/s", "impactor", "bending"],
        "fact":              "Impactor: diameter 100 mm, mass 1.256 kg, speed 7.5 m/s.",
        "should_find":       True,
    },

    # ------------------------------------------------------------------
    # Paper: Tibio-femoral loading during human gait and stair climbing
    # Authors: Taylor et al. (2004)
    # ------------------------------------------------------------------
    {
        "question":          "What was the average peak tibio-femoral contact force during stair climbing according to Taylor et al. 2004?",
        "expected_keywords": ["5.4", "BW", "tibio-femoral", "stair"],
        "fact":              "Average peak tibio-femoral force during stair climbing was 5.4 BW.",
        "should_find":       True,
    },
    {
        "question":          "How much higher were knee contact forces compared to hip contact forces during stair climbing in Taylor 2004?",
        "expected_keywords": ["116", "knee", "hip", "stair climbing"],
        "fact":              "Knee forces were 116% higher than hip forces during stair climbing.",
        "should_find":       True,
    },

    # ------------------------------------------------------------------
    # Out-of-scope — anti-hallucination tests
    # ------------------------------------------------------------------
    {
        "question":          "What does the 2023 WHO report say about knee implant failure rates in elderly patients?",
        "expected_keywords": [],
        "fact":              "Not in knowledge base — should return not found.",
        "should_find":       False,
    },
    {
        "question":          "What is the current stock price of Zimmer Biomet medical devices company?",
        "expected_keywords": [],
        "fact":              "Not in knowledge base — should return not found.",
        "should_find":       False,
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_evaluation():
    from tools.rag_search.rag_search_tool import rag_search

    passed = 0
    failed = 0
    total  = len(TEST_CASES)

    print("=" * 70)
    print("RAG Evaluation Report")
    print(f"Testing {total} cases derived from indexed research papers")
    print("=" * 70)

    for i, case in enumerate(TEST_CASES, 1):
        question = case["question"]
        print(f"\n[{i}/{total}] {question}")
        print(f"  Fact: {case['fact']}")
        print("-" * 60)

        result       = rag_search.invoke({"query": question})
        result_lower = result.lower()
        not_found    = (
            "no relevant information" in result_lower
            or "not found" in result_lower
            or "unavailable" in result_lower
        )

        if not case["should_find"]:
            if not_found:
                print("  PASS  — correctly returned no results (anti-hallucination working)")
                passed += 1
            else:
                print("  FAIL  — returned results for an out-of-scope question")
                print(f"  Got   : {result[:200]}...")
                failed += 1
            continue

        matched = [kw for kw in case["expected_keywords"] if kw.lower() in result_lower]

        if matched:
            print(f"  PASS  — found: {matched}")
            passed += 1
        else:
            print(f"  FAIL  — none of these found: {case['expected_keywords']}")
            if not_found:
                print("          Returned 'not found' — chunk may not be indexed or threshold too strict")
            else:
                print(f"          Retrieved: {result[:300]}...")
            failed += 1

    # Summary
    score = (passed / total) * 100
    print("\n" + "=" * 70)
    print(f"Results  : {passed} passed,  {failed} failed  ({total} total)")
    print(f"Score    : {score:.0f}%")

    if score >= 85:
        quality = "GOOD — retrieval is working well"
    elif score >= 65:
        quality = "ACCEPTABLE — consider tuning chunk size or score threshold"
    else:
        quality = "POOR — check ingestion, threshold, or document quality"

    print(f"Quality  : {quality}")
    print("=" * 70)

    return score


if __name__ == "__main__":
    run_evaluation()
