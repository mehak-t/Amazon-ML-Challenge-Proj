# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Developing Divas
**Team Members:** Mehak Trivedi, Purva Pote
**Submission Date:** [Date]

---

## 1. Executive Summary

_Provide a brief 2-3 sentence overview of your approach and key innovations._

---

## 2. Methodology

### 2.1 Problem Analysis

_Key insights discovered during EDA — noise patterns, address variations, missing fields, etc._

### 2.2 Solution Strategy

_Outline your high-level approach._

**Approach Type:** [Blocking + Classifier / End-to-End / Graph-Based / Hybrid, etc]  
**Core Innovation:** [Brief description of your main technical contribution]

---

## 3. Candidate Generation (Blocking)

_Describe how you reduced the comparison space to a manageable candidate set._

- **Blocking keys used:** [e.g., PIN code, phonetic name encoding, TF-IDF, etc.]
- **Candidate pairs generated:** [total]
- **How you ensured true matches were not lost:**

---

## 4. Matching Model

**Features used:**

- Name features: [e.g., Jaccard, Levenshtein, phonetic encoding]
- Address features: [e.g., token overlap, edit distance, PIN code matching]
- Other: []

**Model type:** [e.g., XGBoost, Siamese Network, Transformer, etc.]  
**Threshold selection method:** [e.g., F_0.5 optimization on validation set]

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [your best validation score]
- **Common false positives (wrong merges):** [brief description]
- **Common false negatives (missed matches):** [brief description]

---

## 6. Conclusion

_Summarize your approach, key achievements, and lessons learned in 2-3 sentences._

---

## Appendix

### A. Code Artefacts

_Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here._

### B. Additional Results

_Include any additional charts, graphs, or detailed results._

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
