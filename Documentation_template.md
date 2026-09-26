# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
We resolve Source 2/3 records to Source 1 with a four-stage pipeline: script-aware normalization,
a weighted inverted-index blocker that combines name tokens with house numbers and street words,
a LightGBM matcher over 40 string, address and group-context features, and a decision layer tuned
directly for macro F0.5 that exploits the fact that every S2/S3 record belongs to at most one S1
entity. Nothing external is used: no APIs, no pretrained models, and the Indic-to-Latin dictionary
is mined from the training matches themselves.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from EDA on the training data (2.2M S1, 10.3M S2+S3 records, 7.64M true pairs):

- **Each S2/S3 record is matched to at most one S1** (7,638,365 pairs, 7,638,365 distinct matched
  IDs). 74% of pool records are matched; 26% are distractors.
- **Country always agrees** between an S1 entity and its matches (100.0%), so we block within country.
- Matches per S1: 0 (5.6%), 1 (5.4%), 2 to 5 (78%), up to 11. S2 and S3 contribute equally.
- S1 always has an address; about 3% of S2/S3 records have an empty address.
- **Names are far from unique.** Generated names reuse a small vocabulary ("Royal Infrastructure",
  "Creative Energy"), so the name alone cannot identify a business. House number plus street is
  the most stable signal; names get replaced by domains ("horizonhealthworks.com"), trade names or
  unrelated strings.
- India records use several scripts, not only Devanagari: names and states appear in Devanagari,
  Telugu, Kannada, Tamil, Bengali, Gujarati, Gurmukhi, Oriya and Malayalam (10 to 13% of India
  pool records). Some addresses carry mojibake (UTF-8 read as cp1252).
- Test adds France (15% of test S1). French records follow the US pattern with French legal forms
  (SARL, SAS, SASU, EURL, SA, SCI) and street abbreviations (R., Q., Bd, All.). The test pool is
  also denser (about 5.8 pool records per S1 vs 4.7 in train), so there are more distractors.

### 2.2 Solution Strategy
**Approach Type:** Blocking + learned pairwise classifier + F0.5-optimised decision layer  
**Core Innovation:** compound blocking keys (name token x street word, name token x house number,
house number x street word, glued name x street word) that stay selective even when names are
common, plus unique assignment of pool records.

---

## 3. Candidate Generation (Blocking)

**Normalization** (`normalize.py`): NFKC, lowercase, mojibake removal, `&` to `and`, Indic tokens
mapped to Latin with a dictionary mined from training matches (names by positional alignment of
word-by-word transliterations, 1,312 tokens; addresses by Dice co-occurrence, mostly state names),
unidecode fallback, accent stripping. Legal forms (US, India and France) are moved into a separate
`legal` field and removed from `core` name. Address tokens are canonicalised (street to st, road to
rd, rue, boulevard to blvd, near/opp...), US and Indian state codes are expanded to full names,
and house/plot numbers are extracted with leading zeros stripped.

**Blocking keys** (`blocking.py`), all within country, hashed to 64 bit:

| Key | Example |
|-----|---------|
| core-name token | `resoft` |
| whole core name without spaces | `horizonhealthworks` |
| 8-char prefix of it | `horizonh` |
| address word | `fordland` |
| house number x address word | `3444|fordland` |
| name token x address word | `resoft|fordland` |
| name token x house number | `resoft|3444` |
| glued core name x address word | `luckypilates|willis` |

Keys with pool document frequency above 200 are dropped. A candidate's score is the sum of
`log(N_country / df)` over shared keys. We keep the top-k per S1.

- **Candidate pairs generated (test):** [fill in] (k = 10 per S1)
- **How we ensured true matches were not lost:** recall was measured on a held-out 10% of train
  S1 against the full training pool, and the compound keys were added after studying missed pairs.

Blocking recall vs candidate set size (50k held-out S1, full training pool):

| k (candidates per S1) | Recall of true pairs |
|---|---|
| 5 | 0.856 |
| 8 | 0.926 |
| 10 | 0.935 |
| 15 | 0.945 |
| 20 | 0.950 |
| 30 | 0.955 |
| 50 | 0.961 |
| 100 | 0.967 |

(The final model's validation run measured 0.944 at k = 10 after adding the glued-name key.)
The curve flattens after k of about 10. Most of the remaining misses are pool records with an empty
address and a very common name, which cannot be matched precisely anyway, so we use k = 10.

---

## 4. Matching Model

**Features used** (`features.py`, 40 total; country is deliberately not a feature so the model
transfers to France):
- Name features: rapidfuzz ratio, token-set, token-sort, partial ratio and Jaro-Winkler on the core
  name; ratio and partial ratio on the space-free core name; token-set and normalised Levenshtein
  on the full name; shared/total token counts; first-token match; acronym match both ways;
  legal-form agreement and legal-form-missing flag.
- Address features: token-set, token-sort and partial ratio on the canonical address; shared
  house numbers, first-number match, number counts; empty-address flag.
- Blocking features: total score, shared weight per key type, rank.
- Group context: score relative to the best candidate of the S1, gap to the best name/address
  similarity in the group, rank by name similarity, number of candidates; and, across S1s, how
  many S1 entities compete for the same pool record and the score gap to the best of them.
- Other: source (S2 vs S3).

**Model type:** LightGBM binary classifier (127 leaves, learning rate 0.08, 600 rounds), trained on
the blocked candidates of 300k training S1 entities (3.0M pairs, 32.6% positive), so the training
distribution matches inference.

**Threshold selection method:** grid search on the validation S1 set maximising macro F0.5 over the
absolute threshold, a relative-to-best threshold and unique assignment (each pool record kept only
for the S1 that scores it highest). Selected: t = 0.70, unique assignment on, no relative threshold.
An S1 with no candidate above threshold gets an empty prediction.

---

## 5. Results & Error Analysis

Validation = 100k S1 entities held out from training (their S2/S3 records stay in the pool).

| Metric | Value |
|---|---|
| **Macro F0.5** | **0.9668** |
| F0.5 US | 0.9715 |
| F0.5 India | 0.9595 |
| Singleton accuracy | 0.9666 |
| Non-singleton F0.5 | 0.9668 |
| Micro precision / recall | 0.9927 / 0.9195 |
| Pairwise PR-AUC | 0.9990 |
| Blocking recall at k = 10 | 0.9437 |

- **Common false positives (wrong merges):** [fill in]
- **Common false negatives (missed matches):** pool records with an empty address and a common name;
  records whose name was replaced by an unrelated string and whose address is truncated.

---

## 6. Conclusion
[fill in]

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: `src/` (all code), `README.md` (exact commands), `requirements.txt`.
Entry points: `python normalize.py` then `python run_pipeline.py all`, which writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv`.

**Models and licenses:** LightGBM (MIT), trained from scratch on the provided data. No pretrained
models are used. Libraries: polars (MIT), numpy (BSD-3), pyarrow (Apache 2.0), scikit-learn
(BSD-3), scipy (BSD-3), rapidfuzz (MIT), unidecode (GPL-2.0, a text utility, not a model).

### B. Additional Results
[fill in]

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
