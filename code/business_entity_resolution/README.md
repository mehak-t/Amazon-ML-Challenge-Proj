# Business Entity Resolution: reproduction guide

Pipeline: raw TSV -> normalization -> blocking (candidate generation) -> pairwise features ->
LightGBM matcher -> F0.5-tuned decision layer -> `output/matching_results.tsv` and
`output/candidate_pairs.tsv`.

No external data, APIs or pretrained models are used. Every artefact is learned from the provided
training files (the Indic-script to Latin word dictionary is mined from the training matches).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Paths

Set with environment variables (defaults in `src/config.py`):

| Variable    | Meaning                                                    |
|-------------|------------------------------------------------------------|
| `ER_DATA`   | the organisers' `dataset/` folder (contains `train/`, `test/`) |
| `ER_OUTPUT` | where the two TSV files are written (default `<project>/output`) |
| `ER_CACHE`  | parquet/model cache (default `~/er_work/cache`)            |

## Run end to end

```bash
cd src
python normalize.py              # parquet cache of normalized train + test records
python run_pipeline.py all       # block -> train -> predict, writes output/*.tsv
```

Stages can be run separately: `python run_pipeline.py block|train|predict`.
Everything is deterministic (fixed seed 42, deterministic split and sampling).

Validate the output with the organisers' script:

```bash
python <student_resource>/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir <student_resource>/dataset/test
```

## Source layout

| File | Purpose |
|------|---------|
| `config.py`      | paths, seed, validation fraction |
| `io_utils.py`    | TSV readers (tab, no quoting, all strings) with parquet caching |
| `normalize.py`   | NFKC/lowercase/accent strip, Indic transliteration dictionary, legal-form split, address canonicalization |
| `blocking.py`    | weighted inverted-index blocking within country |
| `features.py`    | pairwise string/address/number features plus group context |
| `decide.py`      | unique assignment and F0.5-tuned thresholds |
| `evaluate.py`    | macro F0.5 (official definition) and blocking recall/size |
| `data.py`        | train/validation split and S2+S3 pool assembly |
| `run_pipeline.py`| orchestration |

Hardware used: Intel i5-1235U (10 cores), 16 GB RAM, no GPU. Full run takes about 1 to 1.5 hours.
