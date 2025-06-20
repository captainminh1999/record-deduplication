# Record Deduplication

This repository contains a minimal workflow for detecting duplicate records. The
process is split into a few small scripts under `src/` which mirror the typical
steps in a deduplication pipeline.

1. **Preprocessing** (`src/preprocess.py`)
   - Load the raw spreadsheet.
   - Normalize names and phone numbers.
   - Drop exact duplicates and write `data/cleaned.csv`.
2. **Blocking** (`src/blocking.py`)
   - Use `recordlinkage.Index` to build candidate pairs.
3. **Similarity Features** (`src/similarity.py`)
   - Compute similarity features with `recordlinkage` and `rapidfuzz`.
   - Write `data/features.csv`.
4. **Modeling** (`src/model.py`)
   - Train a logistic regression model on labeled pairs.
   - Output high confidence duplicates to `data/dupes_high_conf.csv`.
5. **Reporting** (`src/reporting.py`)
   - Produce `merge_suggestions.xlsx` for human review.

Install the dependencies listed in `requirements.txt` and run the scripts in the
order above to reproduce the workflow.
