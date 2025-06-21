# Spreadsheet Deduplication Pipeline

This repository demonstrates a minimal deduplication workflow using
Python and popular data‑science libraries.  The code is organised as a
set of small modules that can be executed one after another to clean
data, generate candidate pairs, create similarity features and produce
merge suggestions.

## Setup

1. **Install dependencies**

   ```bash
   python -m pip install -r requirements.txt
   ```

   If you would like to use the optional GPT powered helper, also
   install the commented ``openai`` package.

2. **Prepare input data**

   Place your raw spreadsheet in ``data/your_spreadsheet.csv`` (or
   adjust the ``--input-path`` argument when running the modules).  The
   preprocessing step requires a unique identifier column named
   ``record_id``.  If your data uses ``sys_id`` instead, it will be mapped to
   ``record_id`` automatically.  A small example dataset
   ``data/sample_input.csv`` &mdash; which already includes this column &mdash;
   is provided for testing.  The example pipeline expects columns such as
   ``company``, ``domain`` and ``phone`` which will be normalised during
   preprocessing. ``company``, ``domain`` and ``phone`` are combined into a
   ``combined_id`` field used for deduplication.

## Step‑by‑step workflow

Run the following modules in order.  Each step reads the output of the
previous one and writes its own results to the ``data/`` directory.

1. **Preprocess** – cleans the raw spreadsheet and writes
   ``data/cleaned.csv``. Removed duplicates are stored in
   ``data/removed_rows.csv``.  Pass ``--use-openai`` to translate company names to
   English via the OpenAI API using the ``gpt-4o-mini`` model by default. You
   can override the model with ``--openai-model``.

   ```bash
   python -m src.preprocess --use-openai
   ```

2. **Blocking** – generates candidate record pairs for comparison.

   ```bash
   python -m src.blocking
   ```

3. **Similarity features** – computes string similarity metrics for the
   candidate pairs.

   ```bash
   python -m src.similarity
   ```

4. **Model training** – fits a logistic regression model and scores each
   pair to estimate duplicate probability.

   ```bash
   python -m src.model
   ```

5. **Reporting** – creates an Excel workbook with high‑confidence
   duplicates side by side for manual review.

   ```bash
   python -m src.reporting
   ```

6. **(Optional) GPT integration** – demonstrates how the OpenAI API
   could be used to further assist with merge suggestions.

   ```bash
   python -m src.openai_integration
   ```

After inspecting the generated ``merge_suggestions.xlsx`` you can decide
how to merge or remove duplicate records from your original data.

Each step accepts ``--log-path`` to append run information (start time, end time,
rows processed and duration) to a log file. Use ``--clear`` to remove existing
outputs before running a step again. Utility functions are available in
``src.utils`` for clearing intermediate files or the entire ``data`` directory.

## Running tests

The repository contains a small suite of unit tests under the ``tests``
directory. You can execute them with Python's built‑in test runner:

```bash
python -m unittest discover
```

Run the command from the repository root to automatically discover and
execute all tests.
