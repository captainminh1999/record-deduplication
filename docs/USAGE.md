# Step-by-Step Usage Guide

Using the pipeline involves running a series of modular Python scripts in order. Each stage reads the output of the previous stage and writes its results to the **`data/outputs/`** directory.

## Data Requirements

**Minimum Requirements:** The pipeline requires at least two columns:
- A unique **record identifier** column named `record_id` (if your file uses a different name like `sys_id`, the code will handle it by mapping that to `record_id`)
- A **company name** column (or a similarly purposed name column) in your data

**Optional but Recommended:** Additional fields will significantly improve deduplication accuracy:
- `domain` or `website` - for domain-based matching
- `phone` - for phone number matching  
- `address` - for address similarity
- Geographic fields like `state`, `country_code`

**Example minimal CSV:**
```csv
record_id,company
1,Acme Corp
2,ACME Corporation
3,Beta LLC
4,Gamma Inc
```

**Note on Data Quality:** While the pipeline will work with minimal data (just ID and company name), having additional fields like domain, phone, and address will significantly improve the accuracy of duplicate detection.

## Step 1: Preprocessing (Data Cleaning)

**What it does:** Cleans and normalizes raw data, removes obvious duplicates.

**How to run:**
```bash
python -m src.preprocess --input-path data/your_file.csv
```

**Options:**
- `--input-path`: Path to your CSV or Excel file (supports .csv, .xlsx, .xls)
- `--use-openai`: Translate company names to English using GPT
- `--openai-model`: Specify GPT model (default: gpt-4o-mini)
- `--clear`: Clear previous outputs before running
- `--output-path`: Custom output path (default: data/outputs/cleaned.csv)

**Example with custom file:**
```bash
python -m src.preprocess --input-path "C:/my_data/companies.xlsx" --clear
```

## Step 2: Blocking (Candidate Generation)

**What it does:** Generates candidate record pairs for comparison.

**How to run:**
```bash
python -m src.blocking
```

## Step 3: Similarity Features

**What it does:** Computes similarity features for each candidate pair.

**How to run:**
```bash
python -m src.similarity
```

## Step 4: Model Training (Duplicate Scoring)

**What it does:** Trains a model to score duplicate likelihood.

**How to run:**
```bash
python -m src.model
```

## Step 5: Reporting (Review and Export)

**What it does:** Creates an Excel report of suggested duplicates.

**How to run:**
```bash
python -m src.reporting
```

## Step 6: Clustering (Optional)

**What it does:** Alternative approach using DBSCAN clustering.

**How to run:**
```bash
python -m src.clustering --eps 0.5 --min-samples 2 --scale
```

## Running the Full Pipeline

You can run all steps in sequence:

```bash
# Step 1: Clean your data
python -m src.preprocess --input-path data/your_file.csv --clear

# Step 2-5: Run the pipeline
python -m src.blocking
python -m src.similarity  
python -m src.model
python -m src.reporting
```

For detailed information about each step, see [PIPELINE_STEPS.md](PIPELINE_STEPS.md).
