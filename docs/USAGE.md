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

## Clearing Previous Outputs

Before running a fresh pipeline, you may want to clear previous outputs:

```bash
# Clear all output files
python -c "from src.utils import clear_all_data; clear_all_data()"

# Clear specific files
python -c "from src.utils import clear_files; clear_files(['data/outputs/cleaned.csv', 'data/outputs/pairs.csv'])"
```

## Step 1: Preprocessing (Data Cleaning)

**What it does:** Cleans and normalizes raw data, removes obvious duplicates.

**How to run:**
```bash
python -m src.cli.preprocess data/your_file.csv
```

**Options:**
- `--output`: Custom output path (default: data/outputs/cleaned.csv)
- `--normalize`: Apply normalization to company names
- `--deduplicate`: Remove exact duplicates
- `--quiet`: Suppress progress output

**Example with custom file:**
```bash
python -m src.cli.preprocess "C:/my_data/companies.xlsx" --output data/outputs/cleaned.csv --normalize --deduplicate
```

## Step 2: Blocking (Candidate Generation)

**What it does:** Generates candidate record pairs for comparison.

**How to run:**
```bash
python -m src.cli.blocking data/outputs/cleaned.csv
```

**Options:**
- `--output`: Custom output path (default: data/outputs/pairs.csv)
- `--quiet`: Suppress progress output

## Step 3: Similarity Features

**What it does:** Computes similarity features for each candidate pair.

**How to run:**
```bash
python -m src.cli.similarity data/outputs/cleaned.csv data/outputs/pairs.csv
```

**Options:**
- `--output`: Custom output path (default: data/outputs/features.csv)
- `--quiet`: Suppress progress output

## Step 4: Model Training (Duplicate Scoring)

**What it does:** Trains a model to score duplicate likelihood.

**How to run:**
```bash
python -m src.cli.model
```

## Step 5: Reporting (Review and Export)

**What it does:** Creates an Excel report of suggested duplicates.

**How to run:**
```bash
python -m src.cli.reporting
```

## Step 6: Clustering (Optional)

**What it does:** Advanced DBSCAN clustering with hierarchical subdivision for large clusters.

**Basic clustering:**
```bash
python -m src.cli.clustering --eps 0.5 --min-samples 2 --scale
```

**Hierarchical clustering (recommended):**
```bash
python -m src.cli.clustering --hierarchical --max-cluster-size 15 --max-depth 2 --eps 0.4
```

**Advanced options:**
- `--hierarchical`: Enable hierarchical subdivision
- `--max-cluster-size`: Maximum cluster size before subdivision (default: 50)
- `--max-depth`: Maximum subdivision depth (default: 3)
- `--eps`: DBSCAN epsilon parameter (default: 0.1)
- `--min-samples`: DBSCAN minimum samples (default: 2)
- `--scale`: Apply enhanced scaling (recommended)

**Clustering strategies:**
- **AdaptiveDBSCAN**: Cluster-specific PCA optimization
- **AggressivePCA**: For very large clusters (≥1000 records)
- **KMeans**: Efficient subdivision with sampling
- **ForceStrategy**: Guaranteed success fallback

## OpenAI Integration (Optional)

**What it does:** AI-powered record deduplication using OpenAI.

**How to run:**
```bash
python -m src.cli.openai_deduplication
```

**Options:**
- `--features-path`: Path to features CSV (default: data/outputs/features.csv)
- `--cleaned-path`: Path to cleaned records CSV (default: data/outputs/cleaned.csv)
- `--similarity-threshold`: Minimum similarity score (default: 0.6)
- `--confidence-threshold`: Minimum AI confidence (default: 0.7)
- `--model`: OpenAI model to use (default: gpt-4o-mini-2024-07-18)
- `--max-workers`: Number of parallel workers (default: 10)

## Running the Full Pipeline

You can run all steps in sequence:

```bash
# Step 1: Clean your data
python -m src.cli.preprocess data/your_file.csv --normalize --deduplicate

# Step 2-5: Run the pipeline
python -m src.cli.blocking data/outputs/cleaned.csv
python -m src.cli.similarity data/outputs/cleaned.csv data/outputs/pairs.csv
python -m src.cli.model
python -m src.cli.reporting

# Optional: OpenAI deduplication
python -m src.cli.openai_deduplication
```

## Pipeline Orchestrator

For automated pipeline execution, you can use the pipeline orchestrator:

```bash
python -c "from src.pipeline.orchestrator import PipelineOrchestrator; PipelineOrchestrator().run_full_pipeline('data/your_file.csv')"
```

For detailed information about each step, see [PIPELINE_STEPS.md](PIPELINE_STEPS.md).
