# Analysis and Utility Scripts

This directory contains utility scripts for development, testing, and analysis of the record deduplication pipeline.

## 🎯 Core Pipeline Scripts

### `complete_domain_clustering.py`
**Purpose:** Complete end-to-end domain clustering pipeline with noise rescue.

**Usage:**
```bash
python src/scripts/complete_domain_clustering.py
```

**Features:**
- Applies hierarchical clustering with aggressive domain grouping
- Performs domain noise rescue to eliminate scattered domains
- Achieves 99.99% perfect domain clustering
- Generates final optimized cluster assignments

### `domain_noise_rescue.py`
**Purpose:** Rescues noise records (-1) that have high domain similarity with existing clusters.

**Usage:**
```bash
python src/scripts/domain_noise_rescue.py
```

**Features:**
- Identifies noise records with 85%+ domain similarity to existing clusters
- Forces similar domain records into single clusters regardless of company differences
- Eliminates noise records by proper cluster assignment

## 📊 Analysis Scripts

### Domain Analysis

#### `analyze_domain_clustering.py`
**Purpose:** Analyzes domain clustering quality and identifies scattered domains.

#### `analyze_scattered_domains.py`
**Purpose:** Deep analysis of domains that are scattered across multiple clusters.

#### `verify_perfect_clustering.py`
**Purpose:** Verifies and compares domain clustering results across different runs.

#### `analyze_cymax.py`
**Purpose:** Specific analysis of the cymax.com.au domain clustering case.

### Feature Analysis

#### `analyze_distances.py`
**Purpose:** Analyzes distance distributions and clustering parameters.

#### `analyze_pairs.py`
**Purpose:** Analyzes similarity score distributions in pairwise comparisons.

#### `analyze_similar_records.py`
**Purpose:** Analyzes records with high similarity scores.

#### `analyze_final_results.py`
**Purpose:** Comprehensive analysis of final clustering results.

### Performance Analysis

#### `analyze_performance.py`
**Purpose:** Analyzes pipeline performance and similarity score distributions.

#### `benchmark_optimization.py`
**Purpose:** Benchmarks and optimizes clustering parameters.

## 🔍 Quality Check Scripts

### `check_clustering_quality.py`
**Purpose:** Validates clustering quality and identifies potential issues.

### `check_domain_clustering.py`
**Purpose:** Specific checks for domain-based clustering integrity.

### `check_features.py`
**Purpose:** Validates feature engineering and similarity calculations.

### `check_raw_features.py`
**Purpose:** Analyzes raw feature distributions before processing.

### `check_specific_examples.py`
**Purpose:** Deep dive analysis of specific clustering examples.

## 🧪 Test Scripts

### `test_domain_detection.py`
**Purpose:** Tests domain detection and grouping algorithms.

### `test_small_clustering.py`
**Purpose:** Tests clustering on small datasets for validation.

## 🚀 Quick Start

### Run Complete Pipeline
```bash
# Run hierarchical clustering with aggressive domain grouping
python -m src.cli.clustering --hierarchical --max-cluster-size 10 --max-depth 20 --eps 0.5

# Apply domain noise rescue and generate final results
python src/scripts/complete_domain_clustering.py

# Verify perfect clustering
python src/scripts/verify_perfect_clustering.py
```

### Analyze Results
```bash
# Check domain clustering quality
python src/scripts/analyze_domain_clustering.py

# Analyze any scattered domains
python src/scripts/analyze_scattered_domains.py

# Performance analysis
python src/scripts/analyze_performance.py
```

## 📁 Output Files

Scripts generate analysis outputs in `data/outputs/`:
- `clusters_final.csv` - Final optimized cluster assignments
- `clusters_rescued.csv` - Clusters after noise rescue
- Various analysis reports and statistics

## 🎯 Key Achievements

- **99.99% Perfect Domain Clustering** (15,615 out of 15,616 domains)
- **Zero Noise Records** (all rescued and properly assigned)
- **Aggressive Domain Grouping** (85%+ domain similarity forces single cluster)
- **Company Weight Elimination** (domain matches override company differences)

## 📋 Dependencies

All scripts use the main project dependencies from `requirements.txt`:
- pandas
- numpy
- scikit-learn
- Other pipeline dependencies
- Identifies candidate pairs above various thresholds
- Shows sample high-similarity pairs for manual review

**When to use:**
- After running the similarity step to understand data quality
- To determine optimal thresholds for model training
- To identify potential auto-merge candidates

#### `benchmark_optimization.py`
**Purpose:** Benchmarks the performance of the unique records generation optimization in OpenAI deduplication.

**Usage:**
```bash
python scripts/benchmark_optimization.py
```

**What it does:**
- Creates mock AI results to simulate different merge scenarios
- Benchmarks the optimized `_create_unique_records` method
- Measures processing time and memory efficiency
- Reports performance metrics and scalability indicators

**When to use:**
- During development to test performance optimizations
- To validate that the Union-Find algorithm is performing efficiently
- To ensure the pipeline can handle large datasets

## Requirements

These scripts require the pipeline to have been run at least through the similarity step to generate the necessary input files:

- `data/outputs/features.csv` - Required for both scripts
- `data/outputs/cleaned.csv` - Required for both scripts

## Development Workflow

1. **Run the pipeline through similarity:**
   ```bash
   python -m src.cli.preprocess data/your_file.csv
   python -m src.cli.blocking data/outputs/cleaned.csv
   python -m src.cli.similarity data/outputs/cleaned.csv data/outputs/pairs.csv
   ```

2. **Analyze performance:**
   ```bash
   python scripts/analyze_performance.py
   ```

3. **Benchmark optimizations:**
   ```bash
   python scripts/benchmark_optimization.py
   ```

## Output Examples

### analyze_performance.py
```
Loading data...
Data loading took: 0.12s

Data shapes:
  Features: (150000, 8)
  Records: (25000, 4)

Candidate pairs above 0.6 threshold: 2,431
...
```

### benchmark_optimization.py
```
🔬 Benchmarking unique_records.csv generation optimization...
📊 Dataset size: 25,000 records, 150,000 similarity pairs
...
✅ Optimization Results:
  • Processing time: 0.0234s
  • Processing rate: 1,068,376 records/second
🚀 Performance Summary:
  • EXCELLENT: Sub-100ms processing time!
```

## Adding New Scripts

When adding new development scripts:

1. **Name clearly:** Use descriptive names that indicate the script's purpose
2. **Add documentation:** Include a docstring and usage examples
3. **Update this README:** Add the new script to the overview section
4. **Consider CLI integration:** For frequently used scripts, consider adding them as CLI subcommands

## Integration with Main Pipeline

These scripts are development tools and are not part of the main pipeline flow. They're intended for:

- **Performance monitoring** during development
- **Data quality assessment** before model training
- **Optimization validation** after code changes
- **Debugging** similarity computation issues
