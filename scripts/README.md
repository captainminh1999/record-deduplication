# Development Scripts

This directory contains utility scripts for development, testing, and performance analysis of the record deduplication pipeline.

## Scripts Overview

### Performance Analysis

#### `analyze_performance.py`
**Purpose:** Analyzes the performance and distribution of similarity scores in the pipeline output.

**Usage:**
```bash
python scripts/analyze_performance.py
```

**What it does:**
- Loads and analyzes `features.csv` and `cleaned.csv` from `data/outputs/`
- Reports data shapes and loading times
- Analyzes similarity score distributions
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
