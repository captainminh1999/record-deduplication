# Spreadsheet Deduplication Pipeline

A **minimal record deduplication workflow** in Python that identifies and merges duplicate entries in spreadsheet data. The pipeline cleans data, generates candidate pairs, computes similarity features, trains a model to score duplicates, and produces an easy-to-review report.

## 🚀 Quick Start

1. **Install:** `python -m pip install -r requirements.txt`
2. **Run:** `python -m src.cli.preprocess data/your_file.csv`
3. **Continue:** See [Usage Guide](docs/USAGE.md) for the complete 6-step workflow

## 📋 Minimal Data Requirements

Works with just **two columns**:
- **Record ID** (`record_id` or `sys_id`)
- **Company name** (any name/company column)

```csv
record_id,company
1,Acme Corp
2,ACME Corporation  
3,Beta LLC
```

**Optional fields for better accuracy:**
- `domain`/`website` - Domain-based matching
- `phone` - Phone number matching
- `address` - Address similarity
- `state`, `country_code` - Geographic matching

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| **[Installation](docs/INSTALLATION.md)** | Setup, dependencies, virtual environments |
| **[Usage](docs/USAGE.md)** | Step-by-step commands and examples |
| **[Pipeline Steps](docs/PIPELINE_STEPS.md)** | Detailed technical documentation |
| **[Clustering Architecture](docs/CLUSTERING_ARCHITECTURE.md)** | 🆕 Modular clustering system design |
| **[Modular Architecture](docs/MODULAR_ARCHITECTURE.md)** | 🆕 Overall system architecture |
| **[GPT Integration](docs/GPT_INTEGRATION.md)** | Optional AI features and setup |
| **[AI Deduplication](docs/AI_DEDUPLICATION.md)** | 🆕 AI-powered record merging |

## 🔧 Key Features

✅ **Minimal data support** - Works with just company names and IDs  
✅ **Flexible input** - CSV, Excel files (.xlsx, .xls)  
✅ **Robust error handling** - Clear messages for missing columns  
✅ **Modular design** - Run individual steps or full pipeline  
✅ **Experiment tracking** - Comprehensive logging and statistics  
✅ **Multiple approaches** - Supervised ML + unsupervised clustering  
✅ **Advanced clustering** - 🆕 Hierarchical DBSCAN with modular subdivision strategies  
✅ **AI-powered deduplication** - 🆕 Intelligent record merging with OpenAI  

## ⚡ Pipeline Overview

```mermaid
graph LR
    A[Raw Data] --> B[1. Preprocess]
    B --> C[2. Blocking]
    C --> D[3. Similarity]
    D --> E[4. Model]
    E --> F[5. Reporting]
    D --> G[6. Clustering]
    F --> H[Excel Review]
    G --> I[GPT Analysis]
```

1. **Preprocess** - Clean & normalize data
2. **Blocking** - Generate candidate pairs
3. **Similarity** - Compute feature vectors
4. **Model** - Train ML classifier
5. **Reporting** - Excel output for review
6. **Clustering** - Hierarchical DBSCAN with intelligent subdivision

## 🧠 Advanced Clustering Features

The clustering system uses a **modular strategy pattern** for intelligent subdivision:

- **AdaptiveDBSCAN**: Cluster-specific PCA optimization
- **AggressivePCA**: Handles very large, dense clusters  
- **KMeans**: Efficient subdivision with sampling
- **ForceStrategy**: Guaranteed success fallback

**Key Benefits:**
- ✅ Respects max cluster size constraints (e.g., `--max-cluster-size 15`)
- ✅ Preserves natural cluster structure via DBSCAN
- ✅ Cluster-specific PCA transformations for optimal separation
- ✅ Progressive strategy fallback ensures reliable subdivision
- ✅ Noise-aware handling prevents artificial cluster assignments

## 🗂️ Project Structure

```
record-deduplication/
├── 📁 data/
│   ├── your_spreadsheet.csv      # Input data
│   ├── sample_input.csv          # Sample for testing
│   └── 📁 outputs/               # All pipeline outputs
│       ├── cleaned.csv           # Preprocessed data
│       ├── features.csv          # Similarity features
│       ├── high_confidence.csv   # Likely duplicates
│       └── manual_review.xlsx    # Review spreadsheet
├── 📁 src/                       # Pipeline modules
│   ├── 📁 cli/                   # Command-line interfaces
│   ├── 📁 core/                  # Core business logic engines
│   ├── 📁 formatters/            # Terminal output formatting
│   ├── 📁 io/                    # File I/O operations
│   ├── 📁 logging/               # Logging functionality
│   ├── 📁 tracking/              # Progress tracking
│   ├── 📁 pipeline/              # Pipeline orchestration
│   └── 📁 scripts/               # Development & analysis scripts
├── 📁 docs/                      # Documentation
├── 📁 tests/                     # Unit tests
└── 📁 notebooks/                 # Jupyter notebooks
```

## 🚨 Troubleshooting

**Minimal dataset issues:**

| Problem | Solution |
|---------|----------|
| Missing required columns | Ensure `record_id` and `company` columns exist |
| Low accuracy | Add more fields (domain, phone, address) |
| No duplicates found | Lower confidence threshold or review manually |
| API errors (GPT) | Check `OPENAI_KEY` environment variable |

## 🧪 Testing

Run the test suite to verify installation:

```bash
python -m unittest discover
```

## 🔧 Development Scripts

The [`src/scripts/`](src/scripts/) directory contains utility scripts for development and analysis:

- **`complete_domain_clustering.py`** - Complete domain clustering pipeline with 99.99% success
- **`domain_noise_rescue.py`** - Rescue noise records with domain similarity matching
- **`verify_perfect_clustering.py`** - Verify and analyze domain clustering quality
- **`analyze_performance.py`** - Analyze similarity score distributions and data quality
- **`benchmark_optimization.py`** - Benchmark performance of optimization algorithms

```bash
# Run complete domain clustering pipeline
python src/scripts/complete_domain_clustering.py

# Analyze clustering performance
python src/scripts/analyze_performance.py

# Verify domain clustering quality
python src/scripts/verify_perfect_clustering.py
```

See [`src/scripts/README.md`](src/scripts/README.md) for detailed usage instructions.

## � Full Documentation

All documentation is now organized in the [`docs/`](docs/) directory:

- **[📖 Documentation Index](docs/README.md)** - Complete guide to all documentation
- **[🚀 Usage Guide](docs/USAGE.md)** - Step-by-step commands and examples
- **[⚙️ Installation Guide](docs/INSTALLATION.md)** - Setup and dependencies
- **[🔧 Pipeline Steps](docs/PIPELINE_STEPS.md)** - Technical pipeline documentation
- **[🤖 AI Integration](docs/AI_DEDUPLICATION.md)** - AI-powered deduplication features
- **[📋 Architecture](docs/MODULAR_ARCHITECTURE.md)** - Code architecture overview

## �📄 License

This project is provided as-is for educational and research purposes.

---

**Need help?** Check the [complete documentation](docs/) or run with sample data:
```bash
python -m src.cli.preprocess data/sample_input.csv --normalize --deduplicate
```