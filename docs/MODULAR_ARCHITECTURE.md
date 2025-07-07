# Modular Architecture Implementation

## Overview

The record deduplication pipeline has been successfully refactored to demonstrate modern software engineering principles with clean separation of concerns. This document summarizes the new modular architecture implemented across multiple pipeline steps.

## Current Implementation Status

✅ **Completed Steps**:
1. **Preprocessing** - Data cleaning and normalization
2. **Blocking** - Candidate pair generation  
3. **Similarity** - Feature computation
4. **Model Training** - ML duplicate scoring (core logic complete)

🔄 **In Progress**:
5. **Clustering** - DBSCAN grouping
6. **Reporting** - Excel report generation

## Architecture Pattern

The new architecture follows a clear separation of concerns:

```
📁 src/
├── 🧠 core/           # Pure business logic
├── 🖥️  formatters/     # Terminal output formatting
├── 📁 io/             # File I/O utilities
└── 🚀 cli/            # Command-line interface
```

### Business Logic (`src/core/`)

**Implemented Engines**:
- `preprocess_engine.py` - Data cleaning and normalization
- `blocking_engine.py` - Candidate pair generation
- `similarity_engine.py` - Similarity feature computation
- `model_engine.py` - ML model training and scoring

**Purpose**: Contains pure business logic for each pipeline step
- **Responsibilities**: 
  - Core algorithms and data processing
  - Statistical computations
  - Machine learning operations
  - Business rule implementations
- **Dependencies**: Only pandas, scikit-learn, and core libraries
- **No I/O**: No file reading/writing
- **No UI**: No terminal output or user interaction
- **Testable**: Pure functions that are easy to unit test

**Key Pattern**:
- `*Engine`: Main business logic class
- `*Config`: Configuration data class
- `*Stats`: Statistics data class  
- `*Result`: Result data class

### Terminal Output (`src/formatters/`)

**Location**: `src/formatters/preprocess_formatter.py`

- **Purpose**: Handles all terminal output formatting
- **Responsibilities**:
  - User-friendly progress messages
  - Statistics display and formatting
  - Error message formatting
  - Next-step guidance for users
- **Dependencies**: Only formatting logic
- **Reusable**: Can be used by CLI, web interfaces, or other UIs

**Key Classes:**
- `PreprocessTerminalFormatter`: Formats output for terminal display

### File I/O (`src/io/`)

**Location**: `src/io/file_handler.py`

- **Purpose**: Manages all file reading and writing operations
- **Responsibilities**:
  - CSV/Excel file reading
  - Data file writing with proper error handling
  - Directory creation
  - File format detection
- **Dependencies**: Only pandas and OS libraries
- **Flexible**: Supports multiple file formats

**Key Classes:**
- `FileReader`: Handles reading data files
- `FileWriter`: Handles writing data files

### Command-Line Interface (`src/cli/`)

**Location**: `src/cli/`

- **Purpose**: Orchestrates the flow and handles user interaction
- **Responsibilities**:
  - Argument parsing
  - User input validation
  - Workflow orchestration
  - Error handling and user feedback
- **Dependencies**: Uses all other modules but doesn't contain business logic

**Key Files:**
- `preprocess.py`: CLI for preprocessing step
- `main.py`: Main CLI orchestrator
- `__main__.py`: Makes the CLI package executable

## Usage Examples

### New Modular CLI

```bash
# Use the new modular architecture
python -m src.cli preprocess data/sample_input.csv --normalize --deduplicate
python -m src.cli blocking data/outputs/cleaned.csv
python -m src.cli similarity data/outputs/cleaned.csv data/outputs/pairs.csv
python -m src.cli clustering --auto-eps --scale
python -m src.cli reporting

# Get help for any step
python -m src.cli preprocess --help
python -m src.cli clustering --help
python -m src.cli reporting --help
```

### Legacy CLI with New Architecture Option

```bash
# Legacy interfaces still work with new modular components
python -m src.preprocess --input-path data/sample_input.csv
python -m src.blocking data/outputs/cleaned.csv
python -m src.similarity data/outputs/cleaned.csv data/outputs/pairs.csv
python -m src.clustering --eps 0.3 --min-samples 2 --scale
python -m src.reporting
```

### Programmatic Usage

```python
# Preprocessing example
from src.core.preprocess_engine import PreprocessEngine, PreprocessConfig
from src.io.file_handler import FileReader, FileWriter
from src.formatters.preprocess_formatter import PreprocessTerminalFormatter

engine = PreprocessEngine()
result = engine.process(df, PreprocessConfig(remove_duplicates=True))

# Clustering example  
from src.core.clustering_engine import ClusteringEngine
from src.formatters.clustering_formatter import ClusteringFormatter

clustering_engine = ClusteringEngine()
clustered_records, agg_features, stats = clustering_engine.cluster_records(
    features_path="data/outputs/features.csv",
    cleaned_path="data/outputs/cleaned.csv",
    eps=0.3,
    min_samples=2,
    auto_eps=True
)

# Reporting example
from src.core.reporting_engine import ReportingEngine
from src.formatters.reporting_formatter import ReportingFormatter

reporting_engine = ReportingEngine()
stats = reporting_engine.generate_report(
    dupes_path="data/outputs/high_confidence.csv",
    cleaned_path="data/outputs/cleaned.csv",
    report_path="data/outputs/manual_review.xlsx"
)

formatter = ReportingFormatter()
print(formatter.format_comprehensive_results(stats, "manual_review.xlsx"))
```

## Benefits of the New Architecture

### 1. **Separation of Concerns**
- Business logic is isolated from I/O and UI concerns
- Each module has a single, clear responsibility
- Changes to one concern don't affect others

### 2. **Testability**
- Pure business logic functions are easy to unit test
- Mock I/O operations without affecting core logic
- Test formatters independently of business logic

### 3. **Reusability**
- Business logic can be used in different contexts (CLI, web API, notebooks)
- Formatters can be swapped (terminal, web, JSON output)
- I/O handlers work across different interfaces

### 4. **Maintainability**
- Clear module boundaries make code easier to understand
- Localized changes reduce risk of breaking other components
- New developers can focus on specific areas

### 5. **Flexibility**
- Easy to add new output formats (JSON, web interface)
- Simple to modify business logic without UI changes
- Can integrate with different data sources

## Migration Strategy

The refactoring maintains **full backward compatibility**:

1. **Legacy Interface**: `src/preprocess.py` still works exactly as before
2. **New Architecture Option**: Added `--use-new-architecture` flag to legacy CLI
3. **Gradual Migration**: Teams can migrate step-by-step
4. **Fallback Handling**: Graceful degradation if new modules are unavailable

## Current Status

The following pipeline steps have been successfully modularized:

✅ **Preprocessing** - Completed
- `src/core/preprocess_engine.py` (business logic)
- `src/formatters/preprocess_formatter.py` (terminal output)
- `src/cli/preprocess.py` (CLI orchestration)

✅ **Blocking** - Completed
- `src/core/blocking_engine.py` (business logic)
- `src/formatters/blocking_formatter.py` (terminal output)
- `src/cli/blocking.py` (CLI orchestration)

✅ **Similarity** - Completed
- `src/core/similarity_engine.py` (business logic)
- `src/formatters/similarity_formatter.py` (terminal output)
- `src/cli/similarity.py` (CLI orchestration)

✅ **Model Training** - Completed
- `src/core/model_engine.py` (business logic)
- `src/formatters/model_formatter.py` (terminal output)
- `src/cli/model.py` (CLI orchestration)

✅ **Clustering** - Completed
- `src/core/clustering_engine.py` (business logic)
- `src/formatters/clustering_formatter.py` (terminal output)
- `src/cli/clustering.py` (CLI orchestration)

✅ **Reporting** - Completed
- `src/core/reporting_engine.py` (business logic)
- `src/formatters/reporting_formatter.py` (terminal output)
- `src/cli/reporting.py` (CLI orchestration)

## 🎉 Modularization Complete!

All major pipeline steps have been successfully refactored into the modular architecture! 

### ✅ Completed Pipeline Steps:
- **Preprocessing**: Full modular implementation with engine, formatter, and CLI
- **Blocking**: Full modular implementation with engine, formatter, and CLI
- **Similarity**: Full modular implementation with engine, formatter, and CLI
- **Model Training**: Full modular implementation with engine, formatter, and CLI
- **Clustering**: Full modular implementation with engine, formatter, and CLI  
- **Reporting**: Full modular implementation with engine, formatter, and CLI

### ✅ Architecture Features:
- **Main CLI**: Unified CLI supporting all pipeline steps as subcommands
- **Backward Compatibility**: All legacy interfaces still work via bridge pattern
- **File I/O**: Centralized file handling utilities
- **Error Handling**: Robust error handling with automatic fallbacks
- **Data Type Handling**: Smart ID type conversion and mismatch detection

The modular architecture is now production-ready and provides:
- Clean separation of concerns (business logic, UI, I/O)
- Full backward compatibility with existing interfaces
- Easy testing and maintenance
- Flexible integration options (CLI, web, API, notebooks)

Optional future improvements:
1. Add more comprehensive test coverage for edge cases
2. Add web interface using the modular components
3. Create Jupyter notebook examples using the engine classes
4. Add configuration file support for complex workflows

## Example: Current vs. New Architecture

### Before (Monolithic)
```python
# In src/preprocess.py - everything mixed together
def main(input_path, output_path, ...):
    # File I/O
    df = pd.read_csv(input_path)
    
    # Business logic
    df['company_clean'] = df['company'].apply(normalize_company_name)
    
    # Terminal output
    print(f"Processed {len(df)} records")
    
    # More file I/O
    df.to_csv(output_path)
    
    # More terminal output
    print("Complete!")
```

### After (Modular)
```python
# Business logic in src/core/preprocess_engine.py
class PreprocessEngine:
    def process(self, df, config):
        # Pure business logic only
        pass

# Terminal output in src/formatters/preprocess_formatter.py  
class PreprocessTerminalFormatter:
    def format_results(self, result, ...):
        # Only formatting logic
        pass

# File I/O in src/io/file_handler.py
class FileReader:
    def read_data(self, path):
        # Only file reading logic
        pass

# CLI orchestration in src/cli/preprocess.py
class PreprocessCLI:
    def run(self, args):
        # Orchestrates flow, no business logic
        df = self.file_reader.read_data(args.input)
        result = self.engine.process(df, config)
        self.file_writer.write_csv(result.cleaned_df, args.output)
        self.formatter.format_results(result, ...)
```

## Code Quality Improvements

- **Single Responsibility**: Each class has one clear purpose
- **Dependency Injection**: Components can be easily swapped
- **Pure Functions**: Business logic functions have no side effects
- **Clear Interfaces**: Well-defined contracts between modules
- **Error Handling**: Centralized and consistent error management
- **Documentation**: Each module is self-documenting

This refactoring demonstrates how to modernize legacy code while maintaining backward compatibility and providing a foundation for future enhancements.
