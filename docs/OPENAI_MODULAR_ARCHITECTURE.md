# OpenAI Modular Architecture

## Overview

The OpenAI integration has been refactored into a clean, modular architecture with separate concerns and responsibilities.

## New Architecture

### Core Modules

- **`src/core/openai_engine.py`** - Main orchestrator for OpenAI operations
- **`src/core/openai_types.py`** - Data types and configuration classes
- **`src/core/openai_client.py`** - OpenAI API client with error handling and stats
- **`src/core/openai_translator.py`** - Text translation functionality
- **`src/core/openai_cluster_reviewer.py`** - Cluster review logic
- **`src/core/openai_deduplicator.py`** - Record deduplication logic

### CLI Modules

- **`src/cli/openai_cluster_review.py`** - CLI for cluster review
- **`src/cli/openai_deduplication.py`** - CLI for record deduplication
- **`src/cli/main.py`** - Main CLI with integrated OpenAI subcommands

### Legacy Bridges

- **`src/openai_integration.py`** - Legacy wrapper for cluster review (maintained for compatibility)
- **`src/openai_deduplication.py`** - Legacy wrapper for deduplication (maintained for compatibility)

## Usage

### New Modular Approach (Recommended)

```bash
# Main CLI with OpenAI subcommands
python -m src.cli.main openai-cluster --help
python -m src.cli.main openai-dedup --help

# Example cluster review
python -m src.cli.main openai-cluster --clusters-path data/outputs/clusters.csv --min-cluster-size 3

# Example deduplication
python -m src.cli.main openai-dedup --features-path data/outputs/features.csv --confidence-threshold 0.8
```

### Legacy Support (Backward Compatibility)

```python
# Legacy function calls still work
from src.openai_integration import translate_to_english, main
from src.openai_deduplication import main as dedup_main
```

## Benefits

1. **Modularity** - Each component has a single responsibility
2. **Testability** - Pure business logic separated from I/O
3. **Maintainability** - Clean interfaces and separation of concerns
4. **Extensibility** - Easy to add new OpenAI-powered features
5. **Backward Compatibility** - Legacy scripts continue to work

## Migration Guide

- **For new code**: Use the modular CLI or core engines directly
- **For existing code**: Legacy entry points are maintained and bridge to the new architecture
- **For advanced usage**: Import and use the core engines directly for custom workflows

## Configuration

All OpenAI operations use the `OpenAIConfig` dataclass for configuration:

```python
from src.core.openai_types import OpenAIConfig

config = OpenAIConfig(
    model="gpt-4o-mini-2024-07-18",
    max_workers=10,
    similarity_threshold=0.6,
    confidence_threshold=0.7,
    # ... other options
)
```
