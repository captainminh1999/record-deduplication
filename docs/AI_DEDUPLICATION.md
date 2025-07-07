# AI-Powered Deduplication

This document describes the AI-powered deduplication module that uses OpenAI to intelligently identify and merge duplicate records based on similarity scores.

## Overview

The `openai_deduplication.py` module takes the output from similarity calculation (`features.csv`) and uses AI to make intelligent decisions about which records represent the same organization. This provides more nuanced deduplication than simple threshold-based approaches.

## Key Features

### 🤖 **Intelligent Analysis**
- Uses OpenAI GPT models to analyze company pairs
- Considers context beyond just similarity scores
- Distinguishes between duplicates and legitimate business relationships

### ⚡ **Parallel Processing**
- Configurable number of parallel workers (default: 10)
- Batch processing for efficiency
- Progress bar for long-running operations

### 📊 **Structured Output**
- Creates `unique_records.csv` with deduplicated data
- Tracks which records were merged (`merged_from` field)
- Canonical company names for merged entities
- Comprehensive analysis statistics in JSON format

### 🎯 **Configurable Thresholds**
- **Similarity Threshold**: Only analyze pairs above this score
- **Confidence Threshold**: Only merge records with high AI confidence
- Flexible sampling for testing and cost control

## Usage

### Basic Usage
```bash
python -m src.openai_deduplication
```

### With Custom Thresholds
```bash
python -m src.openai_deduplication \
  --similarity-threshold 0.7 \
  --confidence-threshold 0.8
```

### Testing with Sample
```bash
python -m src.openai_deduplication \
  --sample-size 100 \
  --similarity-threshold 0.6
```

### Performance Tuning
```bash
python -m src.openai_deduplication \
  --max-workers 20 \
  --batch-size 15
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--features-path` | `data/outputs/features.csv` | Input features file with similarity scores |
| `--cleaned-path` | `data/outputs/cleaned.csv` | Cleaned records file |
| `--output-path` | `data/outputs/unique_records.csv` | Output deduplicated records |
| `--analysis-path` | `data/outputs/deduplication_analysis.json` | Detailed analysis results |
| `--similarity-threshold` | `0.6` | Minimum similarity score for AI analysis |
| `--confidence-threshold` | `0.7` | Minimum AI confidence to merge records |
| `--sample-size` | `None` | Limit analysis to N pairs (for testing) |
| `--batch-size` | `10` | Pairs per batch for parallel processing |
| `--max-workers` | `10` | Number of parallel API workers |
| `--model` | `gpt-4o-mini` | OpenAI model to use |

## Input Requirements

### Prerequisites
1. **Features File**: Must contain similarity scores from `src.similarity`
2. **Cleaned Records**: Original cleaned data from `src.preprocess`
3. **OpenAI API Key**: Set in environment variables

### Required Columns in Features File
- `record_id_1`, `record_id_2`: Record identifiers
- `company_sim`: Company name similarity score
- `company_clean_1`, `company_clean_2`: Company names
- `domain_clean_1`, `domain_clean_2`: Domain names (optional)
- `phone_clean_1`, `phone_clean_2`: Phone numbers (optional)

## Output Files

### 1. Unique Records (`unique_records.csv`)
Contains deduplicated records with additional fields:

- `record_id`: Original record identifier
- `is_merged`: Boolean indicating if this record represents merged duplicates
- `merged_from`: List of original record IDs that were merged
- `canonical_company`: AI-suggested canonical company name

### 2. Analysis Results (`deduplication_analysis.json`)
Comprehensive statistics including:

```json
{
  "metadata": {
    "timestamp": "2025-07-07 16:17:21",
    "model": "gpt-4o-mini",
    "similarity_threshold": 0.85,
    "confidence_threshold": 0.7,
    "processing_time_seconds": 7.55
  },
  "deduplication_results": {
    "original_records": 24671,
    "unique_records": 24668,
    "deduplication_rate": 0.0001,
    "records_eliminated": 3
  },
  "api_stats": {
    "calls": 1,
    "total_tokens": 887,
    "estimated_cost": 0.027
  }
}
```

## AI Decision Process

The AI analyzes each pair considering:

1. **Company Name Similarity**: Variations, abbreviations, punctuation
2. **Domain/Phone Consistency**: Supporting evidence for same organization
3. **Business Context**: Subsidiaries vs. duplicates vs. unrelated companies

### Decision Criteria
- **Same Organization**: True if records represent the same business entity
- **Confidence Score**: 0.0-1.0 indicating AI certainty
- **Primary Record**: Which record should be the canonical one
- **Canonical Name**: Best representation of the company name

## Best Practices

### 🎯 **Threshold Selection**
- **High Similarity Threshold (0.8+)**: Focus on obvious duplicates
- **Medium Threshold (0.6-0.8)**: Balance precision and recall
- **Low Threshold (0.4-0.6)**: Catch more duplicates but review carefully

### 💰 **Cost Management**
- Use `--sample-size` for initial testing
- Higher similarity thresholds reduce API calls
- Monitor token usage in analysis results

### 🔍 **Quality Assurance**
- Review merged records with low confidence scores
- Validate canonical names for accuracy
- Check `merged_from` field for unexpected groupings

### ⚡ **Performance Optimization**
- Increase `--max-workers` for faster processing (watch API limits)
- Adjust `--batch-size` based on your data complexity
- Use higher similarity thresholds to reduce total pairs analyzed

## Integration with Pipeline

This module fits into the deduplication pipeline as follows:

1. **Preprocessing**: `src.preprocess` → `cleaned.csv`
2. **Blocking**: `src.blocking` → `pairs.csv`
3. **Similarity**: `src.similarity` → `features.csv`
4. **🆕 AI Deduplication**: `src.openai_deduplication` → `unique_records.csv`
5. **Reporting**: `src.reporting` (can be updated to use unique records)

## Troubleshooting

### Common Issues

**High API Costs**
- Reduce similarity threshold to analyze fewer pairs
- Use sample size for testing
- Consider using cheaper models for initial runs

**Low Deduplication Rate**
- Lower similarity threshold to include more candidates
- Reduce confidence threshold (but review results carefully)
- Check if your data actually contains duplicates

**Processing Errors**
- Verify OpenAI API key is set correctly
- Check network connectivity
- Reduce batch size or max workers if hitting rate limits

### Error Messages

- `Features file not found`: Run similarity calculation first
- `OpenAI API key not configured`: Set environment variable
- `No pairs above similarity threshold`: Lower the threshold

## Cost Estimation

Rough cost estimates (using gpt-4o-mini at $0.03/1K tokens):

- **1,000 pairs**: ~$3-5
- **10,000 pairs**: ~$30-50  
- **100,000 pairs**: ~$300-500

Actual costs depend on:
- Complexity of company names
- Batch size and API efficiency
- Model chosen

Use `--sample-size` to test and estimate costs before full runs.
