# Spreadsheet Deduplication Pipeline

This project contains a simple 10-step workflow for detecting and merging
duplicate rows from spreadsheet data.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run sequence

Execute each stage in order:

```bash
python -m src.preprocess
python -m src.blocking
python -m src.similarity
python -m src.model
python -m src.reporting
```

The optional OpenAI helper can be enabled by installing the `openai`
package (uncomment the line in `requirements.txt`) and running:

```bash
python -m src.openai_integration
```
