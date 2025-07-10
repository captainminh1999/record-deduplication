# GPT Integration Guide

The pipeline includes optional OpenAI GPT integration for enhanced data cleaning and review. This feature is entirely optional and the pipeline works perfectly without it.

## Prerequisites

1. Install the OpenAI library:
```bash
pip install openai
```

2. Set up your API key:
```bash
# Option 1: Environment variable (recommended)
export OPENAI_KEY="your-api-key-here"

# Option 2: Windows Command Prompt
set OPENAI_KEY=your-api-key-here

# Option 3: Windows PowerShell
$env:OPENAI_KEY="your-api-key-here"
```

## Features

### 1. Company Name Translation (Preprocessing)

Automatically translate company names to English using GPT. Useful for datasets with mixed languages.

**Note:** The preprocessing CLI doesn't currently support OpenAI translation directly. This feature is being updated for the new modular architecture.

**Alternative approach:**
```bash
python -m src.cli.openai_deduplication
```

### 2. Cluster Review (Post-Clustering)

After running clustering, GPT can analyze each cluster to identify potential duplicates.

**Usage:**
```bash
# First run clustering
python -m src.cli.clustering --eps 0.5 --min-samples 2

# Then analyze with GPT (OpenAI integration is now handled by the dedicated CLI)
python -m src.cli.openai_deduplication
```

**Output:** Creates `gpt_review.json` with AI analysis of each cluster.

## Cost Considerations

- **Translation**: ~$0.001-0.01 per company name (depending on model)
- **Cluster review**: ~$0.01-0.05 per cluster
- **Recommendation**: Start with small datasets to estimate costs

## Privacy & Security

⚠️ **Important**: Company data will be sent to OpenAI's servers. Only use this feature if:
- You have permission to send data externally
- Data is not sensitive/confidential
- You comply with your organization's data policies

## Troubleshooting

**Common issues:**

1. **API Key not found**: Ensure `OPENAI_KEY` environment variable is set
2. **Rate limiting**: GPT integration includes automatic retry with exponential backoff
3. **Quota exceeded**: Check your OpenAI account billing and usage limits
4. **Translation errors**: Some company names may not translate well; review results manually

**Error messages:**
- `OpenAI API key not found`: Set the `OPENAI_KEY` environment variable
- `Rate limit exceeded`: Wait and retry, or upgrade your OpenAI plan
- `Model not found`: Check if the specified model is available in your account

## Best Practices

1. **Test first**: Run on a small subset before processing large datasets
2. **Review results**: Always manually review GPT outputs for accuracy
3. **Backup data**: Keep original data before applying GPT translations
4. **Monitor costs**: Check OpenAI usage dashboard regularly
5. **Use appropriate models**: gpt-4o-mini is usually sufficient and cost-effective
