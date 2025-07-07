"""
OpenAI Deduplication - AI-powered record deduplication using similarity features

This module uses AI to analyze pairs with similarity scores and make intelligent 
deduplication decisions, generating a list of unique records by merging duplicates.
"""

from __future__ import annotations

from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict

import json
import os
import sys
import time
import concurrent.futures

import click
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .utils import log_run, LOG_PATH

# The OpenAI package is optional and may not be installed by default
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

client: Any = None  # type: ignore

# Default chat model used across this module
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"


def _init_api_stats() -> Dict[str, Any]:
    """Initialize API call statistics tracking."""
    return {
        "calls": 0,
        "tokens": {
            "prompt": 0,
            "completion": 0,
            "total": 0
        },
        "durations": [],
        "errors": defaultdict(int),
        "costs": {
            "total": 0.0
        }
    }


def _update_api_stats(stats: Dict[str, Any], duration: float, response: Any, error: str | None = None) -> None:
    """Update API call statistics."""
    stats["calls"] += 1
    stats["durations"].append(duration)
    
    if error:
        stats["errors"][error] += 1
    elif hasattr(response, "usage"):
        stats["tokens"]["prompt"] += response.usage.prompt_tokens
        stats["tokens"]["completion"] += response.usage.completion_tokens
        stats["tokens"]["total"] += response.usage.total_tokens
        
        # Official OpenAI pricing for gpt-4o-mini (as of 2024)
        # Source: https://platform.openai.com/docs/pricing
        input_price_per_1m = 0.15   # $0.15 per 1M input tokens
        output_price_per_1m = 0.60  # $0.60 per 1M output tokens
        
        # Calculate separate costs for input and output tokens
        input_cost = (response.usage.prompt_tokens / 1_000_000) * input_price_per_1m
        output_cost = (response.usage.completion_tokens / 1_000_000) * output_price_per_1m
        
        stats["costs"]["total"] += input_cost + output_cost


def _check_openai() -> None:
    """Ensure the OpenAI dependency and API key are available."""
    if OpenAI is None:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "openai package is not installed. Install 'openai' to enable integration."
        )

    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key is not configured. Set 'OPENAI_API_KEY' or 'OPENAI_KEY' environment variable."
            )
        client = OpenAI(api_key=api_key)


def analyze_duplicate_batch(
    batch_data: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    api_stats: Dict[str, Any] | None = None
) -> List[Dict[str, Any]]:
    """Analyze a batch of candidate duplicate pairs using OpenAI."""
    
    if not batch_data:
        return []
    
    if api_stats is None:
        api_stats = _init_api_stats()
    
    # Create prompt for GPT analysis
    prompt_lines = [
        "Analyze the following pairs of company records with their similarity scores.",
        "For each pair, determine:",
        "1. Are they the SAME organization? (true/false)",
        "2. Your confidence level (0.0-1.0)", 
        "3. Which record should be the primary/canonical one?",
        "4. What should the merged company name be?",
        "",
        "Consider:",
        "- Companies with slight name variations (abbreviations, punctuation) are likely the same",
        "- Subsidiaries and parent companies are DIFFERENT organizations",
        "- Different divisions/brands of the same company are the SAME organization",
        "",
        "Records to analyze:"
    ]
    
    for i, pair in enumerate(batch_data, 1):
        prompt_lines.extend([
            f"Pair {i} (ID: {pair['pair_id']}):",
            f"  Similarity Score: {pair['similarity_score']:.3f}",
            f"  Record A (ID: {pair['record_1']['id']}): '{pair['record_1']['company']}'",
            f"    Domain: '{pair['record_1']['domain']}', Phone: '{pair['record_1']['phone']}'",
            f"  Record B (ID: {pair['record_2']['id']}): '{pair['record_2']['company']}'", 
            f"    Domain: '{pair['record_2']['domain']}', Phone: '{pair['record_2']['phone']}'",
            ""
        ])
    
    prompt_lines.extend([
        "Return a JSON array with this exact format:",
        '[{"pair_id": "id1-id2", "same_organization": true, "confidence": 0.95, "primary_record_id": "id1", "canonical_name": "Canonical Company Name"}]',
        "",
        "Requirements:",
        "- Return valid JSON only, no explanations",
        "- Include ALL pairs in your response", 
        "- Use exact pair_id values provided",
        "- confidence must be 0.0-1.0",
        "- same_organization must be true or false",
        "- primary_record_id must be one of the two record IDs",
        "- canonical_name should be the best/most complete company name"
    ])
    
    prompt_text = "\n".join(prompt_lines)
    
    call_start = time.time()
    try:
        _check_openai()
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.1,
            max_tokens=2000
        )
        
        call_duration = time.time() - call_start
        _update_api_stats(api_stats, call_duration, response)
        
        answer = response.choices[0].message.content.strip()
        
        # Clean up potential markdown formatting
        if answer.startswith('```json'):
            answer = answer[7:]
        if answer.endswith('```'):
            answer = answer[:-3]
        answer = answer.strip()
        
        try:
            analysis_results = json.loads(answer)
            if not isinstance(analysis_results, list):
                analysis_results = []
        except json.JSONDecodeError:
            analysis_results = []
        
        return analysis_results
        
    except Exception as e:
        call_duration = time.time() - call_start
        _update_api_stats(api_stats, call_duration, None, str(type(e).__name__))
        
        # Return error results for this batch
        return [
            {
                "pair_id": pair['pair_id'],
                "same_organization": False,
                "confidence": 0.0,
                "primary_record_id": pair['record_1']['id'],
                "canonical_name": pair['record_1']['company'],
                "error": True
            }
            for pair in batch_data
        ]


def create_unique_records(
    features_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    ai_results: List[Dict[str, Any]],
    confidence_threshold: float = 0.7
) -> pd.DataFrame:
    """Create unique records by merging duplicates based on AI decisions."""
    
    # Create a mapping of duplicates
    duplicate_groups: Dict[str, Set[str]] = {}
    canonical_names: Dict[str, str] = {}
    
    # Process AI results to build duplicate groups
    for result in ai_results:
        if (result.get('same_organization', False) and 
            result.get('confidence', 0) >= confidence_threshold and
            not result.get('error', False)):
            
            pair_id = result['pair_id']
            # Handle pair_id parsing - assume format is "id1-id2"
            if '-' in pair_id:
                # Split on the first hyphen only
                id1, id2 = pair_id.split('-', 1)
            else:
                print(f"Warning: Malformed pair_id: {pair_id}")
                continue
            
            primary_id = result.get('primary_record_id', id1)
            canonical_name = result.get('canonical_name', '')
            
            # Find existing group or create new one
            group_key = None
            for key, group in duplicate_groups.items():
                if id1 in group or id2 in group:
                    group_key = key
                    break
            
            if group_key is None:
                # Create new group with primary record as key
                group_key = primary_id
                duplicate_groups[group_key] = {id1, id2}
            else:
                # Add to existing group
                duplicate_groups[group_key].update({id1, id2})
                # Update group key to primary record if needed
                if primary_id in duplicate_groups[group_key] and group_key != primary_id:
                    duplicate_groups[primary_id] = duplicate_groups.pop(group_key)
                    group_key = primary_id
            
            canonical_names[group_key] = canonical_name
    
    # Create unique records dataframe
    unique_records = []
    processed_ids = set()
    
    # Process duplicate groups
    for primary_id, group_ids in duplicate_groups.items():
        if primary_id in processed_ids:
            continue
            
        # Get primary record
        if primary_id in cleaned_df.index:
            primary_record = cleaned_df.loc[primary_id].copy()
            primary_record['record_id'] = primary_id
            primary_record['is_merged'] = len(group_ids) > 1
            primary_record['merged_from'] = list(group_ids) if len(group_ids) > 1 else []
            primary_record['canonical_company'] = canonical_names.get(primary_id, primary_record.get('company_clean', ''))
            
            unique_records.append(primary_record)
            processed_ids.update(group_ids)
    
    # Add records that weren't identified as duplicates
    for record_id in cleaned_df.index:
        if record_id not in processed_ids:
            record = cleaned_df.loc[record_id].copy()
            record['record_id'] = record_id
            record['is_merged'] = False
            record['merged_from'] = []
            record['canonical_company'] = record.get('company_clean', '')
            unique_records.append(record)
    
    return pd.DataFrame(unique_records)


def main(
    features_path: str = "data/outputs/features.csv",
    cleaned_path: str = "data/outputs/cleaned.csv",
    output_path: str = "data/outputs/unique_records.csv",
    analysis_path: str = "data/outputs/deduplication_analysis.json",
    similarity_threshold: float = 0.6,
    confidence_threshold: float = 0.7,
    sample_size: int | None = None,
    batch_size: int = 10,
    max_workers: int = 10,
    model: str = DEFAULT_MODEL,
    log_path: str = LOG_PATH,
) -> None:
    """Perform AI-powered deduplication using similarity features."""
    
    print("🤖 Starting AI-powered deduplication...")
    start_time = time.time()
    
    # Load data
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned data file not found: {cleaned_path}")
    
    features_df = pd.read_csv(features_path)
    cleaned_df = pd.read_csv(cleaned_path).set_index('record_id')
    
    print(f"📊 Data Overview:")
    print(f"  • Total feature pairs: {len(features_df):,}")
    print(f"  • Unique records: {len(cleaned_df):,}")
    print(f"  • Similarity threshold: ≥{similarity_threshold}")
    print(f"  • Confidence threshold: ≥{confidence_threshold}")
    
    # Filter pairs by similarity threshold
    high_sim_pairs = features_df[features_df['company_sim'] >= similarity_threshold].copy()
    
    print(f"  • High similarity pairs: {len(high_sim_pairs):,}")
    
    if len(high_sim_pairs) == 0:
        print(f"⚠️  No pairs above similarity threshold {similarity_threshold}")
        print(f"   Consider lowering the threshold or checking similarity calculation")
        return
    
    # Sample if requested
    if sample_size and len(high_sim_pairs) > sample_size:
        high_sim_pairs = high_sim_pairs.sample(n=sample_size, random_state=42)
        print(f"  • Sampled for analysis: {len(high_sim_pairs):,}")
    
    # Prepare analysis data
    print(f"🔍 Preparing pairs for AI analysis...")
    analysis_data = []
    
    for _, row in high_sim_pairs.iterrows():
        id1, id2 = row['record_id_1'], row['record_id_2']
        
        pair_info = {
            'pair_id': f"{id1}-{id2}",
            'similarity_score': row['company_sim'],
            'record_1': {
                'id': id1,
                'company': row.get('company_clean_1', ''),
                'domain': row.get('domain_clean_1', ''),
                'phone': row.get('phone_clean_1', '')
            },
            'record_2': {
                'id': id2,
                'company': row.get('company_clean_2', ''),
                'domain': row.get('domain_clean_2', ''),
                'phone': row.get('phone_clean_2', '')
            }
        }
        analysis_data.append(pair_info)
    
    # Split into batches for parallel processing
    batches = [analysis_data[i:i + batch_size] for i in range(0, len(analysis_data), batch_size)]
    
    # Process batches with AI
    all_results = []
    api_stats = _init_api_stats()
    
    print(f"🚀 Analyzing {len(analysis_data):,} pairs in {len(batches):,} batches...")
    
    with tqdm(total=len(batches), desc="AI Analysis", file=sys.stdout) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(analyze_duplicate_batch, batch, model, api_stats): batch 
                for batch in batches
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    # Handle batch failure
                    failed_batch = future_to_batch[future]
                    for pair in failed_batch:
                        all_results.append({
                            "pair_id": pair['pair_id'],
                            "same_organization": False,
                            "confidence": 0.0,
                            "primary_record_id": pair['record_1']['id'],
                            "canonical_name": pair['record_1']['company'],
                            "error": True
                        })
                pbar.update(1)
    
    processing_time = time.time() - start_time
    
    # Create unique records
    print(f"🔧 Creating unique records...")
    unique_df = create_unique_records(features_df, cleaned_df, all_results, confidence_threshold)
    
    # Calculate statistics
    total_pairs = len(all_results)
    duplicates_found = sum(1 for r in all_results if r.get('same_organization', False) and not r.get('error', False))
    high_confidence = sum(1 for r in all_results if r.get('confidence', 0) >= confidence_threshold and not r.get('error', False))
    error_count = sum(1 for r in all_results if r.get('error', False))
    
    merged_records = sum(1 for _, row in unique_df.iterrows() if row.get('is_merged', False))
    total_unique = len(unique_df)
    deduplication_rate = 1 - (total_unique / len(cleaned_df))
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
    
    unique_df.to_csv(output_path, index=False)
    
    # Create comprehensive analysis results
    analysis_result = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "similarity_threshold": similarity_threshold,
            "confidence_threshold": confidence_threshold,
            "processing_time_seconds": round(processing_time, 2),
            "sample_size": len(analysis_data)
        },
        "input_stats": {
            "total_records": len(cleaned_df),
            "total_feature_pairs": len(features_df),
            "high_similarity_pairs": len(high_sim_pairs),
            "analyzed_pairs": total_pairs
        },
        "ai_analysis": {
            "duplicates_identified": duplicates_found,
            "high_confidence_decisions": high_confidence,
            "error_count": error_count,
            "duplicate_rate": duplicates_found / total_pairs if total_pairs > 0 else 0
        },
        "deduplication_results": {
            "original_records": len(cleaned_df),
            "unique_records": total_unique,
            "merged_records": merged_records,
            "deduplication_rate": deduplication_rate,
            "records_eliminated": len(cleaned_df) - total_unique
        },
        "api_stats": api_stats,
        "detailed_results": all_results
    }
    
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2)
    
    # Print comprehensive results
    print(f"\n🎯 AI Deduplication Results:")
    print(f"  • Original records: {len(cleaned_df):,}")
    print(f"  • Unique records: {total_unique:,}")
    print(f"  • Records eliminated: {len(cleaned_df) - total_unique:,}")
    print(f"  • Deduplication rate: {deduplication_rate:.1%}")
    print(f"  • Merged record groups: {merged_records:,}")
    
    print(f"\n🤖 AI Analysis Stats:")
    print(f"  • Pairs analyzed: {total_pairs:,}")
    print(f"  • Duplicates found: {duplicates_found:,} ({duplicates_found/total_pairs:.1%})")
    print(f"  • High confidence: {high_confidence:,}")
    print(f"  • Processing errors: {error_count:,}")
    
    if api_stats["calls"] > 0:
        print(f"\n⚡ API Performance:")
        print(f"  • API calls: {api_stats['calls']:,}")
        print(f"  • Total tokens: {api_stats['tokens']['total']:,}")
        print(f"  • Estimated cost: ${api_stats['costs']['total']:.3f}")
        print(f"  • Avg call duration: {sum(api_stats['durations'])/len(api_stats['durations']):.2f}s")
        print(f"  • Processing speed: {total_pairs/processing_time:.1f} pairs/second")
    
    print(f"\n💾 Files Created:")
    print(f"  • Unique records: {output_path}")
    print(f"  • Analysis details: {analysis_path}")
    
    print(f"\n🚀 Next Steps:")
    if deduplication_rate > 0.1:
        print(f"   ✅ Significant deduplication achieved!")
        print(f"   Review merged records for accuracy")
    else:
        print(f"   ⚠️  Low deduplication rate")
        print(f"   Consider lowering similarity threshold")
    
    print(f"\n💡 Tips:")
    print(f"   • Review 'canonical_company' field for merged names")
    print(f"   • Check 'merged_from' field to see original IDs")
    print(f"   • Adjust confidence threshold for stricter/looser merging")
    
    end_time = time.time()
    log_run("ai_deduplication", start_time, end_time, total_unique, log_path=log_path)


@click.command()
@click.option("--features-path", default="data/outputs/features.csv", show_default=True,
              help="Path to the features CSV file with similarity scores")
@click.option("--cleaned-path", default="data/outputs/cleaned.csv", show_default=True,
              help="Path to the cleaned records CSV file")
@click.option("--output-path", default="data/outputs/unique_records.csv", show_default=True,
              help="Path to save the unique/deduplicated records")
@click.option("--analysis-path", default="data/outputs/deduplication_analysis.json", show_default=True,
              help="Path to save detailed analysis results")
@click.option("--similarity-threshold", default=0.6, show_default=True, type=float,
              help="Minimum similarity score to consider for AI analysis")
@click.option("--confidence-threshold", default=0.7, show_default=True, type=float,
              help="Minimum AI confidence to merge records")
@click.option("--sample-size", default=None, type=int,
              help="Limit analysis to this many pairs (for testing)")
@click.option("--batch-size", default=10, show_default=True,
              help="Number of pairs per batch for parallel processing")
@click.option("--max-workers", default=10, show_default=True,
              help="Number of parallel workers for API calls")
@click.option("--model", default=DEFAULT_MODEL, show_default=True,
              help="OpenAI model to use for analysis")
@click.option("--log-path", default=LOG_PATH, show_default=True)
def cli(
    features_path: str,
    cleaned_path: str,
    output_path: str,
    analysis_path: str,
    similarity_threshold: float,
    confidence_threshold: float,
    sample_size: int | None,
    batch_size: int,
    max_workers: int,
    model: str,
    log_path: str,
) -> None:
    """AI-powered deduplication using similarity features."""
    
    main(features_path, cleaned_path, output_path, analysis_path, 
         similarity_threshold, confidence_threshold, sample_size,
         batch_size, max_workers, model, log_path)


if __name__ == "__main__":  # pragma: no cover
    print("\u23e9 started AI deduplication")
    cli()
