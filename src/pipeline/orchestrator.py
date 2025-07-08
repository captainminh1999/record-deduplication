"""
Pipeline orchestrator for running the complete deduplication pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..core.preprocess_engine import PreprocessEngine, PreprocessConfig
from ..core.blocking_engine import BlockingEngine, BlockingConfig  
from ..core.similarity_engine import SimilarityEngine, SimilarityConfig
from ..core.model_engine import ModelEngine, ModelConfig
from ..core.clustering_engine import ClusteringEngine, ClusteringConfig
from ..core.reporting_engine import ReportingEngine, ReportingConfig
from ..core.openai_engine import OpenAIEngine, OpenAIConfig
from ..logging import log_run


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline."""
    
    # Input/Output paths
    input_path: str = "data/sample_input.csv"
    output_dir: str = "data/outputs"
    
    # Pipeline steps to run (all enabled by default)
    run_preprocess: bool = True
    run_blocking: bool = True
    run_similarity: bool = True
    run_model: bool = True
    run_clustering: bool = True
    run_reporting: bool = True
    run_openai_dedup: bool = False  # Optional AI step
    
    # Step-specific configs
    preprocess: PreprocessConfig = None
    blocking: BlockingConfig = None
    similarity: SimilarityConfig = None
    model: ModelConfig = None
    clustering: ClusteringConfig = None
    reporting: ReportingConfig = None
    openai: OpenAIConfig = None
    
    def __post_init__(self):
        """Initialize default configs if not provided."""
        if self.preprocess is None:
            self.preprocess = PreprocessConfig()
        if self.blocking is None:
            self.blocking = BlockingConfig()
        if self.similarity is None:
            self.similarity = SimilarityConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.clustering is None:
            self.clustering = ClusteringConfig()
        if self.reporting is None:
            self.reporting = ReportingConfig()
        if self.openai is None:
            self.openai = OpenAIConfig()


@dataclass
class PipelineResult:
    """Results from pipeline execution."""
    
    success: bool
    total_time: float
    step_results: Dict[str, Any]
    step_times: Dict[str, float]
    output_files: Dict[str, str]
    error_message: Optional[str] = None


class PipelineOrchestrator:
    """Orchestrates the complete deduplication pipeline."""
    
    def __init__(self):
        self.preprocess_engine = PreprocessEngine()
        self.blocking_engine = BlockingEngine()
        self.similarity_engine = SimilarityEngine()
        self.model_engine = ModelEngine()
        self.clustering_engine = ClusteringEngine()
        self.reporting_engine = ReportingEngine()
        self.openai_engine = OpenAIEngine()
    
    def run_pipeline(
        self, 
        config: PipelineConfig,
        progress_callback: Optional[callable] = None
    ) -> PipelineResult:
        """
        Run the complete deduplication pipeline.
        
        Args:
            config: Pipeline configuration
            progress_callback: Optional callback for progress updates
        
        Returns:
            PipelineResult with execution details
        """
        start_time = time.time()
        step_results = {}
        step_times = {}
        output_files = {}
        
        try:
            # Define pipeline steps
            steps = [
                ("preprocess", config.run_preprocess, self._run_preprocess),
                ("blocking", config.run_blocking, self._run_blocking),
                ("similarity", config.run_similarity, self._run_similarity),
                ("model", config.run_model, self._run_model),
                ("clustering", config.run_clustering, self._run_clustering),
                ("reporting", config.run_reporting, self._run_reporting),
                ("openai_dedup", config.run_openai_dedup, self._run_openai_dedup),
            ]
            
            total_steps = sum(1 for _, enabled, _ in steps if enabled)
            completed_steps = 0
            
            for step_name, enabled, step_func in steps:
                if not enabled:
                    continue
                
                if progress_callback:
                    progress_callback(completed_steps, total_steps, step_name)
                
                # Run step
                step_start = time.time()
                result = step_func(config)
                step_end = time.time()
                
                # Store results
                step_results[step_name] = result
                step_times[step_name] = step_end - step_start
                
                # Track output files
                if hasattr(result, 'output_path'):
                    output_files[step_name] = result.output_path
                
                completed_steps += 1
            
            total_time = time.time() - start_time
            
            return PipelineResult(
                success=True,
                total_time=total_time,
                step_results=step_results,
                step_times=step_times,
                output_files=output_files
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            return PipelineResult(
                success=False,
                total_time=total_time,
                step_results=step_results,
                step_times=step_times,
                output_files=output_files,
                error_message=str(e)
            )
    
    def _run_preprocess(self, config: PipelineConfig) -> Any:
        """Run preprocessing step."""
        start_time = time.time()
        result = self.preprocess_engine.preprocess(
            input_path=config.input_path,
            config=config.preprocess
        )
        end_time = time.time()
        
        log_run("preprocess", start_time, end_time, len(result.cleaned_records))
        return result
    
    def _run_blocking(self, config: PipelineConfig) -> Any:
        """Run blocking step."""
        start_time = time.time()
        
        # Load cleaned data
        import pandas as pd
        cleaned_path = Path(config.output_dir) / "cleaned.csv"
        records_df = pd.read_csv(cleaned_path)
        
        result = self.blocking_engine.generate_pairs(records_df, config.blocking)
        end_time = time.time()
        
        log_run("blocking", start_time, end_time, len(result.pairs))
        return result
    
    def _run_similarity(self, config: PipelineConfig) -> Any:
        """Run similarity step."""
        start_time = time.time()
        
        # Load data
        import pandas as pd
        cleaned_path = Path(config.output_dir) / "cleaned.csv"
        pairs_path = Path(config.output_dir) / "pairs.csv"
        
        records_df = pd.read_csv(cleaned_path)
        pairs_df = pd.read_csv(pairs_path)
        
        result = self.similarity_engine.compute_features(
            records_df, pairs_df, config.similarity
        )
        end_time = time.time()
        
        log_run("similarity", start_time, end_time, len(result.features))
        return result
    
    def _run_model(self, config: PipelineConfig) -> Any:
        """Run model training step."""
        start_time = time.time()
        
        # Load features
        import pandas as pd
        features_path = Path(config.output_dir) / "features.csv"
        features_df = pd.read_csv(features_path)
        
        result = self.model_engine.train_and_predict(features_df, config.model)
        end_time = time.time()
        
        log_run("model", start_time, end_time, len(features_df))
        return result
    
    def _run_clustering(self, config: PipelineConfig) -> Any:
        """Run clustering step."""
        start_time = time.time()
        
        # Load data
        import pandas as pd
        features_path = Path(config.output_dir) / "features.csv"
        features_df = pd.read_csv(features_path)
        
        result = self.clustering_engine.cluster_records(features_df, config.clustering)
        end_time = time.time()
        
        log_run("clustering", start_time, end_time, len(result.clusters))
        return result
    
    def _run_reporting(self, config: PipelineConfig) -> Any:
        """Run reporting step."""
        start_time = time.time()
        
        # Load data
        import pandas as pd
        cleaned_path = Path(config.output_dir) / "cleaned.csv"
        features_path = Path(config.output_dir) / "features.csv"
        
        cleaned_df = pd.read_csv(cleaned_path)
        features_df = pd.read_csv(features_path)
        
        result = self.reporting_engine.generate_report(
            cleaned_df, features_df, config.reporting
        )
        end_time = time.time()
        
        log_run("reporting", start_time, end_time, len(cleaned_df))
        return result
    
    def _run_openai_dedup(self, config: PipelineConfig) -> Any:
        """Run OpenAI deduplication step."""
        start_time = time.time()
        
        # Load data
        import pandas as pd
        features_path = Path(config.output_dir) / "features.csv"
        cleaned_path = Path(config.output_dir) / "cleaned.csv"
        
        features_df = pd.read_csv(features_path)
        cleaned_df = pd.read_csv(cleaned_path).set_index('record_id')
        
        result = self.openai_engine.deduplicate_records(
            features_df, cleaned_df, config.openai
        )
        end_time = time.time()
        
        log_run("openai_deduplication", start_time, end_time, len(result.unique_records))
        return result
