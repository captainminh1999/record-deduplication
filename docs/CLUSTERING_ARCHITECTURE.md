# Hierarchical Clustering Architecture

## Overview

The clustering system has been completely rewritten using a modular strategy pattern to handle large-scale record deduplication with intelligent subdivision strategies.

## Architecture Components

### Core Hierarchical Clusterer (`core_clusterer.py`)
- **Purpose**: Main orchestrator for hierarchical clustering
- **Responsibilities**:
  - Initial DBSCAN clustering
  - Identifying clusters that need subdivision
  - Managing clustering depth and size constraints
  - Coordinating with connectivity manager for high-similarity preservation

### Adaptive Clusterer V3 (`adaptive_clusterer_v3.py`) 🔧 **FIXED**
- **Purpose**: Advanced hierarchical clustering with domain-aware feature engineering
- **Recent Fix**: 
  - **Problem**: All perfect domain matches got identical similarity values (5000.0)
  - **Solution**: Unique incremental offsets preserve domain identity while maintaining priority
  - **Code**: `melted.loc[perfect_domain_mask, "domain_sim"] = 4999.0 + unique_offsets`
- **Benefits**: Enables proper domain-based subdivision without losing domain priority

### Modular Subdivision Engine V3 (`subdivision_engine_v3.py`) 🆕 **ENHANCED**
- **Architecture**: Strategy Pattern implementation with domain awareness
- **Purpose**: Clean, maintainable subdivision with specialized domain handling
- **Key Features**:
  - Progressive strategy fallback system
  - Smart detection of artificially boosted domain values
  - Domain-first clustering with 85% similarity threshold
  - Guaranteed subdivision success via multiple strategies

#### Subdivision Strategies

**1. Domain-First Clustering** 🆕
- **Target**: Any cluster with perfect domain matches (domain_sim ≥ 4999.0)
- **Approach**: 
  - Detects domain column with boosted/uniform values
  - Groups records by domain with 85% similarity threshold
  - Handles artificially boosted values vs. legitimate uniform clusters
- **Benefits**: Ensures each domain gets its own cluster while maintaining similarity grouping

**2. AdaptiveDBSCANStrategy**
- **Target**: Clusters ≥50 records
- **Approach**: 
  - Uses AdaptiveEpsCalculator for cluster-specific parameter optimization
  - Applies cluster-specific PCA transformation when beneficial
  - Progressive eps reduction for difficult clusters
- **Benefits**: Preserves natural cluster structure while enabling subdivision

**3. AggressivePCAStrategy**
- **Target**: Very large clusters ≥1000 records
- **Approach**:
  - Forces aggressive PCA reduction (18 → 2-3 dimensions)
  - Uses very small eps values for maximum separation
  - Designed for clusters resistant to normal subdivision
- **Benefits**: Handles extremely large, dense clusters

**4. KMeansStrategy**
- **Target**: Large clusters ≥500 records
- **Approach**:
  - Intelligent sampling for clusters >5000 records
  - PCA preprocessing for efficiency
  - Targets ~200 records per sub-cluster
  - Maximum 15 sub-clusters per subdivision
- **Benefits**: Always succeeds, handles any cluster size

**4. ForceStrategy**
- **Target**: Any cluster ≥50 records (last resort)
- **Approach**:
  - PCA + K-means combination
  - Random partitioning fallback
  - Targets ~50 records per partition
- **Benefits**: Guaranteed success, ensures no infinite loops

### Adaptive Eps Calculator (`adaptive_eps.py`)
- **Purpose**: Intelligent parameter calculation for cluster-specific optimization
- **Features**:
  - Cluster-specific PCA analysis
  - Variance-based dimensionality reduction
  - Nearest neighbor distance analysis
  - Automatic parameter scaling

### Connectivity Manager (`connectivity_manager.py`)
- **Purpose**: Preserve high-similarity connections during subdivision
- **Features**:
  - Similarity threshold-based connection detection
  - Intelligent connection preservation via eps adjustment
  - Violation detection and repair

## Noise Handling Strategy

### DBSCAN Noise Points
- **During Subdivision**: Noise points (-1) are NOT forced into artificial clusters
- **Preservation**: Noise points return to their original cluster
- **Iteration**: Get multiple chances across different levels and strategies
- **Natural Clustering**: Only points that can be naturally grouped are separated

### Progressive Refinement
1. **Level 1**: Large clusters subdivided with conservative strategies
2. **Level 2+**: Remaining large clusters get more aggressive treatment
3. **Force Strategy**: Last resort ensures size constraints are met
4. **Noise Handling**: Maintains data quality by avoiding artificial groupings

## Key Improvements

### From Monolithic to Modular
- **Before**: Single complex subdivision function with multiple fallback mechanisms
- **After**: Clean strategy pattern with single-responsibility classes
- **Benefits**: Easier testing, maintenance, and extension

### Cluster-Specific PCA
- **Innovation**: Each cluster gets its own PCA optimization
- **Implementation**: AdaptiveEpsCalculator analyzes cluster characteristics
- **Benefits**: Optimal dimensionality reduction per cluster context

### Guaranteed Success
- **Problem**: Previous system could fail to subdivide persistent large clusters
- **Solution**: ForceStrategy ensures every cluster can be subdivided
- **Safeguard**: Prevents infinite loops while meeting size constraints

## Configuration Parameters

### Clustering Settings
- `--max-cluster-size`: Maximum allowed cluster size (default: 15)
- `--max-depth`: Maximum subdivision depth (default: 3)
- `--eps`: Base epsilon parameter for DBSCAN (default: 0.4)

### Strategy Thresholds
- **AdaptiveDBSCAN**: ≥50 records
- **AggressivePCA**: ≥1000 records  
- **KMeans**: ≥500 records
- **Force**: ≥50 records (fallback)

## Performance Characteristics

### Scalability
- **Large Clusters**: Handled via sampling and PCA reduction
- **Memory Efficiency**: Progressive strategies avoid processing entire datasets
- **Speed**: Intelligent parameter calculation reduces trial-and-error

### Quality Preservation
- **Natural Structure**: DBSCAN strategies preserve cluster coherence
- **Noise Handling**: Avoids artificial cluster assignments
- **Connection Preservation**: High-similarity links maintained across subdivisions

## Usage Examples

### Basic Hierarchical Clustering
```bash
python -m src.cli.clustering --hierarchical --max-cluster-size 15 --max-depth 2 --eps 0.4
```

### Aggressive Subdivision
```bash
python -m src.cli.clustering --hierarchical --max-cluster-size 10 --max-depth 3 --eps 0.3
```

### Analysis of Strategy Usage
The system provides detailed statistics on which strategies were used for each subdivision, enabling analysis and optimization of the clustering approach.

## Future Extensions

The modular strategy pattern enables easy addition of new subdivision approaches:
- **GraphClustering**: Community detection algorithms
- **DensityBasedStrategy**: Adaptive density-based subdivision
- **HybridStrategy**: Combination approaches for specific cluster types

This architecture provides a robust foundation for handling diverse clustering challenges while maintaining code quality and extensibility.

## 🆕 Recent Critical Updates (July 2025)

### Domain Clustering Fix in Adaptive Clusterer V3
**Critical Issue Resolved**: Fixed a fundamental bug where all perfect domain matches were assigned identical similarity values, preventing proper domain-based clustering.

**Technical Details**:
- **Problem**: Line 337 in `adaptive_clusterer_v3.py` set all perfect domain matches to 5000.0
- **Impact**: Made different domains indistinguishable, causing massive mixed-domain clusters
- **Solution**: Implemented unique incremental offsets while preserving domain priority
- **Result**: Enables proper domain subdivision while maintaining ultra-high domain priority

### Enhanced Subdivision Engine V3
- **Smart Detection**: Identifies artificially boosted vs. legitimate uniform domain clusters
- **Subdivision Control**: Allows subdivision of boosted values while preserving real uniform clusters
- **Domain Awareness**: Specialized handling for domain-based clustering scenarios

### Complete Domain Clustering Pipeline
- **Integrated Workflow**: End-to-end domain clustering with rescue capabilities
- **Analysis Tools**: Comprehensive validation and debugging utilities in `src/scripts/`
- **Production Ready**: Robust error handling and performance optimization
