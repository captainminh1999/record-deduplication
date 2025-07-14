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

### Modular Subdivision Engine V2 (`subdivision_engine_v2.py`)
- **Architecture**: Strategy Pattern implementation
- **Purpose**: Clean, maintainable subdivision with specialized strategies
- **Key Features**:
  - Progressive strategy fallback system
  - Cluster-specific PCA optimization
  - Guaranteed subdivision success via ForceStrategy

#### Subdivision Strategies

**1. AdaptiveDBSCANStrategy**
- **Target**: Clusters ≥50 records
- **Approach**: 
  - Uses AdaptiveEpsCalculator for cluster-specific parameter optimization
  - Applies cluster-specific PCA transformation when beneficial
  - Progressive eps reduction for difficult clusters
- **Benefits**: Preserves natural cluster structure while enabling subdivision

**2. AggressivePCAStrategy**
- **Target**: Very large clusters ≥1000 records
- **Approach**:
  - Forces aggressive PCA reduction (18 → 2-3 dimensions)
  - Uses very small eps values for maximum separation
  - Designed for clusters resistant to normal subdivision
- **Benefits**: Handles extremely large, dense clusters

**3. KMeansStrategy**
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
