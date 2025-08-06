# Bristol and Shoesmith HOV Temperature Analysis

This folder contains comprehensive temperature analysis tools and results for the Bristol and Shoesmith well field.

## 🗂️ Clear and Descriptive Organization

### 📁 Main Directory
- `LICENSE` - License file
- `README.md` - This comprehensive documentation file

### 📊 data_files/
**Core Data Files with Descriptive Names:**
- `Bristol.xlsx` - Main temperature dataset from Excel (10,569 temperature records)
- `wells_primary_coordinates.csv` - Primary well coordinate data (53 wells)
- `wells_extended_coordinates.csv` - Extended well coordinate data with broader coverage
- `cluster2_site_map.png` - Site map image for cluster 2 analysis
- `site_map_cropped.png` - Cropped version of the main site map

### 🔬 analysis_programs/
**Individual Program Folders - Each containing one Python script and its specific outputs:**

#### 📈 well_correlation_statistical_analysis/
**Program:** `comprehensive_well_correlation_analysis.py`  
**Purpose:** Statistical analysis of correlations between all well pairs

**Outputs:**
- `well_correlations_r06_p005_n800.csv` - Complete correlation data (140 correlations: r > 0.6, p < 0.05, n > 800)
- `comprehensive_correlation_report.txt` - Detailed statistical analysis report

**Features:**
- Cross-correlation analysis with lag detection (-10 to +10 days)
- Permutation testing for correlations > 0.9 (1000 shuffles)
- Statistical significance testing (p < 0.05)
- Large sample size requirements (n > 800)
- Mean correlation strength: r = 0.816

#### 🔄 temperature_flow_pattern_analysis/
**Program:** `temperature_flow_pattern_analyzer.py`  
**Purpose:** Analysis of temperature propagation patterns and temporal flow relationships

**Outputs:**
- `temperature_propagation_data.csv` - Propagation correlation data
- `temperature_propagation_report.txt` - Analysis summary report
- `temperature_propagation_analysis.png` - Flow pattern visualization
- `temporal_cross_correlation_results.csv` - Cross-correlation analysis results
- `temporal_cross_correlation_analysis.png` - Temporal relationship plots

**Features:**
- Temporal cross-correlation analysis
- Lag detection and propagation patterns
- Clockwise vs counterclockwise flow detection
- Time-series correlation mapping

#### 🎯 directional_thermal_influence_mapping/
**Program:** `thermal_influence_mapper.py`  
**Purpose:** Creation of directional thermal influence maps with statistical validation

**Outputs:**
- `directional_influences.csv` - Statistically validated directional correlations
- `directional_temperature_influence_map.png` - Visual influence network map overlay
- `enhanced_temperature_analysis_report.txt` - Enhanced statistical validation report

**Features:**
- Directional correlation mapping on site map
- Statistical validation with p-values and sample sizes
- Visual influence network overlay
- High correlation thresholds (r > 0.9, p < 0.05)
- Permutation testing validation

#### 🎬 animated_temperature_visualization/
**Program:** `temperature_animation_generator.py`  
**Purpose:** Generation of animated temperature visualizations over time

**Outputs:**
- `well_temperatures_animation.gif` - Animated temperature heatmap (GIF format)
- `well_temperatures_animation.mp4` - Temperature animation (MP4 video format)
- `interactive_well_labeling_tool.html` - Interactive web-based well labeling tool

**Features:**
- Animated temperature evolution over time
- Persistent temperature coloring system
- All wells visible on every frame
- Interactive web-based labeling tool
- Time-series temperature visualization

### 📊 cross_program_statistical_validation/
**Cross-Program Statistical Validation Results:**
- `permutation_test_results.csv` - Permutation test validation data
- `statistical_validation_results.csv` - Cross-program validation results
- `final_statistical_validation_report.txt` - Comprehensive statistical summary
- `statistical_validation_permutation_tests.png` - Statistical validation plots

**Features:**
- Permutation testing (1000 shuffles)
- Empirical p-value calculations
- Non-parametric significance testing
- Cross-validation across programs

## Analysis Summary

### Current Results (r > 0.6 threshold)
- **140 significant correlations** found
- **Mean correlation strength:** r = 0.816
- **Mean sample size:** n = 1,015 data points
- **43.6% simultaneous correlations** (lag = 0 days)
- **All correlations validated** with p < 0.05

### Strongest Correlations
1. GW-1 → GW-5: r=0.963, n=1072
2. GW-1 → GW-12: r=0.960 (lag: -1 days), n=1071
3. GW-12 → GW-6: r=0.958, n=1072
4. GW-12 → GW-13: r=0.958, n=1072
5. GW-12 → GW-5: r=0.957, n=1072

### Methodology
- **Data Source:** 10,569 temperature records from Bristol.xlsx
- **Time Period:** Daily resampling with interpolation
- **Statistical Tests:** Pearson correlation with permutation validation
- **Significance Level:** p < 0.05
- **Sample Size Threshold:** n > 800
- **Lag Detection:** -10 to +10 days

## 🏷️ Naming Conventions

### Folder Names
- **Descriptive Purpose**: Each folder name clearly describes what the program does
- **No Generic Prefixes**: Avoided ambiguous terms like "analyze_", "create_", "generate_"
- **Specific Domain Terms**: Used specific terms like "thermal_influence", "flow_pattern", "statistical_analysis"

### File Names
- **Self-Describing**: File names indicate content and parameters
- **Python Scripts**: Named for their specific purpose (e.g., `thermal_influence_mapper.py`)
- **Output Files**: Include key parameters in filename (e.g., `well_correlations_r06_p005_n800.csv`)
- **Data Files**: Specify primary vs extended datasets clearly

### Examples of Improved Names:
| Old Name | New Name | Improvement |
|----------|----------|-------------|
| `everyCor_fixed.py` | `comprehensive_well_correlation_analysis.py` | Clear purpose |
| `wells (1).csv` | `wells_extended_coordinates.csv` | Descriptive content |
| `all_well_correlations.csv` | `well_correlations_r06_p005_n800.csv` | Includes criteria |
| `create_influence_map/` | `directional_thermal_influence_mapping/` | Specific domain |
| `shared_outputs/` | `cross_program_statistical_validation/` | Clear purpose |

## Usage Instructions

1. **Well Correlation Analysis:** Execute `analysis_programs/well_correlation_statistical_analysis/comprehensive_well_correlation_analysis.py`
2. **Temperature Flow Patterns:** Run `analysis_programs/temperature_flow_pattern_analysis/temperature_flow_pattern_analyzer.py`
3. **Thermal Influence Mapping:** Execute `analysis_programs/directional_thermal_influence_mapping/thermal_influence_mapper.py`
4. **Animation Generation:** Run `analysis_programs/animated_temperature_visualization/temperature_animation_generator.py`
5. **Adjust Thresholds:** Modify correlation thresholds (currently r > 0.6) in the respective scripts
6. **View Results:** Check CSV files for data, TXT files for reports, PNG/GIF/MP4 files for visualizations

## Requirements
- Python 3.x with pandas, numpy, scipy, matplotlib
- Source data: `data_files/Bristol.xlsx` (temperature data)
- Coordinate data: `data_files/wells_primary_coordinates.csv` or `wells_extended_coordinates.csv`
- Site maps: `data_files/cluster2_site_map.png` or `site_map_cropped.png`

Last updated: 2025-08-05