<!-- Banner -->
<div align="center">
  <img width="1196" height="106" alt="BWHBanner" src="https://github.com/user-attachments/assets/09314e3e-8271-49c0-8067-4d4aee574304" />

  <h3>Comprehensive temperature analysis tools and results for the Bristol and Shoesmith well field.</h3>
  
</div>

## 📂 Directory Structure

<details>
<summary>📁 Main Directory</summary>

- `LICENSE` — License file  
- `README.md` — Comprehensive documentation

</details>

---

<details>
<summary>📊 data_files/ — Core datasets and maps</summary>

- `Bristol.xlsx` — Main temperature dataset (10,569 records)  
- `wells_primary_coordinates.csv` — Coordinates for 53 wells  
- `wells_extended_coordinates.csv` — Extended coordinates  
- `cluster2_site_map.png` — Site map for cluster 2  
- `site_map_cropped.png` — Cropped site map

</details>

---

<details>
<summary>🔬 analysis_programs/ — Analysis scripts & outputs</summary>

<details>
<summary>📈 well_correlation_statistical_analysis/</summary>

**Script:** `comprehensive_well_correlation_analysis.py`  
**Purpose:** Correlation analysis of all well pairs

**Outputs:**
- `well_correlations_r06_p005_n800.csv` — 140 correlations (r > 0.6, p < 0.05, n > 800)  
- `comprehensive_correlation_report.txt` — Detailed statistics

**Features:**
- Cross-correlation (lag: -10 to +10 days)  
- Permutation testing for r > 0.9 (1,000 shuffles)  
- Statistical significance tests (p < 0.05)  
- Large sample filtering (n > 800)  
- Mean r = 0.816

</details>

<details>
<summary>🔄 temperature_flow_pattern_analysis/</summary>

**Script:** `temperature_flow_pattern_analyzer.py`  
**Purpose:** Analyze propagation and flow patterns

**Outputs:**
- `temperature_propagation_data.csv` — Propagation data  
- `temperature_propagation_report.txt` — Summary  
- `temperature_propagation_analysis.png` — Flow visualization  
- `temporal_cross_correlation_results.csv` — Temporal correlation data  
- `temporal_cross_correlation_analysis.png` — Plots

**Features:**
- Temporal cross-correlation  
- Lag detection & flow pattern mapping  
- Clockwise vs counterclockwise detection

</details>

<details>
<summary>🎯 directional_thermal_influence_mapping/</summary>

**Script:** `thermal_influence_mapper.py`  
**Purpose:** Map directional thermal influences

**Outputs:**
- `directional_influences.csv` — Validated correlations  
- `directional_temperature_influence_map.png` — Visual network map  
- `enhanced_temperature_analysis_report.txt` — Validation report

**Features:**
- Directional mapping with p-values  
- Visual overlay on site map  
- High thresholds (r > 0.9, p < 0.05)  
- Permutation validation

</details>

<details>
<summary>🎬 animated_temperature_visualization/</summary>

**Script:** `temperature_animation_generator.py`  
**Purpose:** Create animated temperature visualizations

**Outputs:**
- `well_temperatures_animation.gif` — Animated heatmap (GIF)  
- `well_temperatures_animation.mp4` — Animated heatmap (MP4)  
- `interactive_well_labeling_tool.html` — Interactive labeling

**Features:**
- Time-lapse temperature visualization  
- Consistent coloring  
- All wells visible in each frame

</details>

</details>

---

<details>
<summary>📊 cross_program_statistical_validation/ — Cross-checking between programs</summary>

- `permutation_test_results.csv` — Validation data  
- `statistical_validation_results.csv` — Cross-program validation  
- `final_statistical_validation_report.txt` — Summary  
- `statistical_validation_permutation_tests.png` — Plots

**Features:**
- 1,000 shuffle permutation testing  
- Empirical p-value computation  
- Non-parametric significance tests

</details>

---

## 📈 Analysis Summary

<details>
<summary>View analysis results</summary>

### Current Results (r > 0.6)
- 140 significant correlations  
- Mean r = 0.816  
- Mean n = 1,015  
- 43.6% simultaneous correlations (lag = 0 days)  
- All p < 0.05

### Top 5 Correlations
1. GW-1 → GW-5: r = 0.963, n = 1072  
2. GW-1 → GW-12: r = 0.960 (lag: -1d), n = 1071  
3. GW-12 → GW-6: r = 0.958, n = 1072  
4. GW-12 → GW-13: r = 0.958, n = 1072  
5. GW-12 → GW-5: r = 0.957, n = 1072  

</details>

---

## 🧪 Methodology

<details>
<summary>View methodology details</summary>

- **Data:** `Bristol.xlsx` (10,569 records)  
- **Time period:** Daily resampling, interpolated  
- **Stats:** Pearson correlation + permutation validation  
- **Significance:** p < 0.05  
- **Sample threshold:** n > 800  
- **Lag:** -10 to +10 days

</details>

---

## 🏷️ Naming Conventions

<details>
<summary>View naming rules & examples</summary>

**Folders**
- Descriptive, domain-specific (e.g., `thermal_influence`, `flow_pattern`)

**Files**
- Self-descriptive names  
- Parameters in output filenames  
- Clear distinction between primary and extended datasets

| Old Name | New Name | Why Better |
|----------|----------|------------|
| `everyCor_fixed.py` | `comprehensive_well_correlation_analysis.py` | Clarifies purpose |
| `wells (1).csv` | `wells_extended_coordinates.csv` | Describes content |
| `all_well_correlations.csv` | `well_correlations_r06_p005_n800.csv` | Includes criteria |
| `create_influence_map/` | `directional_thermal_influence_mapping/` | Domain-specific |
| `shared_outputs/` | `cross_program_statistical_validation/` | Clear purpose |

</details>

---

## ▶️ Usage

<details>
<summary>View step-by-step instructions</summary>

1. **Well Correlation Analysis**  
   `python analysis_programs/well_correlation_statistical_analysis/comprehensive_well_correlation_analysis.py`

2. **Flow Patterns**  
   `python analysis_programs/temperature_flow_pattern_analysis/temperature_flow_pattern_analyzer.py`

3. **Thermal Influence Mapping**  
   `python analysis_programs/directional_thermal_influence_mapping/thermal_influence_mapper.py`

4. **Animation Generation**  
   `python analysis_programs/animated_temperature_visualization/temperature_animation_generator.py`

5. Adjust thresholds in scripts (default r > 0.6)  
6. Review outputs in CSV/TXT/PNG/GIF/MP4 formats

</details>

---

## 📋 Requirements

<details>
<summary>View requirements</summary>

- Python 3.x  
- Libraries: `pandas`, `numpy`, `scipy`, `matplotlib`  
- Data: `data_files/Bristol.xlsx`, coordinate files, site maps

</details>

---

_Last updated: 2025-08-05_
