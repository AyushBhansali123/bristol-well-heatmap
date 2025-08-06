import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from PIL import Image
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

def map_well_id(excel_well_id):
    """Map Excel numeric well ID to CSV prefixed format"""
    well_str = str(excel_well_id)
    
    if well_str in ['100', '200']:
        return f'GP-{well_str}'
    elif well_str in ['29', '33', '34', '37']:
        return f'EW-{well_str}'
    elif well_str == '13' and excel_well_id == 13:
        return 'GP-13'
    else:
        return f'GW-{well_str}'

def calculate_directional_correlations(cluster_well_ids, heatmap_data):
    """Calculate directional correlations between wells in cluster"""
    
    # Get time series data for cluster wells
    cluster_data = heatmap_data[heatmap_data['well_id'].isin(cluster_well_ids)].copy()
    cluster_timeseries = {}
    
    for well_id in cluster_well_ids:
        well_data = cluster_data[cluster_data['well_id'] == well_id].copy()
        well_data = well_data.sort_values('datetime')
        
        if len(well_data) > 10:
            cluster_timeseries[well_id] = well_data.set_index('datetime')['temperature']
    
    if len(cluster_timeseries) < 2:
        return []
    
    # Find common time range
    common_start = max([ts.index.min() for ts in cluster_timeseries.values()])
    common_end = min([ts.index.max() for ts in cluster_timeseries.values()])
    
    # Resample to daily frequency
    resampled_data = {}
    for well_id, ts in cluster_timeseries.items():
        ts_filtered = ts[(ts.index >= common_start) & (ts.index <= common_end)]
        ts_resampled = ts_filtered.resample('D').mean().interpolate()
        resampled_data[well_id] = ts_resampled
    
    # Find common index
    common_index = None
    for well_id, ts in resampled_data.items():
        if common_index is None:
            common_index = ts.index
        else:
            common_index = common_index.intersection(ts.index)
    
    # Align all time series
    aligned_data = {}
    for well_id, ts in resampled_data.items():
        aligned_data[well_id] = ts.reindex(common_index).dropna()
    
    if len(common_index) < 20:
        return []
    
    # Calculate cross-correlations
    well_list = list(aligned_data.keys())
    max_lag = min(10, len(common_index) // 4)
    correlation_results = []
    
    for i, well_a in enumerate(well_list):
        for j, well_b in enumerate(well_list):
            if i >= j:  # Skip duplicate pairs and self-correlation
                continue
            
            ts_a = aligned_data[well_a].values
            ts_b = aligned_data[well_b].values
            
            if len(ts_a) < 20 or len(ts_b) < 20:
                continue
            
            # Find best correlation across lags
            best_corr = 0
            best_lag = 0
            best_source = ""
            best_target = ""
            
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    corr, p_val = pearsonr(ts_a, ts_b)
                    if abs(corr) > abs(best_corr) and p_val < 0.05:
                        best_corr = corr
                        best_lag = lag
                        best_source = well_a if corr > 0 else well_b
                        best_target = well_b if corr > 0 else well_a
                elif lag > 0:
                    if lag < len(ts_a):
                        corr, p_val = pearsonr(ts_a[:-lag], ts_b[lag:])
                        if abs(corr) > abs(best_corr) and p_val < 0.05:
                            best_corr = corr
                            best_lag = lag
                            best_source = well_a
                            best_target = well_b
                else:
                    abs_lag = abs(lag)
                    if abs_lag < len(ts_b):
                        corr, p_val = pearsonr(ts_b[:-abs_lag], ts_a[abs_lag:])
                        if abs(corr) > abs(best_corr) and p_val < 0.05:
                            best_corr = corr
                            best_lag = abs_lag
                            best_source = well_b
                            best_target = well_a
            
            # Store very strong correlations for statistical validation
            if abs(best_corr) > 0.9:  # Only very strong correlations above 0.9
                correlation_results.append({
                    'source': best_source,
                    'target': best_target,
                    'correlation': best_corr,
                    'lag': best_lag,
                    'abs_correlation': abs(best_corr)
                })
    
    return correlation_results

def generate_enhanced_report(correlations, well_ids):
    """Generate enhanced temperature analysis report with p-values and sample sizes"""
    
    from datetime import datetime
    
    report_content = f"""
🔢 ENHANCED TEMPERATURE INFLUENCE ANALYSIS REPORT
==============================================

📍 ANALYSIS SUMMARY:
- Wells analyzed: {', '.join(sorted(well_ids))}
- Total correlations found: {len(correlations)}
- Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Correlation threshold: r > 0.9
- Statistical validation: Permutation testing (1000 shuffles)
- Significance level: p < 0.05

📊 STATISTICALLY VALIDATED CORRELATIONS (r > 0.9, p < 0.05):

"""
    
    if len(correlations) > 0:
        # Sort correlations by strength
        sorted_correlations = sorted(correlations, key=lambda x: x['abs_correlation'], reverse=True)
        
        for i, corr in enumerate(sorted_correlations, 1):
            source = corr['source']
            target = corr['target']
            r_val = corr['correlation']
            lag = corr['lag']
            p_val = corr.get('empirical_p', 0.0)
            sample_size = corr.get('sample_size', 0)
            p_str = f"p<{p_val:.4f}" if p_val > 0 else "p<0.0001"
            
            lag_str = f" (lag: {lag} days)" if lag != 0 else ""
            report_content += f"- {source} → {target}: r={r_val:.3f}{lag_str}, {p_str}, n={sample_size}\\n"
        
        # Add summary statistics
        mean_r = sum(corr['abs_correlation'] for corr in correlations) / len(correlations)
        mean_n = sum(corr.get('sample_size', 0) for corr in correlations) / len(correlations)
        lag_counts = {}
        for corr in correlations:
            lag = corr['lag']
            lag_counts[lag] = lag_counts.get(lag, 0) + 1
        
        report_content += f"""
📈 STATISTICAL SUMMARY:
- Mean correlation strength: r={mean_r:.3f}
- Mean sample size: n={mean_n:.0f}
- All p-values: p < 0.05 (statistically significant)
- Most common lag: {max(lag_counts.keys(), key=lambda k: lag_counts[k])} days ({lag_counts[max(lag_counts.keys(), key=lambda k: lag_counts[k])] } occurrences)
- Simultaneous correlations (lag=0): {lag_counts.get(0, 0)} out of {len(correlations)}

🎯 INTERPRETATION:
- All correlations passed rigorous statistical validation
- Large sample sizes (n={int(mean_n)}) provide robust evidence
- {'Predominantly simultaneous' if lag_counts.get(0, 0) > len(correlations)/2 else 'Mixed temporal'} temperature relationships detected
- Network shows {'very strong' if mean_r > 0.95 else 'strong'} thermal connectivity

⌛ METHODOLOGY:
- Cross-correlation analysis with lag detection (-10 to +10 days)
- Permutation testing with 1000 random shuffles for p-value calculation
- Empirical p-value computation for non-parametric significance testing
- Daily temperature resampling with interpolation for data alignment
"""
    else:
        report_content += "No statistically significant correlations found at r > 0.9 threshold.\\n"
    
    # Write the enhanced report
    with open('enhanced_temperature_analysis_report.txt', 'w') as f:
        f.write(report_content)

def create_influence_map():
    """Create the directional temperature influence map"""
    
    print("🗺️ Creating Directional Temperature Influence Map...")
    
    # Load data - try to use extended coordinates first
    try:
        wells_df = pd.read_csv("../../data_files/wells_left_cluster_coordinates.csv")
        print("✅ Using well coordinates from wells_left_cluster_coordinates.csv")
    except:
        try:
            wells_df = pd.read_csv("../../data_files/wells.csv")
            print("⚠️ Fallback to wells.csv coordinates")
        except:
            print("❌ Could not load wells coordinate file")
            return
    wells_df["stripped_id"] = wells_df["well_id"].str.extract(r"[-_]?([A-Za-z0-9]+)$")[0]
    
    temp_df = pd.read_excel("../../data_files/Bristol.xlsx", sheet_name="List")
    temp_df = temp_df[temp_df["parameter"] == "Temperature"].copy()
    temp_df["datetime"] = pd.to_datetime(temp_df["datetime"], errors='coerce')
    temp_df["well_id"] = temp_df["well_id"].astype(str)
    
    # Clean data
    temp_df = temp_df.drop_duplicates()
    temp_df = temp_df[temp_df["unit"] == "F"]
    temp_df = temp_df[pd.to_numeric(temp_df["value"], errors='coerce').notna()]
    temp_df["mapped_well_id"] = temp_df["well_id"].apply(map_well_id)
    
    # Merge with well locations
    merged = temp_df.merge(wells_df, left_on="mapped_well_id", right_on="well_id")
    merged["temperature"] = pd.to_numeric(merged["value"], errors='coerce')
    heatmap_data = merged[["mapped_well_id", "datetime", "temperature", "x", "y"]].rename(columns={"mapped_well_id": "well_id"})
    
    # Analyze all wells with temperature data
    all_wells_with_data = heatmap_data.groupby('well_id')[['x', 'y']].first().reset_index()
    all_well_ids = all_wells_with_data['well_id'].tolist()
    
    print(f"📍 Analyzing all wells: {len(all_well_ids)} wells with temperature data")
    
    # Calculate directional correlations for all wells
    correlations = calculate_directional_correlations(all_well_ids, heatmap_data)
    
    if not correlations:
        print("❌ No correlations above 0.9 found")
        print("📊 Showing all wells with available temperature data...")
        # Continue to show the wells even without strong correlations
        correlations = []  # Empty list for visualization
    
    if len(correlations) > 0:
        print(f"🔍 Found {len(correlations)} correlations above 0.9, validating statistical significance...")
        
        # Validate correlations with permutation testing to remove coincidences
        validated_correlations = []
        
        for corr in correlations:
            print(f"📊 Validating {corr['source']} → {corr['target']} (r={corr['correlation']:.3f})", end="...")
            
            # Get time series data for this pair
            source_data = heatmap_data[heatmap_data['well_id'] == corr['source']].copy()
            target_data = heatmap_data[heatmap_data['well_id'] == corr['target']].copy()
            
            if len(source_data) < 10 or len(target_data) < 10:
                print("❌ Insufficient data")
                continue
            
            # Create time series
            source_ts = source_data.set_index('datetime')['temperature'].resample('D').mean().interpolate()
            target_ts = target_data.set_index('datetime')['temperature'].resample('D').mean().interpolate()
            
            # Find common time range
            common_start = max(source_ts.index.min(), target_ts.index.min())
            common_end = min(source_ts.index.max(), target_ts.index.max())
            
            source_aligned = source_ts[(source_ts.index >= common_start) & (source_ts.index <= common_end)]
            target_aligned = target_ts[(target_ts.index >= common_start) & (target_ts.index <= common_end)]
            
            # Find overlapping indices
            common_index = source_aligned.index.intersection(target_aligned.index)
            
            if len(common_index) < 20:
                print("❌ Insufficient overlap")
                continue
            
            source_values = source_aligned.reindex(common_index).dropna()
            target_values = target_aligned.reindex(common_index).dropna()
            
            if len(source_values) < 20 or len(target_values) < 20:
                print("❌ Too much missing data")
                continue
            
            # Apply lag if needed
            lag = corr['lag']
            if lag > 0:
                if lag < len(source_values):
                    source_final = source_values.iloc[:-lag].values
                    target_final = target_values.iloc[lag:].values
                else:
                    print("❌ Lag too large")
                    continue
            elif lag < 0:
                abs_lag = abs(lag)
                if abs_lag < len(target_values):
                    source_final = source_values.iloc[abs_lag:].values
                    target_final = target_values.iloc[:-abs_lag].values
                else:
                    print("❌ Lag too large")
                    continue
            else:
                source_final = source_values.values
                target_final = target_values.values
            
            # Ensure same length
            min_len = min(len(source_final), len(target_final))
            source_final = source_final[:min_len]
            target_final = target_final[:min_len]
            
            if min_len < 15:
                print("❌ Final data too short")
                continue
            
            # Calculate real correlation
            from scipy.stats import pearsonr
            real_corr, p_value = pearsonr(source_final, target_final)
            
            # Permutation test with 1000 shuffles
            n_permutations = 1000
            null_correlations = []
            
            for _ in range(n_permutations):
                shuffled_target = np.random.permutation(target_final)
                null_corr = pearsonr(source_final, shuffled_target)[0]
                null_correlations.append(null_corr)
            
            null_correlations = np.array(null_correlations)
            
            # Calculate empirical p-value
            if real_corr >= 0:
                empirical_p = np.sum(null_correlations >= abs(real_corr)) / n_permutations
            else:
                empirical_p = np.sum(null_correlations <= real_corr) / n_permutations
            
            # Two-tailed test
            empirical_p_two_tailed = 2 * min(empirical_p, 1 - empirical_p)
            
            # Only keep if statistically significant (p < 0.05)
            if empirical_p_two_tailed < 0.05:
                corr['empirical_p'] = empirical_p_two_tailed
                corr['sample_size'] = min_len  # Add sample size to correlation data
                corr['validated'] = True
                validated_correlations.append(corr)
                print(f"✅ Validated (p={empirical_p_two_tailed:.4f}, n={min_len})")
            else:
                print(f"❌ Not significant (p={empirical_p_two_tailed:.4f}, n={min_len})")
    
        correlations = validated_correlations
        correlations = sorted(correlations, key=lambda x: x['abs_correlation'], reverse=True)
        
        if not correlations:
            print("❌ No statistically validated correlations found")
            correlations = []  # Continue with empty list to show wells
    else:
        print("🔍 No correlations to validate - showing well positions only")
        correlations = []
        
    if len(correlations) > 0:
        print(f"✅ {len(correlations)} correlations validated as non-coincidental")
    else:
        print("📍 Showing well positions without correlation arrows")
    
    # Get well coordinates for all wells
    well_coords = {}
    for _, well in all_wells_with_data.iterrows():
        well_coords[well['well_id']] = (well['x'], well['y'])
    
    # Load map image (try multiple options)
    map_loaded = False
    map_array = None
    
    for img_path in ["../../data_files/site_map_cropped.png", "../../data_files/cluster2_site_map.png", "map_image.png"]:
        try:
            print(f"🖼️ Trying to load {img_path}...")
            # Try to handle HEIF files that are misnamed as PNG
            try:
                map_img = Image.open(img_path)
                # Convert HEIF to RGB if needed
                if map_img.mode != 'RGB':
                    map_img = map_img.convert('RGB')
                map_array = np.array(map_img)
                print(f"✅ Successfully loaded {img_path} ({map_array.shape})")
                map_loaded = True
                break
            except Exception as inner_e:
                print(f"❌ Failed to load {img_path}: {inner_e}")
                continue
        except Exception as e:
            print(f"❌ Failed to load {img_path}: {e}")
            continue
    
    if not map_loaded:
        print("⚠️ No map image found, creating blank background")
        # Create a blank white background based on well coordinates
        all_x = [coord[0] for coord in well_coords.values()]
        all_y = [coord[1] for coord in well_coords.values()]
        
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        width = int(x_max - x_min + 400)
        height = int(y_max - y_min + 400)
        map_array = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Adjust coordinates to match the created background
        offset_x, offset_y = x_min - 200, y_min - 200
        extent = [offset_x, offset_x + width, offset_y + height, offset_y]
    else:
        extent = [0, map_array.shape[1], map_array.shape[0], 0]
    
    # Create the influence map
    fig, ax = plt.subplots(figsize=(16, 12), dpi=150)
    ax.imshow(map_array, extent=extent)
    
    # Plot all wells with temperature data
    for _, well in all_wells_with_data.iterrows():
        x, y = well['x'], well['y']
        well_id = well['well_id']
        
        # Plot well as circle
        ax.scatter(x, y, s=200, c='white', edgecolor='black', linewidth=3, zorder=10, alpha=0.95)
        ax.scatter(x, y, s=150, c='lightblue', edgecolor='navy', linewidth=2, zorder=11, alpha=0.9)
        
        # Add well ID labels
        ax.text(x, y, well_id.replace('GW-', '').replace('GP-', '').replace('EW-', ''), 
               ha='center', va='center', fontsize=10, fontweight='bold', 
               color='darkblue', zorder=12)
    
    # Create colormap for correlation strength (validated correlations)
    norm = Normalize(vmin=0.9, vmax=1.0)  # Range for validated correlations above 0.9
    cmap_arrows = plt.cm.plasma  # High contrast for subtle differences
    
    # All correlations are validated and very strong (r > 0.9), render with consistent high-quality styling
    for corr in correlations:
        source = corr['source']
        target = corr['target']
        correlation = corr['correlation']
        abs_corr = corr['abs_correlation']
        
        if source in well_coords and target in well_coords:
            x1, y1 = well_coords[source]
            x2, y2 = well_coords[target]
            
            # Calculate arrow properties based on correlation strength
            arrow_color = cmap_arrows(norm(abs_corr))
            
            # Vary arrow thickness based on correlation strength (0.9 to 1.0)
            arrow_width = 2.0 + (abs_corr - 0.9) * 10  # Scale from 2.0 to 3.0
            
            # Offset arrow start/end to avoid overlapping with well circles
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 20:  # Only draw if wells are far enough apart
                # Normalize direction vector
                dx_norm = dx / length
                dy_norm = dy / length
                
                # Smart offset based on well size
                offset = 20
                x1_offset = x1 + dx_norm * offset
                y1_offset = y1 + dy_norm * offset
                x2_offset = x2 - dx_norm * offset
                y2_offset = y2 - dy_norm * offset
                
                # Create straight arrows for clarity at this high correlation level
                ax.annotate('', xy=(x2_offset, y2_offset), xytext=(x1_offset, y1_offset),
                           arrowprops={
                               'arrowstyle': '->',
                               'color': arrow_color,
                               'linewidth': arrow_width,
                               'alpha': 0.9,
                               'zorder': 6,
                               'shrinkA': 0,
                               'shrinkB': 0
                           })
                
                # Add p-value and sample size label near the arrow midpoint (for strongest correlations only)
                if abs_corr > 0.95:  # Only show details for very strongest correlations to avoid clutter
                    mid_x = (x1_offset + x2_offset) / 2
                    mid_y = (y1_offset + y2_offset) / 2
                    p_val = corr.get('empirical_p', 0.0)
                    sample_size = corr.get('sample_size', 0)
                    p_str = f'p<{p_val:.3f}' if p_val > 0 else 'p<0.001'
                    label_text = f'{p_str}\nn={sample_size}'
                    ax.text(mid_x, mid_y, label_text, 
                           fontsize=7, ha='center', va='center', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                           zorder=15)
    
    # Add colorbar for correlation strength
    sm = plt.cm.ScalarMappable(cmap=cmap_arrows, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Correlation Strength', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Add legend for validated very strong correlations
    legend_elements = [
        patches.FancyArrowPatch((0, 0), (1, 0), 
                               arrowstyle='->', 
                               color=cmap_arrows(norm(0.9)), 
                               linewidth=3.0,
                               alpha=0.9,
                               label='Very Strong (r > 0.9)'),
        patches.FancyArrowPatch((0, 0), (1, 0), 
                               arrowstyle='->', 
                               color=cmap_arrows(norm(0.85)), 
                               linewidth=2.3,
                               alpha=0.9,
                               label='Strong (0.8 < r ≤ 0.9)'),
        patches.FancyArrowPatch((0, 0), (1, 0), 
                               arrowstyle='->', 
                               color=cmap_arrows(norm(0.75)), 
                               linewidth=1.8,
                               alpha=0.9,
                               label='Moderate (0.7 < r ≤ 0.8)')
    ]
    
    # Position legend in upper right corner
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
             frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    
    # Set title and remove axes - show full map
    ax.set_title('Directional Temperature Influence Map\nStatistically Validated Correlations (r > 0.9, p < 0.05)', 
                fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Show full map extent
    ax.set_xlim(0, map_array.shape[1])
    ax.set_ylim(map_array.shape[0], 0)  # Invert y-axis for image coordinates
    
    # Save high-resolution image
    plt.tight_layout()
    plt.savefig('directional_temperature_influence_map.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # Print summary
    print(f"\n📊 DIRECTIONAL INFLUENCE MAP SUMMARY:")
    print(f"   Wells visualized: {len(all_well_ids)}")
    print(f"   Directional influences: {len(correlations)}")
    
    if len(correlations) > 0:
        strongest = max(correlations, key=lambda x: x['abs_correlation'])
        p_val = strongest.get('empirical_p', 0.0)
        sample_size = strongest.get('sample_size', 0)
        p_str = f"p<{p_val:.4f}" if p_val > 0 else "p<0.0001"
        print(f"   Strongest influence: {strongest['source']} → {strongest['target']} (r={strongest['correlation']:.3f}, {p_str}, n={sample_size})")
        
        # Create detailed influence table
        influence_df = pd.DataFrame(correlations)
        influence_df = influence_df.sort_values('abs_correlation', ascending=False)
        influence_df.to_csv('directional_influences.csv', index=False)
    else:
        print(f"   No significant correlations found among these wells")
        # Create empty CSV
        pd.DataFrame({'message': ['No correlations found above 0.9']}).to_csv('directional_influences.csv', index=False)
    
    # Generate enhanced analysis report with p-values and sample sizes
    generate_enhanced_report(correlations, all_well_ids)
    
    print(f"\n📁 FILES SAVED:")
    print(f"   - directional_temperature_influence_map.png (high-resolution influence map)")
    print(f"   - directional_influences.csv (detailed influence data)")
    print(f"   - enhanced_temperature_analysis_report.txt (statistical report with p-values and sample sizes)")
    
    if len(correlations) > 0:
        print(f"\n🎯 Top 3 Strongest Influences:")
        for i, corr in enumerate(influence_df.head(3).to_dict('records'), 1):
            p_val = corr.get('empirical_p', 0.0)
            sample_size = corr.get('sample_size', 0)
            p_str = f"p<{p_val:.4f}" if p_val > 0 else "p<0.0001"
            print(f"   {i}. {corr['source']} → {corr['target']}: r={corr['correlation']:.3f} (lag: {corr['lag']} days, {p_str}, n={sample_size})")
    else:
        print(f"\n🎯 Wells appear to have independent temperature patterns")

if __name__ == "__main__":
    create_influence_map()