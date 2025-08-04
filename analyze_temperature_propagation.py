import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans

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

# Load and prepare data
print("🔍 Loading and preparing temperature data...")
wells_df = pd.read_csv("wells.csv")
wells_df["stripped_id"] = wells_df["well_id"].str.extract(r"[-_]?([A-Za-z0-9]+)$")[0]

temp_df = pd.read_excel("Bristol.xlsx", sheet_name="List")
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

print(f"📊 Data loaded: {len(heatmap_data)} temperature records from {heatmap_data['well_id'].nunique()} wells")

# Step 1: Identify well clusters and focus on upper-left cluster
wells_with_data = heatmap_data.groupby('well_id')[['x', 'y']].first().reset_index()
coordinates = wells_with_data[['x', 'y']].values

# Use KMeans to identify clusters
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(coordinates)
wells_with_data['cluster'] = cluster_labels

# Identify upper-left cluster (lower x and y coordinates typically)
cluster_centers = []
for i in [0, 1]:
    cluster_wells = wells_with_data[wells_with_data['cluster'] == i]
    center_x = cluster_wells['x'].mean()
    center_y = cluster_wells['y'].mean()
    cluster_centers.append((i, center_x, center_y, len(cluster_wells)))

# Select the cluster that's more upper-left (lower x + y sum)
cluster_0_sum = cluster_centers[0][1] + cluster_centers[0][2]
cluster_1_sum = cluster_centers[1][1] + cluster_centers[1][2]
upper_left_cluster_id = 0 if cluster_0_sum < cluster_1_sum else 1

upper_left_wells = wells_with_data[wells_with_data['cluster'] == upper_left_cluster_id]
print(f"\n🎯 Analyzing upper-left cluster with {len(upper_left_wells)} wells:")
print(upper_left_wells[['well_id', 'x', 'y']].to_string(index=False))

# Step 2: Calculate angular positions relative to cluster centroid
centroid_x = upper_left_wells['x'].mean()
centroid_y = upper_left_wells['y'].mean()
print(f"\n📍 Cluster centroid: ({centroid_x:.1f}, {centroid_y:.1f})")

# Calculate angles for each well
upper_left_wells = upper_left_wells.copy()
upper_left_wells['theta_rad'] = np.arctan2(upper_left_wells['y'] - centroid_y, 
                                          upper_left_wells['x'] - centroid_x)
upper_left_wells['theta_deg'] = np.degrees(upper_left_wells['theta_rad'])

# Normalize angles to 0-360 degrees
upper_left_wells['theta_deg'] = (upper_left_wells['theta_deg'] + 360) % 360

print("\n🧭 Well angular positions:")
for _, well in upper_left_wells.iterrows():
    print(f"  {well['well_id']}: {well['theta_deg']:.1f}°")

# Step 3: Track temperature increases over time
cluster_well_ids = upper_left_wells['well_id'].tolist()
cluster_data = heatmap_data[heatmap_data['well_id'].isin(cluster_well_ids)].copy()
cluster_data = cluster_data.sort_values(['well_id', 'datetime'])

# Calculate temperature changes (ΔT)
temperature_changes = []
for well_id in cluster_well_ids:
    well_data = cluster_data[cluster_data['well_id'] == well_id].copy()
    well_data = well_data.sort_values('datetime')
    
    if len(well_data) < 2:
        continue
        
    well_data['temp_change'] = well_data['temperature'].diff()
    well_data['temp_change_pct'] = well_data['temperature'].pct_change() * 100
    
    # Get angular position for this well
    theta = upper_left_wells[upper_left_wells['well_id'] == well_id]['theta_deg'].iloc[0]
    
    for _, row in well_data.iterrows():
        if pd.notna(row['temp_change']):
            temperature_changes.append({
                'well_id': well_id,
                'datetime': row['datetime'],
                'temperature': row['temperature'],
                'temp_change': row['temp_change'],
                'temp_change_pct': row['temp_change_pct'],
                'theta_deg': theta,
                'x': upper_left_wells[upper_left_wells['well_id'] == well_id]['x'].iloc[0],
                'y': upper_left_wells[upper_left_wells['well_id'] == well_id]['y'].iloc[0]
            })

temp_changes_df = pd.DataFrame(temperature_changes)
print(f"\n📈 Calculated {len(temp_changes_df)} temperature change events")

# Step 4: Identify significant temperature increases
temp_threshold = temp_changes_df['temp_change'].quantile(0.75)  # Top 25% of increases
pct_threshold = 5.0  # 5% increase threshold

significant_increases = temp_changes_df[
    (temp_changes_df['temp_change'] > temp_threshold) | 
    (temp_changes_df['temp_change_pct'] > pct_threshold)
].copy()

print(f"\n🔥 Found {len(significant_increases)} significant temperature increases")
print(f"   Temperature threshold: >{temp_threshold:.1f}°F")
print(f"   Percentage threshold: >{pct_threshold:.1f}%")

# Step 5: Analyze temporal-angular correlation
if len(significant_increases) > 3:
    # Convert datetime to numeric for correlation analysis (hours since start)
    start_time = significant_increases['datetime'].min()
    significant_increases['hours_elapsed'] = (significant_increases['datetime'] - start_time).dt.total_seconds() / 3600
    
    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(significant_increases['hours_elapsed'], significant_increases['theta_deg'])
    spearman_corr, spearman_p = spearmanr(significant_increases['hours_elapsed'], significant_increases['theta_deg'])
    
    print(f"\n🔢 PROPAGATION ANALYSIS RESULTS:")
    print(f"   Pearson correlation (time vs angle): {pearson_corr:.3f} (p={pearson_p:.4f})")
    print(f"   Spearman correlation (time vs angle): {spearman_corr:.3f} (p={spearman_p:.4f})")
    
    if pearson_corr > 0.3 and pearson_p < 0.05:
        print("   ✅ CLOCKWISE propagation detected!")
    elif pearson_corr < -0.3 and pearson_p < 0.05:
        print("   ↩️ COUNTER-CLOCKWISE propagation detected!")
    else:
        print("   ❓ No clear directional propagation pattern")
    
    # Additional analysis: Check for circular patterns
    # Use circular statistics for angles
    angles_rad = np.radians(significant_increases['theta_deg'].values)
    times = significant_increases['hours_elapsed'].values
    
    # Calculate circular correlation (simplified)
    mean_angle = np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))
    angular_deviation = np.abs(angles_rad - mean_angle)
    
    print(f"\n🌀 CIRCULAR PATTERN ANALYSIS:")
    print(f"   Mean activation angle: {np.degrees(mean_angle):.1f}°")
    print(f"   Angular spread: {np.degrees(np.std(angular_deviation)):.1f}°")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Well positions with angles
    ax1.scatter(upper_left_wells['x'], upper_left_wells['y'], c=upper_left_wells['theta_deg'], 
                cmap='hsv', s=100, edgecolor='black')
    ax1.plot(centroid_x, centroid_y, 'r*', markersize=15, label='Centroid')
    for _, well in upper_left_wells.iterrows():
        ax1.annotate(f"{well['well_id']}\n{well['theta_deg']:.0f}°", 
                    (well['x'], well['y']), xytext=(5, 5), textcoords='offset points')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('Well Positions and Angular Positions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time vs Angle scatter
    scatter = ax2.scatter(significant_increases['hours_elapsed'], significant_increases['theta_deg'], 
                         c=significant_increases['temp_change'], cmap='Reds', s=80, alpha=0.7)
    ax2.set_xlabel('Time (hours since start)')
    ax2.set_ylabel('Angular Position (degrees)')
    ax2.set_title(f'Temperature Increases Over Time\n(r={pearson_corr:.3f}, p={pearson_p:.4f})')
    plt.colorbar(scatter, ax=ax2, label='Temperature Change (°F)')
    
    # Add trend line
    if len(significant_increases) > 1:
        z = np.polyfit(significant_increases['hours_elapsed'], significant_increases['theta_deg'], 1)
        p = np.poly1d(z)
        ax2.plot(significant_increases['hours_elapsed'], p(significant_increases['hours_elapsed']), 
                "r--", alpha=0.8, label=f'Trend: {z[0]:.2f}°/hour')
        ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Temperature changes over time
    for well_id in cluster_well_ids:
        well_changes = temp_changes_df[temp_changes_df['well_id'] == well_id]
        if len(well_changes) > 0:
            ax3.plot(well_changes['datetime'], well_changes['temp_change'], 
                    marker='o', label=well_id, alpha=0.7)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Temperature Change (°F)')
    ax3.set_title('Temperature Changes by Well')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Polar plot of activations
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    theta_rad = np.radians(significant_increases['theta_deg'])
    colors = plt.cm.plasma(significant_increases['hours_elapsed'] / significant_increases['hours_elapsed'].max())
    scatter = ax4.scatter(theta_rad, significant_increases['hours_elapsed'], 
                         c=significant_increases['hours_elapsed'], cmap='plasma', s=80)
    ax4.set_title('Polar View: Activation Time vs Angle')
    ax4.set_xlabel('Time (hours)')
    plt.colorbar(scatter, ax=ax4, label='Hours Elapsed')
    
    plt.tight_layout()
    plt.savefig('temperature_propagation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary report
    report = f"""
🔢 TEMPERATURE PROPAGATION ANALYSIS REPORT
========================================

📍 CLUSTER ANALYZED:
- Wells: {', '.join(cluster_well_ids)}
- Centroid: ({centroid_x:.1f}, {centroid_y:.1f})
- Analysis period: {significant_increases['datetime'].min()} to {significant_increases['datetime'].max()}

📊 STATISTICAL RESULTS:
- Significant temperature increases detected: {len(significant_increases)}
- Pearson correlation (time vs angle): {pearson_corr:.3f} (p-value: {pearson_p:.4f})
- Spearman correlation (time vs angle): {spearman_corr:.3f} (p-value: {spearman_p:.4f})

🌀 PROPAGATION PATTERN:
"""
    
    if pearson_corr > 0.3 and pearson_p < 0.05:
        report += "- ✅ CLOCKWISE propagation detected (statistically significant)"
    elif pearson_corr < -0.3 and pearson_p < 0.05:
        report += "- ↩️ COUNTER-CLOCKWISE propagation detected (statistically significant)"
    else:
        report += "- ❓ No clear directional propagation pattern detected"
    
    report += f"""

📈 TREND ANALYSIS:
- Angular change rate: {z[0] if len(significant_increases) > 1 else 'N/A'}°/hour
- Mean activation angle: {np.degrees(mean_angle):.1f}°
- Angular spread: {np.degrees(np.std(angular_deviation)):.1f}°

🎯 INTERPRETATION:
"""
    
    if abs(pearson_corr) > 0.5:
        report += f"- Strong {'clockwise' if pearson_corr > 0 else 'counter-clockwise'} pattern"
    elif abs(pearson_corr) > 0.3:
        report += f"- Moderate {'clockwise' if pearson_corr > 0 else 'counter-clockwise'} pattern"
    else:
        report += "- Temperature changes appear random or non-directional"
    
    print(report)
    
    # Save detailed results
    significant_increases.to_csv('temperature_propagation_data.csv', index=False)
    with open('temperature_propagation_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n📁 FILES SAVED:")
    print(f"   - temperature_propagation_analysis.png (visualization)")
    print(f"   - temperature_propagation_data.csv (detailed data)")
    print(f"   - temperature_propagation_report.txt (summary report)")

else:
    print("\n⚠️ Insufficient data for propagation analysis (need >3 significant temperature increases)")

# ⌛ TEMPORAL CROSS-CORRELATION ANALYSIS
print(f"\n⌛ TEMPORAL CROSS-CORRELATION ANALYSIS")
print("="*50)

# Get time series data for upper left cluster wells
cluster_timeseries = {}
for well_id in cluster_well_ids:
    well_data = cluster_data[cluster_data['well_id'] == well_id].copy()
    well_data = well_data.sort_values('datetime')
    
    if len(well_data) > 10:  # Need sufficient data points
        # Create regular time series (interpolate to common time grid)
        cluster_timeseries[well_id] = well_data.set_index('datetime')['temperature']

if len(cluster_timeseries) >= 2:
    # Find common time range for all wells
    common_start = max([ts.index.min() for ts in cluster_timeseries.values()])
    common_end = min([ts.index.max() for ts in cluster_timeseries.values()])
    
    print(f"📅 Analysis period: {common_start} to {common_end}")
    print(f"🔍 Analyzing {len(cluster_timeseries)} wells for temporal dependencies")
    
    # Resample all time series to common frequency (daily)
    resampled_data = {}
    for well_id, ts in cluster_timeseries.items():
        # Filter to common time range and resample
        ts_filtered = ts[(ts.index >= common_start) & (ts.index <= common_end)]
        ts_resampled = ts_filtered.resample('D').mean().interpolate()
        resampled_data[well_id] = ts_resampled
    
    # Find common index (dates where all wells have data)
    common_index = None
    for well_id, ts in resampled_data.items():
        if common_index is None:
            common_index = ts.index
        else:
            common_index = common_index.intersection(ts.index)
    
    # Align all time series to common index
    aligned_data = {}
    for well_id, ts in resampled_data.items():
        aligned_data[well_id] = ts.reindex(common_index).dropna()
    
    print(f"📊 Common time points: {len(common_index)}")
    
    if len(common_index) > 20:  # Need sufficient time points for cross-correlation
        # Calculate cross-correlations for all well pairs
        well_list = list(aligned_data.keys())
        max_lag = min(10, len(common_index) // 4)  # Maximum lag to test
        
        correlation_results = []
        causal_chains = []
        
        print(f"\n🔗 CROSS-CORRELATION RESULTS (max lag: {max_lag} days):")
        print("-" * 60)
        
        for i, well_a in enumerate(well_list):
            for j, well_b in enumerate(well_list):
                if i >= j:  # Skip duplicate pairs and self-correlation
                    continue
                
                ts_a = aligned_data[well_a].values
                ts_b = aligned_data[well_b].values
                
                if len(ts_a) < 20 or len(ts_b) < 20:
                    continue
                
                # Calculate cross-correlation at different lags
                best_corr = 0
                best_lag = 0
                best_direction = ""
                
                correlations_at_lags = []
                
                for lag in range(-max_lag, max_lag + 1):
                    if lag == 0:
                        corr = pearsonr(ts_a, ts_b)[0]
                        direction = "simultaneous"
                    elif lag > 0:
                        # A leads B by lag days
                        if lag < len(ts_a):
                            corr = pearsonr(ts_a[:-lag], ts_b[lag:])[0]
                            direction = f"{well_a} leads {well_b}"
                        else:
                            continue
                    else:  # lag < 0
                        # B leads A by |lag| days
                        abs_lag = abs(lag)
                        if abs_lag < len(ts_b):
                            corr = pearsonr(ts_b[:-abs_lag], ts_a[abs_lag:])[0]
                            direction = f"{well_b} leads {well_a}"
                        else:
                            continue
                    
                    correlations_at_lags.append((lag, corr, direction))
                    
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
                        best_direction = direction
                
                # Store results
                correlation_results.append({
                    'well_a': well_a,
                    'well_b': well_b,
                    'best_correlation': best_corr,
                    'best_lag': best_lag,
                    'direction': best_direction,
                    'all_correlations': correlations_at_lags
                })
                
                # Print significant correlations
                if abs(best_corr) > 0.3:  # Threshold for significant correlation
                    if best_lag == 0:
                        print(f"  {well_a} ↔ {well_b}: r={best_corr:.3f} (simultaneous)")
                    elif best_lag > 0:
                        print(f"  {well_a} → {well_b}: r={best_corr:.3f} (lag: {best_lag} days)")
                    else:
                        print(f"  {well_b} → {well_a}: r={best_corr:.3f} (lag: {abs(best_lag)} days)")
                    
                    # Add to causal chains
                    if abs(best_corr) > 0.5 and best_lag != 0:
                        if best_lag > 0:
                            causal_chains.append((well_a, well_b, best_lag, best_corr))
                        else:
                            causal_chains.append((well_b, well_a, abs(best_lag), best_corr))
        
        # Create causal chain visualization
        if causal_chains:
            print(f"\n🔗 CAUSAL CHAINS DETECTED:")
            print("-" * 30)
            for leader, follower, lag, corr in sorted(causal_chains, key=lambda x: x[3], reverse=True):
                print(f"  {leader} → {follower} (lag: {lag} days, r={corr:.3f})")
        else:
            print(f"\n❌ No strong causal chains detected (correlation threshold: 0.5)")
        
        # Create advanced visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Time series overlay
        for well_id, ts in aligned_data.items():
            ax1.plot(ts.index, ts.values, label=well_id, alpha=0.7, linewidth=2)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Temperature (°F)')
        ax1.set_title('Temperature Time Series - Upper Left Cluster')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cross-correlation heatmap
        n_wells = len(well_list)
        corr_matrix = np.zeros((n_wells, n_wells))
        lag_matrix = np.zeros((n_wells, n_wells))
        
        for result in correlation_results:
            i = well_list.index(result['well_a'])
            j = well_list.index(result['well_b'])
            corr_matrix[i, j] = result['best_correlation']
            corr_matrix[j, i] = result['best_correlation']  # Symmetric
            lag_matrix[i, j] = result['best_lag']
            lag_matrix[j, i] = -result['best_lag']  # Opposite lag
        
        # Fill diagonal with 1.0
        np.fill_diagonal(corr_matrix, 1.0)
        
        im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_xticks(range(n_wells))
        ax2.set_yticks(range(n_wells))
        ax2.set_xticklabels(well_list, rotation=45)
        ax2.set_yticklabels(well_list)
        ax2.set_title('Cross-Correlation Matrix\n(Best Correlation at Optimal Lag)')
        
        # Add correlation values as text
        for i in range(n_wells):
            for j in range(n_wells):
                if i != j:
                    text = f"{corr_matrix[i, j]:.2f}\n(lag: {int(lag_matrix[i, j])})"
                    ax2.text(j, i, text, ha="center", va="center", fontsize=8)
        
        plt.colorbar(im, ax=ax2)
        
        # Plot 3: Lag distribution
        all_lags = [result['best_lag'] for result in correlation_results if abs(result['best_correlation']) > 0.3]
        if all_lags:
            ax3.hist(all_lags, bins=range(-max_lag-1, max_lag+2), alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Lag (days)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Optimal Lags\n(for correlations > 0.3)')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No significant\ncorrelations found', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Lag Distribution')
        
        # Plot 4: Causal network diagram
        if causal_chains:
            # Create a simple network layout
            well_positions = {}
            for i, well_id in enumerate(well_list):
                angle = 2 * np.pi * i / len(well_list)
                well_positions[well_id] = (np.cos(angle), np.sin(angle))
            
            # Draw wells
            for well_id, (x, y) in well_positions.items():
                ax4.scatter(x, y, s=200, alpha=0.7, edgecolor='black')
                ax4.text(x, y, well_id, ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Draw causal arrows
            for leader, follower, lag, corr in causal_chains:
                if leader in well_positions and follower in well_positions:
                    x1, y1 = well_positions[leader]
                    x2, y2 = well_positions[follower]
                    
                    # Arrow properties based on correlation strength
                    alpha = min(abs(corr), 1.0)
                    width = abs(corr) * 3
                    
                    ax4.annotate('', xy=(x2, y2), xytext=(x1, y1),
                               arrowprops=dict(arrowstyle='->', alpha=alpha, lw=width, color='red'))
                    
                    # Add lag label
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ax4.text(mid_x, mid_y, f'{lag}d', fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            
            ax4.set_xlim(-1.5, 1.5)
            ax4.set_ylim(-1.5, 1.5)
            ax4.set_title('Causal Network\n(Arrows show temporal dependencies)')
            ax4.set_aspect('equal')
        else:
            ax4.text(0.5, 0.5, 'No causal chains\ndetected', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Causal Network')
        
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig('temporal_cross_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Enhanced report with cross-correlation results
        enhanced_report = report + f"""

⌛ TEMPORAL CROSS-CORRELATION ANALYSIS:
====================================

📊 ANALYSIS PARAMETERS:
- Wells analyzed: {', '.join(well_list)}
- Time period: {common_start} to {common_end}
- Data points: {len(common_index)} days
- Maximum lag tested: {max_lag} days

🔗 SIGNIFICANT CORRELATIONS (|r| > 0.3):
"""
        
        significant_correlations = [r for r in correlation_results if abs(r['best_correlation']) > 0.3]
        if significant_correlations:
            for result in sorted(significant_correlations, key=lambda x: abs(x['best_correlation']), reverse=True):
                enhanced_report += f"\n- {result['well_a']} ↔ {result['well_b']}: r={result['best_correlation']:.3f}"
                if result['best_lag'] != 0:
                    enhanced_report += f" (lag: {result['best_lag']} days)"
        else:
            enhanced_report += "\n- No significant correlations detected"
        
        enhanced_report += f"\n\n🔗 CAUSAL CHAINS (|r| > 0.5 with lag > 0):"
        if causal_chains:
            for leader, follower, lag, corr in sorted(causal_chains, key=lambda x: x[3], reverse=True):
                enhanced_report += f"\n- {leader} → {follower} (lag: {lag} days, r={corr:.3f})"
        else:
            enhanced_report += "\n- No strong causal chains detected"
        
        enhanced_report += f"""

🎯 TEMPORAL DEPENDENCY SUMMARY:
- Total well pairs analyzed: {len(correlation_results)}
- Significant correlations found: {len(significant_correlations)}
- Strong causal chains detected: {len(causal_chains)}
"""
        
        # Save enhanced results
        cross_corr_df = pd.DataFrame(correlation_results)
        cross_corr_df.to_csv('temporal_cross_correlation_results.csv', index=False)
        
        with open('enhanced_temperature_analysis_report.txt', 'w') as f:
            f.write(enhanced_report)
        
        print(f"\n📁 ADDITIONAL FILES SAVED:")
        print(f"   - temporal_cross_correlation_analysis.png (cross-correlation visualization)")
        print(f"   - temporal_cross_correlation_results.csv (detailed correlation data)")
        print(f"   - enhanced_temperature_analysis_report.txt (complete report)")
        
    else:
        print(f"\n⚠️ Insufficient common time points ({len(common_index)}) for cross-correlation analysis")
else:
    print(f"\n⚠️ Insufficient wells with adequate data for cross-correlation analysis")

print(f"\n✅ Complete analysis finished!")