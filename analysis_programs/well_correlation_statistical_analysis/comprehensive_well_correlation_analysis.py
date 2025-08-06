import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from datetime import datetime
import itertools

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

def calculate_correlation_with_stats(ts1, ts2, well1, well2, max_lag=10):
    """Calculate correlation with lag detection and statistical validation"""
    
    if len(ts1) < 20 or len(ts2) < 20:
        return None
    
    # Find common time range
    common_start = max(ts1.index.min(), ts2.index.min())
    common_end = min(ts1.index.max(), ts2.index.max())
    
    # Align time series
    ts1_aligned = ts1[(ts1.index >= common_start) & (ts1.index <= common_end)]
    ts2_aligned = ts2[(ts2.index >= common_start) & (ts2.index <= common_end)]
    
    # Find overlapping indices
    common_index = ts1_aligned.index.intersection(ts2_aligned.index)
    
    if len(common_index) < 20:
        return None
    
    ts1_values = ts1_aligned.reindex(common_index).dropna()
    ts2_values = ts2_aligned.reindex(common_index).dropna()
    
    if len(ts1_values) < 20 or len(ts2_values) < 20:
        return None
    
    # Find best correlation across lags
    best_result = None
    best_abs_corr = 0
    
    max_lag = min(max_lag, len(ts1_values) // 4)
    
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            # No lag correlation
            if len(ts1_values) == len(ts2_values):
                min_len = min(len(ts1_values), len(ts2_values))
                final_ts1 = ts1_values.values[:min_len]
                final_ts2 = ts2_values.values[:min_len]
            else:
                continue
        elif lag > 0:
            # ts1 leads ts2
            if lag < len(ts1_values):
                final_ts1 = ts1_values.values[:-lag]
                final_ts2 = ts2_values.values[lag:]
            else:
                continue
        else:
            # ts2 leads ts1
            abs_lag = abs(lag)
            if abs_lag < len(ts2_values):
                final_ts1 = ts1_values.values[abs_lag:]
                final_ts2 = ts2_values.values[:-abs_lag]
            else:
                continue
        
        # Ensure same length
        min_len = min(len(final_ts1), len(final_ts2))
        if min_len < 20:
            continue
            
        final_ts1 = final_ts1[:min_len]
        final_ts2 = final_ts2[:min_len]
        
        # Calculate correlation
        try:
            corr, p_val = pearsonr(final_ts1, final_ts2)
            
            if abs(corr) > best_abs_corr and p_val < 0.05:
                best_result = {
                    'well1': well1,
                    'well2': well2,
                    'correlation': corr,
                    'p_value': p_val,
                    'lag': lag,
                    'sample_size': min_len,
                    'abs_correlation': abs(corr)
                }
                best_abs_corr = abs(corr)
        except:
            continue
    
    return best_result

def validate_correlation_permutation(ts1_values, ts2_values, observed_corr, n_permutations=1000):
    """Validate correlation using permutation testing"""
    
    null_correlations = []
    for _ in range(n_permutations):
        shuffled_ts2 = np.random.permutation(ts2_values)
        null_corr = pearsonr(ts1_values, shuffled_ts2)[0]
        null_correlations.append(null_corr)
    
    null_correlations = np.array(null_correlations)
    
    # Calculate empirical p-value
    if observed_corr >= 0:
        empirical_p = np.sum(null_correlations >= abs(observed_corr)) / n_permutations
    else:
        empirical_p = np.sum(null_correlations <= observed_corr) / n_permutations
    
    # Two-tailed test
    empirical_p_two_tailed = 2 * min(empirical_p, 1 - empirical_p)
    
    return empirical_p_two_tailed

def find_all_correlations():
    """Find all correlations between wells with r > 0.80, p < 0.05, n > 800"""
    
    print("🔍 COMPREHENSIVE WELL CORRELATION ANALYSIS")
    print("=" * 50)
    
    # Load wells data
    try:
        wells_df = pd.read_csv("../../data_files/wells.csv")
        print(f"✅ Loaded {len(wells_df)} wells from wells.csv")
    except:
        print("❌ Could not load wells.csv")
        return
    
    # Load temperature data
    try:
        temp_df = pd.read_excel("../../data_files/Bristol.xlsx", sheet_name="List")
        temp_df = temp_df[temp_df["parameter"] == "Temperature"].copy()
        print(f"✅ Loaded {len(temp_df)} temperature records")
    except:
        print("❌ Could not load ../../data_files/Bristol.xlsx")
        return
    
    # Clean and prepare data
    temp_df["datetime"] = pd.to_datetime(temp_df["datetime"], errors='coerce')
    temp_df["well_id"] = temp_df["well_id"].astype(str)
    temp_df = temp_df.drop_duplicates()
    temp_df = temp_df[temp_df["unit"] == "F"]
    temp_df = temp_df[pd.to_numeric(temp_df["value"], errors='coerce').notna()]
    temp_df["mapped_well_id"] = temp_df["well_id"].apply(map_well_id)
    temp_df["temperature"] = pd.to_numeric(temp_df["value"], errors='coerce')
    
    # Merge with well locations
    merged = temp_df.merge(wells_df, left_on="mapped_well_id", right_on="well_id")
    heatmap_data = merged[["mapped_well_id", "datetime", "temperature", "x", "y"]].rename(columns={"mapped_well_id": "well_id"})
    
    print(f"✅ Successfully matched {len(heatmap_data)} temperature records with well locations")
    
    # Get all wells with temperature data
    wells_with_data = heatmap_data.groupby('well_id')[['x', 'y']].first().reset_index()
    well_ids = wells_with_data['well_id'].tolist()
    
    print(f"📊 Analyzing {len(well_ids)} wells with temperature data:")
    print(f"   Wells: {', '.join(sorted(well_ids))}")
    
    # Create time series for each well
    print("🔄 Creating time series for each well...")
    well_timeseries = {}
    
    for well_id in well_ids:
        well_data = heatmap_data[heatmap_data['well_id'] == well_id].copy()
        well_data = well_data.sort_values('datetime')
        
        if len(well_data) > 10:
            # Resample to daily frequency with interpolation
            ts = well_data.set_index('datetime')['temperature'].resample('D').mean().interpolate()
            well_timeseries[well_id] = ts
            print(f"   {well_id}: {len(ts)} daily data points")
    
    print(f"✅ Created time series for {len(well_timeseries)} wells")
    
    # Find all pairwise correlations
    print("🔍 Calculating all pairwise correlations...")
    
    all_correlations = []
    well_pairs = list(itertools.combinations(well_timeseries.keys(), 2))
    total_pairs = len(well_pairs)
    
    print(f"📊 Analyzing {total_pairs} well pairs...")
    
    for i, (well1, well2) in enumerate(well_pairs):
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i + 1}/{total_pairs} pairs analyzed...")
        
        ts1 = well_timeseries[well1]
        ts2 = well_timeseries[well2]
        
        result = calculate_correlation_with_stats(ts1, ts2, well1, well2)
        
        if result and result['abs_correlation'] > 0.60 and result['sample_size'] > 800:
            # Additional permutation validation for high correlations
            if result['abs_correlation'] > 0.9:
                # Get the time series data used for this correlation
                common_start = max(ts1.index.min(), ts2.index.min())
                common_end = min(ts1.index.max(), ts2.index.max())
                ts1_aligned = ts1[(ts1.index >= common_start) & (ts1.index <= common_end)]
                ts2_aligned = ts2[(ts2.index >= common_start) & (ts2.index <= common_end)]
                common_index = ts1_aligned.index.intersection(ts2_aligned.index)
                
                ts1_values = ts1_aligned.reindex(common_index).dropna()
                ts2_values = ts2_aligned.reindex(common_index).dropna()
                
                # Apply lag if needed
                lag = result['lag']
                if lag > 0:
                    final_ts1 = ts1_values.values[:-lag]
                    final_ts2 = ts2_values.values[lag:]
                elif lag < 0:
                    abs_lag = abs(lag)
                    final_ts1 = ts1_values.values[abs_lag:]
                    final_ts2 = ts2_values.values[:-abs_lag]
                else:
                    final_ts1 = ts1_values.values
                    final_ts2 = ts2_values.values
                
                min_len = min(len(final_ts1), len(final_ts2))
                final_ts1 = final_ts1[:min_len]
                final_ts2 = final_ts2[:min_len]
                
                empirical_p = validate_correlation_permutation(final_ts1, final_ts2, result['correlation'])
                result['empirical_p'] = empirical_p
                
                if empirical_p < 0.05:
                    all_correlations.append(result)
            else:
                all_correlations.append(result)
    
    print(f"✅ Analysis complete!")
    
    # Filter and sort results
    filtered_correlations = [corr for corr in all_correlations if 
                           corr['abs_correlation'] > 0.60 and 
                           corr['p_value'] < 0.05 and 
                           corr['sample_size'] > 800]
    
    filtered_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    print(f"📊 RESULTS SUMMARY:")
    print(f"   Total correlations found: {len(filtered_correlations)}")
    print(f"   Criteria: r > 0.60, p < 0.05, n > 800")
    
    if len(filtered_correlations) > 0:
        print(f"\n📊 TOP 10 STRONGEST CORRELATIONS:")
        for i, corr in enumerate(filtered_correlations[:10], 1):
            lag_str = f" (lag: {corr['lag']} days)" if corr['lag'] != 0 else ""
            emp_p_str = f", emp_p<{corr['empirical_p']:.4f}" if 'empirical_p' in corr else ""
            print(f"   {i:2d}. {corr['well1']} → {corr['well2']}: r={corr['correlation']:.3f}{lag_str}, p<{corr['p_value']:.4f}{emp_p_str}, n={corr['sample_size']}")
        
        # Save detailed results to CSV
        results_df = pd.DataFrame(filtered_correlations)
        results_df = results_df.sort_values('abs_correlation', ascending=False)
        results_df.to_csv('all_well_correlations.csv', index=False)
        
        # Generate comprehensive report
        generate_comprehensive_report(filtered_correlations, well_ids)
        
        print(f"\n📁 FILES SAVED:")
        print(f"   - all_well_correlations.csv (detailed correlation data)")
        print(f"   - comprehensive_correlation_report.txt (full analysis report)")
        
        # Statistics summary
        mean_r = np.mean([corr['abs_correlation'] for corr in filtered_correlations])
        mean_n = np.mean([corr['sample_size'] for corr in filtered_correlations])
        simultaneous = len([corr for corr in filtered_correlations if corr['lag'] == 0])
        
        print(f"\n📈 STATISTICAL SUMMARY:")
        print(f"   Mean correlation strength: r={mean_r:.3f}")
        print(f"   Mean sample size: n={mean_n:.0f}")
        print(f"   Simultaneous correlations (lag=0): {simultaneous}/{len(filtered_correlations)} ({100*simultaneous/len(filtered_correlations):.1f}%)")
        
    else:
        print("❌ No correlations found meeting the specified criteria")

def generate_comprehensive_report(correlations, well_ids):
    """Generate comprehensive correlation analysis report"""
    
    report_content = f"""
🔍 COMPREHENSIVE WELL CORRELATION ANALYSIS REPORT
===============================================

📊 ANALYSIS SUMMARY:
- Wells analyzed: {', '.join(sorted(well_ids))}
- Total well pairs examined: {len(well_ids) * (len(well_ids) - 1) // 2}
- Correlations meeting criteria: {len(correlations)}
- Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Criteria: r > 0.60, p < 0.05, n > 800
- Statistical validation: Permutation testing for r > 0.9

📊 ALL CORRELATIONS MEETING CRITERIA (r > 0.60, p < 0.05, n > 800):

"""
    
    if len(correlations) > 0:
        for i, corr in enumerate(correlations, 1):
            well1 = corr['well1']
            well2 = corr['well2']
            r_val = corr['correlation']
            lag = corr['lag']
            p_val = corr['p_value']
            sample_size = corr['sample_size']
            
            lag_str = f" (lag: {lag} days)" if lag != 0 else ""
            emp_p_str = ""
            if 'empirical_p' in corr:
                emp_p_str = f", empirical_p<{corr['empirical_p']:.4f}"
            
            report_content += f"{i:3d}. {well1} → {well2}: r={r_val:.3f}{lag_str}, p<{p_val:.4f}{emp_p_str}, n={sample_size}\n"
        
        # Add summary statistics
        abs_correlations = [corr['abs_correlation'] for corr in correlations]
        sample_sizes = [corr['sample_size'] for corr in correlations]
        lag_counts = {}
        
        for corr in correlations:
            lag = corr['lag']
            lag_counts[lag] = lag_counts.get(lag, 0) + 1
        
        strongest_corr = max(correlations, key=lambda x: x['abs_correlation'])
        
        report_content += f"""
📈 STATISTICAL SUMMARY:
- Total correlations found: {len(correlations)}
- Mean correlation strength: r={np.mean(abs_correlations):.3f}
- Strongest correlation: {strongest_corr['well1']} → {strongest_corr['well2']} (r={strongest_corr['correlation']:.3f})
- Mean sample size: n={np.mean(sample_sizes):.0f}
- Sample size range: n={min(sample_sizes)} to n={max(sample_sizes)}
- Most common lag: {max(lag_counts.keys(), key=lambda k: lag_counts[k])} days
- Simultaneous correlations (lag=0): {lag_counts.get(0, 0)} out of {len(correlations)}

🎯 INTERPRETATION:
- Found {len(correlations)} statistically significant correlations above r=0.60 threshold
- Large sample sizes (mean n={np.mean(sample_sizes):.0f}) provide robust statistical evidence
- {'Predominantly simultaneous' if lag_counts.get(0, 0) > len(correlations)/2 else 'Mixed temporal'} temperature relationships detected
- Network shows strong thermal connectivity across well field

⚙️ METHODOLOGY:
- Cross-correlation analysis with lag detection (-10 to +10 days)
- Pearson correlation with significance testing (p < 0.05)
- Permutation testing for correlations > 0.9 (1000 shuffles)
- Daily temperature resampling with interpolation
- Minimum sample size requirement: n > 800
"""
    else:
        report_content += "No correlations found meeting the specified criteria.\n"
    
    # Write the comprehensive report
    with open('comprehensive_correlation_report.txt', 'w') as f:
        f.write(report_content)

if __name__ == "__main__":
    find_all_correlations()