import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import imageio
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.interpolate import griddata
from sklearn.cluster import KMeans


# Load your data
wells_df = pd.read_csv("../../data_files/wells.csv")
wells_df["stripped_id"] = wells_df["well_id"].str.extract(r"[-_]?([A-Za-z0-9]+)$")[0]

temp_df = pd.read_excel("../../data_files/Bristol.xlsx", sheet_name="List")
# Use "Temperature" (capital T) which has 10,569 records vs "temperature" (367 records)
temp_df = temp_df[temp_df["parameter"] == "Temperature"].copy()
temp_df["datetime"] = pd.to_datetime(temp_df["datetime"], errors='coerce')
temp_df["well_id"] = temp_df["well_id"].astype(str)

# Clean data
temp_df = temp_df.drop_duplicates()  # Remove 4,913 duplicates
temp_df = temp_df[temp_df["unit"] == "F"]  # Only Fahrenheit temperatures
temp_df = temp_df[pd.to_numeric(temp_df["value"], errors='coerce').notna()]  # Remove non-numeric values

def map_well_id(excel_well_id):
    """Map Excel numeric well ID to CSV prefixed format"""
    well_str = str(excel_well_id)
    
    # Map based on patterns found in the data
    if well_str in ['100', '200']:
        return f'GP-{well_str}'
    elif well_str in ['29', '33', '34', '37']:
        return f'EW-{well_str}'
    elif well_str == '13' and excel_well_id == 13:  # Handle GP-13 case
        return 'GP-13'
    else:
        return f'GW-{well_str}'

# Map Excel well IDs to CSV format
temp_df["mapped_well_id"] = temp_df["well_id"].apply(map_well_id)

# Merge temperature data with well locations using mapped well IDs
merged = temp_df.merge(wells_df, left_on="mapped_well_id", right_on="well_id")
merged["temperature"] = pd.to_numeric(merged["value"], errors='coerce')
heatmap_data = merged[["mapped_well_id", "datetime", "temperature", "x", "y"]].rename(columns={"mapped_well_id": "well_id"})

print(f"📊 Data Summary:")
print(f"  • Wells loaded: {len(wells_df)}")
print(f"  • Temperature records: {len(temp_df)}")
print(f"  • Matched records: {len(merged)}")
print(f"  • Wells with temperature data: {heatmap_data['well_id'].nunique()}")


# Load map - try different available map images
map_loaded = False
for img_path in ["../../data_files/map_image.png", "../../data_files/site_map_cropped.png", "../../data_files/cluster2_site_map.png"]:
    try:
        print(f"🖼️ Trying to load {img_path}...")
        map_img = Image.open(img_path)
        map_array = np.array(map_img)
        print(f"✅ Successfully loaded map image: {img_path}")
        map_loaded = True
        break
    except:
        continue

if not map_loaded:
    print("❌ Could not load any map image. Creating blank background...")
    # Create a blank white background if no map image is found
    map_array = np.ones((1000, 1000, 3), dtype=np.uint8) * 255

# Normalize colors
norm = mcolors.Normalize(vmin=heatmap_data["temperature"].min(), vmax=heatmap_data["temperature"].max())
cmap = plt.cm.coolwarm

# Prepare animation with persistent colors - DIRECT TO VIDEO (memory efficient)
unique_times = heatmap_data["datetime"].dropna().sort_values().unique()

# Get all unique wells with their coordinates
all_wells = wells_df[["well_id", "x", "y"]].copy()

# Create a comprehensive time series for proper forward-filling
# Initialize well states with all wells from wells.csv
well_states = {}
for _, well in all_wells.iterrows():
    well_states[well["well_id"]] = {"temperature": None, "last_update": None}

print(f"🎬 Generating EXTENDED CLUSTERED HEATMAP animation with {len(unique_times)} time steps...")
print(f"  • Tracking {len(well_states)} wells total")
print(f"  • Separate heatmaps for each well cluster")
print(f"  • Extended margins for better visual coverage")

# Initialize video writer with maximum detail settings
video_writer = imageio.get_writer("well_temperatures_animation.mp4", fps=15, codec='libx264', quality=10, macro_block_size=1)

for i, time in enumerate(unique_times):
    if i % 50 == 0:  # More frequent progress updates
        percent_complete = (i / len(unique_times)) * 100
        print(f"  • Processing frame {i+1}/{len(unique_times)} ({percent_complete:.1f}% complete)...")
    frame_data = heatmap_data[heatmap_data["datetime"] == time]
    
    # Update well states with new readings
    for _, row in frame_data.iterrows():
        well_id = row["well_id"]
        if well_id in well_states:
            well_states[well_id]["temperature"] = row["temperature"]
            well_states[well_id]["last_update"] = time

    fig, ax = plt.subplots(figsize=(16, 12), dpi=150)  # Higher resolution figure
    ax.imshow(map_array, interpolation='bilinear')  # Better image quality
    ax.set_title(f"Temperature at {pd.to_datetime(time).strftime('%Y-%m-%d %H:%M')}", fontsize=18, pad=20)
    ax.axis('off')

    # Create separate heatmaps for well clusters
    wells_with_data = []
    temps = []
    x_coords = []
    y_coords = []
    
    # Collect wells with temperature data
    for _, well in all_wells.iterrows():
        well_id = well["well_id"]
        x, y = well["x"], well["y"]
        
        if well_states[well_id]["temperature"] is not None:
            wells_with_data.append(well_id)
            temps.append(well_states[well_id]["temperature"])
            x_coords.append(x)
            y_coords.append(y)
    
    # Create cluster-based heatmaps if we have enough temperature data
    if len(temps) > 5:  # Need at least 6 points for 2 clusters
        # Cluster wells into 2 groups based on location
        coordinates = np.column_stack((x_coords, y_coords))
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coordinates)
        
        # Create separate heatmap for each cluster
        for cluster_id in [0, 1]:
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) >= 3:  # Need at least 3 points for interpolation
                
                cluster_x = np.array(x_coords)[cluster_mask]
                cluster_y = np.array(y_coords)[cluster_mask]
                cluster_temps = np.array(temps)[cluster_mask]
                
                # Create interpolation grid for this cluster
                x_min, x_max = cluster_x.min(), cluster_x.max()
                y_min, y_max = cluster_y.min(), cluster_y.max()
                
                # Add larger margin for better visual coverage beyond wells
                x_margin = (x_max - x_min) * 0.4
                y_margin = (y_max - y_min) * 0.4
                
                xi = np.linspace(x_min - x_margin, x_max + x_margin, 120)  # Higher resolution
                yi = np.linspace(y_min - y_margin, y_max + y_margin, 120)  # Higher resolution
                Xi, Yi = np.meshgrid(xi, yi)
                
                # Interpolate temperature values for this cluster with higher quality
                cluster_points = np.column_stack((cluster_x, cluster_y))
                zi = griddata(cluster_points, cluster_temps, (Xi, Yi), method='cubic', fill_value=np.nan)
                
                # Create high-detail smooth heatmap overlay for this cluster
                heatmap = ax.contourf(Xi, Yi, zi, levels=25, cmap=cmap, alpha=0.75, extend='both', antialiased=True)
        
        # Add well locations as high-quality dots
        ax.scatter(x_coords, y_coords, c='black', s=40, edgecolor='white', linewidth=2, zorder=10, alpha=0.9)
    
    elif len(temps) > 0:  # Fallback for few data points - show as high-quality individual dots
        ax.scatter(x_coords, y_coords, c=temps, cmap=cmap, s=120, edgecolor='black', linewidth=2, zorder=10, alpha=0.9)
    
    # Show wells without data as high-quality gray dots
    for _, well in all_wells.iterrows():
        well_id = well["well_id"]
        x, y = well["x"], well["y"]
        if well_states[well_id]["temperature"] is None:
            ax.scatter(x, y, color='lightgray', s=35, edgecolor='black', alpha=0.8, linewidth=1.5, zorder=10)
    
    # Add high-quality summary text
    ax.text(20, 60, f"Wells with data: {len(wells_with_data)}/{len(all_wells)}", 
           fontsize=16, color='black', bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.95, edgecolor='gray'))

    canvas = FigureCanvas(fig)
    canvas.draw()
    frame = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3]
    
    # Write frame directly to video
    video_writer.append_data(frame)
    plt.close(fig)

# Close video writer
video_writer.close()
print("✅ MP4 video saved as 'well_temperatures_animation.mp4'")
print(f"🎉 Video complete! {len(unique_times)} frames processed successfully")
