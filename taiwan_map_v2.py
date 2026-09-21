
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
import zipfile
import io
import os
import pandas as pd
import numpy as np
from io import BytesIO
from matplotlib.patches import FancyArrowPatch, Rectangle

# Setup professional style and font
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']


# --- Function to add a compass rose ---
def add_compass(ax, x, y, size=0.1):
    """
    Draws a simple compass rose on the map.
    """
    scale = size
    # North arrow
    north_arrow = FancyArrowPatch(
        (x, y - scale * 0.4), (x, y + scale * 0.4),
        arrowstyle='-|>',
        mutation_scale=20,
        color='black',
        linewidth=1.2
    )
    ax.add_patch(north_arrow)
    ax.text(x, y + scale * 0.55, 'N', fontsize=12, ha='center', va='bottom', weight='bold')

# --- Function to add a scale bar ---
def add_scale_bar(ax, lon, lat, length_km):
    """
    Adds a scale bar to the map at a specified location.
    """
    # Earth radius in km
    R = 6371
    # Length of 1 degree of longitude at the given latitude
    deg_len = (np.pi / 180) * R * np.cos(np.radians(lat))
    # Length of the scale bar in degrees
    bar_degrees = length_km / deg_len

    # Bar coordinates
    x0, y0 = lon, lat
    x1 = x0 + bar_degrees

    # Draw the bar and ticks
    ax.plot([x0, x1], [y0, y0], color='black', linewidth=2)
    ax.plot([x0, x0], [y0, y0 - 0.02], color='black', linewidth=2)
    ax.plot([x1, x1], [y0, y0 - 0.02], color='black', linewidth=2)
    
    # Add text
    ax.text(x0 + bar_degrees / 2, y0 + 0.02, f'{length_km} km',
            ha='center', va='bottom', fontsize=10)

# --- 1. Read stations from Google Sheets ---
key = '1a5DReajqstsnUSUdTcRm8pZqeIP9ZmOct834UcOLmjg'
link = f'https://docs.google.com/spreadsheet/ccc?key={key}&output=csv'
try:
    r = requests.get(link)
    df = pd.read_csv(BytesIO(r.content), header=0, low_memory=False)
    # Select and remove duplicates
    stations = df[['siteengname', 'twd97lon', 'twd97lat']].drop_duplicates(
        subset='siteengname', keep='first'
    )
except Exception as e:
    print(f"Failed to load station data: {e}")
    stations = pd.DataFrame(columns=['siteengname', 'twd97lon', 'twd97lat'])


# --- 2. Download Taiwan shapefile (level 2 = municipalities) ---
url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_TWN_shp.zip"
data_dir = "./data"
shp_path = os.path.join(data_dir, "gadm41_TWN_2.shp")

if not os.path.exists(shp_path):
    os.makedirs(data_dir, exist_ok=True)
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(data_dir)

taiwan = gpd.read_file(shp_path)

# --- 3. Reproject to WGS84 ---
taiwan = taiwan.to_crs(epsg=4326)

# --- 4. Create figure ---
fig, ax = plt.subplots(figsize=(10, 10))

# Plot Taiwan mainland
taiwan.plot(ax=ax, color='#f0f0f0', edgecolor='gray', linewidth=0.7)

# Plot stations in a modern blue
valid_stations = stations.dropna(subset=['twd97lon', 'twd97lat'])
if not valid_stations.empty:
    ax.scatter(
        valid_stations['twd97lon'], valid_stations['twd97lat'],
        color='#006BA4', s=40, alpha=0.9, label="Weather Stations",
        edgecolor='white', linewidth=0.7
    )
else:
    print("No valid stations to plot!")

# Adjust map limits for a closer view
minx, miny, maxx, maxy = taiwan.total_bounds
margin_x = 0.1
margin_y = 0.1
ax.set_xlim(minx - margin_x, maxx + margin_x)
ax.set_ylim(miny - margin_y, maxy + margin_y)

# Title and axes
ax.set_title("Map of Weather Stations in Taiwan", fontsize=18, weight='bold', pad=20)
ax.set_xlabel("Longitude (°E)", fontsize=14)
ax.set_ylabel("Latitude (°N)", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)

# Subtle grid
ax.grid(True, linestyle='--', alpha=0.6, color='gray')

# Legend (moved to lower left)
legend = ax.legend(fontsize=12, loc='lower left', frameon=True, title="Legend")
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.8)

# Add compass rose
add_compass(ax, 121.8, 21.7, size=0.5)

# Add scale bar
add_scale_bar(ax, lon=120.3, lat=20.9, length_km=50)

# --- 5. Inset Map (Top-Left and Bigger) ---
# Load low-resolution world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world.to_crs(epsg=4326)

# Create axes for the inset (made bigger)
ax_inset = fig.add_axes([0.18, 0.62, 0.3, 0.3])

# Plot the world and highlight Taiwan
world.plot(ax=ax_inset, color='lightgray', edgecolor='white')
taiwan.plot(ax=ax_inset, color='#006BA4', edgecolor='black', linewidth=0.5)

# Set inset limits to focus on East Asia
ax_inset.set_xlim(100, 140)
ax_inset.set_ylim(10, 50)

# Remove inset axes ticks and add a black frame
ax_inset.set_xticks([])
ax_inset.set_yticks([])
for spine in ax_inset.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.0)

# --- 6. Extent Indicator --- 
min_lon, max_lon = ax.get_xlim()
min_lat, max_lat = ax.get_ylim()

# Draw rectangle on the inset
rect = Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                 fill=True, color='red', alpha=0.3, ec='red', lw=0.8)
ax_inset.add_patch(rect)

# Layout and saving
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()

# Save in high quality
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'taiwan_map_final.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# --- Info in console ---
print(f"Total unique stations: {len(stations)}")
print(f"With valid coordinates: {len(valid_stations)}")

#%%