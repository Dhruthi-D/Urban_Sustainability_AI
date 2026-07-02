import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Point
import itertools
import csv

# --- Load Raster ---
raster_path = "data_raw/Stack_Bangalore_Full_Fix.tif"
dataset = rasterio.open(raster_path)

# --- Load Landuse Vector ---
gdf_landuse = gpd.read_file("data_raw/bbbike/landuse_clipped.shp")
if gdf_landuse.crs != "EPSG:32643":
    gdf_landuse = gdf_landuse.to_crs("EPSG:32643")

# Transformer WGS84 → UTM43N
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)

# --- Sample Grid Points ---
minx, miny, maxx, maxy = gdf_landuse.total_bounds
x_step = 50  # 50 m steps
y_step = 50

# Create a spatial index for faster lookups
from shapely.strtree import STRtree
landuse_geometries = list(gdf_landuse.geometry)
landuse_index = STRtree(landuse_geometries)

# Generate points more efficiently
utm_points = []
for x in range(int(minx), int(maxx), x_step):
    for y in range(int(miny), int(maxy), y_step):
        point = Point(x, y)
        # Use spatial index for faster intersection check
        intersecting_indices = landuse_index.query(point)
        if intersecting_indices.size > 0:
            # Check if any of the intersecting geometries actually contain the point
            if any(landuse_geometries[idx].contains(point) for idx in intersecting_indices):
                utm_points.append((x, y))

print(f"Total points inside city: {len(utm_points)}")

# --- Project Scenarios ---
project_types = ["Mall","Park","Residential","Commercial Complex"]
areas = [500, 1000, 2000]
heights = [6, 15]
materials = ["Concrete","Steel","Glass"]

# --- Heuristic Impacts ---
impact_dict = {
    "Mall": {"NDVI": -0.1, "NDBI": 0.15, "LST": 1.0, "Albedo": -0.02},
    "Park": {"NDVI": 0.1, "NDBI": -0.05, "LST": -0.5, "Albedo": 0.01},
    "Flyover": {"NDVI": -0.05, "NDBI": 0.05, "LST": 0.5, "Albedo": -0.01},
    "Residential": {"NDVI": -0.05, "NDBI": 0.05, "LST": 0.3, "Albedo": -0.005},
    "School": {"NDVI": -0.02, "NDBI": 0.03, "LST": 0.2, "Albedo": -0.003},
    "Hospital": {"NDVI": -0.02, "NDBI": 0.04, "LST": 0.3, "Albedo": -0.004},
    "Commercial Complex": {"NDVI": -0.08, "NDBI": 0.1, "LST": 0.8, "Albedo": -0.015},
    "Parking Lot": {"NDVI": -0.05, "NDBI": 0.05, "LST": 0.5, "Albedo": -0.01},
    "Community Center": {"NDVI": -0.03, "NDBI": 0.02, "LST": 0.2, "Albedo": -0.005}
}

# --- Sustainability Score Function ---
def compute_sustainability_score(ndvi, ndbi, lst, albedo, landuse):
    score = 50
    score += ndvi * 30
    score -= ndbi * 20
    score -= (lst - 25) * 1.5
    score += (0.3 - albedo) * 10
    if landuse == "recreation ground":
        score += 5
    return max(0, min(100, score))

# --- CSV Writing in Batches ---
with open("training_dataset.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "lat","lon","NDVI","NDBI","LST","Albedo","Landuse",
        "Project_Type","Area","Height","Material",
        "New_NDVI","New_NDBI","New_LST","New_Albedo",
        "Old_Score","New_Score","Delta_Score"
    ])
    writer.writeheader()

    # Create a mapping for faster landuse lookups
    landuse_mapping = {}
    for idx, geom in enumerate(landuse_geometries):
        if "type" in gdf_landuse.columns:
            landuse_mapping[idx] = gdf_landuse.iloc[idx]["type"]
    
    total_points = len(utm_points)
    for i, (x, y) in enumerate(utm_points):
        if i % 100 == 0:  # Progress tracking
            print(f"Processing point {i+1}/{total_points} ({((i+1)/total_points)*100:.1f}%)")
            
        lat, lon = transformer.transform(x, y, direction='INVERSE')
        row, col = rowcol(dataset.transform, x, y)

        ndvi = float(dataset.read(1)[row, col])
        ndbi = float(dataset.read(2)[row, col])
        albedo = float(dataset.read(3)[row, col])
        lst = float(dataset.read(4)[row, col])

        # Landuse type - use spatial index for faster lookup
        pt = Point(x, y)
        landuse_type = None
        intersecting_indices = landuse_index.query(pt)
        if intersecting_indices.size > 0 and "type" in gdf_landuse.columns:
            # Get the first intersecting geometry's landuse type
            for idx in intersecting_indices:
                if idx in landuse_mapping:
                    lu = str(landuse_mapping[idx])
                    landuse_type = "recreation ground" if lu=="recreation_groun" else lu
                    break

        old_score = compute_sustainability_score(ndvi, ndbi, lst, albedo, landuse_type)

        # Loop over all project scenarios
        total_combinations = len(project_types) * len(areas) * len(heights) * len(materials)
        print(f"  Generating {total_combinations} combinations per point...")
        
        for p_type, area, height, material in itertools.product(project_types, areas, heights, materials):
            impact = impact_dict[p_type]
            new_ndvi = ndvi + impact["NDVI"]
            new_ndbi = ndbi + impact["NDBI"]
            new_lst = lst + impact["LST"]
            new_albedo = albedo + impact["Albedo"]
            new_score = compute_sustainability_score(new_ndvi, new_ndbi, new_lst, new_albedo, landuse_type)
            delta_score = new_score - old_score

            row_data = {
                "lat": lat, "lon": lon,
                "NDVI": ndvi, "NDBI": ndbi, "LST": lst, "Albedo": albedo,
                "Landuse": landuse_type,
                "Project_Type": p_type, "Area": area, "Height": height, "Material": material,
                "New_NDVI": new_ndvi, "New_NDBI": new_ndbi, "New_LST": new_lst, "New_Albedo": new_albedo,
                "Old_Score": old_score, "New_Score": new_score, "Delta_Score": delta_score
            }

            writer.writerow(row_data)
