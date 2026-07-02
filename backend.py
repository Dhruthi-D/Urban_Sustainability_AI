from flask import Flask, request, jsonify, render_template
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Point
import requests

app = Flask(__name__)

# --- Load Raster ---
raster_path = "data_raw/Stack_Bangalore_Full_Fix.tif"
dataset = rasterio.open(raster_path)

# --- Load Vector Layers ---
vector_files = {
    "Buildings": "data_raw/bbbike/buildings_clipped.shp",
    "Landuse": "data_raw/bbbike/landuse_clipped.shp",
    "Natural": "data_raw/bbbike/natural_clipped.shp",
    "Places": "data_raw/bbbike/places_clipped.shp",
    "Points": "data_raw/bbbike/points_clipped.shp",
    "Railways": "data_raw/bbbike/railways_clipped.shp",
    "Roads": "data_raw/bbbike/roads_clipped.shp",
    "Waterways": "data_raw/bbbike/waterways_clipped.shp"
}

layers = {}
for name, path in vector_files.items():
    gdf = gpd.read_file(path)
    # Ensure UTM CRS
    if gdf.crs != "EPSG:32643":
        gdf = gdf.to_crs("EPSG:32643")
    layers[name] = gdf

# Transformer WGS84 → UTM43N
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)

# --- Sustainability Score Function ---
def compute_sustainability_score(raster_values, features_present, landuse_type):
    score = 50  # base
    ndvi = raster_values.get("Band_1", 0)
    ndbi = raster_values.get("Band_2", 0)
    albedo = raster_values.get("Band_3", 0)
    lst = raster_values.get("Band_4", 0)

    # Raster contribution
    score += ndvi * 30
    score -= ndbi * 20
    score -= (lst - 25) * 1.5
    score += (0.3 - albedo) * 10

    # Vector features
    if "Buildings" in features_present or "Roads" in features_present:
        score -= 10
    if "Natural" in features_present or "Waterways" in features_present:
        score += 10
    if landuse_type == "recreation ground":
        score += 5

    score = max(0, min(100, score))
    return round(score, 1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_params", methods=["GET"])
def get_params():
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))

    # Convert to UTM
    x, y = transformer.transform(lon, lat)

    # --- Raster Bands ---
    row, col = rowcol(dataset.transform, x, y)
    values = {}
    for i in range(1, dataset.count + 1):
        val = dataset.read(i)[row, col]
        values[f"Band_{i}"] = float(val)

    # --- Vector Intersection ---
    point = Point(x, y)
    features_present = []
    highlighted_geojson = {}
    landuse_info = None

    for name, gdf in layers.items():
        matches = gdf[gdf.intersects(point)]
        if not matches.empty:
            features_present.append(name)

            # Extract landuse type
            if name.lower() == "landuse":
                if "type" in gdf.columns:
                    lu = str(matches.iloc[0]["type"])
                    if lu == "recreation_groun":
                        lu = "recreation ground"
                    landuse_info = lu

            # Skip highlighting for Buildings
            if name != "Buildings":
                highlighted_geojson[name] = matches.to_crs("EPSG:4326").__geo_interface__

    # --- Compute Sustainability Score ---
    score = compute_sustainability_score(values, features_present, landuse_info)

    result = {
        "coordinates": {"lat": lat, "lon": lon},
        "raster_bands": values,
        "features_present": features_present,
        "landuse_type": landuse_info,
        "highlighted_geojson": highlighted_geojson,
        "sustainability_score": score
    }

    return jsonify(result)

# --- TomTom Traffic Info Route ---
@app.route("/get_traffic_info", methods=["GET"])
def get_traffic_info():
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))
    api_key = "lF9KC9U2sLDxh9u8IlQoILfdkvJfetna"

    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/json?point={lat},{lon}&key={api_key}"

    try:
        response = requests.get(url)
        data = response.json()
        flow = data.get("flowSegmentData", {})

        traffic_info = {
            "currentSpeed": flow.get("currentSpeed"),
            "freeFlowSpeed": flow.get("freeFlowSpeed"),
            "confidence": flow.get("confidence"),
            "roadClosure": flow.get("roadClosure")
        }
    except Exception as e:
        traffic_info = {"error": str(e)}

    return jsonify(traffic_info)

if __name__ == "__main__":
    app.run(debug=True)
