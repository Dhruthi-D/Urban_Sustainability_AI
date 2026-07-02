from flask import Flask, request, jsonify, render_template
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Point, Polygon
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from llm_helper import explain_sustainability_change

app = Flask(__name__)

# --- Paths ---
RASTER_PATH = "data_raw/Stack_Bangalore_Full_Fix.tif"
MODEL_PATH = "delta_score_lgb_chunked.txt"

# --- Load Raster ---
dataset = rasterio.open(RASTER_PATH)

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
    if gdf.crs != "EPSG:32643":
        gdf = gdf.to_crs("EPSG:32643")
    layers[name] = gdf

# --- Transformer (WGS84 → UTM 43N) ---
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)

# --- Load ML Model ---
print("Loading LightGBM model...")
bst = lgb.Booster(model_file=MODEL_PATH)
print("Model loaded successfully.")

# ------------------------------------------------------------------------
# --- BASELINE SCORE FUNCTION (for current condition) --------------------
# ------------------------------------------------------------------------
def compute_sustainability_score(raster_values, features_present, landuse_type):
    score = 50
    ndvi = raster_values.get("Band_1", 0)
    ndbi = raster_values.get("Band_2", 0)
    albedo = raster_values.get("Band_3", 0)
    lst = raster_values.get("Band_4", 0)

    score += ndvi * 30
    score -= ndbi * 20
    score -= (lst - 25) * 1.5
    score += (0.3 - albedo) * 10

    if "Buildings" in features_present or "Roads" in features_present:
        score -= 10
    if "Natural" in features_present or "Waterways" in features_present:
        score += 10
    if landuse_type == "recreation ground":
        score += 5

    return round(max(0, min(100, score)), 1)

# ------------------------------------------------------------------------
# --- ROUTE: GET BASELINE PARAMETERS (same as Phase 1) ------------------
# ------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_params", methods=["GET"])
def get_params():
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))

    x, y = transformer.transform(lon, lat)
    row, col = rowcol(dataset.transform, x, y)

    # Raster band values
    values = {f"Band_{i}": float(dataset.read(i)[row, col]) for i in range(1, dataset.count + 1)}

    # Vector features
    point = Point(x, y)
    features_present = []
    highlighted_geojson = {}
    landuse_info = None
    natural_type = None
    natural_name = None

    for name, gdf in layers.items():
        matches = gdf[gdf.intersects(point)]
        if not matches.empty:
            features_present.append(name)
            # --- Extract landuse type ---
            if name.lower() == "landuse" and "type" in gdf.columns:
                lu = str(matches.iloc[0]["type"])
                if lu == "recreation_groun":
                    lu = "recreation ground"
                landuse_info = lu

            # --- Extract natural info ---
            if name.lower() == "natural":
                if "type" in gdf.columns:
                    natural_type = str(matches.iloc[0]["type"])
                if "name" in gdf.columns:
                    natural_name = str(matches.iloc[0]["name"])

            if name != "Buildings":
                highlighted_geojson[name] = matches.to_crs("EPSG:4326").__geo_interface__

    score = compute_sustainability_score(values, features_present, landuse_info)
    result = {
        "coordinates": {"lat": lat, "lon": lon},
        "raster_bands": values,
        "features_present": features_present,
        "landuse_type": landuse_info,
        "natural_type": natural_type,
        "name": natural_name,
        "sustainability_score": score,
        "highlighted_geojson": highlighted_geojson
    }
    return jsonify(result)



@app.route("/impact")
def impact_page():
    return render_template("impact.html")


# ------------------------------------------------------------------------
# --- ROUTE: PREDICT PROJECT IMPACT (Phase 3 core) ----------------------
# ------------------------------------------------------------------------
@app.route("/predict_project_impact", methods=["POST"])
def predict_project_impact():
    data = request.get_json()
    lat, lon = float(data["lat"]), float(data["lon"])
    project_type = data["project_type"]
    material = data["material"]
    geometry = data.get("geometry", None)
    area = float(data.get("area", 500))
    height = float(data.get("height", 10))

    # --- Convert to UTM and extract raster values ---
    x, y = transformer.transform(lon, lat)
    row, col = rowcol(dataset.transform, x, y)
    raster_values = {f"Band_{i}": float(dataset.read(i)[row, col]) for i in range(1, dataset.count + 1)}

    # --- Get landuse ---
    point = Point(x, y)
    landuse_info = None
    for name, gdf in layers.items():
        if name.lower() == "landuse":
            matches = gdf[gdf.intersects(point)]
            if not matches.empty:
                lu = str(matches.iloc[0]["type"])
                if lu == "recreation_groun":
                    lu = "recreation ground"
                landuse_info = lu
                break

    # --- Prepare features ---
    feature_dict = {
        "NDVI": raster_values.get("Band_1", 0),
        "NDBI": raster_values.get("Band_2", 0),
        "ALBEDO": raster_values.get("Band_3", 0),
        "LST": raster_values.get("Band_4", 0),
        "Landuse": landuse_info or "unknown",
        "Project_Type": project_type,
        "Material": material,
        "Area": area,
        "Height": height
    }

    df = pd.DataFrame([feature_dict])

    # --- Encode categorical columns ---
    categorical_cols = ["Landuse", "Project_Type", "Material"]
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    # --- Match model features ---
    train_features = bst.feature_name()
    for col in train_features:
        if col not in df.columns:
            df[col] = 0
    df = df[train_features]

    # --- Predict delta score ---
    delta_pred = float(bst.predict(df, num_iteration=bst.best_iteration)[0])
    current_score = compute_sustainability_score(raster_values, [], landuse_info)
    predicted_score = max(0, min(100, current_score + delta_pred))

    # --- Form JSON response ---
    result = {
        "input": feature_dict,
        "predicted_delta": round(delta_pred, 2),
        "current_score": current_score,
        "predicted_score": round(predicted_score, 2),
        "impact": "Positive" if delta_pred > 0 else "Negative"
    }

    # --- Get Gemini explanation (via helper) ---
    explanation = explain_sustainability_change(
        project_type=project_type,
        material=material,
        landuse=landuse_info,
        area=area,
        height=height,
        current_score=current_score,
        predicted_score=predicted_score,
        delta=delta_pred,
        ndvi=feature_dict["NDVI"],
        ndbi=feature_dict["NDBI"],
        lst=feature_dict["LST"],
        albedo=feature_dict["ALBEDO"]
    )

    result["explanation"] = explanation
    return jsonify(result)



# ------------------------------------------------------------------------
# --- OPTIONAL: GET TRAFFIC INFO ----------------------------------------
# ------------------------------------------------------------------------
@app.route("/get_traffic_info", methods=["GET"])
def get_traffic_info():
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))
    api_key = "lF9KC9U2sLDxh9u8IlQoILfdkvJfetna"

    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/json?point={lat},{lon}&key={api_key}"
    try:
        data = requests.get(url).json()
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

# ------------------------------------------------------------------------
# --- MAIN ---------------------------------------------------------------
# ------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
