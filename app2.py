# app2.py — Cleaned, robust, ready-to-run version
import os
import random
import base64
from io import BytesIO
from ollama_helper import generate_explanation
from flask import Flask, request, jsonify, render_template, send_file
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Point, Polygon
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
# LLM helpers using Ollama (local model)

try:
    from llm_helper import (
        explain_sustainability_change,
        get_sustainability_explanation
    )
except Exception as e:
    print(f"Warning: Ollama LLM helper not available: {e}")

    def explain_sustainability_change(*args, **kwargs):
        return "Local LLM (Ollama) unavailable."

    def get_sustainability_explanation(*args, **kwargs):
        return "Local LLM (Ollama) unavailable."

# PDF generation deps
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
except Exception:
    # We'll raise at runtime if /download_pdf is used and reportlab missing.
    SimpleDocTemplate = Paragraph = Spacer = RLImage = getSampleStyleSheet = A4 = None

app = Flask(__name__)

# ---------------- CONFIG ----------------
RASTER_PATH = "data_raw/Stack_Bangalore_Full_Fix.tif"
MODEL_PATH = "delta_score_lgb_chunked.txt"

PLACE_BOUNDARIES = {
    "rvce": {"lat_min": 12.9210, "lat_max": 12.9250, "lon_min": 77.4980, "lon_max": 77.5035},
    "pes": {"lat_min": 12.9315, "lat_max": 12.9360, "lon_min": 77.5315, "lon_max": 77.5355},
    "lalbagh": {"lat_min": 12.9462, "lat_max": 12.9527, "lon_min": 77.5830, "lon_max": 77.5900},
    "bms": {"lat_min": 12.9404, "lat_max": 12.9428, "lon_min": 77.5651, "lon_max": 77.5669},
    "cmrit": {"lat_min": 12.9657, "lat_max": 12.9671, "lon_min": 77.7109, "lon_max": 77.7120},
}

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

# ---------------- LOAD RASTER ----------------
dataset = None
try:
    dataset = rasterio.open(RASTER_PATH)
    print(f"Opened raster: {RASTER_PATH} (bands={dataset.count}, size={dataset.width}x{dataset.height})")
except Exception as e:
    print(f"Warning: Could not open raster '{RASTER_PATH}': {e}")
    dataset = None

# ---------------- LOAD VECTOR LAYERS ----------------
layers = {}
for name, path in vector_files.items():
    if not os.path.exists(path):
        print(f"Warning: vector file not found: {path} — skipping {name}")
        continue
    try:
        gdf = gpd.read_file(path)
        # normalize CRS to UTM 32643 (EPSG:32643) if possible
        if gdf.crs and str(gdf.crs) != "EPSG:32643":
            gdf = gdf.to_crs("EPSG:32643")
        layers[name] = gdf
        print(f"Loaded vector layer: {name} ({len(gdf)} features)")
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")

# ---------------- TRANSFORMER (WGS84 -> UTM 43N) ----------------
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)

# ---------------- LOAD MODEL ----------------
bst = None
if os.path.exists(MODEL_PATH):
    try:
        bst = lgb.Booster(model_file=MODEL_PATH)
        print("Loaded LightGBM model:", MODEL_PATH)
    except Exception as e:
        print(f"Warning: Could not load LightGBM model '{MODEL_PATH}': {e}")
else:
    print(f"Warning: Model file not found: {MODEL_PATH}")

# ---------------- Utility: safe raster read ----------------
def safe_raster_values_at_xy(x, y):
    """
    Given raster coords x,y (same CRS as raster), return dict of band values.
    Returns None if dataset missing or point out of bounds.
    """
    if dataset is None:
        return None, "Raster dataset not available."

    try:
        # dataset.index returns (row, col)
        row, col = dataset.index(x, y)
    except Exception as e:
        return None, f"Could not compute raster index: {e}"

    if not (0 <= row < dataset.height and 0 <= col < dataset.width):
        return None, "Point outside raster bounds."

    try:
        values = {}
        for i in range(1, dataset.count + 1):
            arr = dataset.read(i)
            val = arr[row, col]
            # convert masked/nodata to numpy.nan safely
            try:
                if np.ma.is_masked(val):
                    val = float("nan")
                else:
                    val = float(val)
            except Exception:
                try:
                    val = float(val)
                except Exception:
                    val = float("nan")
            values[f"Band_{i}"] = val
        return values, None
    except IndexError:
        return None, "Raster read index error (out of range)."
    except Exception as e:
        return None, f"Raster read error: {e}"

# ---------------- SUSTAINABILITY SCORE ----------------
def compute_sustainability_score(raster_values, features_present, landuse_type):
    # default score baseline
    score = 50
    ndvi = float(raster_values.get("Band_1", 0)) if raster_values else 0
    ndbi = float(raster_values.get("Band_2", 0)) if raster_values else 0
    albedo = float(raster_values.get("Band_3", 0)) if raster_values else 0
    lst = float(raster_values.get("Band_4", 0)) if raster_values else 0

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

# ---------------- COMPUTE PARAMS (reusable) ----------------
def compute_params(lat, lon):
    """
    Returns a dict with the same structure used by /get_params.
    Will raise ValueError for invalid inputs or missing data.
    """
    # Transform lat/lon -> raster CRS (UTM)
    try:
        x, y = transformer.transform(lon, lat)
    except Exception as e:
        raise ValueError(f"Coordinate transformation failed: {e}")

    raster_values, err = safe_raster_values_at_xy(x, y)
    if raster_values is None:
        raise ValueError(err or "Raster read failed.")

    # Vector features
    point = Point(x, y)
    features_present = []
    highlighted_geojson = {}
    landuse_info = None
    natural_type = None
    natural_name = None

    for name, gdf in layers.items():
        try:
            matches = gdf[gdf.intersects(point)]
            if not matches.empty:
                features_present.append(name)
                if name.lower() == "landuse":
                    if "type" in matches.columns:
                        lu = str(matches.iloc[0].get("type", "")).strip()
                        if lu == "recreation_groun":
                            lu = "recreation ground"
                        landuse_info = lu or None
                if name.lower() == "natural":
                    if "type" in matches.columns:
                        natural_type = str(matches.iloc[0].get("type", "")).strip()
                    if "name" in matches.columns:
                        natural_name = str(matches.iloc[0].get("name", "")).strip()
                # don't include buildings geojson if huge, but keep others
                if name != "Buildings":
                    highlighted_geojson[name] = matches.to_crs("EPSG:4326").__geo_interface__
                
        except Exception:
            # don't break entire routine if one layer errors
            continue

    score = compute_sustainability_score(raster_values, features_present, landuse_info)
    return {
        "coordinates": {"lat": lat, "lon": lon},
        "raster_bands": raster_values,
        "features_present": features_present,
        "landuse_type": landuse_info,
        "natural_type": natural_type,
        "name": natural_name,
        "sustainability_score": score,
        "highlighted_geojson": highlighted_geojson
    }

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/sustainability")
def index():
    return render_template("index.html")

@app.route("/get_params", methods=["GET"])
def get_params():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except Exception:
        return jsonify({"error": "Invalid or missing lat/lon"}), 400

    try:
        result = compute_params(lat, lon)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500

@app.route("/explain_sustainability", methods=["POST"])
def explain_sustainability():
    data = request.get_json() or {}
    score = data.get("score")
    landuse = data.get("landuse")
    features = data.get("features", [])
    coordinates = data.get("coordinates", {})
    raster_bands = data.get("raster_bands", {})

    ndvi = raster_bands.get("Band_1")
    ndbi = raster_bands.get("Band_2")
    albedo = raster_bands.get("Band_3")
    lst = raster_bands.get("Band_4")

    explanation = get_sustainability_explanation(score, ndvi, ndbi, albedo, lst)
    return jsonify({
        "score": score,
        "landuse": landuse,
        "features": features,
        "coordinates": coordinates,
        "explanation": explanation
    })

@app.route("/search_place", methods=["GET"])
def search_place():
    place_name = request.args.get("name", "").lower().strip()
    if not place_name:
        return jsonify({"error": "No place name provided"}), 400

    # bbox shortcut
    if place_name in PLACE_BOUNDARIES:
        bbox = PLACE_BOUNDARIES[place_name]
        scores = []
        for _ in range(30):
            lat = random.uniform(bbox["lat_min"], bbox["lat_max"])
            lon = random.uniform(bbox["lon_min"], bbox["lon_max"])
            try:
                x, y = transformer.transform(lon, lat)
                raster_values, err = safe_raster_values_at_xy(x, y)
                if raster_values is None:
                    continue
                score = compute_sustainability_score(raster_values, [], None)
                scores.append(score)
            except Exception:
                continue

        if not scores:
            return jsonify({"error": "No valid sample points within bbox"}), 400

        avg_score = round(sum(scores) / len(scores), 2)
        rect_coords = [
            [bbox["lat_min"], bbox["lon_min"]],
            [bbox["lat_min"], bbox["lon_max"]],
            [bbox["lat_max"], bbox["lon_max"]],
            [bbox["lat_max"], bbox["lon_min"]],
            [bbox["lat_min"], bbox["lon_min"]]
        ]
        geom_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[ [lon, lat] for lat, lon in rect_coords ]]},
                "properties": {"place": place_name}
            }]
        }
        return jsonify({
            "place": place_name,
            "method": "bbox",
            "samples": len(scores),
            "average_score": avg_score,
            "geometry": geom_geojson
        })

    # fallback polygonal search on loaded vector layers
    found_geom = None
    found_layer = None
    for layer_name, gdf in layers.items():
        if "name" in gdf.columns:
            matches = gdf[gdf["name"].astype(str).str.lower().str.contains(place_name)]
            if not matches.empty:
                found_geom = matches.iloc[0].geometry
                found_layer = layer_name
                break

    if found_geom is None:
        return jsonify({"error": f"No match found for '{place_name}'"}), 404

    if not isinstance(found_geom, Polygon):
        return jsonify({"error": "Geometry not polygonal"}), 400

    scores = []
    minx, miny, maxx, maxy = found_geom.bounds
    # sample up to 20 successful points (tries limited)
    attempts = 0
    while len(scores) < 20 and attempts < 2000:
        attempts += 1
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        p = Point(x, y)
        if not found_geom.contains(p):
            continue
        try:
            raster_values, err = safe_raster_values_at_xy(x, y)
            if raster_values is None:
                continue
            score = compute_sustainability_score(raster_values, [found_layer], None)
            scores.append(score)
        except Exception:
            continue

    if not scores:
        return jsonify({"error": "No valid sample points in polygon"}), 400

    avg_score = round(sum(scores) / len(scores), 2)
    geom_geojson = gpd.GeoSeries([found_geom]).set_crs(gdf.crs).to_crs("EPSG:4326").__geo_interface__
    return jsonify({
        "place": place_name,
        "method": "polygon",
        "layer": found_layer,
        "samples": len(scores),
        "average_score": avg_score,
        "geometry": geom_geojson
    })

@app.route("/impact")
def impact_page():
    return render_template("impact.html")

@app.route("/predict_project_impact", methods=["POST"])
def predict_project_impact():
    data = request.get_json() or {}
    try:
        lat, lon = float(data["lat"]), float(data["lon"])
    except Exception:
        return jsonify({"error": "Invalid lat/lon"}), 400

    project_type = data.get("project_type", "unknown")
    material = data.get("material", "unknown")
    area = float(data.get("area", 500))
    height = float(data.get("height", 10))

    # raster values
    x, y = transformer.transform(lon, lat)
    raster_values, err = safe_raster_values_at_xy(x, y)
    if raster_values is None:
        return jsonify({"error": err}), 400

    # landuse
    landuse_info = None
    landuse_gdf = layers.get("Landuse")
    if landuse_gdf is not None:
        p = Point(x, y)
        matches = landuse_gdf[landuse_gdf.intersects(p)]
        if not matches.empty and "type" in matches.columns:
            lu = str(matches.iloc[0].get("type", "")).strip()
            if lu == "recreation_groun":
                lu = "recreation ground"
            landuse_info = lu or None

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

    # Prepare df aligned to model
    if bst is None:
        return jsonify({"error": "Model not loaded."}), 500

    df = pd.DataFrame([feature_dict])
    for col in ["Landuse", "Project_Type", "Material"]:
        df[col] = df[col].astype("category")

    train_features = bst.feature_name()
    for col in train_features:
        if col not in df.columns:
            df[col] = 0
    df = df[train_features]

    try:
        delta_pred = float(bst.predict(df, num_iteration=bst.best_iteration)[0])
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {e}"}), 500

    current_score = compute_sustainability_score(raster_values, [], landuse_info)
    predicted_score = max(0, min(100, current_score + delta_pred))

    explanation = explain_sustainability_change(
        project_type=project_type, material=material, landuse=landuse_info,
        area=area, height=height, current_score=current_score,
        predicted_score=predicted_score, delta=delta_pred,
        ndvi=feature_dict["NDVI"], ndbi=feature_dict["NDBI"],
        lst=feature_dict["LST"], albedo=feature_dict["ALBEDO"]
    )

    return jsonify({
        "input": feature_dict,
        "predicted_delta": round(delta_pred, 2),
        "current_score": current_score,
        "predicted_score": round(predicted_score, 2),
        "impact": "Positive" if delta_pred > 0 else "Negative",
        "explanation": explanation
    })

@app.route("/get_impact_explanation", methods=["POST"])
def get_impact_explanation():
    data = request.get_json() or {}
    try:
        explanation = explain_sustainability_change(**data)
        return jsonify({"explanation": explanation})
    except Exception:
        return jsonify({"explanation": "LLM explanation unavailable right now."})

@app.route("/report")
def report_page():
    return render_template("report.html")

@app.route("/general_report", methods=["POST"])
def general_report():
    data = request.get_json() or {}
    places = data.get("places", [])
    if not places:
        return jsonify({"error": "No places selected"}), 400

    results = []
    for p in places:
        lat, lon = p.get("lat"), p.get("lon")
        try:
            params = compute_params(lat, lon)
            score = params["sustainability_score"]
            results.append({
                "name": p.get("name", "point"),
                "score": score,
                "ndvi": params["raster_bands"].get("Band_1"),
                "ndbi": params["raster_bands"].get("Band_2"),
                "albedo": params["raster_bands"].get("Band_3"),
                "lst": params["raster_bands"].get("Band_4")
            })
        except Exception as e:
            results.append({
                "name": p.get("name", "point"),
                "error": str(e)
            })

    # -------- AI insight using Ollama --------
    try:
        text = "Compare sustainability scores:\n" + \
            "\n".join([f"{r.get('name')}: {r.get('score')}" for r in results if "score" in r])

        prompt = f"""
        You are an expert environmental analyst.

        Compare the sustainability scores of these locations:

        {text}

        Explain:
        - Which location performs best
        - Which performs worst
        - Why (based on environmental indicators)
        - Keep under 500 words.
        """

        ai_response = generate_explanation(prompt)

    except Exception as e:
        ai_response = f"Ollama explanation failed: {e}"

    # IMPORTANT: RETURN RESPONSE
    return jsonify({
        "bars": results,
        "insight": ai_response
    })

@app.route("/impact_compare")
def impact_compare_page():
    return render_template("impact_compare.html")

def _predict_for_point(lat, lon, project_type, material, area, height):
    """Helper used by impact_compare: returns before/after/delta and explanation."""
    params = compute_params(lat, lon)
    raster = params["raster_bands"]
    ndvi = raster.get("Band_1")
    ndbi = raster.get("Band_2")
    albedo = raster.get("Band_3")
    lst = raster.get("Band_4")

    feature_dict = {
        "NDVI": ndvi,
        "NDBI": ndbi,
        "ALBEDO": albedo,
        "LST": lst,
        "Landuse": params.get("landuse_type") or "unknown",
        "Project_Type": project_type,
        "Material": material,
        "Area": float(area),
        "Height": float(height)
    }

    df = pd.DataFrame([feature_dict])
    for col in ["Landuse", "Project_Type", "Material"]:
        df[col] = df[col].astype("category")

    train_features = bst.feature_name()
    for col in train_features:
        if col not in df.columns:
            df[col] = 0
    df = df[train_features]

    delta_pred = float(bst.predict(df, num_iteration=bst.best_iteration)[0])
    current_score = compute_sustainability_score(raster, [], feature_dict["Landuse"])
    predicted_score = max(0, min(100, current_score + delta_pred))

    try:
        explanation = explain_sustainability_change(
            project_type=project_type, material=material, landuse=feature_dict["Landuse"],
            area=area, height=height, current_score=current_score, predicted_score=predicted_score,
            delta=delta_pred, ndvi=ndvi, ndbi=ndbi, lst=lst, albedo=albedo
        )
    except Exception:
        explanation = "LLM explanation unavailable."

    return {
        "lat": lat,
        "lon": lon,
        "before": {"ndvi": ndvi, "ndbi": ndbi, "albedo": albedo, "lst": lst, "score": current_score},
        "delta_score": round(delta_pred, 3),
        "after": {"score": round(predicted_score, 3)},
        "explanation": explanation
    }

@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    if SimpleDocTemplate is None:
        return jsonify({"error": "reportlab not installed on server. Install 'reportlab' and restart."}), 500

    data = request.get_json() or {}
    insights = data.get("insights", "")
    graph1 = data.get("graph1", "")
    graph2 = data.get("graph2", "")

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Urban Sustainability Report", styles["Title"]))
    story.append(Spacer(1, 12))

    if insights:
        story.append(Paragraph("AI Insights:", styles["Heading3"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(insights.replace("\n", "<br/>"), styles["Normal"]))
        story.append(Spacer(1, 12))

    def add_b64_image(b64data, w=450, h=260):
        if not b64data or not isinstance(b64data, str) or not b64data.startswith("data:"):
            return
        try:
            header, b64 = b64data.split(",", 1)
            img_bytes = base64.b64decode(b64)
            img_buf = BytesIO(img_bytes)
            img = RLImage(img_buf, width=w, height=h)
            story.append(img)
            story.append(Spacer(1, 12))
        except Exception as e:
            story.append(Paragraph(f"Could not include an image: {e}", styles["Normal"]))
            story.append(Spacer(1, 8))

    add_b64_image(graph1)
    add_b64_image(graph2)

    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="Sustainability_Report.pdf", mimetype="application/pdf")

@app.route("/get_traffic_info", methods=["GET"])
def get_traffic_info():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except Exception:
        return jsonify({"error": "Invalid lat/lon"}), 400

    api_key = os.getenv("TOMTOM_API_KEY", "lF9KC9U2sLDxh9u8IlQoILfdkvJfetna")
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/json?point={lat},{lon}&key={api_key}"
    try:
        data = requests.get(url, timeout=6).json()
        flow = data.get("flowSegmentData", {})
        return jsonify({
            "currentSpeed": flow.get("currentSpeed"),
            "freeFlowSpeed": flow.get("freeFlowSpeed"),
            "confidence": flow.get("confidence"),
            "roadClosure": flow.get("roadClosure")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- Main ----------------
if __name__ == "__main__":
    # run dev server
    app.run(debug=True)
