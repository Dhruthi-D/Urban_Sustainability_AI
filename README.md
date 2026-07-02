# Urban Sustainability AI

An AI-powered Flask web application for analyzing urban sustainability in Bangalore, India. The app combines satellite-derived environmental indicators, OpenStreetMap vector layers, a LightGBM impact prediction model, traffic data, and local Ollama-based LLM explanations.

## Overview

Urban Sustainability AI helps users inspect the environmental condition of a location and simulate the sustainability impact of proposed development projects. Users can open an interactive map, click locations in Bangalore, search predefined places, compare project outcomes, generate reports, and receive plain-language explanations from a local Ollama model.

The current Ollama-enabled application entry point is:

```bash
python app2.py
```

Older Gemini-based files are still present (`app1.py`, `gemini_helper.py`), but `app2.py`, `llm_helper.py`, and `ollama_helper.py` are the local LLM path.

## Features

### Interactive Sustainability Map

- Real-time sustainability scoring from map clicks.
- Raster indicator lookup for NDVI, NDBI, Albedo, and Land Surface Temperature.
- Vector feature detection using buildings, roads, waterways, natural areas, land use, places, points, and railways.
- Place search for predefined areas such as RVCE, PES, Lalbagh, BMS, and CMRIT.
- Local Ollama explanations for sustainability scores.
- TomTom traffic lookup for current speed, free-flow speed, confidence, and road closure status.

### Project Impact Prediction

- LightGBM model predicts the sustainability score delta for proposed development.
- Supports project type, construction material, area, and height inputs.
- Computes current and predicted sustainability scores.
- Uses Ollama to explain why a proposed project may improve or reduce sustainability.

### Reports

- Multi-location report generation.
- Sustainability score comparison across selected places.
- Ollama-generated report insights.
- Optional PDF export using ReportLab.

## Ollama Integration

This project can run AI explanations locally with Ollama instead of depending on a cloud LLM.

### Ollama Files

```text
llm_helper.py       # Main Ollama helper for sustainability and impact explanations
ollama_helper.py    # Generic Ollama prompt helper used by report generation
app2.py             # Flask app wired to the Ollama helpers
```

### Ollama Configuration

Both Ollama helper files currently use:

```python
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi"
```

You can change `MODEL_NAME` in `llm_helper.py` and `ollama_helper.py` if you want to use another locally installed model, such as `llama3`.

### Install And Run Ollama

1. Install Ollama from: https://ollama.com/download
2. Start Ollama. On most systems it runs automatically as a local service on port `11434`.
3. Pull the model used by this project:

   ```bash
   ollama pull phi
   ```

4. Check that Ollama is working:

   ```bash
   ollama run phi
   ```

5. In another terminal, run the Flask app:

   ```bash
   python app2.py
   ```

6. Open `http://localhost:5000`.

### Ollama API Behavior

The app sends POST requests to:

```text
http://localhost:11434/api/generate
```

with a JSON body like:

```json
{
  "model": "phi",
  "prompt": "Your prompt text",
  "stream": false
}
```

The helpers return the `response` field from Ollama. If Ollama is not running, the UI/API will return messages such as:

```text
Ollama connection failed: ...
Local LLM (Ollama) unavailable.
```

### What Ollama Explains

- Current sustainability score from `/explain_sustainability`.
- Project impact prediction from `/predict_project_impact`.
- Detailed impact explanation from `/get_impact_explanation`.
- Multi-place report insight from `/general_report`.

## Quick Start

### Prerequisites

- Python 3.8+
- Git
- Ollama installed and running
- Local Ollama model, currently `phi`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ganashree-pm26/Urban-Sustainability-AI.git
   cd Urban-Sustainability-AI
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   Windows:

   ```bash
   venv\Scripts\activate
   ```

   macOS/Linux:

   ```bash
   source venv/bin/activate
   ```

3. Install Python dependencies:

   ```bash
   pip install flask rasterio geopandas lightgbm pandas numpy requests pyproj shapely reportlab google-generativeai python-dotenv
   ```

4. Install and prepare Ollama:

   ```bash
   ollama pull phi
   ```

5. Add required data files.

   Large geospatial/model files may not be included in Git. Make sure these files exist:

   ```text
   data_raw/Stack_Bangalore_Full_Fix.tif
   data_raw/bbbike/buildings_clipped.shp
   data_raw/bbbike/landuse_clipped.shp
   data_raw/bbbike/natural_clipped.shp
   data_raw/bbbike/places_clipped.shp
   data_raw/bbbike/points_clipped.shp
   data_raw/bbbike/railways_clipped.shp
   data_raw/bbbike/roads_clipped.shp
   data_raw/bbbike/waterways_clipped.shp
   delta_score_lgb_chunked.txt
   training_dataset.csv
   ```

6. Run the Ollama-enabled app:

   ```bash
   python app2.py
   ```

7. Open the app at `http://localhost:5000`.

## Project Structure

```text
urban-sustainability-ai/
├── app2.py                  # Current Ollama-enabled Flask application
├── llm_helper.py            # Ollama helper for score and project explanations
├── ollama_helper.py         # Generic Ollama helper for report insights
├── app1.py                  # Older Gemini-enabled Flask application
├── app.py                   # Earlier app version
├── gemini_helper.py         # Gemini helper retained for older app path
├── backend.py               # Backend processing logic
├── train_lgb_batch.py       # LightGBM training script
├── obtain_data.py           # Data acquisition/processing script
├── temp.py                  # Utility script
├── templates/
│   ├── home.html            # Landing/home screen
│   ├── index.html           # Sustainability map interface
│   ├── impact.html          # Project impact interface
│   ├── report.html          # Report generation interface
│   ├── script.js            # Frontend JavaScript
│   └── style.css            # Styling
├── data_raw/                # Raw geospatial data
│   ├── Stack_Bangalore_Full_Fix.tif
│   └── bbbike/
│       ├── buildings_clipped.shp
│       ├── landuse_clipped.shp
│       ├── natural_clipped.shp
│       ├── places_clipped.shp
│       ├── points_clipped.shp
│       ├── railways_clipped.shp
│       ├── roads_clipped.shp
│       └── waterways_clipped.shp
├── data_processed/          # Processed raster outputs
├── delta_score_lgb_chunked.txt
├── training_dataset.csv
└── README.md
```

## API Endpoints

### `GET /`

Home page.

### `GET /sustainability`

Interactive sustainability map page.

### `GET /get_params?lat={lat}&lon={lon}`

Returns sustainability parameters for a coordinate, including coordinates, raster bands, detected vector features, land use type, natural feature information, sustainability score, and highlighted GeoJSON.

### `POST /explain_sustainability`

Uses Ollama to explain a sustainability score.

Expected body includes:

```json
{
  "score": 75,
  "landuse": "recreation ground",
  "features": ["Natural"],
  "coordinates": {"lat": 12.95, "lon": 77.58},
  "raster_bands": {
    "Band_1": 0.4,
    "Band_2": 0.1,
    "Band_3": 0.2,
    "Band_4": 28
  }
}
```

### `GET /search_place?name={place_name}`

Searches predefined place boxes or matching vector layer names.

Predefined names:

```text
rvce, pes, lalbagh, bms, cmrit
```

### `GET /impact`

Project impact prediction page.

### `POST /predict_project_impact`

Predicts the environmental impact of a proposed project and returns an Ollama explanation.

Expected body:

```json
{
  "lat": 12.95,
  "lon": 77.58,
  "project_type": "park",
  "material": "Concrete",
  "area": 500,
  "height": 10
}
```

Returns input features, predicted score delta, current score, predicted score, positive/negative impact label, and Ollama explanation.

### `POST /get_impact_explanation`

Gets an Ollama explanation for project impact details.

### `GET /report`

Report generation page.

### `POST /general_report`

Generates sustainability comparison results and Ollama insight for multiple locations.

### `POST /download_pdf`

Generates a PDF report from insight text and base64 chart images. Requires `reportlab`.

### `GET /get_traffic_info?lat={lat}&lon={lon}`

Fetches TomTom traffic flow information for a coordinate.

## Machine Learning Model

The project uses a LightGBM regression model for impact prediction.

- Model file: `delta_score_lgb_chunked.txt`
- Training script: `train_lgb_batch.py`
- Training data: `training_dataset.csv`
- Target: sustainability score delta
- Features: NDVI, NDBI, Albedo, LST, land use, project type, material, area, and height

To retrain:

```bash
python train_lgb_batch.py
```

## Sustainability Scoring

The sustainability score is computed on a 0-100 scale using:

- NDVI: higher vegetation generally improves score.
- NDBI: higher built-up intensity generally reduces score.
- LST: higher surface temperature reduces score.
- Albedo: reflectivity affects thermal behavior.
- Vector features: buildings/roads reduce score, natural/water features improve score.
- Land use: recreation grounds receive a small bonus.

The scoring function is in `app2.py`:

```text
compute_sustainability_score()
```

## Data Requirements

### Raster Data

- File: `data_raw/Stack_Bangalore_Full_Fix.tif`
- Format: GeoTIFF
- Bands used by the app:
  - `Band_1`: NDVI
  - `Band_2`: NDBI
  - `Band_3`: Albedo
  - `Band_4`: LST
- CRS expected by app/vector processing: EPSG:32643

### Vector Data

Vector files are read from `data_raw/bbbike/` and converted to EPSG:32643 when possible.

Required clipped shapefile layers:

- Buildings
- Landuse
- Natural
- Places
- Points
- Railways
- Roads
- Waterways

## API Keys

### TomTom Traffic API

Purpose: traffic flow lookup.

`app2.py` reads:

```text
TOMTOM_API_KEY
```

If the environment variable is not set, the code currently falls back to a hardcoded key in `app2.py`.

PowerShell example:

```powershell
$env:TOMTOM_API_KEY="your_key_here"
python app2.py
```

### Gemini API

Gemini is not required for the Ollama-enabled `app2.py` workflow. It is still used by the older `app1.py` / `gemini_helper.py` path.

If running the older Gemini version, configure:

```text
GEMINI_API_KEY
```

## Troubleshooting Ollama

### `Ollama connection failed`

Make sure Ollama is running:

```bash
ollama list
```

Then try:

```bash
ollama run phi
```

### Model Not Found

Pull the configured model:

```bash
ollama pull phi
```

If you changed `MODEL_NAME`, pull that model too.

### Slow Explanations

Local inference speed depends on CPU/GPU, RAM, and selected model. Smaller models such as `phi` are faster, while larger models may produce better explanations but take longer.

### Changing Models

Edit both files if you want all Ollama routes to use the same model:

```text
llm_helper.py
ollama_helper.py
```

Change:

```python
MODEL_NAME = "phi"
```

to another local model name.

## Development Notes

- Add new project types in the training data/model pipeline, then retrain `delta_score_lgb_chunked.txt`.
- Adjust score weights in `compute_sustainability_score()` in `app2.py`.
- Add predefined place searches by editing `PLACE_BOUNDARIES` in `app2.py`.
- Keep Ollama running before testing LLM features.
- Keep large raster/model files out of Git if they exceed repository limits.

## License

See `LICENSE`.

## Note

This project is intended for research and educational use. Before production use, review API key handling, error handling, data validation, model validation, and security settings.
