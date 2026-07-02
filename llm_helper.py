import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi"
def _call_ollama(prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"Ollama error: {response.text}"

    except Exception as e:
        return f"Ollama connection failed: {e}"


def get_sustainability_explanation(score, ndvi, ndbi, albedo, lst):
    prompt = f"""
    You are an environmental sustainability expert.
    NDVI → Normalized Difference Vegetation Index (vegetation health)
    NDBI → Normalized Difference Built-up Index (urban/built-up areas)
    LST → Land Surface Temperature (surface temperature)
    Albedo → Surface Albedo (surface reflectivity)
    Sustainability Score: {score}
    NDVI: {ndvi}
    NDBI: {ndbi}
    Albedo: {albedo}
    LST: {lst}

    Explain:
    - What these indicators mean but dont give their formulae or any such expressions or in very low terms
    - Why the sustainability score is high or low by mentioning if its high or low
    - Environmental interpretation

    Keep under 100 words and simple without formulae or technical expressions.
    Dont use emojis or any such symbols.
    Do not include any words saying sure or happy to help, or any such expressions.
    """

    return _call_ollama(prompt)


def explain_sustainability_change(
    project_type, material, landuse,
    area, height,
    current_score, predicted_score, delta,
    ndvi, ndbi, lst, albedo
):
    prompt = f"""
    You are an environmental impact analyst.

    Project Type: {project_type}
    Material: {material}
    Landuse: {landuse}
    Area: {area}
    Height: {height}

    Current Score: {current_score}
    Predicted Score: {predicted_score}
    Change: {delta}

    Environmental Indicators:
    NDVI: {ndvi}
    NDBI: {ndbi}
    LST: {lst}
    Albedo: {albedo}

    Explain clearly:
    - Why the score changes
    - Environmental reasoning
    - Whether impact is positive or negative

    Keep under 300 words and simple without formulae or technical expressions.
    """

    return _call_ollama(prompt)