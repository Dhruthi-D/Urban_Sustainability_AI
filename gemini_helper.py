# gemini_helper.py
import google.generativeai as genai
import os

# ✅ Configure Gemini (make sure your API key is set)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

def get_sustainability_explanation(score, ndvi, ndbi, albedo, lst):
    """
    Generates a short, meaningful sustainability explanation using Gemini 2.0 Flash.
    """
    prompt = f"""
    Explain the sustainability score for an urban region given these parameters:

    Sustainability Score: {score}
    NDVI (Vegetation Index): {ndvi}
    NDBI (Built-up Index): {ndbi}
    Albedo (Surface Reflectivity): {albedo}
    LST (Land Surface Temperature): {lst}

    Describe:
    - Why the score is high or low
    - How vegetation, buildings, and temperature influence it
    - Keep it concise and readable in simple language and dont use the terms in parameters (use vegetation or builtup or reflectivity or temperature) (3–4 sentences max)
    """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Explanation unavailable: {e}"
