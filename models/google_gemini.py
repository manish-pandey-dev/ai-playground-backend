# models/google_gemini.py

import requests

def ask_gemini(api_key, model_param, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_param}:generateContent?key={api_key}"
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}]
        }
    )
    result = response.json()
    candidates = result.get("candidates", [])
    if candidates:
        return candidates[0]["content"]["parts"][0]["text"]
    else:
        return "No response from Gemini."
