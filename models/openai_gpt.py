# models/openai_gpt.py

import requests

def ask_gpt(api_key, model_param, prompt):
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model_param,
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    result = response.json()
    return result["choices"][0]["message"]["content"]
