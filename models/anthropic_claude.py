# models/anthropic_claude.py

import requests

def ask_claude(api_key, model_param, prompt):
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        },
        json={
            "model": model_param,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024
        }
    )
    result = response.json()
    return result.get("content", [{}])[0].get("text", "")
