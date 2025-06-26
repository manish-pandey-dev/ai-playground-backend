# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import requests

# Import model functions
from models import openai_gpt, anthropic_claude, google_gemini

app = FastAPI()

# Load Google Sheet
def load_sheet():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name("google-creds.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open("ai_models_configurations").sheet1
    return sheet.get_all_records()

# Request body
class AIRequest(BaseModel):
    model: str
    prompt: str

@app.post("/ask-ai")
async def ask_ai(req: AIRequest):
    records = load_sheet()
    selected = next((r for r in records if r["Model Name"] == req.model), None)
    if not selected:
        return {"error": "Model not found"}

    api_url = selected["API URL"]
    api_key = selected["API Key"]
    model_param = selected["Model Parameter"]

    if "openai.com" in api_url:
        reply = openai_gpt.ask_gpt(api_key, model_param, req.prompt)
    elif "anthropic.com" in api_url:
        reply = anthropic_claude.ask_claude(api_key, model_param, req.prompt)
    elif "googleapis.com" in api_url:
        reply = google_gemini.ask_gemini(api_key, model_param, req.prompt)
    else:
        reply = "Unsupported API."

    return {"response": reply}
