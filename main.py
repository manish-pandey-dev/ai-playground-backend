from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import gspread
from google.oauth2.service_account import Credentials
import requests
import os
import json

# ----------- Logging Setup -----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
# --------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to restrict domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Google Sheet with detailed logging
def get_model_config():
    try:
        logger.info("Starting to load GOOGLE_CREDENTIALS_JSON from environment variable...")
        credentials_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        if not credentials_json:
            logger.error("GOOGLE_CREDENTIALS_JSON not found in environment variables.")
            raise Exception("GOOGLE_CREDENTIALS_JSON environment variable not set")
        logger.info("GOOGLE_CREDENTIALS_JSON successfully retrieved.")

        logger.info("Parsing credentials JSON...")
        credentials = Credentials.from_service_account_info(json.loads(credentials_json))
        logger.info("Credentials parsed and service account authenticated.")

        logger.info("Authorizing gspread client...")
        client = gspread.authorize(credentials)
        logger.info("gspread client authorized successfully.")

        logger.info("Opening Google Sheet: ai_models_configurations")
        sheet = client.open("ai_models_configurations").sheet1
        logger.info("Sheet opened successfully. Fetching all records...")

        data = sheet.get_all_records()
        logger.info(f"Retrieved {len(data)} rows from the sheet.")

        logger.info("Successfully loaded model configurations from Google Sheet.")
        return data

    except Exception as e:
        logger.error(f"Error loading Google Sheet: {e}")
        raise


class AIRequest(BaseModel):
    model: str
    prompt: str

@app.get("/")
def root():
    logger.info("Health check endpoint called")
    return {"status": "AI Playground is live"}

@app.get("/models")
def list_models():
    data = get_model_config()
    model_names = [row["model"] for row in data]
    logger.info(f"Model list returned: {model_names}")
    return {"models": model_names}

@app.post("/ask-ai")
def ask_ai(request: AIRequest):
    logger.info(f"Received prompt for model: {request.model}")
    logger.info(f"Received prompt for model: {request.prompt}")
    config = get_model_config()
    selected = next((item for item in config if item["model"] == request.model), None)

    if not selected:
        logger.warning(f"Model '{request.model}' not found in config")
        return {"error": f"Model '{request.model}' not found"}

    try:
        url = selected["api_endpoint"]
        api_key = selected["api_key"]
        logger.info(f"Calling endpoint: {url}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": selected["model"],
            "messages": [
                {"role": "user", "content": request.prompt}
            ]
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        ai_reply = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.info(f"AI response: {ai_reply[:100]}...")

        return {"response": ai_reply}
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return {"error": str(e)}
