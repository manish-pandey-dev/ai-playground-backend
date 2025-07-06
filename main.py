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
        creds_dict = json.loads(credentials_json)
        logger.info("Credentials JSON parsed successfully.")

        # Define the required scopes for Google Sheets access
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/drive.readonly'
        ]
        logger.info(f"Using scopes: {scopes}")

        # Create credentials with the proper scopes
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        logger.info("Credentials created with proper scopes.")

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
    try:
        logger.info("Loading model configurations for /models endpoint...")
        data = get_model_config()
        model_names = [row["model"] for row in data]
        logger.info(f"Model list returned: {model_names}")
        return {"models": model_names}
    except Exception as e:
        logger.error(f"Error in /models endpoint: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"error": f"Failed to load models: {str(e)}"}


@app.post("/ask-ai")
def ask_ai(request: AIRequest):
    try:
        logger.info(f"Received request for model: {request.model}")
        logger.info(f"Received prompt: {request.prompt}")

        logger.info("Loading model configuration...")
        config = get_model_config()
        logger.info(f"Successfully loaded {len(config)} model configurations")

        selected = next((item for item in config if item["model"] == request.model), None)

        if not selected:
            logger.warning(f"Model '{request.model}' not found in config")
            return {"error": f"Model '{request.model}' not found"}

        url = selected["api_endpoint"]
        api_key = selected["api_key"]
        logger.info(f"Using endpoint: {url}")
        logger.info(f"API key starts with: {api_key[:10]}..." if api_key else "No API key found")

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

        logger.info("Making API request...")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        logger.info(f"API response status: {response.status_code}")

        response.raise_for_status()
        result = response.json()

        ai_reply = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.info(f"AI response received successfully. Length: {len(ai_reply)} characters")

        return {"response": ai_reply}

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return {"error": f"API request failed: {str(e)}"}
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return {"error": f"Invalid JSON response from API: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error in ask_ai: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"error": f"Internal server error: {str(e)}"}