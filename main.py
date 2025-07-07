from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import gspread
from google.oauth2.service_account import Credentials
import requests
import os
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta

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

# Rate limiting storage (in production, use Redis or database)
request_counts = defaultdict(list)
daily_request_count = 0
MAX_REQUESTS_PER_IP_PER_HOUR = 10  # Limit per IP
MAX_DAILY_REQUESTS = 100  # Total daily limit for your app
MAX_TOKENS_PER_REQUEST = 4000  # Prevent very expensive requests


def check_rate_limit(client_ip: str):
    """Check if client has exceeded rate limits"""
    global daily_request_count

    now = datetime.now()
    hour_ago = now - timedelta(hours=1)

    # Clean old requests for this IP
    request_counts[client_ip] = [req_time for req_time in request_counts[client_ip] if req_time > hour_ago]

    # Check per-IP hourly limit
    if len(request_counts[client_ip]) >= MAX_REQUESTS_PER_IP_PER_HOUR:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {MAX_REQUESTS_PER_IP_PER_HOUR} requests per hour per IP."
        )

    # Check daily global limit
    if daily_request_count >= MAX_DAILY_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Daily request limit exceeded. Maximum {MAX_DAILY_REQUESTS} requests per day."
        )

    # Record this request
    request_counts[client_ip].append(now)
    daily_request_count += 1


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

        # Debug: Print the sheet data structure
        if data:
            logger.info(f"First row keys: {list(data[0].keys())}")
            logger.info(f"First row data: {data[0]}")
        else:
            logger.info("No data found in the sheet!")

        logger.info("Successfully loaded model configurations from Google Sheet.")
        return data

    except Exception as e:
        logger.error(f"Error loading Google Sheet: {e}")
        raise


class AIRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 150  # Default small limit


@app.get("/")
def root():
    logger.info("Health check endpoint called")
    return {"status": "AI Playground is live", "daily_requests_used": daily_request_count}


@app.get("/models")
def list_models():
    try:
        logger.info("Loading model configurations for /models endpoint...")
        data = get_model_config()

        # Debug: Show what we got from the sheet
        logger.info(f"Sheet data type: {type(data)}")
        logger.info(f"Sheet data length: {len(data) if data else 0}")

        if not data:
            return {"error": "No data found in Google Sheet"}

        # Check if the data has the expected structure
        if isinstance(data, list) and len(data) > 0:
            first_row = data[0]
            logger.info(f"First row keys: {list(first_row.keys()) if isinstance(first_row, dict) else 'Not a dict'}")

            # Try to get model names more safely
            model_names = []
            for row in data:
                if isinstance(row, dict):
                    # Try different possible column names
                    model = row.get("model") or row.get("Model") or row.get("MODEL")
                    if model:
                        model_names.append(model)
                    else:
                        logger.warning(f"No 'model' field found in row: {row}")

            logger.info(f"Model list returned: {model_names}")
            return {"models": model_names}
        else:
            return {"error": "Invalid data structure from Google Sheet"}

    except Exception as e:
        logger.error(f"Error in /models endpoint: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"error": f"Failed to load models: {str(e)}"}


@app.post("/ask-ai")
def ask_ai(request: AIRequest, client_request: Request):
    try:
        # Get client IP for rate limiting
        client_ip = client_request.client.host

        # Apply rate limiting
        check_rate_limit(client_ip)

        # Validate token limit
        if request.max_tokens > MAX_TOKENS_PER_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens too high. Maximum allowed: {MAX_TOKENS_PER_REQUEST}"
            )

        # Validate prompt length (approximate token count)
        if len(request.prompt) > 10000:  # ~2500 tokens
            raise HTTPException(
                status_code=400,
                detail="Prompt too long. Maximum 10,000 characters."
            )

        logger.info(f"Received request from IP: {client_ip}")
        logger.info(f"Request for model: {request.model}")
        logger.info(f"Prompt length: {len(request.prompt)} characters")
        logger.info(f"Max tokens requested: {request.max_tokens}")

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
            ],
            "max_tokens": min(request.max_tokens, MAX_TOKENS_PER_REQUEST),  # Enforce limit
            "temperature": 0.7
        }

        logger.info("Making API request...")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        logger.info(f"API response status: {response.status_code}")

        # Check for quota errors specifically
        if response.status_code == 429:
            error_detail = response.json().get("error", {})
            if "quota" in error_detail.get("message", "").lower():
                logger.error("OpenAI quota exceeded - API spending limit reached")
                return {"error": "Service temporarily unavailable due to usage limits. Please try again later."}

        response.raise_for_status()
        result = response.json()

        ai_reply = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.info(f"AI response received successfully. Length: {len(ai_reply)} characters")

        return {
            "response": ai_reply,
            "model_used": request.model,
            "daily_requests_remaining": MAX_DAILY_REQUESTS - daily_request_count
        }

    except HTTPException:
        raise  # Re-raise HTTP exceptions (like rate limits)
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return {"error": "AI service temporarily unavailable. Please try again later."}
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return {"error": "Invalid response from AI service."}
    except Exception as e:
        logger.error(f"Unexpected error in ask_ai: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"error": "Internal server error. Please try again later."}


@app.get("/stats")
def get_stats():
    """Public endpoint to show current usage stats"""
    return {
        "daily_requests_used": daily_request_count,
        "daily_requests_remaining": MAX_DAILY_REQUESTS - daily_request_count,
        "rate_limit_per_ip_per_hour": MAX_REQUESTS_PER_IP_PER_HOUR,
        "max_tokens_per_request": MAX_TOKENS_PER_REQUEST
    }