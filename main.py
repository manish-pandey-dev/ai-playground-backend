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

# Import your AI model modules
from models.anthropic_claude import ask_claude
from models.google_gemini import ask_gemini
from models.openai_gpt import ask_gpt
from models.deepseek import ask_deepseek

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
MAX_REQUESTS_PER_IP_PER_HOUR = 100  # Limit per IP
MAX_DAILY_REQUESTS = 1000  # Total daily limit for your app
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


def is_claude_model(model_name):
    """Check if model is a Claude model"""
    return model_name.startswith('claude-')


def is_gemini_model(model_name):
    """Check if model is a Gemini model"""
    return model_name.startswith('gemini-')


def is_openai_model(model_name):
    """Check if model is an OpenAI model"""
    openai_models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o']
    return model_name in openai_models or model_name.startswith('gpt-')

def is_deepseek_model(model_name):
    """Check if model is a DeepSeek model"""
    return model_name.startswith('deepseek-')

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

        api_key = selected["api_key"]
        logger.info(f"API key starts with: {api_key[:10]}..." if api_key else "No API key found")

        # Route to appropriate AI service based on model type
        try:
            if is_claude_model(request.model):
                logger.info("Using Claude API")
                ai_response = ask_claude(
                    api_key=api_key,
                    model_param=request.model,
                    prompt=request.prompt
                )
            elif is_gemini_model(request.model):
                logger.info("Using Gemini API")
                ai_response = ask_gemini(
                    api_key=api_key,
                    model_param=request.model,
                    prompt=request.prompt
                )
            elif is_openai_model(request.model):
                logger.info("Using OpenAI API")
                ai_response = ask_gpt(
                    api_key=api_key,
                    model_param=request.model,
                    prompt=request.prompt
                )
            elif is_deepseek_model(request.model):
                logger.info("Using DeepSeek API")
                ai_response = ask_deepseek(
                    api_key=api_key,
                    model_param=request.model,
                    prompt=request.prompt
                )
            else:
                logger.warning(f"Unknown model type: {request.model}")
                return {"error": f"Unsupported model type: {request.model}"}

            logger.info(f"AI response received successfully. Length: {len(ai_response)} characters")

            return {
                "response": ai_response,
                "model_used": request.model,
                "daily_requests_remaining": MAX_DAILY_REQUESTS - daily_request_count
            }

        except requests.exceptions.RequestException as api_error:
            logger.error(f"AI API call failed: {str(api_error)}")

            # Check for quota/rate limit errors
            error_msg = str(api_error).lower()
            if "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                return {"error": "Service temporarily unavailable due to usage limits. Please try again later."}
            else:
                return {"error": "AI service temporarily unavailable. Please try again later."}

    except HTTPException:
        raise  # Re-raise HTTP exceptions (like rate limits)
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


@app.get("/supported-models")
def get_supported_models():
    """Get information about supported AI models"""
    return {
        "claude_models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
        "openai_models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"],
        "gemini_models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
        "deepseek_models": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
        "note": "Add these models to your Google Sheet with appropriate API keys"
    }