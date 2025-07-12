# health_checker.py

import requests
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import gspread
from google.oauth2.service_account import Credentials
import os

# Import your model functions
from models.anthropic_claude import ask_claude
from models.google_gemini import ask_gemini
from models.openai_gpt import ask_gpt
from models.deepseek import ask_deepseek
from models.mistral import ask_mistral
from models.grok import ask_grok

logger = logging.getLogger(__name__)


class APIHealthChecker:
    def __init__(self):
        self.test_prompt = "Hello! This is a test message. Please respond with 'Test successful' if you can read this."
        self.max_tokens = 50
        self.results = []

    def get_model_config(self):
        """Load model configurations from Google Sheet"""
        try:
            logger.info("Loading model configurations from Google Sheet...")
            credentials_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
            if not credentials_json:
                raise Exception("GOOGLE_CREDENTIALS_JSON environment variable not set")

            creds_dict = json.loads(credentials_json)
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets.readonly',
                'https://www.googleapis.com/auth/drive.readonly'
            ]
            credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            client = gspread.authorize(credentials)
            sheet = client.open("ai_models_configurations").sheet1
            data = sheet.get_all_records()
            
            logger.info(f"Retrieved {len(data)} model configurations")
            return data
        except Exception as e:
            logger.error(f"Error loading Google Sheet: {e}")
            return []

    def is_claude_model(self, model_name):
        return model_name.startswith('claude-')

    def is_gemini_model(self, model_name):
        return model_name.startswith('gemini-')

    def is_openai_model(self, model_name):
        openai_models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini']
        return model_name in openai_models or model_name.startswith('gpt-')

    def is_deepseek_model(self, model_name):
        return model_name.startswith('deepseek-')

    def is_mistral_model(self, model_name):
        return model_name.startswith('mistral-') or model_name.startswith('mixtral-')

    def is_grok_model(self, model_name):
        return model_name.startswith('grok-')

    def test_single_model(self, model_name: str, api_key: str) -> Dict[str, Any]:
        """Test a single model and return results"""
        start_time = datetime.now()
        result = {
            "model": model_name,
            "status": "❌ FAILED",
            "response_time_ms": 0,
            "error": None,
            "response_preview": None,
            "timestamp": start_time.isoformat()
        }

        try:
            logger.info(f"Testing model: {model_name}")
            
            # Route to appropriate AI service
            if self.is_claude_model(model_name):
                response = ask_claude(api_key, model_name, self.test_prompt)
            elif self.is_gemini_model(model_name):
                response = ask_gemini(api_key, model_name, self.test_prompt)
            elif self.is_openai_model(model_name):
                response = ask_gpt(api_key, model_name, self.test_prompt, self.max_tokens)
            elif self.is_deepseek_model(model_name):
                response = ask_deepseek(api_key, model_name, self.test_prompt, self.max_tokens)
            elif self.is_mistral_model(model_name):
                response = ask_mistral(api_key, model_name, self.test_prompt, self.max_tokens)
            elif self.is_grok_model(model_name):
                response = ask_grok(api_key, model_name, self.test_prompt, self.max_tokens)
            else:
                raise Exception(f"Unknown model type: {model_name}")

            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000

            # Check if response is valid
            if response and len(response.strip()) > 0:
                result.update({
                    "status": "✅ SUCCESS",
                    "response_time_ms": round(response_time, 2),
                    "response_preview": response[:100] + "..." if len(response) > 100 else response
                })
                logger.info(f"✅ {model_name} - SUCCESS ({response_time:.0f}ms)")
            else:
                result.update({
                    "status": "⚠️ EMPTY_RESPONSE",
                    "response_time_ms": round(response_time, 2),
                    "error": "Received empty or invalid response"
                })
                logger.warning(f"⚠️ {model_name} - EMPTY_RESPONSE")

        except requests.exceptions.RequestException as e:
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            error_msg = str(e)
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                result["status"] = "⚠️ RATE_LIMITED"
            elif "quota" in error_msg.lower() or "insufficient" in error_msg.lower():
                result["status"] = "💳 NO_CREDITS"
            elif "unauthorized" in error_msg.lower() or "401" in error_msg:
                result["status"] = "🔑 AUTH_ERROR"
            
            result.update({
                "response_time_ms": round(response_time, 2),
                "error": error_msg
            })
            logger.error(f"❌ {model_name} - {result['status']}: {error_msg}")

        except Exception as e:
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000
            result.update({
                "response_time_ms": round(response_time, 2),
                "error": str(e)
            })
            logger.error(f"❌ {model_name} - FAILED: {str(e)}")

        return result

    def run_health_check(self) -> Dict[str, Any]:
        """Run health check on all models"""
        logger.info("🔍 Starting API Health Check...")
        
        # Load model configurations
        models = self.get_model_config()
        if not models:
            return {
                "success": False,
                "error": "Failed to load model configurations",
                "results": [],
                "summary": {}
            }

        results = []
        total_models = len(models)
        successful = 0
        failed = 0
        
        # Test each model
        for model_config in models:
            model_name = model_config.get("model")
            api_key = model_config.get("api_key")
            
            if not model_name or not api_key:
                results.append({
                    "model": model_name or "UNKNOWN",
                    "status": "❌ FAILED",
                    "error": "Missing model name or API key in configuration",
                    "response_time_ms": 0
                })
                failed += 1
                continue
            
            result = self.test_single_model(model_name, api_key)
            results.append(result)
            
            if result["status"] == "✅ SUCCESS":
                successful += 1
            else:
                failed += 1

        # Generate summary
        summary = {
            "total_models": total_models,
            "successful": successful,
            "failed": failed,
            "success_rate": round((successful / total_models) * 100, 1) if total_models > 0 else 0,
            "test_completed_at": datetime.now().isoformat()
        }

        logger.info(f"🏁 Health Check Complete: {successful}/{total_models} models working ({summary['success_rate']}%)")

        return {
            "success": True,
            "results": results,
            "summary": summary
        }


def run_api_health_check():
    """Main function to run the health check"""
    checker = APIHealthChecker()
    return checker.run_health_check()