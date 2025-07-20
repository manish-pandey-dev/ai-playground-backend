# health_checker.py

import requests
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import gspread
from google.oauth2.service_account import Credentials
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Import your model functions
from models.anthropic_claude import ask_claude
from models.google_gemini import ask_gemini
from models.openai_gpt import ask_gpt
from models.deepseek import ask_deepseek
from models.mistral import ask_mistral
from models.grok import ask_grok
from models.huggingface import ask_huggingface

logger = logging.getLogger(__name__)


class APIHealthChecker:
    def __init__(self):
        self.test_prompt = "Hello! This is a test message. Please respond with 'Test successful' if you can read this."
        self.max_tokens = 50
        self.results = []
        self.recipient_email = "pandeyelectric@gmail.com"

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

    def is_huggingface_model(self, model_name):
        return "/" in model_name  # HF models use format like "meta-llama/Llama-2-7b-chat-hf"

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
            elif self.is_huggingface_model(model_name):
                response = ask_huggingface(api_key, model_name, self.test_prompt, self.max_tokens)
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

    def generate_email_report(self, health_data: Dict[str, Any]) -> str:
        """Generate HTML email report from health check data"""
        results = health_data.get("results", [])
        summary = health_data.get("summary", {})

        # Count status types
        success_count = len([r for r in results if "SUCCESS" in r["status"]])
        failed_count = len([r for r in results if "FAILED" in r["status"]])
        credit_issues = len([r for r in results if "NO_CREDITS" in r["status"]])
        auth_issues = len([r for r in results if "AUTH_ERROR" in r["status"]])
        rate_limited = len([r for r in results if "RATE_LIMITED" in r["status"]])

        # Generate HTML report
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #f5f5f5; margin: 0; padding: 20px; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; padding: 20px; border-radius: 8px 8px 0 0; text-align: center; }}
                .summary {{ padding: 20px; background: #f8fafc; border-bottom: 1px solid #e2e8f0; }}
                .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-top: 15px; }}
                .summary-card {{ background: white; padding: 15px; border-radius: 6px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                .success {{ color: #10b981; }}
                .error {{ color: #ef4444; }}
                .warning {{ color: #f59e0b; }}
                .details {{ padding: 20px; }}
                .model-item {{ display: flex; justify-content: space-between; align-items: center; padding: 12px; border-bottom: 1px solid #e2e8f0; }}
                .model-item:last-child {{ border-bottom: none; }}
                .model-name {{ font-weight: bold; }}
                .status-success {{ color: #10b981; font-weight: bold; }}
                .status-error {{ color: #ef4444; font-weight: bold; }}
                .status-warning {{ color: #f59e0b; font-weight: bold; }}
                .response-time {{ color: #6b7280; font-size: 12px; }}
                .error-details {{ color: #ef4444; font-size: 11px; margin-top: 4px; }}
                .footer {{ padding: 15px; background: #f8fafc; border-radius: 0 0 8px 8px; text-align: center; color: #6b7280; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧠 AI Playground Health Report</h1>
                    <p>Comprehensive API Status Check</p>
                    <p>{datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
                </div>

                <div class="summary">
                    <h2>📊 Summary</h2>
                    <div class="summary-grid">
                        <div class="summary-card">
                            <div class="success" style="font-size: 24px; font-weight: bold;">{success_count}</div>
                            <div>✅ Working</div>
                        </div>
                        <div class="summary-card">
                            <div class="error" style="font-size: 24px; font-weight: bold;">{failed_count}</div>
                            <div>❌ Failed</div>
                        </div>
                        <div class="summary-card">
                            <div class="warning" style="font-size: 24px; font-weight: bold;">{credit_issues}</div>
                            <div>💳 No Credits</div>
                        </div>
                        <div class="summary-card">
                            <div class="warning" style="font-size: 24px; font-weight: bold;">{auth_issues}</div>
                            <div>🔑 Auth Issues</div>
                        </div>
                        <div class="summary-card">
                            <div style="font-size: 24px; font-weight: bold; color: #3b82f6;">{summary.get('success_rate', 0)}%</div>
                            <div>📈 Success Rate</div>
                        </div>
                    </div>
                </div>

                <div class="details">
                    <h2>🔍 Detailed Results</h2>
        """

        # Add each model result
        for result in results:
            status_class = "status-success" if "SUCCESS" in result["status"] else "status-error"
            if "RATE_LIMITED" in result["status"] or "NO_CREDITS" in result["status"] or "AUTH_ERROR" in result[
                "status"]:
                status_class = "status-warning"

            html_report += f"""
                    <div class="model-item">
                        <div>
                            <div class="model-name">{result['model']}</div>
                            <div class="response-time">{result['response_time_ms']}ms</div>
                            {f'<div class="error-details">{result.get("error", "")}</div>' if result.get("error") else ""}
                        </div>
                        <div class="{status_class}">{result['status']}</div>
                    </div>
            """

        html_report += f"""
                </div>

                <div class="footer">
                    <p>🚀 AI Playground Health Monitor | Generated automatically</p>
                    <p>Platform Status: {"🟢 Healthy" if summary.get('success_rate', 0) >= 80 else "🟡 Needs Attention" if summary.get('success_rate', 0) >= 50 else "🔴 Critical Issues"}</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html_report

    def send_email_report(self, health_data: Dict[str, Any]) -> bool:
        """Send health check report via email using Gmail SMTP"""
        try:
            # Email configuration
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = os.environ.get("GMAIL_SENDER_EMAIL", "your-email@gmail.com")
            sender_password = os.environ.get("GMAIL_APP_PASSWORD", "")  # App-specific password

            if not sender_password:
                logger.warning("Gmail app password not configured - skipping email")
                return False

            summary = health_data.get("summary", {})
            success_rate = summary.get("success_rate", 0)

            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = f"🧠 AI Playground Health Report - {success_rate}% Success Rate"
            message["From"] = sender_email
            message["To"] = self.recipient_email

            # Generate HTML content
            html_content = self.generate_email_report(health_data)

            # Create plain text version
            text_content = f"""
AI Playground Health Report
{datetime.now().strftime("%B %d, %Y at %I:%M %p")}

Summary:
- Working Models: {len([r for r in health_data.get('results', []) if 'SUCCESS' in r['status']])}
- Failed Models: {len([r for r in health_data.get('results', []) if 'FAILED' in r['status']])}
- Success Rate: {success_rate}%

Detailed Results:
"""
            for result in health_data.get("results", []):
                text_content += f"- {result['model']}: {result['status']} ({result['response_time_ms']}ms)\n"
                if result.get('error'):
                    text_content += f"  Error: {result['error']}\n"

            text_content += f"\nGenerated by AI Playground Health Monitor"

            # Attach both versions
            text_part = MIMEText(text_content, "plain")
            html_part = MIMEText(html_content, "html")

            message.attach(text_part)
            message.attach(html_part)

            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(message)

            logger.info(f"✅ Health report email sent successfully to {self.recipient_email}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to send email report: {str(e)}")
            return False

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

        # Send email report
        email_sent = self.send_email_report({
            "success": True,
            "results": results,
            "summary": summary
        })

        return {
            "success": True,
            "results": results,
            "summary": summary,
            "email_sent": email_sent
        }


def run_api_health_check():
    """Main function to run the health check"""
    checker = APIHealthChecker()
    return checker.run_health_check()