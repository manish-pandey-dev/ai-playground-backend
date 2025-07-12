# models/grok.py

import requests
import logging

logger = logging.getLogger(__name__)


def ask_grok(api_key, model_param, prompt, max_tokens=150):
    """
    Call Grok X.AI API with enhanced error handling

    Args:
        api_key (str): Grok API key from xAI Console
        model_param (str): Model name (e.g., 'grok-4', 'grok-beta', 'grok-2-vision-012')
        prompt (str): User prompt
        max_tokens (int): Maximum tokens in response

    Returns:
        str: Grok response text

    Raises:
        requests.exceptions.RequestException: For API errors
    """

    logger.info(f"Calling Grok X.AI API with model: {model_param}")
    logger.info(f"Max tokens: {max_tokens}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_param,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False
    }

    try:
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        logger.info(f"Grok API response status: {response.status_code}")

        # Check for quota/rate limit errors
        if response.status_code == 429:
            error_data = response.json() if response.content else {}
            error_message = error_data.get("error", {}).get("message", "Rate limit exceeded")
            logger.error(f"Grok rate limit/quota: {error_message}")
            raise requests.exceptions.RequestException(f"Grok quota/rate limit: {error_message}")

        response.raise_for_status()
        result = response.json()

        # Extract response
        choices = result.get("choices", [])
        if choices and len(choices) > 0:
            response_text = choices[0].get("message", {}).get("content", "")
            logger.info(f"Grok response length: {len(response_text)} characters")
            return response_text.strip()
        else:
            logger.warning("No choices in Grok response")
            return "No response from Grok."

    except requests.exceptions.Timeout:
        logger.error("Grok API timeout")
        raise requests.exceptions.RequestException("Grok API timeout")
    except requests.exceptions.RequestException as e:
        logger.error(f"Grok API error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calling Grok: {str(e)}")
        raise requests.exceptions.RequestException(f"Unexpected error calling Grok: {str(e)}")


def is_grok_model(model_name):
    """Check if model is a Grok model"""
    grok_models = [
        'grok-4',
        'grok-3', 
        'grok-beta',
        'grok-2-vision-012',
        'grok-2-012'
    ]
    return model_name in grok_models or model_name.startswith('grok-')


def get_grok_models():
    """Get list of available Grok models"""
    return [
        'grok-4',
        'grok-3',
        'grok-beta',
        'grok-2-vision-012'
    ]