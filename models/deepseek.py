# models/deepseek.py

import requests
import logging

logger = logging.getLogger(__name__)


def ask_deepseek(api_key, model_param, prompt, max_tokens=150):
    """
    Call DeepSeek API with enhanced error handling

    Args:
        api_key (str): DeepSeek API key
        model_param (str): Model name (e.g., 'deepseek-chat', 'deepseek-coder')
        prompt (str): User prompt
        max_tokens (int): Maximum tokens in response

    Returns:
        str: DeepSeek response text

    Raises:
        requests.exceptions.RequestException: For API errors
    """

    logger.info(f"Calling DeepSeek API with model: {model_param}")
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
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        logger.info(f"DeepSeek API response status: {response.status_code}")

        # Check for quota/rate limit errors
        if response.status_code == 429:
            error_data = response.json() if response.content else {}
            error_message = error_data.get("error", {}).get("message", "Rate limit exceeded")
            logger.error(f"DeepSeek rate limit/quota: {error_message}")
            raise requests.exceptions.RequestException(f"DeepSeek quota/rate limit: {error_message}")

        response.raise_for_status()
        result = response.json()

        # Extract response
        choices = result.get("choices", [])
        if choices and len(choices) > 0:
            response_text = choices[0].get("message", {}).get("content", "")
            logger.info(f"DeepSeek response length: {len(response_text)} characters")
            return response_text.strip()
        else:
            logger.warning("No choices in DeepSeek response")
            return "No response from DeepSeek."

    except requests.exceptions.Timeout:
        logger.error("DeepSeek API timeout")
        raise requests.exceptions.RequestException("DeepSeek API timeout")
    except requests.exceptions.RequestException as e:
        logger.error(f"DeepSeek API error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calling DeepSeek: {str(e)}")
        raise requests.exceptions.RequestException(f"Unexpected error calling DeepSeek: {str(e)}")


def is_deepseek_model(model_name):
    """Check if model is a DeepSeek model"""
    deepseek_models = ['deepseek-chat', 'deepseek-coder', 'deepseek-reasoner']
    return model_name in deepseek_models or model_name.startswith('deepseek-')


def get_deepseek_models():
    """Get list of available DeepSeek models"""
    return [
        'deepseek-chat',
        'deepseek-coder', 
        'deepseek-reasoner'
    ]