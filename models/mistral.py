# models/mistral.py

import requests
import logging

logger = logging.getLogger(__name__)


def ask_mistral(api_key, model_param, prompt, max_tokens=150):
    """
    Call Mistral AI API with enhanced error handling

    Args:
        api_key (str): Mistral API key
        model_param (str): Model name (e.g., 'mistral-large-latest', 'mistral-small-latest')
        prompt (str): User prompt
        max_tokens (int): Maximum tokens in response

    Returns:
        str: Mistral response text

    Raises:
        requests.exceptions.RequestException: For API errors
    """

    logger.info(f"Calling Mistral API with model: {model_param}")
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
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        logger.info(f"Mistral API response status: {response.status_code}")

        # Check for quota/rate limit errors
        if response.status_code == 429:
            error_data = response.json() if response.content else {}
            error_message = error_data.get("error", {}).get("message", "Rate limit exceeded")
            logger.error(f"Mistral rate limit/quota: {error_message}")
            raise requests.exceptions.RequestException(f"Mistral quota/rate limit: {error_message}")

        response.raise_for_status()
        result = response.json()

        # Extract response
        choices = result.get("choices", [])
        if choices and len(choices) > 0:
            response_text = choices[0].get("message", {}).get("content", "")
            logger.info(f"Mistral response length: {len(response_text)} characters")
            return response_text.strip()
        else:
            logger.warning("No choices in Mistral response")
            return "No response from Mistral."

    except requests.exceptions.Timeout:
        logger.error("Mistral API timeout")
        raise requests.exceptions.RequestException("Mistral API timeout")
    except requests.exceptions.RequestException as e:
        logger.error(f"Mistral API error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calling Mistral: {str(e)}")
        raise requests.exceptions.RequestException(f"Unexpected error calling Mistral: {str(e)}")


def is_mistral_model(model_name):
    """Check if model is a Mistral model"""
    mistral_models = [
        'mistral-large-latest', 
        'mistral-small-latest',
        'mistral-medium-latest',
        'mistral-tiny',
        'mistral-7b-instruct',
        'mixtral-8x7b-instruct',
        'mixtral-8x22b-instruct'
    ]
    return model_name in mistral_models or model_name.startswith('mistral-') or model_name.startswith('mixtral-')


def get_mistral_models():
    """Get list of available Mistral models"""
    return [
        'mistral-large-latest',
        'mistral-small-latest', 
        'mistral-medium-latest',
        'mixtral-8x7b-instruct',
        'mixtral-8x22b-instruct'
    ]