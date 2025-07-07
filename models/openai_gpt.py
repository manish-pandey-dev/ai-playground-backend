# models/openai_gpt.py

import requests
import logging

logger = logging.getLogger(__name__)


def ask_gpt(api_key, model_param, prompt, max_tokens=150):
    """
    Call OpenAI API with enhanced error handling

    Args:
        api_key (str): OpenAI API key
        model_param (str): Model name (e.g., 'gpt-3.5-turbo')
        prompt (str): User prompt
        max_tokens (int): Maximum tokens in response

    Returns:
        str: GPT response text

    Raises:
        requests.exceptions.RequestException: For API errors
    """

    logger.info(f"Calling OpenAI API with model: {model_param}")
    logger.info(f"Max tokens: {max_tokens}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_param,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        logger.info(f"OpenAI API response status: {response.status_code}")

        # Check for quota/rate limit errors
        if response.status_code == 429:
            error_data = response.json() if response.content else {}
            error_message = error_data.get("error", {}).get("message", "Rate limit exceeded")
            logger.error(f"OpenAI rate limit/quota: {error_message}")
            raise requests.exceptions.RequestException(f"OpenAI quota/rate limit: {error_message}")

        response.raise_for_status()
        result = response.json()

        # Extract response
        choices = result.get("choices", [])
        if choices and len(choices) > 0:
            response_text = choices[0].get("message", {}).get("content", "")
            logger.info(f"OpenAI response length: {len(response_text)} characters")
            return response_text
        else:
            logger.warning("No choices in OpenAI response")
            return "No response from OpenAI."

    except requests.exceptions.Timeout:
        logger.error("OpenAI API timeout")
        raise requests.exceptions.RequestException("OpenAI API timeout")
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calling OpenAI: {str(e)}")
        raise requests.exceptions.RequestException(f"Unexpected error calling OpenAI: {str(e)}")


def is_openai_model(model_name):
    """Check if model is an OpenAI model"""
    openai_models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini']
    return model_name in openai_models or model_name.startswith('gpt-')


def get_openai_models():
    """Get list of available OpenAI models"""
    return [
        'gpt-3.5-turbo',
        'gpt-4o',
        'gpt-4-turbo',
        'gpt-4',
        'gpt-4o-mini'
    ]