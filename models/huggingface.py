# models/huggingface.py

import requests
import logging
import time

logger = logging.getLogger(__name__)


def ask_huggingface(api_key, model_param, prompt, max_tokens=150):
    """
    Call Hugging Face Inference API with enhanced error handling

    Args:
        api_key (str): Hugging Face API key (free tier available)
        model_param (str): Model name (e.g., 'meta-llama/Llama-2-7b-chat-hf')
        prompt (str): User prompt
        max_tokens (int): Maximum tokens in response

    Returns:
        str: Hugging Face model response text

    Raises:
        requests.exceptions.RequestException: For API errors
    """

    logger.info(f"Calling Hugging Face API with model: {model_param}")
    logger.info(f"Max tokens: {max_tokens}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Format prompt for chat models
    if "chat" in model_param.lower() or "instruct" in model_param.lower():
        formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
    else:
        formatted_prompt = prompt

    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False
        },
        "options": {
            "wait_for_model": True,
            "use_cache": False
        }
    }

    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model_param}",
            headers=headers,
            json=payload,
            timeout=60  # HF models can take time to load
        )

        logger.info(f"Hugging Face API response status: {response.status_code}")

        # Handle model loading (503 status)
        if response.status_code == 503:
            try:
                error_data = response.json()
                if "loading" in error_data.get("error", "").lower():
                    logger.info(f"Model {model_param} is loading, waiting...")
                    time.sleep(10)  # Wait for model to load
                    # Retry once
                    response = requests.post(
                        f"https://api-inference.huggingface.co/models/{model_param}",
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
            except:
                pass

        # Check for rate limits
        if response.status_code == 429:
            error_data = response.json() if response.content else {}
            error_message = error_data.get("error", "Rate limit exceeded")
            logger.error(f"Hugging Face rate limit: {error_message}")
            raise requests.exceptions.RequestException(f"Hugging Face rate limit: {error_message}")

        response.raise_for_status()
        result = response.json()

        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                # Standard text generation response
                response_text = result[0].get("generated_text", "")
            else:
                # Some models return different formats
                response_text = str(result[0])
        elif isinstance(result, dict):
            # Some models return dict format
            response_text = result.get("generated_text", result.get("text", ""))
        else:
            response_text = str(result)

        # Clean up the response
        response_text = response_text.strip()

        # Remove the original prompt if it's included
        if response_text.startswith(formatted_prompt):
            response_text = response_text[len(formatted_prompt):].strip()

        if response_text:
            logger.info(f"Hugging Face response length: {len(response_text)} characters")
            return response_text
        else:
            logger.warning("Empty response from Hugging Face")
            return "No response from Hugging Face model."

    except requests.exceptions.Timeout:
        logger.error("Hugging Face API timeout (model may be loading)")
        raise requests.exceptions.RequestException("Hugging Face API timeout - model may be loading")
    except requests.exceptions.RequestException as e:
        logger.error(f"Hugging Face API error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calling Hugging Face: {str(e)}")
        raise requests.exceptions.RequestException(f"Unexpected error calling Hugging Face: {str(e)}")


def is_huggingface_model(model_name):
    """Check if model is a Hugging Face model"""
    huggingface_models = [
        'meta-llama/Llama-2-7b-chat-hf',
        'meta-llama/Llama-2-13b-chat-hf',
        'codellama/CodeLlama-7b-Instruct-hf',
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'microsoft/DialoGPT-medium',
        'HuggingFaceH4/zephyr-7b-beta',
        'microsoft/Phi-3-mini-4k-instruct',
        'Qwen/Qwen1.5-7B-Chat',
        'google/flan-t5-large',
        'bigscience/bloom-7b1'
    ]
    return model_name in huggingface_models or "/" in model_name


def get_huggingface_models():
    """Get list of popular Hugging Face models"""
    return [
        'meta-llama/Llama-2-7b-chat-hf',
        'meta-llama/Llama-2-13b-chat-hf',
        'codellama/CodeLlama-7b-Instruct-hf',
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'HuggingFaceH4/zephyr-7b-beta',
        'microsoft/Phi-3-mini-4k-instruct',
        'Qwen/Qwen1.5-7B-Chat',
        'google/flan-t5-large'
    ]