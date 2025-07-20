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
        model_param (str): Model name (e.g., 'gpt2', 'facebook/bart-large-cnn')
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

    # Format prompt based on model type
    if "gpt2" in model_param.lower():
        # GPT-2 style text generation
        formatted_prompt = prompt
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }
    elif "distilbert-base-uncased-finetuned-sst-2-english" == model_param:
        # Sentiment analysis model
        formatted_prompt = prompt
        payload = {
            "inputs": formatted_prompt
        }
    elif "bart" in model_param.lower():
        # Summarization model
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": max_tokens,
                "min_length": 10,
                "do_sample": False
            }
        }
    elif "bert-base-uncased" == model_param:
        # Fill-mask model - add [MASK] if not present
        if "[MASK]" not in prompt:
            formatted_prompt = f"{prompt} [MASK]."
        else:
            formatted_prompt = prompt
        payload = {
            "inputs": formatted_prompt
        }
    elif "gemma" in model_param.lower():
        # Google Gemma chat model
        formatted_prompt = prompt
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }
    else:
        # Default format for other models
        formatted_prompt = prompt
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }

    payload["options"] = {
        "wait_for_model": True,
        "use_cache": False
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

        # Handle different response formats based on model type
        if isinstance(result, list) and len(result) > 0:
            first_result = result[0]

            if isinstance(first_result, dict):
                # Text generation models
                if "generated_text" in first_result:
                    response_text = first_result["generated_text"]
                # Sentiment analysis models
                elif "label" in first_result and "score" in first_result:
                    response_text = f"Sentiment: {first_result['label']} (confidence: {first_result['score']:.2f})"
                # Summarization models
                elif "summary_text" in first_result:
                    response_text = first_result["summary_text"]
                # Fill-mask models
                elif "token_str" in first_result:
                    response_text = f"Suggested word: {first_result['token_str']} (score: {first_result['score']:.3f})"
                else:
                    response_text = str(first_result)
            else:
                response_text = str(first_result)
        elif isinstance(result, dict):
            # Some models return dict format
            response_text = result.get("generated_text", result.get("text", str(result)))
        else:
            response_text = str(result)

        # Clean up the response
        response_text = response_text.strip()

        # Remove the original prompt if it's included (for text generation)
        if hasattr(locals(), 'formatted_prompt') and response_text.startswith(formatted_prompt):
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
    working_models = [
        'gpt2',  # Text generation
        'distilbert-base-uncased-finetuned-sst-2-english',  # Sentiment analysis
        'facebook/bart-large-cnn',  # Summarization
        'bert-base-uncased',  # Fill-mask
        'google/gemma-2-2b-it'  # Chat/Instruct model
    ]
    return model_name in working_models or "/" in model_name


def get_huggingface_models():
    """Get list of working Hugging Face models for text generation and classification"""
    return [
        'gpt2',  # Text generation
        'distilbert-base-uncased-finetuned-sst-2-english',  # Sentiment analysis
        'facebook/bart-large-cnn',  # Summarization
        'bert-base-uncased',  # Fill-mask
        'google/gemma-2-2b-it'  # Chat/Instruct model
    ]