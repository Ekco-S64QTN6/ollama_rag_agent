import functools
import logging
import os
import requests
import config
from typing import Any, Dict, List, Optional, Tuple, Union


logger = logging.getLogger(__name__)

# Cache for Ollama model availability checks
_model_cache: Dict[str, Tuple[str, Optional[str]]] = {}

# Color Percentage Calculation
def get_color_for_percentage(percent: Union[int, float]) -> str:
    if not isinstance(percent, (int, float)):
        return config.COLOR_RESET

    if percent <= 70:
        return config.COLOR_GREEN
    elif 70 < percent <= 80:
        return config.COLOR_YELLOW
    else:
        return config.COLOR_RED

# Ollama Model Availability Check
@functools.lru_cache(maxsize=32)
def check_ollama_model_availability(model_name: str, fallback_model: Optional[str] = None) -> Tuple[str, Optional[str]]:
    cache_key = (model_name, fallback_model)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    try:
        ollama_models_response = requests.get("http://localhost:11434/api/tags", timeout=config.TIMEOUT_SECONDS)
        ollama_models_response.raise_for_status()
        available_models = [m['name'] for m in ollama_models_response.json().get('models', [])]

        if model_name in available_models:
            _model_cache[cache_key] = (model_name, None)
            return model_name, None
        else:
            logger.warning(f"Configured Ollama model '{model_name}' not found. Available models: {available_models}")
            if fallback_model and fallback_model in available_models:
                logger.warning(f"Attempting to use fallback model '{fallback_model}'.")
                _model_cache[cache_key] = (fallback_model, None)
                return fallback_model, None
            else:
                if 'llama2:7b-chat' in available_models:
                    logger.warning("Attempting to use 'llama2:7b-chat' as a generic fallback.")
                    _model_cache[cache_key] = ('llama2:7b-chat', None)
                    return 'llama2:7b-chat', None
                elif 'mistral:instruct' in available_models:
                    logger.warning("Attempting to use 'mistral:instruct' as a generic fallback.")
                    _model_cache[cache_key] = ('mistral:instruct', None)
                    return 'mistral:instruct', None
                else:
                    error_msg = f"No suitable Ollama model found. Configured: '{model_name}', Fallback: '{fallback_model}'. Available: {available_models}"
                    logger.error(error_msg)
                    _model_cache[cache_key] = ("", error_msg)
                    return "", error_msg
    except requests.exceptions.ConnectionError:
        error_msg = "Could not connect to Ollama server. Please ensure Ollama is running."
        logger.error(error_msg)
        _model_cache[cache_key] = ("", error_msg)
        return "", error_msg
    except requests.exceptions.Timeout:
        error_msg = "Ollama server connection timed out."
        logger.error(error_msg)
        _model_cache[cache_key] = ("", error_msg)
        return "", error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred while checking Ollama models: {e}"
        logger.error(error_msg, exc_info=True)
        _model_cache[cache_key] = ("", error_msg)
        return "", error_msg
