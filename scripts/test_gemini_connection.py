"""Run a redacted live connectivity check against the configured Gemini model."""

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.config import get_gemini_api_key, get_gemini_model


def main() -> int:
    """Send a test prompt and print the model response, never the API key."""
    api_key = get_gemini_api_key()
    if not api_key:
        print("Gemini connection test failed: GEMINI_API_KEY is not configured.")
        return 1

    try:
        from google import genai

        model_name = get_gemini_model()
        client = genai.Client(api_key=api_key)
        response: Any = client.models.generate_content(
            model=model_name,
            contents="Explain artificial intelligence in two sentences.",
        )
        response_text = response.text.strip()
    except Exception as error:
        error_name = type(error).__name__.lower()
        error_message = str(error).lower()
        if "unauthenticated" in error_message or "api key" in error_message:
            category = "authentication error"
        elif (
            "resourceexhausted" in error_name
            or "quota" in error_message
            or "rate" in error_message
        ):
            category = "quota or rate-limit error"
        elif (
            "timeout" in error_message
            or "connection" in error_message
            or "network" in error_message
        ):
            category = "network error"
        else:
            category = "Gemini API error"
        print(f"Gemini connection test failed ({category}): {error}")
        return 1

    if not response_text:
        print("Gemini connection test failed: the response was empty.")
        return 1

    print("Gemini SDK: google-genai")
    print(f"Gemini model: {model_name}")
    print("GEMINI_API_KEY available: yes")
    print("Gemini response:")
    print(response_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
