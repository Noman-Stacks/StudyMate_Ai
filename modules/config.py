"""Environment-based application configuration."""

import os

from dotenv import load_dotenv

load_dotenv()


def get_gemini_api_key() -> str | None:
    """Return the Gemini API key from environment variables or Streamlit secrets."""
    value = os.getenv("GEMINI_API_KEY")
    if value:
        return value

    try:
        import streamlit as st

        return st.secrets.get("GEMINI_API_KEY")
    except (FileNotFoundError, ModuleNotFoundError):
        return None


def get_gemini_model() -> str:
    """Return the configured Gemini model name."""
    value = os.getenv("GEMINI_MODEL")
    if value:
        return value

    try:
        import streamlit as st

        return st.secrets.get("GEMINI_MODEL", "gemini-3.8-flash")
    except (FileNotFoundError, ModuleNotFoundError):
        return "gemini-3.8-flash"
