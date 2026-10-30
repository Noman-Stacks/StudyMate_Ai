"""Streamlit entry point for StudyMate AI."""

import sys
from pathlib import Path

# Get the project root directory
ROOT_DIR = Path(__file__).resolve().parent

# Add project root to Python's module search path
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import the Streamlit application
from src.ui import render_app


def main() -> None:
    """Start the StudyMate AI application."""
    render_app()


if __name__ == "__main__":
    main()

