# tests/conftest.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Define the path to the .env file in the ai_service root directory
dotenv_path = Path(__file__).resolve().parent.parent / ".env"  # Go up two levels to ai_service root

# Load environment variables from the .env file
load_dotenv(dotenv_path)
