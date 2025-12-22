"""
Configuration file for the Princess Storybook Generator
Copy this file to config.py and add your OpenAI API key
"""
import os
from pathlib import Path

# Try to load .env file if it exists (for local development)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

# Dynamic function to get OpenAI API key - checks Streamlit secrets every time
# This ensures we always get the latest key from Streamlit secrets, not a cached value
def get_openai_api_key():
    """
    Dynamically retrieve OpenAI API key with priority:
    1. Streamlit secrets (for Streamlit Cloud)
    2. Environment variable OPENAI_API_KEY
    3. Default fallback key (for local testing)
    
    Returns:
        str: The API key
    """
    # Priority 1: Try Streamlit secrets (for Streamlit Cloud)
    try:
        import streamlit as st
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
            if api_key:
                return api_key
        except Exception:
            # Secrets file doesn't exist or key not found - continue to fallbacks
            pass
    except (ImportError, AttributeError, RuntimeError):
        # Not running in Streamlit or secrets not available
        pass
    
    # Priority 2: Environment variable
    api_key = os.getenv("OPENAI_API_KEY", None)
    if api_key:
        return api_key
    
    # Priority 3: Default fallback key (for local testing)
    return "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"

# For backward compatibility, keep OPENAI_API_KEY but it may be stale
# New code should use get_openai_api_key() instead
OPENAI_API_KEY = get_openai_api_key()

# Template image paths - Base directory for all book assets
# Override with env var BOOKS_BASE_DIR when deploying (e.g., /data/books on Render)
# Each book has its own folder with the structure:
#   - 00.png: Front cover only (for AI personalization)
#   - 0.png: Back cover only (unchanged)
#   - 1.png: Full wrap cover (front + spine + back) - used for spine extraction
#   - 2.png - 33.png: Interior pages

# Default to Books folder in repository (relative to this config file)
# Can be overridden with BOOKS_BASE_DIR environment variable
_default_books_dir = os.path.join(os.path.dirname(__file__), "Books")
BOOKS_BASE_DIR = os.getenv("BOOKS_BASE_DIR", _default_books_dir)

# Legacy path for backward compatibility
TEMPLATE_IMAGES_DIR = os.path.join(
    os.path.dirname(__file__),
    "A True Princess",
    "A True Princess",
    "A True Princess PNG"
)

# Number of pages in the book
# Full book = 33 pages (cover + 32 interior). App UI can choose cover-only mode.
TOTAL_PAGES = 33

# Image generation settings
IMAGE_SIZE = "1024x1024"  # DALL-E 3 supports 1024x1024, 1792x1024, or 1024x1792
IMAGE_QUALITY = "standard"  # "standard" or "hd"

# PDF settings
PDF_PAGE_SIZE = (8.5 * 72, 11 * 72)  # Letter size in points (width, height)
PDF_IMAGE_WIDTH = 7.5 * 72  # 7.5 inches wide
PDF_IMAGE_HEIGHT = 10 * 72  # 10 inches tall

