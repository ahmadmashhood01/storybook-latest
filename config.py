"""
Configuration file for the Princess Storybook Generator
Copy this file to config.py and add your OpenAI API key
"""
import os
import base64
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

# Hardcoded fallback key (always available) - FULL 219 CHARACTER KEY
# Split into two parts to prevent any truncation issues during file read
_key_part1 = "sk-proj-qHGBdugGkCyG0wVplvRBzgP2YnH13jrrw7SKdrw1n0XlvfSF31TUiIuBLS5gvnEO4cc2DvtgRWT3BlbkFJs_xFtPbo1WLKsDMn9WG9PL73-"
_key_part2 = "WY5pWud4QF9YfWpkJiUZ4L-ldHK1rY40J9vqn4b1tphfkFtMA"
# Concatenate parts to form the complete key
HARDCODED_API_KEY = _key_part1 + _key_part2
# Debug: Show lengths for troubleshooting
print(f"🔍 DEBUG: Key part1 length: {len(_key_part1)}, part2 length: {len(_key_part2)}")
print(f"🔍 DEBUG: Full key length: {len(HARDCODED_API_KEY)}")
print(f"🔍 DEBUG: Key preview: {HARDCODED_API_KEY[:20]}...{HARDCODED_API_KEY[-10:]}")

# Validate the hardcoded key is correct length
if len(HARDCODED_API_KEY) != 219:
    raise ValueError(f"CRITICAL: Hardcoded API key is corrupted! Expected 219 chars, got {len(HARDCODED_API_KEY)}")

def get_openai_api_key():
    """
    Get OpenAI API key with priority:
    1. Streamlit secrets (if valid and not truncated)
    2. Hardcoded fallback key (always works)
    
    Returns:
        str: The API key (219 characters)
    """
    # Priority 1: Try Streamlit secrets (per Streamlit Cloud best practices)
    try:
        import streamlit as st
        try:
            secret_key = st.secrets.get("OPENAI_API_KEY", None)
            if secret_key:
                secret_key = str(secret_key).strip()
                # Remove quotes if present
                if (secret_key.startswith('"') and secret_key.endswith('"')) or \
                   (secret_key.startswith("'") and secret_key.endswith("'")):
                    secret_key = secret_key[1:-1].strip()
                
                # Only use Streamlit secret if it's the full length (not truncated)
                if len(secret_key) >= 200 and secret_key.startswith("sk-"):
                    print(f"🔑 Using Streamlit secret (length: {len(secret_key)})")
                    return secret_key
                else:
                    print(f"⚠️ Streamlit secret is truncated ({len(secret_key)} chars), using hardcoded fallback")
        except Exception:
            pass
    except Exception:
        pass
    
    # Priority 2: Always use hardcoded fallback (reliable)
    # Double-check the key is correct before returning
    if len(HARDCODED_API_KEY) != 219:
        raise ValueError(f"CRITICAL: Hardcoded key corrupted! Expected 219, got {len(HARDCODED_API_KEY)}")
    
    print(f"🔑 Using hardcoded API key (length: {len(HARDCODED_API_KEY)})")
    if len(HARDCODED_API_KEY) != 219:
        print(f"❌ ERROR: Key length mismatch! Expected 219, got {len(HARDCODED_API_KEY)}")
        print(f"   Key preview: {HARDCODED_API_KEY[:50]}...{HARDCODED_API_KEY[-20:]}")
    
    return HARDCODED_API_KEY

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

