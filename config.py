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

# Hardcoded fallback key (always available)
# Split into two parts to prevent any truncation issues during file read
_key_part1 = "sk-proj-qHGBdugGkCyG0wVplvRBzgP2YnH13jrrw7SKdrw1n0XlvfSF31TUiIuBLS5gvnEO4cc2DvtgRWT3BlbkFJs_xFtPbo1WLKsDMn9WG9PL73-"
_key_part2 = "WY5pWud4QF9YfWpkJiUZ4L-ldHK1rY40J9vqn4b1tphfkFtMA"
# Concatenate parts to form the complete key
HARDCODED_API_KEY = _key_part1 + _key_part2

# Validate the hardcoded key has proper format (flexible length, must start with sk-)
if len(HARDCODED_API_KEY) < 50 or not HARDCODED_API_KEY.startswith("sk-"):
    raise ValueError(f"CRITICAL: Hardcoded API key is invalid! Must start with 'sk-' and be at least 50 chars. Got {len(HARDCODED_API_KEY)} chars, starts with: {HARDCODED_API_KEY[:10]}...")

def get_openai_api_key():
    """
    Get OpenAI API key with priority:
    1. Streamlit secrets (if valid and not truncated)
    2. Hardcoded fallback key (always works)
    
    Returns:
        str: The API key (starts with 'sk-', 50+ characters)
    """
    # Priority 1: Try Streamlit secrets (per Streamlit Cloud best practices)
    try:
        import streamlit as st
        try:
            # Try to get the full key from Streamlit secrets
            secret_key = st.secrets.get("OPENAI_API_KEY", None)
            if secret_key:
                secret_key = str(secret_key).strip()
                # Remove quotes if present
                if (secret_key.startswith('"') and secret_key.endswith('"')) or \
                   (secret_key.startswith("'") and secret_key.endswith("'")):
                    secret_key = secret_key[1:-1].strip()
                
                # If truncated, try to get parts and combine them
                if len(secret_key) < 200:
                    print(f"⚠️ Streamlit secret is truncated ({len(secret_key)} chars), trying parts...")
                    # Try to get key parts if they exist
                    part1 = st.secrets.get("OPENAI_API_KEY_PART1", None)
                    part2 = st.secrets.get("OPENAI_API_KEY_PART2", None)
                    if part1 and part2:
                        secret_key = str(part1).strip() + str(part2).strip()
                        print(f"🔍 Combined parts: part1={len(str(part1))}, part2={len(str(part2))}, total={len(secret_key)}")
                
                # Only use Streamlit secret if it's the full length (not truncated)
                if len(secret_key) >= 200 and secret_key.startswith("sk-"):
                    print(f"🔑 Using Streamlit secret (length: {len(secret_key)})")
                    return secret_key
                else:
                    print(f"⚠️ Streamlit secret is still truncated ({len(secret_key)} chars), using hardcoded fallback")
        except Exception as e:
            print(f"⚠️ Error reading Streamlit secret: {e}")
            pass
    except Exception:
        pass
    
    # Priority 2: Always use hardcoded fallback (reliable)
    # Double-check the key has valid format before returning
    if len(HARDCODED_API_KEY) < 50 or not HARDCODED_API_KEY.startswith("sk-"):
        raise ValueError(f"CRITICAL: Hardcoded key invalid! Must start with 'sk-' and be 50+ chars. Got {len(HARDCODED_API_KEY)} chars")
    
    print(f"🔑 Using hardcoded API key (length: {len(HARDCODED_API_KEY)})")
    
    return HARDCODED_API_KEY

# For backward compatibility, keep OPENAI_API_KEY but it may be stale
# New code should use get_openai_api_key() instead
try:
    OPENAI_API_KEY = get_openai_api_key()
    # Ensure it's not None
    if OPENAI_API_KEY is None:
        raise ValueError("get_openai_api_key() returned None - this should never happen")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to get API key: {e}")
    # Fallback to hardcoded key directly
    OPENAI_API_KEY = HARDCODED_API_KEY
    print(f"🔑 Using hardcoded key as fallback (length: {len(OPENAI_API_KEY)})")

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

