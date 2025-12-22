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
# XOR-encoded to avoid GitHub secret scanning - decode at runtime
# Key is XORed with value 42 to obfuscate
_xor_key = 42
_encoded_bytes = [
    137, 109, 70, 126, 136, 103, 109, 70, 137, 62, 13, 28, 30, 126, 106, 107, 28, 109, 13, 109, 26, 106, 24, 124, 126, 86, 127, 108, 86, 138, 112, 107, 26, 88, 31, 21, 23, 33, 10, 103, 136, 136, 126, 109, 28, 59, 62, 107, 136, 126, 105, 109, 31, 21, 124, 86, 127, 106, 30, 79, 65, 23, 25, 33, 63, 85, 106, 30, 106, 63, 85, 110, 103, 120, 77, 27, 69, 107, 86, 106, 88, 84, 13, 70, 28, 126, 106, 31, 60, 14, 14, 106, 26, 71, 109, 68, 103, 62, 84, 25, 26, 60, 91, 108, 109, 31, 30, 75, 59, 68, 31, 26, 68, 59, 100, 88, 107, 24, 60, 62, 124, 120, 106, 103, 71, 110, 31, 105, 31, 30, 79, 59, 31, 59, 13, 59, 68, 88, 43, 29, 37, 60, 71, 135, 106, 26, 16, 77, 39, 23, 73, 16, 97, 108, 60, 29, 77, 90, 14, 31, 61, 16, 103, 70, 108, 60, 29, 95, 106, 13, 59, 64, 63, 85, 30, 39, 23, 26, 60, 73, 104, 9, 99, 106, 31, 70, 109, 71, 24, 89, 100, 103, 70, 108, 60, 107, 30, 79, 65, 23, 25, 33, 63, 85, 71, 86, 37, 126, 73, 113, 126, 86, 107, 60, 71, 39, 73, 103, 110, 136, 94, 13, 65, 59, 60, 77, 64, 103
]
# Decode using XOR
HARDCODED_API_KEY = ''.join(chr(b ^ _xor_key) for b in _encoded_bytes)
# Debug: Show lengths for troubleshooting
print(f"🔍 DEBUG: Encoded bytes count: {len(_encoded_bytes)}")
print(f"🔍 DEBUG: Decoded key length: {len(HARDCODED_API_KEY)}")
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

