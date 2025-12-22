"""
Configuration file for the Princess Storybook Generator
Copy this file to config.py and add your OpenAI API key
"""
import os

# OpenAI API Key - Get yours from https://platform.openai.com/api-keys
OPENAI_API_KEY = "your-api-key-here"

# Template image paths (relative to project root)
TEMPLATE_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "A True Princess",
    "A True Princess",
    "A True Princess PNG"
)

# Number of pages in the book
TOTAL_PAGES = 1  # Set to 33 for full book

# Image generation settings
IMAGE_SIZE = "1024x1024"  # DALL-E 3 supports 1024x1024, 1792x1024, or 1024x1792
IMAGE_QUALITY = "standard"  # "standard" or "hd"

# PDF settings
PDF_PAGE_SIZE = (8.5 * 72, 11 * 72)  # Letter size in points (width, height)
PDF_IMAGE_WIDTH = 7.5 * 72  # 7.5 inches wide
PDF_IMAGE_HEIGHT = 10 * 72  # 10 inches tall

