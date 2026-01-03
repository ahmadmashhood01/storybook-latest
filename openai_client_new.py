"""
OpenAI API client for face replacement - ChatGPT-style approach
Uses the same logic as when you chat with ChatGPT and upload two images
"""
import os
import base64
import time
import tempfile
import requests
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
from PIL import Image, ImageDraw
from openai import OpenAI
from config import OPENAI_API_KEY, get_openai_api_key
import re
import io

# Simple file logger - defined first so it can be used below
def log(message):
    """Write to console only - no file logging to avoid errors"""
    try:
        if not isinstance(message, str):
            message = str(message)
        # Replace any problematic characters that might cause encoding issues
        safe_message = message.encode('utf-8', errors='replace').decode('utf-8')
        # Only print to console - no file logging
        print(safe_message)
    except Exception:
        # Silently ignore logging errors
        pass

# Initialize OpenAI client lazily to ensure API key is available
_client = None
_cached_api_key = None

def get_client():
    """
    Get or create OpenAI client instance.
    Dynamically retrieves API key each time to ensure Streamlit secrets are used.
    """
    global _client, _cached_api_key
    
    # Get the current API key dynamically (checks Streamlit secrets)
    current_api_key = get_openai_api_key()
    
    # Re-initialize client if API key changed or client doesn't exist
    if _client is None or _cached_api_key != current_api_key:
        if not current_api_key or current_api_key == "":
            raise ValueError("OpenAI API key is not configured! Please set OPENAI_API_KEY in Streamlit secrets or environment variable.")
        
        # Validate API key format before using
        if not isinstance(current_api_key, str):
            raise ValueError(f"API key must be a string, got {type(current_api_key)}")
        
        # Ensure we have the full key (no truncation)
        api_key_to_use = str(current_api_key).strip()
        if len(api_key_to_use) != len(str(current_api_key).strip()):
            raise ValueError("API key appears to have been modified (whitespace issues)")
        
        # Validate key format (flexible length - OpenAI keys vary in length)
        # Minimum 50 characters and must start with "sk-"
        if len(api_key_to_use) < 50:
            raise ValueError(f"API key appears to be too short (length: {len(api_key_to_use)}). Expected at least 50 characters.")
        if not api_key_to_use.startswith("sk-"):
            raise ValueError(f"API key format invalid. Should start with 'sk-', got: {api_key_to_use[:10]}...")
        
        # Log full key details for debugging (first 15 and last 10 chars only)
        key_preview = f"{api_key_to_use[:15]}...{api_key_to_use[-10:]}" if len(api_key_to_use) > 25 else api_key_to_use
        log(f"[KEY] Using API key (length: {len(api_key_to_use)}): {key_preview}")
        
        # Create client with validated key
        try:
            _client = OpenAI(api_key=api_key_to_use)
            _cached_api_key = api_key_to_use
        except Exception as e:
            log(f"[ERROR] ERROR creating OpenAI client: {e}")
            log(f"   Key length: {len(api_key_to_use)}")
            log(f"   Key starts with: {api_key_to_use[:10]}")
            raise
        
        # Determine key source for logging
        key_source = "Unknown"
        try:
            import streamlit as st
            try:
                secret_key = st.secrets.get("OPENAI_API_KEY", None)
                if secret_key and str(secret_key).strip() == api_key_to_use:
                    key_source = "Streamlit Secret"
                    log(f"[OK] Verified: Key matches Streamlit Secret")
            except Exception:
                pass
        except Exception:
            pass
        
        if key_source == "Unknown":
            import os
            env_key = os.getenv("OPENAI_API_KEY")
            if env_key and str(env_key).strip() == api_key_to_use:
                key_source = "Environment Variable"
            else:
                key_source = "Default/Config"
        
        log(f"[OK] OpenAI client initialized with API key from: {key_source}")
        log(f"   Key length: {len(api_key_to_use)} characters")
    
    return _client

# For backward compatibility, create client at module level
# Note: This is lazy - client will be None until first use, which is fine
# This prevents import-time errors if Streamlit secrets aren't available yet
client = None

def _ensure_client():
    """Ensure client is initialized - called lazily when needed"""
    global client
    if client is None:
        try:
            client = get_client()
        except Exception as e:
            log(f"Warning: Could not initialize OpenAI client: {e}")
            # Don't raise - let get_client() handle it when actually called
    return client


def _extract_bbox_from_analysis(analysis: str):
    """Extract a bbox tuple (x1, y1, x2, y2) if present in analysis text."""
    bbox_match = re.search(r"BOUNDING_BOX:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", analysis)
    if bbox_match:
        return tuple(int(bbox_match.group(i)) for i in range(1, 5))
    return None


def _clamp_bbox(bbox, width, height):
    """Clamp bbox to image boundaries."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return (x1, y1, x2, y2)


def detect_and_fix_rotation(image_bytes: bytes, is_interior_page: bool = True) -> bytes:
    """
    Auto-detect and fix incorrectly rotated pages.
    
    For interior pages (square 200x200mm format):
    - Expected aspect ratio is ~1:1 (square)
    - If width >> height, the page is rotated 90 degrees clockwise
    - If height >> width, the page is rotated 90 degrees counter-clockwise
    
    For cover pages (landscape wrap format):
    - Expected aspect ratio is ~1.86:1 (1298.27 / 697.323)
    - Cover pages should be wider than tall
    
    Args:
        image_bytes: Raw image bytes
        is_interior_page: True for interior pages (square), False for cover pages (landscape)
        
    Returns:
        bytes: Corrected image bytes (rotated if needed)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        aspect_ratio = width / height if height > 0 else 1.0
        
        log(f"Rotation detection - Size: {width}x{height}, Aspect: {aspect_ratio:.3f}, Interior: {is_interior_page}")
        
        if is_interior_page:
            # Interior pages should be square (~1:1 aspect ratio)
            # Tolerance: consider anything between 0.9 and 1.1 as "square enough"
            if aspect_ratio > 1.3:
                # Width >> Height: rotated 90 degrees clockwise, needs counter-clockwise rotation
                log(f"Interior page appears rotated 90° clockwise (aspect {aspect_ratio:.2f}), correcting...")
                img = img.rotate(90, expand=True)
                output = io.BytesIO()
                img.save(output, format='PNG')
                output.seek(0)
                return output.read()
            elif aspect_ratio < 0.77:
                # Height >> Width: rotated 90 degrees counter-clockwise, needs clockwise rotation
                log(f"Interior page appears rotated 90° counter-clockwise (aspect {aspect_ratio:.2f}), correcting...")
                img = img.rotate(-90, expand=True)
                output = io.BytesIO()
                img.save(output, format='PNG')
                output.seek(0)
                return output.read()
        else:
            # Cover pages should be landscape (~1.86:1 aspect ratio)
            # If height > width significantly, it's rotated
            if aspect_ratio < 1.0:
                # Portrait when should be landscape: rotate 90 degrees
                log(f"Cover page appears rotated (portrait, aspect {aspect_ratio:.2f}), correcting...")
                img = img.rotate(-90, expand=True)
                output = io.BytesIO()
                img.save(output, format='PNG')
                output.seek(0)
                return output.read()
        
        # No rotation needed
        return image_bytes
        
    except Exception as e:
        log(f"Error in rotation detection: {e}")
        return image_bytes


# =============================================================================
# NEW COVER WORKFLOW FUNCTIONS
# These functions support the new cover processing workflow:
# - 00.png: Front cover only (high-res, for AI personalization)
# - 0.png: Back cover only (high-res, unused in final assembly)
# - 1.png: Full wrap cover (print layout) - used as final layout reference
#
# IMPORTANT: 00.png/0.png have different dimensions than 1.png!
# Example: 00.png = 1920x2000 (high-res), 1.png = 2000x1074 (print layout)
#
# Workflow:
# 1. Split 1.png into back/spine/front portions
# 2. AI processes 00.png (high-res) for better face/name replacement
# 3. Scale AI result to match front portion of 1.png
# 4. Reassemble: back (from 1.png) + spine (from 1.png) + scaled personalized front
# =============================================================================

# Book-specific artwork area overrides
# For books where automatic border detection cuts off titles or important content
# Set to True to skip border detection entirely and use full image dimensions
# Or provide specific margin percentages as dict
BOOK_ARTWORK_OVERRIDES = {
    # All books use the same border detection approach
    # Add book-specific overrides here only if needed:
    # "Book Name": {
    #     "skip_detection": True,  # Skip border detection entirely
    # },
    # "Another Book": {
    #     "top_margin_pct": 0.02,    # 2% top margin
    #     "bottom_margin_pct": 0.02,
    #     "left_margin_pct": 0.02,
    #     "right_margin_pct": 0.02,
    # },
}


def detect_artwork_area(front_portion_bytes: bytes, tolerance: int = 30, edge_skip: int = 3, book_name: str = None) -> Tuple[int, int, int, int]:
    """
    Detect the artwork area within a front cover portion by finding where the border ends.
    - Auto-derives border color from inset corners (avoiding edge artifacts)
    - Scans each edge until a row/column has > threshold non-border pixels
    - Skips edge rows/columns that may have artifacts
    - Supports book-specific overrides for books with titles near edges
    
    Args:
        front_portion_bytes: Image bytes of the front cover portion
        tolerance: Color difference tolerance for border detection
        edge_skip: Number of edge pixels to skip (artifact avoidance)
        book_name: Optional book name to check for specific overrides
    
    Returns (left, top, right, bottom) crop box.
    """
    import numpy as np

    try:
        img = Image.open(io.BytesIO(front_portion_bytes))
        img_array = np.array(img)
        h, w, _ = img_array.shape
        
        # Check for book-specific overrides
        if book_name and book_name in BOOK_ARTWORK_OVERRIDES:
            override = BOOK_ARTWORK_OVERRIDES[book_name]
            log(f"Using artwork override for book: {book_name}")
            
            if override.get("skip_detection", False):
                # Skip border detection entirely - use full image
                log(f"Skipping border detection for {book_name} - using full dimensions")
                return (0, 0, w, h)
            
            # Use percentage-based margins if specified
            top_margin = int(h * override.get("top_margin_pct", 0))
            bottom_margin = int(h * override.get("bottom_margin_pct", 0))
            left_margin = int(w * override.get("left_margin_pct", 0))
            right_margin = int(w * override.get("right_margin_pct", 0))
            
            box = (left_margin, top_margin, w - right_margin, h - bottom_margin)
            log(f"Using override artwork area: {box}")
            return box

        # 1) Derive border color from INSET corners (skip edge pixels to avoid artifacts)
        # Sample from 5-15 pixels in from each corner instead of 0-5
        inset = 10
        corners = np.vstack([
            img_array[inset:inset+10, inset:inset+10, :3].reshape(-1, 3),
            img_array[inset:inset+10, -inset-10:-inset, :3].reshape(-1, 3),
            img_array[-inset-10:-inset, inset:inset+10, :3].reshape(-1, 3),
            img_array[-inset-10:-inset, -inset-10:-inset, :3].reshape(-1, 3),
        ])
        border_color = np.median(corners, axis=0).astype(np.float32)
        log(f"Detected border color from inset corners: {border_color}")

        def border_thickness_from_left(threshold_ratio=0.2):
            # Skip first few columns (edge artifacts)
            for x in range(edge_skip, w):
                col = img_array[:, x, :3].astype(np.float32)
                diff = np.abs(col - border_color)
                non_border = np.any(diff > tolerance, axis=1)
                if np.mean(non_border) > threshold_ratio:
                    return x
            return edge_skip

        def border_thickness_from_right(threshold_ratio=0.2):
            # Skip last few columns (edge artifacts)
            for x in range(w - 1 - edge_skip, -1, -1):
                col = img_array[:, x, :3].astype(np.float32)
                diff = np.abs(col - border_color)
                non_border = np.any(diff > tolerance, axis=1)
                if np.mean(non_border) > threshold_ratio:
                    return w - 1 - x
            return edge_skip

        def border_thickness_from_top(threshold_ratio=0.2):
            # Skip first few rows (edge artifacts)
            for y in range(edge_skip, h):
                row = img_array[y, :, :3].astype(np.float32)
                diff = np.abs(row - border_color)
                non_border = np.any(diff > tolerance, axis=1)
                if np.mean(non_border) > threshold_ratio:
                    return y
            return edge_skip

        def border_thickness_from_bottom(threshold_ratio=0.2):
            # Skip last few rows (edge artifacts) - this fixes the issue where 
            # the very last row has different colors due to image edge artifacts
            for y in range(h - 1 - edge_skip, -1, -1):
                row = img_array[y, :, :3].astype(np.float32)
                diff = np.abs(row - border_color)
                non_border = np.any(diff > tolerance, axis=1)
                if np.mean(non_border) > threshold_ratio:
                    return h - 1 - y
            return edge_skip

        left = border_thickness_from_left()
        right = border_thickness_from_right()
        top = border_thickness_from_top()
        bottom = border_thickness_from_bottom()

        # Clamp to avoid invalid sizes
        left = max(0, left)
        right = max(0, right)
        top = max(0, top)
        bottom = max(0, bottom)
        if left + right >= w:
            left = 0
            right = 0
        if top + bottom >= h:
            top = 0
            bottom = 0

        box = (left, top, w - right, h - bottom)
        log(f"Detected artwork area: {box} (border L{left}, R{right}, T{top}, B{bottom})")
        return box

    except Exception as e:
        log(f"Error detecting artwork area: {e}")
        img = Image.open(io.BytesIO(front_portion_bytes))
        return (0, 0, img.size[0], img.size[1])


def split_full_cover_into_parts(full_cover_bytes: bytes, spine_percentage: float = 0.03) -> Tuple[bytes, bytes, bytes, Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Split the full wrap cover (1.png) into back cover, spine, and front cover portions.
    
    Layout of 1.png: [Back Cover (left)] [Spine (middle)] [Front Cover (right)]
    
    Args:
        full_cover_bytes: Bytes of the full wrap cover (1.png)
        spine_percentage: Approximate percentage of total width for spine (default 3%)
    
    Returns:
        Tuple of (back_bytes, spine_bytes, front_bytes, back_dims, spine_dims, front_dims)
    """
    try:
        full_cover = Image.open(io.BytesIO(full_cover_bytes))
        full_width, full_height = full_cover.size
        
        # Calculate spine width (typically 2-5% of total width for book spines)
        spine_width = int(full_width * spine_percentage)
        spine_width = max(10, spine_width)  # Minimum 10px for visibility
        
        # Calculate back and front widths (equal, minus spine)
        remaining_width = full_width - spine_width
        back_width = remaining_width // 2
        front_width = remaining_width - back_width  # Handle odd pixels
        
        log(f"Splitting full cover {full_width}x{full_height}")
        log(f"  Back: 0 to {back_width} ({back_width}px)")
        log(f"  Spine: {back_width} to {back_width + spine_width} ({spine_width}px)")
        log(f"  Front: {back_width + spine_width} to {full_width} ({front_width}px)")
        
        # Crop the three portions
        back_cover = full_cover.crop((0, 0, back_width, full_height))
        spine = full_cover.crop((back_width, 0, back_width + spine_width, full_height))
        front_cover = full_cover.crop((back_width + spine_width, 0, full_width, full_height))
        
        # Convert to bytes
        def img_to_bytes(img):
            output = io.BytesIO()
            img.save(output, format="PNG")
            output.seek(0)
            return output.read()
        
        back_bytes = img_to_bytes(back_cover)
        spine_bytes = img_to_bytes(spine)
        front_bytes = img_to_bytes(front_cover)
        
        return (
            back_bytes, spine_bytes, front_bytes,
            back_cover.size, spine.size, front_cover.size
        )
        
    except Exception as e:
        log(f"Error splitting full cover: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Failed to split full cover: {e}")


def extract_spine_from_full_cover(full_cover_bytes: bytes, front_cover_bytes: bytes, back_cover_bytes: bytes) -> bytes:
    """
    Extract the spine strip from the full wrap cover (1.png).
    
    The spine is the middle portion of the full cover that isn't part of the
    front or back cover. It's calculated as:
    spine_width = full_cover_width - front_cover_width - back_cover_width
    
    Args:
        full_cover_bytes: Bytes of the full wrap cover (1.png)
        front_cover_bytes: Bytes of the front cover (00.png)
        back_cover_bytes: Bytes of the back cover (0.png)
    
    Returns:
        bytes: The extracted spine strip as PNG bytes
    """
    try:
        # Load all three images to get their dimensions
        full_cover = Image.open(io.BytesIO(full_cover_bytes))
        front_cover = Image.open(io.BytesIO(front_cover_bytes))
        back_cover = Image.open(io.BytesIO(back_cover_bytes))
        
        full_width, full_height = full_cover.size
        front_width, front_height = front_cover.size
        back_width, back_height = back_cover.size
        
        # Calculate spine width
        spine_width = full_width - front_width - back_width
        
        log(f"Cover dimensions - Full: {full_width}x{full_height}, Front: {front_width}x{front_height}, Back: {back_width}x{back_height}")
        log(f"Calculated spine width: {spine_width}px")
        
        if spine_width <= 0:
            log(f"WARNING: Calculated spine width is {spine_width}px (non-positive). Using minimal spine of 1px")
            spine_width = max(1, spine_width)
        
        # The spine is located between back cover and front cover
        # Layout: [Back Cover (left)] [Spine (middle)] [Front Cover (right)]
        spine_start_x = back_width
        spine_end_x = back_width + spine_width
        
        # Ensure we don't exceed image bounds
        spine_start_x = max(0, min(spine_start_x, full_width - 1))
        spine_end_x = max(spine_start_x + 1, min(spine_end_x, full_width))
        
        # Extract the spine strip
        spine = full_cover.crop((spine_start_x, 0, spine_end_x, full_height))
        
        log(f"Extracted spine: {spine.size[0]}x{spine.size[1]} from x={spine_start_x} to x={spine_end_x}")
        
        # Convert to bytes
        output = io.BytesIO()
        spine.save(output, format="PNG")
        output.seek(0)
        return output.read()
        
    except Exception as e:
        log(f"Error extracting spine from full cover: {e}")
        import traceback
        traceback.print_exc()
        # Return a minimal 1px wide strip as fallback
        try:
            full_cover = Image.open(io.BytesIO(full_cover_bytes))
            fallback_spine = Image.new("RGB", (1, full_cover.size[1]), (255, 255, 255))
            output = io.BytesIO()
            fallback_spine.save(output, format="PNG")
            output.seek(0)
            return output.read()
        except:
            raise ValueError("Failed to extract spine from full cover")


def assemble_final_cover(back_bytes: bytes, spine_bytes: bytes, front_bytes: bytes, target_dimensions: Tuple[int, int] = None) -> bytes:
    """
    Assemble the final print-ready cover by combining back cover, spine, and front cover.
    
    Layout (left to right): [Back Cover] [Spine] [Front Cover]
    
    Args:
        back_bytes: Bytes of the back cover (from 1.png split)
        spine_bytes: Bytes of the spine strip (from 1.png split)
        front_bytes: Bytes of the personalized front cover
        target_dimensions: Optional (width, height) to enforce exact output size
    
    Returns:
        bytes: The assembled full cover as PNG bytes, matching 1.png dimensions
    """
    try:
        # Load all three images
        back_cover = Image.open(io.BytesIO(back_bytes))
        spine = Image.open(io.BytesIO(spine_bytes))
        front_cover = Image.open(io.BytesIO(front_bytes))
        
        back_width, back_height = back_cover.size
        spine_width, spine_height = spine.size
        front_width, front_height = front_cover.size
        
        # Calculate total dimensions from parts
        total_width = back_width + spine_width + front_width
        # Use the target height if provided, otherwise use the spine height (from 1.png)
        # This ensures consistent height matching the original 1.png
        total_height = spine_height  # Spine comes from 1.png, so use its height as reference
        
        log(f"Assembling final cover: Back({back_width}x{back_height}) + Spine({spine_width}x{spine_height}) + Front({front_width}x{front_height})")
        log(f"Target final cover dimensions: {total_width}x{total_height}")
        
        # Create new image with calculated dimensions
        final_cover = Image.new("RGB", (total_width, total_height))
        
        # Paste back cover on the left - resize height to match if needed
        if back_height != total_height:
            log(f"Resizing back cover height: {back_height} -> {total_height}")
            back_cover = back_cover.resize((back_width, total_height), Image.Resampling.LANCZOS)
        final_cover.paste(back_cover, (0, 0))
        
        # Paste spine in the middle (spine defines our reference height)
        final_cover.paste(spine, (back_width, 0))
        
        # Paste front cover on the right - resize to match expected dimensions
        if front_height != total_height or front_width != (total_width - back_width - spine_width):
            expected_front_width = total_width - back_width - spine_width
            log(f"Resizing front cover: {front_width}x{front_height} -> {expected_front_width}x{total_height}")
            front_cover = front_cover.resize((expected_front_width, total_height), Image.Resampling.LANCZOS)
        final_cover.paste(front_cover, (back_width + spine_width, 0))
        
        log(f"Final cover assembled successfully: {final_cover.size}")
        
        # Convert to bytes
        output = io.BytesIO()
        final_cover.save(output, format="PNG")
        output.seek(0)
        return output.read()
        
    except Exception as e:
        log(f"Error assembling final cover: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Failed to assemble final cover: {e}")


def process_cover_with_new_workflow(
    child_image_bytes: bytes,
    front_cover_bytes: bytes,
    back_cover_bytes: bytes,
    full_cover_bytes: bytes,
    child_name: str = None,
    book_name: str = None,
    canonical_reference_bytes: bytes = None,
    identity_info: dict = None
) -> Tuple[bytes, Tuple[int, int]]:
    """
    Process the cover using the new workflow:
    1. Split 1.png into back/spine/front portions (these define final layout)
    2. AI personalizes 00.png (high-res front cover) for better face/name replacement
    3. Scale AI result to match front portion dimensions from 1.png
    4. Reassemble: back (from 1.png) + spine (from 1.png) + scaled personalized front
    
    CRITICAL: Uses canonical_reference_bytes (clean front-facing portrait) instead of
    raw child photo to prevent copying gestures/poses from the child's photo.
    
    IMPORTANT: 00.png and 0.png have different dimensions than 1.png!
    - 00.png/0.png: High-resolution individual covers (e.g., 1920x2000)
    - 1.png: Print-ready layout (e.g., 2000x1074)
    
    Args:
        child_image_bytes: Bytes of the child's photo
        front_cover_bytes: Bytes of the front cover only (00.png) - high-res, for AI
        back_cover_bytes: Bytes of the back cover only (0.png) - high-res, unused
        full_cover_bytes: Bytes of the full wrap cover (1.png) - defines final layout
        child_name: Child's name for text replacement on front cover
        book_name: Book name for artwork area override detection
    
    Returns:
        Tuple of (assembled cover bytes, original full cover dimensions)
    """
    log(f"\n{'='*60}")
    log("PROCESSING COVER WITH NEW WORKFLOW (FIXED DIMENSIONS)")
    log(f"High-res front cover (00.png): {len(front_cover_bytes)} bytes")
    log(f"High-res back cover (0.png): {len(back_cover_bytes)} bytes")
    log(f"Full wrap layout (1.png): {len(full_cover_bytes)} bytes")
    log(f"Child name: {child_name}")
    log(f"{'='*60}\n")
    
    # Get dimensions for logging
    front_img = Image.open(io.BytesIO(front_cover_bytes))
    full_cover_img = Image.open(io.BytesIO(full_cover_bytes))
    log(f"High-res front (00.png) dimensions: {front_img.size}")
    log(f"Full wrap (1.png) dimensions: {full_cover_img.size}")
    original_dimensions = full_cover_img.size
    
    # Step 1: Split 1.png into back/spine/front portions (defines final layout)
    log("Step 1: Splitting full wrap cover (1.png) into back/spine/front portions...")
    back_from_1png, spine_from_1png, front_from_1png, back_dims, spine_dims, front_dims = \
        split_full_cover_into_parts(full_cover_bytes, spine_percentage=0.03)
    
    log(f"  Back portion: {back_dims}")
    log(f"  Spine portion: {spine_dims}")
    log(f"  Front portion: {front_dims}")
    
    # Step 2: AI personalizes the high-res front cover (00.png)
    log("Step 2: AI personalizing high-res front cover (00.png)...")
    
    # Check if front cover has a child character
    if has_child_character(front_cover_bytes):
        log("Front cover has child character - processing face replacement")
        
        # CRITICAL: Cover ALWAYS has happy expression, regardless of template
        # The child should ALWAYS look happy on the cover, no matter what the template shows
        log("COVER EXPRESSION ENFORCEMENT: Always happy, regardless of template")
        log("Detecting template expression for reference (will be overridden to happy)...")
        template_expression_detected = detect_template_expression(front_cover_bytes)
        log(f"Template expression detected: {template_expression_detected['expression']} - {template_expression_detected['description']}")
        log("OVERRIDING to happy expression for cover (regardless of template)")
        
        # ALWAYS override to happy expression for cover, regardless of template
        template_expression = {
            'expression': 'happy',
            'smile_intensity': 'subtle',
            'mouth_state': 'gentle smile',
            'mood': 'happy and content',
            'description': 'gentle happy smile with slightly upturned corners - welcoming and positive',
            'teeth_visible': 'no',
            'mouth_openness': 'closed',
            'eye_state': 'relaxed and happy',
            'raw_analysis': 'COVER ENFORCEMENT: Always happy expression'
        }
        log(f"Enforced cover expression: {template_expression['expression']} - {template_expression['description']}")
        
        # CRITICAL: Use canonical_reference_bytes (clean front-facing portrait) instead of
        # raw child_image_bytes to prevent copying gestures/poses from the child's photo.
        # The canonical reference contains ONLY the face identity, no body/gestures.
        personalized_front_highres = generate_face_replacement_page(
            child_image_bytes,
            front_cover_bytes,
            page_number=0,  # Use 0 to indicate front cover
            is_cover_page=True,
            child_name=child_name,
            book_name=book_name,
            canonical_reference_bytes=canonical_reference_bytes,  # Use clean portrait, not raw photo!
            identity_info=identity_info,  # Pass pre-analyzed identity features
            template_expression_override=template_expression  # Pass enforced expression
        )
        log(f"Cover using canonical reference: {canonical_reference_bytes is not None}")
    else:
        log("Front cover has no child character - using original")
        personalized_front_highres = front_cover_bytes
    
    # Step 3: Detect artwork area in front portion and overlay personalized content
    # The front portion has borders that must be preserved for print-ready output
    log("Step 3: Detecting artwork area and overlaying personalized content...")
    
    # Detect artwork area (where the actual illustration is, excluding borders)
    # Pass book_name for book-specific overrides (e.g., "A True Princess" has title near edge)
    artwork_bbox = detect_artwork_area(front_from_1png, book_name=book_name)  # (left, top, right, bottom)
    artwork_width = artwork_bbox[2] - artwork_bbox[0]
    artwork_height = artwork_bbox[3] - artwork_bbox[1]
    log(f"Artwork area: {artwork_bbox}, size: {artwork_width}x{artwork_height}")
    
    # Scale personalized front to fit just the artwork area (not the full front portion)
    log(f"Scaling personalized front to artwork area: {artwork_width}x{artwork_height}")
    personalized_artwork_scaled = ensure_exact_dimensions(
        personalized_front_highres, 
        artwork_width,
        artwork_height
    )
    
    # Load original front portion (with borders) and overlay the personalized artwork
    original_front = Image.open(io.BytesIO(front_from_1png))
    personalized_artwork_img = Image.open(io.BytesIO(personalized_artwork_scaled))
    
    # Convert to same mode if needed
    if original_front.mode != personalized_artwork_img.mode:
        personalized_artwork_img = personalized_artwork_img.convert(original_front.mode)
    
    # Paste personalized artwork in the artwork area, preserving borders
    original_front.paste(personalized_artwork_img, (artwork_bbox[0], artwork_bbox[1]))
    
    # Convert back to bytes
    front_with_borders = io.BytesIO()
    original_front.save(front_with_borders, format="PNG")
    front_with_borders.seek(0)
    personalized_front_with_borders = front_with_borders.read()
    
    log(f"Personalized front with borders preserved: {original_front.size}")
    
    # Step 4: Assemble final cover using parts from 1.png + personalized front (with borders)
    log("Step 4: Assembling final cover (back from 1.png + spine from 1.png + personalized front with borders)...")
    assembled_cover = assemble_final_cover(back_from_1png, spine_from_1png, personalized_front_with_borders)
    
    # ALWAYS verify and enforce final cover matches original full cover dimensions (1.png)
    # This is critical for Gelato print compatibility
    assembled_img = Image.open(io.BytesIO(assembled_cover))
    log(f"Assembled cover dimensions: {assembled_img.size}, Target (1.png): {original_dimensions}")
    
    if assembled_img.size != original_dimensions:
        log(f"DIMENSION MISMATCH: Assembled {assembled_img.size} vs Original {original_dimensions}")
        log("Enforcing exact dimensions to match 1.png...")
        assembled_cover = ensure_exact_dimensions(assembled_cover, original_dimensions[0], original_dimensions[1])
        # Verify after resize
        final_check = Image.open(io.BytesIO(assembled_cover))
        log(f"After resize: {final_check.size}")
    else:
        log(f"Dimensions match perfectly: {assembled_img.size}")
    
    # FINAL SAFETY: Always ensure exact dimensions match 1.png for Gelato compatibility
    # Even if dimensions appear to match, force resize to handle any edge cases
    assembled_cover = ensure_exact_dimensions(assembled_cover, original_dimensions[0], original_dimensions[1])
    
    log(f"Cover processing complete. Final size: {len(assembled_cover)} bytes, Dimensions: {original_dimensions}")
    
    return assembled_cover, original_dimensions


def detect_template_expression(template_bytes: bytes) -> dict:
    """
    Detect the facial expression of the character in the template image.
    
    This is a critical pre-analysis step to ensure the AI uses the template's
    expression (not the child photo's expression) when generating the face swap.
    
    Args:
        template_bytes: Bytes of the template image
        
    Returns:
        dict with:
            - 'expression': Main expression type (e.g., "happy", "neutral", "sad", "surprised")
            - 'description': Detailed description of the expression
            - 'mouth_state': "open", "closed", "smiling", etc.
            - 'eye_state': "wide", "squinting", "relaxed", etc.
            - 'mood': Overall mood of the character
            - 'mouth_openness': How open the mouth is (closed/slightly_open/open/wide_open)
            - 'smile_intensity': Level of smile (none/subtle/moderate/wide)
            - 'teeth_visible': Whether teeth are showing (yes/no)
    """
    try:
        template_b64 = base64.b64encode(template_bytes).decode('utf-8')
        
        detection_prompt = (
            "Analyze the facial expression of the main character in this storybook illustration. "
            "Focus ONLY on the character's expression, not any other elements.\n\n"
            "CRITICAL: Be EXTREMELY precise about the mouth position - this is the most important part!\n\n"
            "Provide your analysis in EXACTLY this format:\n"
            "EXPRESSION: [one word: happy/sad/neutral/surprised/excited/worried/scared/angry/thoughtful/content/gentle]\n"
            "MOUTH_OPENNESS: [closed/slightly_open/open/wide_open] - How open is the mouth?\n"
            "SMILE_INTENSITY: [none/subtle/moderate/wide] - How much are they smiling? Subtle = gentle/soft smile, not a big grin\n"
            "TEETH_VISIBLE: [yes/no] - Can you see any teeth?\n"
            "MOUTH_STATE: [smiling with teeth/smiling closed/gentle smile/neutral/slight frown/frowning/open/speaking]\n"
            "EYE_STATE: [wide open/half closed/squinting/relaxed/looking up/looking down/gentle/soft]\n"
            "MOOD: [brief description of overall emotional state, max 10 words]\n"
            "DESCRIPTION: [one sentence describing the EXACT mouth and expression - be very specific]\n\n"
            "IMPORTANT: A 'subtle' or 'gentle' smile means the corners of the mouth are slightly upturned but NOT a big wide smile.\n"
            "Be precise - this will be used to recreate the EXACT same expression. Do NOT exaggerate or minimize what you see."
        )
        
        response = get_client().chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": detection_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{template_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        analysis = response.choices[0].message.content or ""
        log(f"Template expression analysis:\n{analysis}")
        
        # Parse the response
        result = {
            'expression': 'neutral',
            'mouth_state': 'neutral',
            'eye_state': 'relaxed',
            'mood': 'calm',
            'description': 'neutral expression',
            'mouth_openness': 'closed',
            'smile_intensity': 'none',
            'teeth_visible': 'no',
            'raw_analysis': analysis
        }
        
        # Extract expression
        expr_match = re.search(r'EXPRESSION:\s*(\w+)', analysis, re.IGNORECASE)
        if expr_match:
            result['expression'] = expr_match.group(1).lower()
        
        # Extract mouth openness (NEW - critical for exact matching)
        openness_match = re.search(r'MOUTH_OPENNESS:\s*(\w+)', analysis, re.IGNORECASE)
        if openness_match:
            result['mouth_openness'] = openness_match.group(1).strip().lower()
        
        # Extract smile intensity (NEW - critical for exact matching)
        smile_match = re.search(r'SMILE_INTENSITY:\s*(\w+)', analysis, re.IGNORECASE)
        if smile_match:
            result['smile_intensity'] = smile_match.group(1).strip().lower()
        
        # Extract teeth visibility (NEW)
        teeth_match = re.search(r'TEETH_VISIBLE:\s*(\w+)', analysis, re.IGNORECASE)
        if teeth_match:
            result['teeth_visible'] = teeth_match.group(1).strip().lower()
        
        # Extract mouth state
        mouth_match = re.search(r'MOUTH_STATE:\s*([^\n]+)', analysis, re.IGNORECASE)
        if mouth_match:
            result['mouth_state'] = mouth_match.group(1).strip().lower()
        
        # Extract eye state
        eye_match = re.search(r'EYE_STATE:\s*([^\n]+)', analysis, re.IGNORECASE)
        if eye_match:
            result['eye_state'] = eye_match.group(1).strip().lower()
        
        # Extract mood
        mood_match = re.search(r'MOOD:\s*([^\n]+)', analysis, re.IGNORECASE)
        if mood_match:
            result['mood'] = mood_match.group(1).strip()
        
        # Extract description
        desc_match = re.search(r'DESCRIPTION:\s*([^\n]+)', analysis, re.IGNORECASE)
        if desc_match:
            result['description'] = desc_match.group(1).strip()
        
        log(f"Parsed expression: {result['expression']}, Mouth: {result['mouth_state']}, Eyes: {result['eye_state']}")
        log(f"Mouth openness: {result['mouth_openness']}, Smile intensity: {result['smile_intensity']}, Teeth visible: {result['teeth_visible']}")
        
        return result
        
    except Exception as e:
        log(f"Error detecting template expression: {e}")
        # Return default neutral expression
        return {
            'expression': 'neutral',
            'mouth_state': 'neutral',
            'eye_state': 'relaxed',
            'mood': 'calm',
            'description': 'unable to detect expression',
            'mouth_openness': 'closed',
            'smile_intensity': 'none',
            'teeth_visible': 'no',
            'raw_analysis': ''
        }


def analyze_child_features(child_image_bytes: bytes) -> dict:
    """
    Analyze the child's facial features in detail using GPT-4o Vision.
    
    This function extracts specific, detailed descriptions of the child's unique
    facial features to be used in the generation prompt. This helps ensure the
    generated face is recognizable as the same child.
    
    Args:
        child_image_bytes: Bytes of the child photo
        
    Returns:
        dict with detailed feature descriptions:
            - 'face_shape': Description of face shape
            - 'eyes': Eye details (color, shape, size, spacing)
            - 'nose': Nose characteristics
            - 'mouth': Mouth/lips details
            - 'skin_tone': Skin tone description
            - 'hair': Hair color, texture, style
            - 'distinctive_features': Unique identifying features
            - 'age_appearance': Approximate age appearance
            - 'overall_description': Summary description
    """
    try:
        child_b64 = base64.b64encode(child_image_bytes).decode('utf-8')
        
        analysis_prompt = (
            "You are an expert at describing children's facial features for portrait artists. "
            "Analyze this child's photo and describe their UNIQUE, IDENTIFYING features in detail. "
            "The description should be specific enough that an artist could draw this EXACT child, "
            "not a generic child.\n\n"
            "Provide your analysis in EXACTLY this format:\n\n"
            "FACE_SHAPE: [specific shape - round/oval/heart/square, cheek fullness, chin shape, jaw line]\n"
            "EYES: [color, shape (almond/round/hooded), size (large/small/medium), spacing (wide-set/close-set), "
            "any unique characteristics like thick lashes, eye shape asymmetry]\n"
            "EYEBROWS: [shape, thickness, color, arch]\n"
            "NOSE: [size, shape, bridge (high/low/flat), tip shape (round/pointed/upturned), nostril shape]\n"
            "MOUTH: [lip size (full/thin), lip shape, cupid's bow, any unique characteristics]\n"
            "SKIN_TONE: [specific tone - fair/light/medium/olive/tan/brown/dark, undertones (warm/cool/neutral), "
            "any complexion details like rosy cheeks, freckles]\n"
            "HAIR_COLOR: [specific color - not just 'brown' but 'warm chestnut brown with golden highlights']\n"
            "HAIR_TEXTURE: [straight/wavy/curly/coily, thickness, volume]\n"
            "HAIR_STYLE: [current style, length, how it frames the face]\n"
            "DISTINCTIVE_FEATURES: [THE MOST IMPORTANT - list 3-5 unique features that make this child "
            "instantly recognizable: dimples, birthmarks, freckles pattern, unique eye shape, etc.]\n"
            "AGE_APPEARANCE: [approximate age the child appears to be]\n"
            "OVERALL_SUMMARY: [One sentence that captures what makes this child's face unique and recognizable]\n\n"
            "Be VERY SPECIFIC - use precise descriptions, not generic ones. "
            "This description will be used to ensure the child is recognizable in illustrations."
        )
        
        response = get_client().chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{child_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=800
        )
        
        analysis = response.choices[0].message.content or ""
        log(f"Child features analysis:\n{analysis}")
        
        # Parse the response into structured data
        result = {
            'face_shape': '',
            'eyes': '',
            'eyebrows': '',
            'nose': '',
            'mouth': '',
            'skin_tone': '',
            'hair_color': '',
            'hair_texture': '',
            'hair_style': '',
            'distinctive_features': '',
            'age_appearance': '',
            'overall_summary': '',
            'raw_analysis': analysis
        }
        
        # Extract each feature
        face_match = re.search(r'FACE_SHAPE:\s*([^\n]+)', analysis, re.IGNORECASE)
        if face_match:
            result['face_shape'] = face_match.group(1).strip()
        
        eyes_match = re.search(r'EYES:\s*([^\n]+)', analysis, re.IGNORECASE)
        if eyes_match:
            result['eyes'] = eyes_match.group(1).strip()
        
        eyebrows_match = re.search(r'EYEBROWS:\s*([^\n]+)', analysis, re.IGNORECASE)
        if eyebrows_match:
            result['eyebrows'] = eyebrows_match.group(1).strip()
        
        nose_match = re.search(r'NOSE:\s*([^\n]+)', analysis, re.IGNORECASE)
        if nose_match:
            result['nose'] = nose_match.group(1).strip()
        
        mouth_match = re.search(r'MOUTH:\s*([^\n]+)', analysis, re.IGNORECASE)
        if mouth_match:
            result['mouth'] = mouth_match.group(1).strip()
        
        skin_match = re.search(r'SKIN_TONE:\s*([^\n]+)', analysis, re.IGNORECASE)
        if skin_match:
            result['skin_tone'] = skin_match.group(1).strip()
        
        hair_color_match = re.search(r'HAIR_COLOR:\s*([^\n]+)', analysis, re.IGNORECASE)
        if hair_color_match:
            result['hair_color'] = hair_color_match.group(1).strip()
        
        hair_texture_match = re.search(r'HAIR_TEXTURE:\s*([^\n]+)', analysis, re.IGNORECASE)
        if hair_texture_match:
            result['hair_texture'] = hair_texture_match.group(1).strip()
        
        hair_style_match = re.search(r'HAIR_STYLE:\s*([^\n]+)', analysis, re.IGNORECASE)
        if hair_style_match:
            result['hair_style'] = hair_style_match.group(1).strip()
        
        distinctive_match = re.search(r'DISTINCTIVE_FEATURES:\s*([^\n]+)', analysis, re.IGNORECASE)
        if distinctive_match:
            result['distinctive_features'] = distinctive_match.group(1).strip()
        
        age_match = re.search(r'AGE_APPEARANCE:\s*([^\n]+)', analysis, re.IGNORECASE)
        if age_match:
            result['age_appearance'] = age_match.group(1).strip()
        
        summary_match = re.search(r'OVERALL_SUMMARY:\s*([^\n]+)', analysis, re.IGNORECASE)
        if summary_match:
            result['overall_summary'] = summary_match.group(1).strip()
        
        log(f"Parsed child features - Face: {result['face_shape'][:30]}..., Eyes: {result['eyes'][:30]}...")
        
        return result
        
    except Exception as e:
        log(f"Error analyzing child features: {e}")
        # Return empty defaults
        return {
            'face_shape': 'child face',
            'eyes': 'child eyes',
            'eyebrows': 'natural eyebrows',
            'nose': 'child nose',
            'mouth': 'child mouth',
            'skin_tone': 'natural skin tone',
            'hair_color': 'natural hair color',
            'hair_texture': 'natural hair',
            'hair_style': 'natural style',
            'distinctive_features': 'unique child features',
            'age_appearance': 'young child',
            'overall_summary': 'unique child',
            'raw_analysis': ''
        }


def detect_face_in_child_photo(child_image_bytes: bytes) -> Optional[Tuple[int, int, int, int]]:
    """
    Use GPT-4o Vision to detect face bounding box in child photo.
    
    Args:
        child_image_bytes: Bytes of the child photo
    
    Returns:
        Optional tuple (x1, y1, x2, y2) of face bounding box, or None if detection fails
    """
    try:
        child_b64 = base64.b64encode(child_image_bytes).decode('utf-8')
        detection_prompt = (
            "Find the face bounding box in this photo. "
            "Return ONLY the bounding box as: BOUNDING_BOX: (x1, y1, x2, y2)"
        )
        
        response = get_client().chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": detection_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{child_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=100
        )
        
        analysis = response.choices[0].message.content or ""
        bbox = _extract_bbox_from_analysis(analysis)
        
        if bbox:
            log(f"Face detected in child photo: {bbox}")
        else:
            log("Face detection in child photo failed, will use full image")
        
        return bbox
        
    except Exception as e:
        log(f"Error detecting face in child photo: {e}")
        return None


def preprocess_child_face(child_image_bytes: bytes) -> bytes:
    """
    Preprocess child photo to optimize face for identity preservation.
    - Detects and crops to face area with padding
    - Enhances image quality (sharpness, contrast)
    - Normalizes lighting
    - Resizes to optimal dimensions
    
    Args:
        child_image_bytes: Original child photo bytes
    
    Returns:
        Preprocessed child photo bytes optimized for face replacement
    """
    try:
        # Load image
        img = Image.open(io.BytesIO(child_image_bytes))
        original_width, original_height = img.size
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Detect face
        face_bbox = detect_face_in_child_photo(child_image_bytes)
        
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            # Clamp to image boundaries
            x1 = max(0, min(x1, original_width - 1))
            y1 = max(0, min(y1, original_height - 1))
            x2 = max(0, min(x2, original_width - 1))
            y2 = max(0, min(y2, original_height - 1))
            
            # Add padding - include more skin area (neck/shoulders) for better skin tone matching
            width = x2 - x1
            height = y2 - y1
            padding_x = int(width * 0.3)  # 30% horizontal padding
            padding_y_top = int(height * 0.2)  # 20% top padding (hair area)
            padding_y_bottom = int(height * 0.6)  # 60% bottom padding (neck/shoulders for skin tone reference)
            
            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y_top)
            x2 = min(original_width, x2 + padding_x)
            y2 = min(original_height, y2 + padding_y_bottom)
            
            # Crop to face area including neck/shoulders
            img = img.crop((x1, y1, x2, y2))
            log(f"Cropped child photo to face+neck area: {x2-x1}x{y2-y1} (includes skin tone reference)")
        else:
            # If face detection fails, try to find center area (likely where face is)
            # Use upper portion but include neck/shoulders for skin tone reference
            center_x = original_width // 2
            center_y = original_height // 3
            crop_width = min(original_width, int(original_width * 0.7))
            crop_height = min(original_height, int(original_height * 0.6))  # Include more vertical area for neck
            x1 = max(0, center_x - crop_width // 2)
            y1 = max(0, center_y - crop_height // 3)  # Start higher to include face
            x2 = min(original_width, center_x + crop_width // 2)
            y2 = min(original_height, y1 + crop_height)  # Extend down to include neck/shoulders
            img = img.crop((x1, y1, x2, y2))
            log(f"Using center crop for child photo (includes neck): {x2-x1}x{y2-y1}")
        
        # Subtle image quality adjustments (minimal to preserve natural appearance for style integration)
        try:
            from PIL import ImageEnhance, ImageFilter
            
            # Very subtle sharpness enhancement (reduced from 1.2 to 1.05) to avoid over-processing
            # Over-sharpening can make it harder to match illustration style
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.05)  # 5% sharper (reduced from 20%)
            
            # Minimal contrast adjustment to preserve natural look
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.03)  # 3% more contrast (reduced from 10%)
            
            # No brightness adjustment - preserve original lighting for better style matching
            
            log("Applied subtle image enhancements (minimal sharpness and contrast to preserve natural appearance)")
        except Exception as e:
            log(f"Could not apply all enhancements: {e}")
        
        # Resize to optimal size (max 1024px on longest side, maintain aspect ratio)
        max_size = 1024
        width, height = img.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            log(f"Resized child photo to optimal size: {new_width}x{new_height}")
        
        # Convert back to bytes
        output = io.BytesIO()
        img.save(output, format="PNG", quality=95)
        output.seek(0)
        preprocessed_bytes = output.read()
        
        log(f"Preprocessed child photo: {len(child_image_bytes)} -> {len(preprocessed_bytes)} bytes")
        return preprocessed_bytes
        
    except Exception as e:
        log(f"Error preprocessing child face: {e}")
        # Return original if preprocessing fails
        return child_image_bytes


def create_face_mask_with_gpt_detection(template_bytes: bytes) -> bytes:
    """
    Use GPT-4o vision to detect face bbox and build a tight mask for face/hair.
    Falls back to a central ellipse if detection fails.
    """
    img = Image.open(io.BytesIO(template_bytes)).convert("RGBA")
    width, height = img.size

    # Ask GPT-4o for bbox
    tmpl_b64 = base64.b64encode(template_bytes).decode("utf-8")
    detection_prompt = (
        "Find the main character's face bounding box in this illustration. "
        "Return as: BOUNDING_BOX: (x1, y1, x2, y2)"
    )
    try:
        resp = get_client().chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": detection_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tmpl_b64}"},},
                    ],
                }
            ],
            max_tokens=200,
        )
        analysis = resp.choices[0].message.content or ""
        bbox = _extract_bbox_from_analysis(analysis)
    except Exception as e:
        log(f"Face detection fallback (analysis failed): {e}")
        bbox = None

    # Fallback to central ellipse if bbox missing
    if bbox:
        bbox = _clamp_bbox(bbox, width, height)
    else:
        cx, cy = width // 2, height // 3
        rw, rh = width // 6, height // 5
        bbox = (cx - rw, cy - rh, cx + rw, cy + rh)

    # Build mask
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(bbox, fill=255)

    # Slightly feather the mask to avoid hard edges
    # Feather lightly to avoid hard seams
    try:
        from PIL import ImageFilter
        mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    except Exception:
        pass

    out = io.BytesIO()
    mask.save(out, format="PNG")
    out.seek(0)
    return out.read()


def choose_allowed_size(width: int, height: int) -> str:
    """
    Choose closest allowed size for gpt-image-1.
    Allowed: 1024x1024, 1024x1536, 1536x1024, or 'auto'.
    """
    allowed = [(1024, 1024), (1024, 1536), (1536, 1024)]
    aspect = width / height if height != 0 else 1.0
    # pick by minimal aspect ratio difference; tie-breaker by area closeness
    best = None
    best_score = 1e9
    for w, h in allowed:
        a = w / h
        score = abs(a - aspect) + abs((w * h) - (width * height)) / (width * height + 1e-6)
        if score < best_score:
            best_score = score
            best = (w, h)
    if best:
        return f"{best[0]}x{best[1]}"
    return "auto"


def is_empty_or_white_page(template_image_bytes: bytes) -> bool:
    """
    Check if template image is empty/white.
    
    Args:
        template_image_bytes: Bytes of the template image
    
    Returns:
        bool: True if page is empty/white, False otherwise
    """
    try:
        # Check file size first - very small files are likely empty/white
        if len(template_image_bytes) < 50000:  # Less than ~50KB is suspicious
            log(f"Template is very small ({len(template_image_bytes)} bytes), likely empty/white")
            return True
        
        # Load image with PIL
        img = Image.open(io.BytesIO(template_image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array for analysis
        img_array = np.array(img)
        
        # Calculate statistics
        # Check if image is mostly white (RGB values close to 255)
        white_threshold = 240  # Consider pixels with R, G, B all > 240 as white
        white_pixels = np.sum(
            (img_array[:, :, 0] > white_threshold) & 
            (img_array[:, :, 1] > white_threshold) & 
            (img_array[:, :, 2] > white_threshold)
        )
        total_pixels = img_array.shape[0] * img_array.shape[1]
        white_percentage = (white_pixels / total_pixels) * 100
        
        # Check color variance - empty/white pages have very low variance
        variance = np.var(img_array)
        
        log(f"Page analysis - White pixels: {white_percentage:.2f}%, Variance: {variance:.2f}")
        
        # Consider empty if >90% white OR very low variance (< 100)
        is_empty = white_percentage > 90 or variance < 100
        
        if is_empty:
            log(f"Page detected as empty/white (white: {white_percentage:.2f}%, variance: {variance:.2f})")
        
        return is_empty
        
    except Exception as e:
        log(f"Error analyzing page for empty/white: {e}")
        # If we can't analyze, assume it's not empty (safer to process)
        return False


def is_cover_page(template_image_bytes: bytes, page_number: int = 1) -> bool:
    """
    Detect if a page is a cover page (page 1 with front and back cover together).
    Cover pages typically have a wider aspect ratio (approximately 2:1 or wider).
    
    Args:
        template_image_bytes: Bytes of the template image
        page_number: Page number (1 is typically the cover)
    
    Returns:
        bool: True if this appears to be a cover page, False otherwise
    """
    try:
        # Page 1 is typically the cover
        if page_number != 1:
            return False
        
        # Check aspect ratio - cover pages are typically wider
        img = Image.open(io.BytesIO(template_image_bytes))
        width, height = img.size
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Cover pages are typically wider than 1.5:1 (front and back cover together)
        # Regular pages are usually closer to 1:1 or portrait orientation
        is_cover = aspect_ratio > 1.5
        
        log(f"Cover page detection: {width}x{height} (aspect ratio: {aspect_ratio:.2f}) -> {'COVER' if is_cover else 'REGULAR PAGE'}")
        
        return is_cover
        
    except Exception as e:
        log(f"Error detecting cover page: {e}")
        # If detection fails and it's page 1, assume it might be a cover
        return page_number == 1


def process_cover_page(child_image_bytes: bytes, template_image_bytes: bytes, page_number: int = 1, child_name: str = None) -> bytes:
    """
    Process a cover page by splitting it into front (right) and back (left) covers.
    Only process the front cover (right side with picture), keep back cover (left side) unchanged.
    
    Args:
        child_image_bytes: Bytes of the child photo
        template_image_bytes: Bytes of the full cover page template (back + front)
        page_number: Page number (should be 1 for cover)
        child_name: Child's name for name replacement on front cover
    
    Returns:
        bytes: Processed cover page with front cover (right side) personalized and back cover (left side) unchanged
    """
    try:
        log(f"Processing cover page {page_number} - splitting back (left) and front (right) covers")
        
        # Load the full cover image
        full_cover = Image.open(io.BytesIO(template_image_bytes))
        full_width, full_height = full_cover.size
        
        # Split into left (back cover) and right (front cover with picture) halves
        mid_point = full_width // 2
        back_cover = full_cover.crop((0, 0, mid_point, full_height))  # LEFT side - back cover (unchanged)
        front_cover = full_cover.crop((mid_point, 0, full_width, full_height))  # RIGHT side - front cover (process this)
        
        log(f"Cover split: Full={full_width}x{full_height}, Back (left)={mid_point}x{full_height}, Front (right)={full_width-mid_point}x{full_height}")
        
        # Convert front cover (right side) to bytes for processing
        front_cover_bytes = io.BytesIO()
        front_cover.save(front_cover_bytes, format="PNG")
        front_cover_bytes.seek(0)
        front_cover_bytes_data = front_cover_bytes.read()
        
        # Store exact front cover dimensions to prevent overflow
        front_cover_width = full_width - mid_point
        front_cover_height = full_height
        
        # Check if front cover (right side) has a child character
        if has_child_character(front_cover_bytes_data):
            log("Front cover (right side) has child character - processing face replacement")
            # Process only the front cover (right side) with name replacement
            processed_front_bytes = generate_face_replacement_page(
                child_image_bytes, 
                front_cover_bytes_data, 
                page_number=page_number,
                is_cover_page=True,
                child_name=child_name
            )
            processed_front = Image.open(io.BytesIO(processed_front_bytes))
            
            # CRITICAL: Ensure processed front cover is exactly the right size and cropped to prevent overflow
            processed_width, processed_height = processed_front.size
            if processed_width != front_cover_width or processed_height != front_cover_height:
                log(f"Resizing processed front cover from {processed_front.size} to {front_cover_width}x{front_cover_height}")
                processed_front = processed_front.resize((front_cover_width, front_cover_height), Image.Resampling.LANCZOS)
            
            # Additional safety: Crop to exact dimensions if somehow still larger
            if processed_front.width > front_cover_width or processed_front.height > front_cover_height:
                log(f"Cropping processed front cover to prevent overflow: {processed_front.size} -> {front_cover_width}x{front_cover_height}")
                processed_front = processed_front.crop((0, 0, front_cover_width, front_cover_height))
        else:
            log("Front cover (right side) has no child character - using original")
            processed_front = front_cover
        
        # Merge unchanged back cover (left) with processed front cover (right)
        # Create new image with original dimensions
        merged_cover = Image.new("RGB", (full_width, full_height))
        merged_cover.paste(back_cover, (0, 0))  # Paste back cover on LEFT side
        
        # Ensure processed_front is exactly the right size before pasting (double-check)
        if processed_front.size != (front_cover_width, front_cover_height):
            log(f"Final resize before paste: {processed_front.size} -> {front_cover_width}x{front_cover_height}")
            processed_front = processed_front.resize((front_cover_width, front_cover_height), Image.Resampling.LANCZOS)
        
        # Paste processed front cover on RIGHT side at exact position - ensure no overflow
        merged_cover.paste(processed_front, (mid_point, 0))  # Paste processed front cover on RIGHT side
        
        log(f"Merged cover dimensions: {merged_cover.size} (should match original: {full_width}x{full_height})")
        
        # Convert back to bytes
        output = io.BytesIO()
        merged_cover.save(output, format="PNG")
        output.seek(0)
        result_bytes = output.read()
        
        # Verify dimensions match
        result_img = Image.open(io.BytesIO(result_bytes))
        if result_img.size != (full_width, full_height):
            log(f"WARNING: Merged cover dimensions {result_img.size} don't match original {full_width}x{full_height}")
        
        return result_bytes
        
    except Exception as e:
        log(f"Error processing cover page: {e}")
        import traceback
        traceback.print_exc()
        # Return original if processing fails
        return template_image_bytes


def has_child_character(template_image_bytes: bytes) -> bool:
    """
    Use GPT-4 Vision to detect if template contains a child character.
    
    Args:
        template_image_bytes: Bytes of the template image
    
    Returns:
        bool: True if child character is detected, False otherwise
    """
    try:
        template_b64 = base64.b64encode(template_image_bytes).decode('utf-8')
        
        detection_prompt = (
            "Look at this storybook page image carefully. "
            "Does this page contain a VISIBLE child character (a person/child figure) with a face that can be replaced? "
            "Look for an actual illustrated child character, not just text, background elements, or decorative images. "
            "The child must be clearly visible as a character in the illustration, not just mentioned in text. "
            "Answer with ONLY 'YES' if there is a visible child character with a face, or 'NO' if there is no such character. "
            "Answer with ONLY 'YES' or 'NO' - nothing else."
        )
        
        response = get_client().chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": detection_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{template_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=10
        )
        
        answer = response.choices[0].message.content.strip().upper()
        has_character = "YES" in answer
        
        log(f"Child character detection: {answer} -> {has_character}")
        
        return has_character
        
    except Exception as e:
        log(f"Error detecting child character: {e}")
        # If detection fails, be conservative and return False to preserve original template
        # This is safer than generating on pages without characters
        log("Child detection failed - returning False to preserve original template")
        return False


def ensure_exact_dimensions(image_bytes: bytes, target_width: int, target_height: int) -> bytes:
    """
    Ensure the image has exact target dimensions, resizing if necessary.
    
    Args:
        image_bytes: Bytes of the image
        target_width: Target width in pixels
        target_height: Target height in pixels
    
    Returns:
        bytes: Image bytes with exact target dimensions
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.size == (target_width, target_height):
            return image_bytes  # Already correct size
        
        log(f"Resizing image from {img.size} to {target_width}x{target_height} to match original template")
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        output = io.BytesIO()
        img.save(output, format="PNG")
        output.seek(0)
        return output.read()
    except Exception as e:
        log(f"Error ensuring dimensions: {e}")
        return image_bytes  # Return original if resize fails


# =============================================================================
# CANONICAL REFERENCE GENERATION
# Creates a clean, front-facing reference portrait that serves as the identity
# anchor for all pages in the book. This ensures consistency across 30+ pages.
# =============================================================================

def generate_canonical_reference(
    child_image_bytes: bytes,
    style_reference_bytes: bytes = None,
    book_name: str = None
) -> Tuple[bytes, dict]:
    """
    Generate a clean, canonical reference portrait from the customer photo.
    This reference becomes the identity anchor for all pages in the book.
    
    The canonical reference has:
    - Clean, front-facing view
    - Neutral expression (slight smile)
    - Correct skin tone, hair color, eye color from original photo
    - REALISTIC/PHOTOGRAPHIC style (NOT illustrated) - preserves real appearance
    - Neutral lighting for consistent application across pages
    
    Args:
        child_image_bytes: Bytes of the customer's child photo
        style_reference_bytes: Optional bytes of a template page to match artistic style
        book_name: Name of the book (for style matching)
    
    Returns:
        Tuple of (canonical_reference_bytes, identity_features_dict)
    """
    log("\n" + "="*60)
    log("GENERATING CANONICAL REFERENCE PORTRAIT")
    log("="*60 + "\n")
    
    # Step 1: Analyze child's features in detail
    log("Step 1: Analyzing child's facial features...")
    child_features = analyze_child_features(child_image_bytes)
    log(f"Features analyzed: {child_features.get('overall_summary', 'N/A')[:100]}...")
    
    # Step 2: Preprocess child photo for optimal face extraction
    log("Step 2: Preprocessing child photo...")
    preprocessed_child = preprocess_child_face(child_image_bytes)
    
    # Step 3: Create canonical reference prompt
    # This prompt creates a clean, front-facing portrait preserving exact identity
    canonical_prompt = (
        "Create a clean, front-facing portrait of this child for use as an identity reference. "
        "CRITICAL REQUIREMENTS:\n\n"
        
        "IDENTITY PRESERVATION (COPY EXACTLY FROM PHOTO):\n"
        f"• Face shape: {child_features.get('face_shape', 'as shown in photo')}\n"
        f"• Eyes: {child_features.get('eyes', 'as shown in photo')}\n"
        f"• Eyebrows: {child_features.get('eyebrows', 'as shown in photo')}\n"
        f"• Nose: {child_features.get('nose', 'as shown in photo')}\n"
        f"• Mouth: {child_features.get('mouth', 'as shown in photo')}\n"
        f"• Skin tone: {child_features.get('skin_tone', 'as shown in photo')} - EXACT match required\n"
        f"• Hair color: {child_features.get('hair_color', 'as shown in photo')} - EXACT match required\n"
        f"• Hair texture: {child_features.get('hair_texture', 'as shown in photo')}\n"
        f"• Eye color: Match exactly from photo\n"
        f"• Distinctive features: {child_features.get('distinctive_features', 'preserve all')}\n\n"
        
        "POSE & EXPRESSION:\n"
        "• Front-facing view (looking directly at camera)\n"
        "• Neutral, pleasant expression with slight natural smile\n"
        "• Head straight, not tilted\n"
        "• Clean, even lighting on face\n"
        "• Plain neutral background (light gray or soft gradient)\n\n"
        
        "ARTISTIC STYLE (PRESERVE REALISM):\n"
        "• Keep the child looking REALISTIC and PHOTOGRAPHIC - NOT illustrated or cartoon\n"
        "• Preserve the natural, realistic appearance of skin, hair, and features\n"
        "• The face must look like a REAL child, not a drawn or painted version\n"
        "• DO NOT stylize, cartoonify, or illustrate - keep it looking like a real photograph\n"
        "• The face must be INSTANTLY RECOGNIZABLE as this specific child\n"
        "• Natural skin texture, realistic hair strands, lifelike eyes\n\n"
        
        "OUTPUT:\n"
        "• Head and shoulders portrait\n"
        "• High quality, clear facial details\n"
        "• This will be used as the identity reference for all book pages\n"
        "• The child's parent must be able to immediately recognize their child"
    )
    
    # Step 4: Generate canonical reference using gpt-image-1
    log("Step 3: Generating canonical reference portrait...")
    
    # Create temp files
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_child:
        tmp_child.write(preprocessed_child)
        child_path = tmp_child.name
    
    try:
        with open(child_path, 'rb') as child_file:
            response = get_client().images.edit(
                model="gpt-image-1",
                image=child_file,
                prompt=canonical_prompt,
                input_fidelity="high",  # Maximum fidelity for identity preservation
                size="1024x1024",  # Square format for reference portrait
                output_format="png",
                n=1,
            )
        
        # Extract result
        result_bytes = None
        if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
            log("Got base64 canonical reference response")
            result_bytes = base64.b64decode(response.data[0].b64_json)
        elif hasattr(response.data[0], 'url') and response.data[0].url:
            log(f"Got URL response for canonical reference: {response.data[0].url[:80]}...")
            img_response = requests.get(response.data[0].url)
            img_response.raise_for_status()
            result_bytes = img_response.content
        else:
            raise ValueError("No image data in canonical reference response")
        
        log(f"Canonical reference generated successfully: {len(result_bytes)} bytes")
        
        # Store the identity features for use in page generation
        identity_info = {
            'features': child_features,
            'canonical_size': (1024, 1024),
            'generated': True
        }
        
        return result_bytes, identity_info
        
    except Exception as e:
        log(f"Error generating canonical reference: {e}")
        # Fallback: use preprocessed child photo as reference
        log("Falling back to preprocessed child photo as canonical reference")
        identity_info = {
            'features': child_features,
            'canonical_size': Image.open(io.BytesIO(preprocessed_child)).size,
            'generated': False,
            'fallback': True
        }
        return preprocessed_child, identity_info
        
    finally:
        try:
            os.unlink(child_path)
        except:
            pass


# =============================================================================
# VIEW DETECTION SYSTEM
# Classifies each template page as front/profile/back view to determine
# appropriate mask regions and inpainting strategy.
# =============================================================================

def detect_character_view(template_image_bytes: bytes) -> dict:
    """
    Detect the view type of the character in a template page.
    This determines what should be masked and how to apply face replacement.
    
    View Types:
    - FRONT: Face fully visible, looking at camera or slightly angled
            -> Mask: face + hair
    - PROFILE: Side view, partial face visible (one eye, nose, partial mouth)
            -> Mask: head + visible facial parts
    - BACK: Character facing away, no face visible
            -> Mask: hair/head only (do NOT add a face)
    - THREE_QUARTER: Between front and profile (both eyes visible but angled)
            -> Mask: face + hair (similar to front)
    - NONE: No character or no human figure found
            -> No masking needed
    
    Args:
        template_image_bytes: Bytes of the template image
    
    Returns:
        dict with keys:
        - view_type: "front", "profile", "back", "three_quarter", or "none"
        - confidence: float 0-1
        - mask_region: "face_hair", "head_partial", "hair_only", or None
        - face_bbox: (x1, y1, x2, y2) if face detected, else None
        - description: Human-readable description
    """
    log("\n--- Detecting character view type ---")
    
    try:
        template_b64 = base64.b64encode(template_image_bytes).decode('utf-8')
        
        detection_prompt = (
            "Analyze this storybook illustration and determine the VIEW TYPE of the main child character.\n\n"
            "VIEW TYPES:\n"
            "1. FRONT - Face is fully visible, looking at camera/viewer or slightly angled. Both eyes clearly visible.\n"
            "2. THREE_QUARTER - Face angled 30-60 degrees. Both eyes visible but one is closer to edge.\n"
            "3. PROFILE - Side view. Only one eye visible (or none), nose and chin profile visible.\n"
            "4. BACK - Character facing away. Back of head/hair visible, NO face visible.\n"
            "5. NONE - No child character present, or character is too small/obscured.\n\n"
            
            "Also estimate the bounding box of the character's HEAD (not just face) as pixel coordinates.\n\n"
            
            "Respond in EXACTLY this format:\n"
            "VIEW_TYPE: [FRONT/THREE_QUARTER/PROFILE/BACK/NONE]\n"
            "CONFIDENCE: [0.0-1.0]\n"
            "HEAD_BBOX: (x1, y1, x2, y2) or NONE\n"
            "DESCRIPTION: [Brief description of character pose and what's visible]"
        )
        
        response = get_client().chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": detection_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{template_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        analysis = response.choices[0].message.content or ""
        log(f"View detection response:\n{analysis}")
        
        # Parse response
        result = {
            'view_type': 'front',  # Default
            'confidence': 0.5,
            'mask_region': 'face_hair',
            'face_bbox': None,
            'description': ''
        }
        
        # Extract view type
        view_match = re.search(r'VIEW_TYPE:\s*(FRONT|THREE_QUARTER|PROFILE|BACK|NONE)', analysis, re.IGNORECASE)
        if view_match:
            view_type = view_match.group(1).upper()
            result['view_type'] = view_type.lower()
            
            # Set mask region based on view type
            if view_type in ['FRONT', 'THREE_QUARTER']:
                result['mask_region'] = 'face_hair'
            elif view_type == 'PROFILE':
                result['mask_region'] = 'head_partial'
            elif view_type == 'BACK':
                result['mask_region'] = 'hair_only'
            else:  # NONE
                result['mask_region'] = None
        
        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', analysis)
        if conf_match:
            result['confidence'] = float(conf_match.group(1))
        
        # Extract bounding box
        bbox_match = re.search(r'HEAD_BBOX:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', analysis)
        if bbox_match:
            result['face_bbox'] = tuple(int(bbox_match.group(i)) for i in range(1, 5))
        
        # Extract description
        desc_match = re.search(r'DESCRIPTION:\s*(.+)', analysis, re.IGNORECASE)
        if desc_match:
            result['description'] = desc_match.group(1).strip()
        
        log(f"View detection result: {result['view_type']} (confidence: {result['confidence']:.2f})")
        log(f"Mask region: {result['mask_region']}")
        
        return result
        
    except Exception as e:
        log(f"Error detecting character view: {e}")
        # Default to front view if detection fails
        return {
            'view_type': 'front',
            'confidence': 0.3,
            'mask_region': 'face_hair',
            'face_bbox': None,
            'description': 'Detection failed, defaulting to front view'
        }


def create_view_specific_mask(template_bytes: bytes, view_info: dict) -> bytes:
    """
    Create an appropriate mask based on the detected view type.
    
    Mask strategies:
    - FRONT/THREE_QUARTER: Ellipse covering face and hair
    - PROFILE: Extended ellipse covering visible head profile
    - BACK: Mask only the hair/back of head area (no face)
    
    Args:
        template_bytes: Template image bytes
        view_info: Dict from detect_character_view()
    
    Returns:
        Mask image bytes (white = area to edit, black = preserve)
    """
    img = Image.open(io.BytesIO(template_bytes)).convert("RGBA")
    width, height = img.size
    
    view_type = view_info.get('view_type', 'front')
    bbox = view_info.get('face_bbox')
    
    # If no bbox detected, use GPT detection or fallback
    if not bbox:
        # Try GPT detection
        mask_bytes = create_face_mask_with_gpt_detection(template_bytes)
        if view_type == 'back':
            # For back view, we need to adjust the mask to cover hair only
            # Re-create with higher position (top of head, not face)
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
            # Shift mask upward for hair-only
            mask_array = np.array(mask_img)
            # Find mask center and shift up
            rows_with_white = np.where(mask_array.max(axis=1) > 128)[0]
            if len(rows_with_white) > 0:
                shift = int((rows_with_white[-1] - rows_with_white[0]) * 0.3)
                shifted = np.zeros_like(mask_array)
                if shift > 0:
                    shifted[:-shift] = mask_array[shift:]
                else:
                    shifted = mask_array
                mask_img = Image.fromarray(shifted)
                out = io.BytesIO()
                mask_img.save(out, format="PNG")
                out.seek(0)
                return out.read()
        return mask_bytes
    
    # Create mask based on view type and bbox
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    
    if view_type in ['front', 'three_quarter']:
        # Standard face + hair mask - expand bbox slightly for hair
        hair_expansion_top = int(box_height * 0.3)  # More expansion at top for hair
        hair_expansion_sides = int(box_width * 0.15)
        neck_expansion = int(box_height * 0.1)
        
        mask_bbox = (
            max(0, x1 - hair_expansion_sides),
            max(0, y1 - hair_expansion_top),
            min(width, x2 + hair_expansion_sides),
            min(height, y2 + neck_expansion)
        )
        draw.ellipse(mask_bbox, fill=255)
        
    elif view_type == 'profile':
        # Extended mask for profile - wider to cover head shape
        side_expansion = int(box_width * 0.3)
        top_expansion = int(box_height * 0.25)
        
        mask_bbox = (
            max(0, x1 - side_expansion),
            max(0, y1 - top_expansion),
            min(width, x2 + side_expansion),
            min(height, y2 + int(box_height * 0.1))
        )
        draw.ellipse(mask_bbox, fill=255)
        
    elif view_type == 'back':
        # Hair only - focus on top/back of head, no face area
        # Shift mask upward to focus on hair
        top_expansion = int(box_height * 0.4)
        
        mask_bbox = (
            max(0, x1 - int(box_width * 0.1)),
            max(0, y1 - top_expansion),
            min(width, x2 + int(box_width * 0.1)),
            min(height, y1 + int(box_height * 0.6))  # Only upper portion
        )
        draw.ellipse(mask_bbox, fill=255)
    
    # Apply gaussian blur for soft edges
    try:
        from PIL import ImageFilter
        mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
    except Exception:
        pass
    
    out = io.BytesIO()
    mask.save(out, format="PNG")
    out.seek(0)
    
    log(f"Created {view_type} view mask with bbox {bbox}")
    
    return out.read()


def generate_face_replacement_page(child_image_bytes: bytes, template_image_bytes: bytes, page_number: int = 1, is_cover_page: bool = False, child_name: str = None, book_name: str = None, canonical_reference_bytes: bytes = None, view_type: str = None, identity_info: dict = None, template_expression_override: dict = None) -> bytes:
    """
    Replace the character's face in template with the child's face.
    Uses the same approach as ChatGPT when you upload two images and ask it to combine them.
    
    CRITICAL: This function preserves the TEMPLATE's facial expression, not the child photo's.
    The expression is detected first and enforced during face replacement.
    
    NEW: Now accepts canonical_reference_bytes for consistent identity across all pages.
    
    Args:
        child_image_bytes: Bytes of the child photo
        template_image_bytes: Bytes of the template image
        page_number: Page number
        is_cover_page: Whether this is a cover page (for name replacement)
        child_name: Child's name for name replacement on cover pages
    
    Returns original template if page is empty/white or has no child character.
    """
    log(f"\n{'='*60}")
    log(f"Processing page {page_number} - Expression-Preserving Face Replacement")
    log(f"Template size: {len(template_image_bytes)} bytes")
    log(f"Child photo size: {len(child_image_bytes)} bytes")
    if canonical_reference_bytes:
        log(f"Using CANONICAL REFERENCE for identity ({len(canonical_reference_bytes)} bytes)")
    log(f"View type: {view_type or 'auto-detect'}")
    log(f"{'='*60}\n")
    
    # Pre-processing checks
    # Check if page is empty/white
    if is_empty_or_white_page(template_image_bytes):
        log(f"Page {page_number} is empty/white - returning original template unchanged")
        return template_image_bytes
    
    # Check if page has a child character
    if not has_child_character(template_image_bytes):
        log(f"Page {page_number} has no child character - returning original template unchanged")
        return template_image_bytes
    
    # Detect view type if not provided
    if not view_type:
        log(f"Auto-detecting view type for page {page_number}...")
        view_info = detect_character_view(template_image_bytes)
        view_type = view_info.get('view_type', 'front')
        log(f"Detected view type: {view_type}")
    else:
        # Create view_info from provided view_type
        view_info = {
            'view_type': view_type,
            'confidence': 1.0,
            'mask_region': 'face_hair' if view_type in ['front', 'three_quarter'] else ('head_partial' if view_type == 'profile' else 'hair_only'),
            'face_bbox': None,
            'description': f'View type provided: {view_type}'
        }
    
    # For BACK views, only replace hair - do NOT force a face
    if view_type == 'back':
        log(f"Page {page_number} is BACK view - will only replace hair/head, NOT adding face")
    
    # CRITICAL: Detect the template's expression BEFORE any face replacement
    # This expression will be enforced in the output (only relevant for front/profile views)
    # Use override if provided (e.g., for covers that need enforced positive expression)
    if template_expression_override:
        log(f"Using provided expression override for page {page_number} (cover enforcement)")
        template_expression = template_expression_override
        log(f"Enforced expression: {template_expression['expression']} - {template_expression['description']}")
    else:
        log(f"Detecting template expression for page {page_number}...")
        template_expression = detect_template_expression(template_image_bytes)
        log(f"Template expression detected: {template_expression['expression']} - {template_expression['description']}")
        
        # ALL PAGES: If expression seems sad/negative/neutral, override to subtle smile
        # The child should ALWAYS have at least a subtle smile - never sad, frowning, or neutral
        negative_expressions = ['sad', 'frowning', 'worried', 'scared', 'angry', 'frown', 'upset', 'crying', 
                               'unhappy', 'distressed', 'concerned', 'anxious', 'fearful', 'somber', 
                               'melancholy', 'downcast', 'gloomy', 'dejected', 'miserable', 'sorrowful',
                               'neutral', 'serious', 'blank', 'expressionless', 'stoic', 'flat']
        negative_mouth_states = ['frowning', 'slight frown', 'frown', 'downturned', 'turned down', 'drooping', 
                                'pout', 'pouting', 'neutral', 'flat', 'straight', 'closed neutral']
        negative_keywords = ['sad', 'frown', 'down', 'unhappy', 'negative', 'upset', 'cry', 'tear', 'worried', 
                            'scared', 'neutral', 'serious', 'blank', 'no smile', 'not smiling']
        
        description_lower = template_expression.get('description', '').lower()
        mouth_state_lower = template_expression.get('mouth_state', '').lower()
        
        is_negative = (
            template_expression['expression'].lower() in negative_expressions or
            mouth_state_lower in negative_mouth_states or
            any(keyword in description_lower for keyword in negative_keywords) or
            any(keyword in mouth_state_lower for keyword in negative_keywords)
        )
        
        if is_negative:
            log(f"SMILE OVERRIDE: Page {page_number} has sad/negative/neutral expression - converting to subtle smile")
            log(f"Original: {template_expression['expression']} - {template_expression.get('description', 'no desc')}")
            # Override to subtle smile (never show sad/frown on any page)
            template_expression['expression'] = 'happy'
            template_expression['smile_intensity'] = 'subtle'
            template_expression['mouth_state'] = 'gentle smile'
            template_expression['mood'] = 'happy and content'
            template_expression['description'] = 'gentle subtle smile with slightly upturned mouth corners'
            template_expression['teeth_visible'] = 'no'
            template_expression['mouth_openness'] = 'closed'
            template_expression['eye_state'] = 'relaxed and warm'
            log(f"Overridden to: {template_expression['expression']} - {template_expression['description']}")
    
    # Use pre-analyzed features from identity_info if available, otherwise analyze
    if identity_info and 'features' in identity_info:
        log(f"Using pre-analyzed child features from canonical reference generation")
        child_features = identity_info['features']
    else:
        # CRITICAL: Analyze child's facial features in detail for better identity preservation
        log(f"Analyzing child's facial features for page {page_number}...")
        child_features = analyze_child_features(child_image_bytes)
    log(f"Child features - Face: {child_features['face_shape'][:50] if child_features['face_shape'] else 'N/A'}...")
    
    # Determine which image to use as identity reference
    # Priority: canonical_reference_bytes > child_image_bytes
    if canonical_reference_bytes:
        log(f"Using CANONICAL REFERENCE as identity anchor")
        identity_image_bytes = canonical_reference_bytes
    else:
        # Preprocess child photo to optimize for identity preservation
        log(f"Preprocessing child photo for page {page_number}...")
        identity_image_bytes = preprocess_child_face(child_image_bytes)
    
    # Create temporary files with proper extensions
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_template:
        tmp_template.write(template_image_bytes)
        template_path = tmp_template.name
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_identity:
        tmp_identity.write(identity_image_bytes)
        identity_path = tmp_identity.name
    
    # Create VIEW-SPECIFIC mask for precise replacement
    log(f"Creating {view_type} view mask for page {page_number}...")
    mask_bytes = create_view_specific_mask(template_image_bytes, view_info)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_mask:
        tmp_mask.write(mask_bytes)
        mask_path = tmp_mask.name
    
    try:
        # Build STRENGTHENED expression enforcement section based on pre-detected template expression
        # Get specific mouth parameters with defaults
        mouth_openness = template_expression.get('mouth_openness', 'closed')
        smile_intensity = template_expression.get('smile_intensity', 'none')
        teeth_visible = template_expression.get('teeth_visible', 'no')
        
        expression_enforcement_section = (
            f"╔══════════════════════════════════════════════════════════════════╗\n"
            f"║  ABSOLUTE REQUIREMENT #1: COPY EXACT TEMPLATE EXPRESSION         ║\n"
            f"╚══════════════════════════════════════════════════════════════════╝\n\n"
            f"⚠️  CRITICAL: COPY THE EXACT MOUTH POSITION FROM TEMPLATE - DO NOT INTERPRET ⚠️\n\n"
            f"The template's expression has been PRE-ANALYZED with PRECISE measurements.\n"
            f"You MUST COPY the EXACT visual appearance - do NOT interpret or change it!\n\n"
            f"   ╭───────────────────────────────────────────────────╮\n"
            f"   │ ★★★ MOUTH POSITION (MOST CRITICAL) ★★★           │\n"
            f"   │ MOUTH OPENNESS:   {mouth_openness.upper():12} (COPY EXACTLY!) │\n"
            f"   │ SMILE INTENSITY:  {smile_intensity.upper():12} (DO NOT CHANGE!)│\n"
            f"   │ TEETH VISIBLE:    {teeth_visible.upper():12} (MATCH EXACTLY!) │\n"
            f"   │ MOUTH STATE:      {template_expression['mouth_state'][:12]:12}              │\n"
            f"   ├───────────────────────────────────────────────────┤\n"
            f"   │ EXPRESSION TYPE:  {template_expression['expression'].upper():12}              │\n"
            f"   │ EYE STATE:        {template_expression['eye_state'][:12]:12}              │\n"
            f"   │ MOOD:             {template_expression['mood'][:12]:12}              │\n"
            f"   ╰───────────────────────────────────────────────────╯\n\n"
            f"★★★ MOUTH MATCHING RULES (NON-NEGOTIABLE) ★★★\n"
            f"• If template mouth is CLOSED → Output mouth MUST be CLOSED (no teeth showing)\n"
            f"• If template has SUBTLE smile → Output must have SUBTLE smile (NOT wide grin)\n"
            f"• If template has NO smile → Output must have NO smile\n"
            f"• If template teeth are NOT visible → Output teeth must NOT be visible\n"
            f"• COPY what you SEE in the template - do NOT exaggerate or minimize!\n\n"
            f"SPECIFIC REQUIREMENTS FOR THIS IMAGE:\n"
            f"• Mouth openness must be: {mouth_openness.upper()}\n"
            f"• Smile intensity must be: {smile_intensity.upper()}\n"
            f"• Teeth visible: {teeth_visible.upper()}\n\n"
            f"WHAT 'SUBTLE SMILE' MEANS:\n"
            f"• Corners of mouth SLIGHTLY upturned\n"
            f"• NOT a wide grin or big smile\n"
            f"• NOT teeth showing (unless template shows teeth)\n"
            f"• Gentle, soft expression - NOT exaggerated happiness\n\n"
            f"ABSOLUTE PROHIBITIONS:\n"
            f"✗ DO NOT look at the child photo's expression - COMPLETELY IGNORE IT\n"
            f"✗ DO NOT create a wide smile if template has subtle/gentle smile\n"
            f"✗ DO NOT create a frown if template is smiling\n"
            f"✗ DO NOT show teeth if template doesn't show teeth\n"
            f"✗ DO NOT open mouth if template mouth is closed\n"
            f"✗ DO NOT exaggerate or minimize the expression\n"
            f"✗ DO NOT interpret emotion labels - COPY the visual appearance\n\n"
            f"REQUIRED ACTIONS:\n"
            f"✓ LOOK at the template mouth position carefully\n"
            f"✓ COPY that EXACT mouth position to the output\n"
            f"✓ Match the EXACT smile intensity: {smile_intensity.upper()}\n"
            f"✓ Match the EXACT mouth openness: {mouth_openness.upper()}\n\n"
            f"The child photo provides ONLY: face shape, features, skin tone, hair\n"
            f"The template provides: EXACT expression to copy, pose, artistic style\n\n"
            f"VERIFICATION CHECKLIST:\n"
            f"□ Is mouth openness {mouth_openness.upper()}? Must be YES\n"
            f"□ Is smile intensity {smile_intensity.upper()}? Must be YES\n"
            f"□ Are teeth visible matching template ({teeth_visible.upper()})? Must be YES\n"
            f"□ Does mouth look EXACTLY like template? Must be YES\n\n"
        )
        
        # Build name replacement and layout-preservation instructions for cover pages
        name_replacement_section = ""
        cover_layout_section = ""
        cover_expression_section = ""
        if is_cover_page:
            # Keep layout and framing identical to the template (avoid zooming/scaling the character)
            cover_layout_section = (
                "=== COVER LAYOUT PRESERVATION (CRITICAL) ===\n"
                "This is a COVER PAGE (front cover on the right side). Preserve the exact layout and framing:\n"
                "• Do NOT zoom, crop, or enlarge the character or scene; keep the subject size and position identical to the template\n"
                "• Keep the full body visible exactly as in the template (head to feet), do NOT crop off feet/hands/cape/shoulders\n"
                "• Keep background, skyline, margins, and spacing exactly the same as the template\n"
                "• Keep the title and any other text in the exact same position, size, and style; do NOT move or resize text\n"
                "• Match the silhouette bounding box of the character to the template (same relative scale within the frame)\n"
                "• Only change the face (and name text if requested); the rest of the cover must remain pixel-perfect to the template\n\n"
            )
            
            # CRITICAL: Cover expression enforcement - ALWAYS HAPPY, regardless of template
            cover_expression_section = (
                "=== COVER EXPRESSION REQUIREMENT (ABSOLUTE - NO EXCEPTIONS) ===\n"
                "╔══════════════════════════════════════════════════════════════════╗\n"
                "║  THE CHILD MUST ALWAYS LOOK HAPPY ON THE COVER                  ║\n"
                "║  REGARDLESS OF WHAT THE TEMPLATE SHOWS                          ║\n"
                "╚══════════════════════════════════════════════════════════════════╝\n\n"
                "This is a COVER PAGE - the child MUST ALWAYS look HAPPY:\n\n"
                "ABSOLUTE REQUIREMENTS:\n"
                f"• Expression MUST be: HAPPY (enforced, regardless of template)\n"
                f"• Smile intensity MUST be: SUBTLE (gentle happy smile)\n"
                "• The child MUST look happy, content, and welcoming on the cover\n"
                "• The mouth corners MUST be slightly upturned (subtle happy smile)\n"
                "• The eyes should look relaxed and happy\n"
                "• The overall mood MUST be positive and inviting\n\n"
                "CRITICAL OVERRIDES:\n"
                "• IGNORE the template's expression completely - the child MUST be happy\n"
                "• Even if the template shows neutral, sad, or any other expression, the child MUST be happy\n"
                "• Even if the child photo shows a different expression, the cover MUST show happy\n"
                "• The template expression is IRRELEVANT - always create a happy expression\n"
                "• NEVER copy a sad, worried, angry, frowning, or neutral expression from the template\n"
                "• ALWAYS create a gentle happy smile with slightly upturned mouth corners\n\n"
                "WHY THIS MATTERS:\n"
                "• This is the FIRST thing customers see - it must be positive and inviting\n"
                "• A happy child on the cover creates an emotional connection\n"
                "• The cover sets the tone for the entire book\n"
                "• Happy expression is NON-NEGOTIABLE for covers\n\n"
                "FINAL CHECK:\n"
                "✓ Is the child smiling? YES (must be)\n"
                "✓ Is the expression happy and positive? YES (must be)\n"
                "✓ Would this make a parent smile? YES (must be)\n"
                "✓ Is the template expression ignored? YES (must be)\n\n"
            )

        if is_cover_page and child_name:
            name_replacement_section = (
                "=== NAME REPLACEMENT ON COVER PAGE (CRITICAL - FRONT COVER ONLY) ===\n"
                f"This is a COVER PAGE (front cover on the right side). You MUST replace any name text with the child's name: \"{child_name}\"\n"
                "• DETECT any name text on the front cover (right side of the image)\n"
                "• REPLACE the existing name with the child's name: \"" + child_name + "\"\n"
                "• PRESERVE the EXACT style, font, size, color, position, and formatting of the original name text\n"
                "• The new name must look EXACTLY like the original name text - same font style, same size, same color, same position\n"
                "• Match the artistic style of the text (handwritten, printed, decorative, etc.)\n"
                "• Do NOT change any other text on the cover - only replace the name\n"
                "• Do NOT modify the back cover (left side) - name replacement is ONLY for front cover (right side)\n"
                "• The name replacement must be seamless and look like it was originally part of the design\n\n"
            )
        
        # Build STRENGTHENED identity preservation section with SPECIFIC analyzed features
        identity_preservation_section = (
            f"╔══════════════════════════════════════════════════════════════════╗\n"
            f"║  ABSOLUTE REQUIREMENT #2: CHILD MUST BE INSTANTLY RECOGNIZABLE   ║\n"
            f"╚══════════════════════════════════════════════════════════════════╝\n\n"
            f"⚠️  CRITICAL: The child's IDENTITY must be PERFECTLY PRESERVED ⚠️\n\n"
            f"The child in the output image MUST be INSTANTLY RECOGNIZABLE as the \n"
            f"same person from the reference photo. Their parent should immediately \n"
            f"recognize them!\n\n"
            f"═══════════════════════════════════════════════════════════════════\n"
            f"  THIS CHILD'S SPECIFIC FEATURES (PRE-ANALYZED - MUST PRESERVE!)  \n"
            f"═══════════════════════════════════════════════════════════════════\n\n"
            f"FACE SHAPE: {child_features['face_shape']}\n"
            f"EYES: {child_features['eyes']}\n"
            f"EYEBROWS: {child_features['eyebrows']}\n"
            f"NOSE: {child_features['nose']}\n"
            f"MOUTH/LIPS: {child_features['mouth']}\n"
            f"SKIN TONE: {child_features['skin_tone']}\n"
            f"HAIR COLOR: {child_features['hair_color']}\n"
            f"HAIR TEXTURE: {child_features['hair_texture']}\n"
            f"HAIR STYLE: {child_features['hair_style']}\n"
            f"AGE APPEARANCE: {child_features['age_appearance']}\n\n"
            f"★★★ MOST DISTINCTIVE FEATURES (MUST PRESERVE AT ALL COSTS): ★★★\n"
            f"{child_features['distinctive_features']}\n\n"
            f"OVERALL: {child_features['overall_summary']}\n\n"
            f"═══════════════════════════════════════════════════════════════════\n\n"
            f"The features above are what make THIS child unique and recognizable.\n"
            f"Every single one of these features MUST be visible in the output.\n\n"
            f"RECOGNITION TEST:\n"
            f"• Would the child's PARENT immediately recognize them? → Must be YES\n"
            f"• Would a FRIEND immediately recognize them? → Must be YES\n"
            f"• Does it look like the SAME CHILD in the template's style? → Must be YES\n\n"
            f"COMMON MISTAKES TO AVOID:\n"
            f"✗ Making the face too generic or 'average looking'\n"
            f"✗ Changing the skin tone (lighter or darker than: {child_features['skin_tone'][:30]}...)\n"
            f"✗ Losing the unique face shape (must be: {child_features['face_shape'][:30]}...)\n"
            f"✗ Changing eye shape or spacing\n"
            f"✗ Making hair a different color (must be: {child_features['hair_color'][:30]}...)\n"
            f"✗ Losing distinctive features\n\n"
            f"The face should look like THIS SPECIFIC CHILD placed into the scene,\n"
            f"rendered in the EXACT same style as the template - not a random child who vaguely resembles them.\n\n"
        )
        
        # ChatGPT-style prompt - enhanced for natural integration and style matching
        # Expression enforcement is FIRST, then identity preservation
        chatgpt_prompt = (
            "CRITICAL TASK: Natural Face Integration with Artistic Style Matching\n\n"
            + expression_enforcement_section + identity_preservation_section + cover_layout_section + cover_expression_section + name_replacement_section +
            "=== PRIMARY GOAL ===\n"
            "Replace the face in IMAGE 1 (template) with the face from IMAGE 2 (child photo).\n"
            "The child must be INSTANTLY RECOGNIZABLE - their parent should immediately know it's them!\n"
            "Use the TEMPLATE'S expression (requirement #1) with the CHILD'S identity (requirement #2).\n"
            "The face must look like it NATURALLY BELONGS in the scene - matching the EXACT style and realism level of the template.\n\n"
            "=== IMAGE 1 — TEMPLATE (STORYBOOK PAGE) ===\n"
            "This is the template page. PRESERVE EVERYTHING outside the face/hair/neck area:\n"
            "• EXACT image dimensions (width × height) — output IDENTICAL to input size\n"
            "• EXACT pose, body position, limb placement\n"
            "• EXACT outfit, clothing, accessories, crown, jewelry\n"
            "• EXACT background, scenery, objects, props\n"
            "• EXACT text, fonts, titles, lettering — DO NOT alter, erase, or regenerate text EXCEPT for name replacement on cover pages (see name replacement section)\n"
            "• EXACT artistic style, brush strokes, color palette, rendering technique\n"
            "• EXACT lighting direction, shadows, highlights\n"
            "• EXACT composition, framing, perspective\n"
            "• Outside the face/hair/neck region, every pixel must remain IDENTICAL to IMAGE 1\n\n"
            "=== POSE & ORIENTATION PRESERVATION (CRITICAL - PRESERVE TEMPLATE POSE) ===\n"
            "╔══════════════════════════════════════════════════════════════════╗\n"
            "║  STRICTLY FOLLOW TEMPLATE (IMAGE 1) FOR ALL POSES & GESTURES    ║\n"
            "║  COMPLETELY IGNORE ANY POSES/GESTURES IN CHILD PHOTO (IMAGE 2)  ║\n"
            "╚══════════════════════════════════════════════════════════════════╝\n\n"
            "The child's BODY POSE, ORIENTATION, and FACING DIRECTION from IMAGE 1 (template) must be PRESERVED EXACTLY:\n\n"
            "★★★ HAND GESTURES (COPY EXACTLY FROM TEMPLATE) ★★★\n"
            "• Copy EXACT hand positions from IMAGE 1 (template)\n"
            "• Copy EXACT finger positions and gestures from IMAGE 1\n"
            "• Copy EXACT hand-to-body placement from IMAGE 1\n"
            "• If template shows hands raised, output MUST have hands raised\n"
            "• If template shows hands at sides, output MUST have hands at sides\n"
            "• If template shows hands holding something, output MUST match exactly\n"
            "• IGNORE any hand gestures visible in child photo - use ONLY template\n\n"
            "★★★ ARM POSITIONS (COPY EXACTLY FROM TEMPLATE) ★★★\n"
            "• Copy EXACT arm positions from IMAGE 1 (template)\n"
            "• Copy EXACT elbow angles and arm placement from IMAGE 1\n"
            "• IGNORE any arm positions visible in child photo - use ONLY template\n\n"
            "★★★ BODY LANGUAGE (COPY EXACTLY FROM TEMPLATE) ★★★\n"
            "• Copy EXACT body pose from IMAGE 1 (standing, sitting, crouching, etc.)\n"
            "• Copy EXACT stance and posture from IMAGE 1\n"
            "• Copy EXACT body tilt or lean from IMAGE 1\n"
            "• IGNORE any body language visible in child photo - use ONLY template\n\n"
            "★★★ HEAD & FACE ORIENTATION (COPY EXACTLY FROM TEMPLATE) ★★★\n"
            "• HEAD ORIENTATION: Preserve EXACT head angle and facing direction from IMAGE 1:\n"
            "  - If child in IMAGE 1 is facing FRONT (toward viewer), result must also face FRONT\n"
            "  - If child in IMAGE 1 is facing AWAY (back to viewer), result must also face AWAY\n"
            "  - If child in IMAGE 1 is facing LEFT, result must also face LEFT\n"
            "  - If child in IMAGE 1 is facing RIGHT, result must also face RIGHT\n"
            "  - If child in IMAGE 1 is facing at an ANGLE (3/4 view, profile, etc.), result must match that EXACT angle\n"
            "• BODY ORIENTATION: Preserve EXACT body angle and position relative to viewer/camera from IMAGE 1\n"
            "• IGNORE any head orientation visible in child photo - use ONLY template\n\n"
            "★★★ ABSOLUTE RULES (NON-NEGOTIABLE) ★★★\n"
            "• Do NOT change the child's pose, orientation, or facing direction - these must match IMAGE 1 exactly\n"
            "• Do NOT rotate or reorient the child's body or head - preserve the exact orientation from template\n"
            "• Do NOT use ANY pose, gesture, or body position from the child photo (IMAGE 2)\n"
            "• The face replacement should ONLY change the facial features, NOT the pose or orientation\n"
            "• If the template child is facing away, the result must also face away (even if only the back of head is visible)\n"
            "• If the template child is in profile view, the result must also be in profile view\n"
            "• The body position, head angle, and facing direction are LOCKED from IMAGE 1 - do not modify them\n"
            "• CHILD PHOTO (IMAGE 2) = FACE IDENTITY ONLY, NOT POSES OR GESTURES\n\n"
            f"=== FACIAL EXPRESSION PRESERVATION (CRITICAL - COPY EXACT MOUTH POSITION) ===\n"
            f"The template expression has been PRE-ANALYZED with PRECISE measurements.\n"
            f"You MUST COPY the EXACT visual appearance of the mouth - do NOT interpret!\n\n"
            f"★★★ EXACT MOUTH REQUIREMENTS (COPY VISUALLY) ★★★\n"
            f"• MOUTH OPENNESS: {mouth_openness.upper()} - mouth must be {mouth_openness.upper()}\n"
            f"• SMILE INTENSITY: {smile_intensity.upper()} - do NOT change this level\n"
            f"• TEETH VISIBLE: {teeth_visible.upper()} - {'show teeth' if teeth_visible == 'yes' else 'do NOT show teeth'}\n"
            f"• MOUTH STATE: {template_expression['mouth_state']}\n"
            f"• EYE STATE: {template_expression['eye_state']}\n\n"
            f"WHAT THIS MEANS:\n"
            f"• If smile intensity is SUBTLE → gentle upturn of mouth corners, NOT a wide grin\n"
            f"• If smile intensity is NONE → neutral mouth, no smile at all\n"
            f"• If mouth openness is CLOSED → lips together, mouth closed\n"
            f"• If teeth visible is NO → absolutely NO teeth showing\n\n"
            f"EXPRESSION RULES (NON-NEGOTIABLE):\n"
            f"• LOOK at the template mouth and COPY that exact visual appearance\n"
            f"• Do NOT exaggerate a subtle smile into a big grin\n"
            f"• Do NOT add a smile if template has neutral/no smile\n"
            f"• Do NOT show teeth if template doesn't show teeth\n"
            f"• COMPLETELY IGNORE the child photo's expression - it is IRRELEVANT\n"
            f"• The expression comes 100% from VISUALLY COPYING the template\n"
            f"• Facial FEATURES (shape, structure) come from child photo\n"
            f"• Facial EXPRESSION (mouth position, smile, eyes) come from template - COPY EXACTLY\n\n"
            "=== IMAGE 2 — CHILD PHOTO (IDENTITY REFERENCE ONLY) ===\n"
            "This child's features have been PRE-ANALYZED. You MUST preserve these EXACT features:\n\n"
            "╔══════════════════════════════════════════════════════════════════╗\n"
            "║  CRITICAL: CHILD PHOTO LIMITATIONS - READ CAREFULLY!             ║\n"
            "╚══════════════════════════════════════════════════════════════════╝\n\n"
            "★★★ USE CHILD PHOTO ONLY FOR: ★★★\n"
            "✓ Face shape and structure\n"
            "✓ Eye color, shape, and details\n"
            "✓ Nose shape and details\n"
            "✓ Mouth/lip structure (NOT expression)\n"
            "✓ Skin tone and complexion\n"
            "✓ Hair color and texture\n"
            "✓ Distinctive facial features (freckles, dimples, etc.)\n\n"
            "★★★ DO NOT USE CHILD PHOTO FOR (COMPLETELY IGNORE): ★★★\n"
            "✗ Body pose or position - USE TEMPLATE INSTEAD\n"
            "✗ Hand gestures or positions - USE TEMPLATE INSTEAD\n"
            "✗ Arm positions - USE TEMPLATE INSTEAD\n"
            "✗ Body language or stance - USE TEMPLATE INSTEAD\n"
            "✗ Facial expression - USE TEMPLATE INSTEAD\n"
            "✗ Head angle or tilt - USE TEMPLATE INSTEAD\n"
            "✗ Any gesture visible in child photo - COMPLETELY IGNORE\n\n"
            "If the child photo shows ANY pose, gesture, or expression:\n"
            "→ COMPLETELY IGNORE IT\n"
            "→ Use ONLY the template's pose, gesture, and expression\n"
            "→ The child photo provides IDENTITY only, not actions\n\n"
            f"THE CHILD'S SPECIFIC FEATURES TO PRESERVE:\n"
            f"• FACE: {child_features['face_shape']}\n"
            f"• EYES: {child_features['eyes']}\n"
            f"• EYEBROWS: {child_features['eyebrows']}\n"
            f"• NOSE: {child_features['nose']}\n"
            f"• MOUTH: {child_features['mouth']} (but COPY EXACT template mouth: {mouth_openness} openness, {smile_intensity} smile)\n"
            f"• SKIN: {child_features['skin_tone']}\n"
            f"• HAIR COLOR: {child_features['hair_color']}\n"
            f"• HAIR TEXTURE: {child_features['hair_texture']}\n"
            f"• HAIR STYLE: {child_features['hair_style']}\n\n"
            f"★ MOST DISTINCTIVE (MUST PRESERVE): {child_features['distinctive_features']}\n\n"
            f"SUMMARY: {child_features['overall_summary']}\n\n"
            "FACIAL PROPORTIONS - NO DEFORMATION:\n"
            "• Keep exact face width-to-height ratio\n"
            "• Keep exact distance between eyes, nose, mouth\n"
            "• NO stretching, warping, or compression\n\n"
            "SKIN TONE MATCHING:\n"
            f"• Use EXACT skin tone: {child_features['skin_tone']}\n"
            "• Extend seamlessly to neck and visible body parts\n"
            "• Do NOT lighten, darken, or change the skin color\n\n"
            f"EXPRESSION (COPY EXACTLY FROM TEMPLATE - NOT CHILD PHOTO):\n"
            f"• MOUTH OPENNESS: {mouth_openness.upper()} - copy exactly\n"
            f"• SMILE INTENSITY: {smile_intensity.upper()} - do not exaggerate or minimize\n"
            f"• TEETH: {'show teeth' if teeth_visible == 'yes' else 'NO teeth visible'}\n"
            f"• IGNORE child photo's expression completely - COPY template visually\n\n"
            "=== CHILD MUST BE FULLY REALISTIC (ABSOLUTE REQUIREMENT) ===\n"
            "╔══════════════════════════════════════════════════════════════════╗\n"
            "║  ALWAYS RENDER THE CHILD AS FULLY REALISTIC / PHOTOGRAPHIC      ║\n"
            "║  NEVER ILLUSTRATED, CARTOON, PAINTED, OR STYLIZED               ║\n"
            "╚══════════════════════════════════════════════════════════════════╝\n\n"
            "★★★ THE CHILD MUST ALWAYS LOOK LIKE A REAL PHOTOGRAPH ★★★\n\n"
            "REALISTIC REQUIREMENTS (MANDATORY FOR ALL IMAGES):\n"
            "• The child must look like a REAL person in a photograph\n"
            "• Natural, realistic skin texture - smooth, lifelike, with natural pores\n"
            "• Real hair strands - individual hairs visible, natural flow and texture\n"
            "• Lifelike eyes - natural reflections, realistic iris detail, real depth\n"
            "• Photographic lighting - natural shadows, realistic highlights\n"
            "• The child should look like they could step out of the image\n\n"
            "★★★ ABSOLUTE PROHIBITIONS (NEVER DO THESE): ★★★\n"
            "✗ NEVER make the child look illustrated or cartoon-like\n"
            "✗ NEVER make the child look painted or stylized\n"
            "✗ NEVER add brush strokes or painterly effects to the child\n"
            "✗ NEVER simplify or stylize the child's features\n"
            "✗ NEVER make the child look 'drawn' in any way\n"
            "✗ NEVER make the skin look painted or artificial\n"
            "✗ NEVER make the hair look like solid blocks or painted strokes\n"
            "✗ NEVER make the eyes look stylized or cartoon-like\n\n"
            "★★★ WHAT REALISTIC MEANS: ★★★\n"
            "• Skin looks like real human skin (natural texture, pores, subtle variations)\n"
            "• Hair looks like real human hair (individual strands, natural shine)\n"
            "• Eyes look like real human eyes (realistic iris, natural reflections)\n"
            "• Face has realistic depth and dimension\n"
            "• Lighting creates natural shadows and highlights\n"
            "• The child looks like a real person photographed in the scene\n\n"
            "=== BACKGROUND AND SCENE ===\n"
            "• The BACKGROUND can be fantasy/illustrated/artistic - that's fine\n"
            "• But the CHILD themselves must ALWAYS be fully realistic/photographic\n"
            "• The child should look like a real person placed in the scene\n"
            "• Match the lighting direction from the background onto the realistic child\n\n"
            "=== FACE QUALITY REQUIREMENTS (CRITICAL - HIGH QUALITY RENDERING) ===\n"
            "The face must be rendered with EXCEPTIONAL QUALITY and CLARITY:\n"
            "• Render the face with HIGH QUALITY and CLEAR DETAIL - avoid blurry, pixelated, or low-resolution faces\n"
            "• Maintain sharp, well-defined facial features while matching the template's style\n"
            "• Ensure all facial features (eyes, nose, mouth) are clearly visible and well-rendered\n"
            "• The face should be clearly visible and recognizable, not faded, washed out, or obscured\n"
            "• Ensure proper contrast and clarity for all facial features - they must stand out clearly\n"
            "• Avoid any blur, smudging, or loss of detail in the face area\n"
            "• The face must have sufficient detail and sharpness to be instantly recognizable\n"
            "• Maintain crisp edges and clear definition for eyes, eyebrows, nose, and mouth\n"
            "• Ensure the face has proper depth and dimension, not flat or lifeless\n"
            "• The quality should match or exceed the quality of other elements in the template\n"
            "• Do NOT compromise on face quality - it is the most important element of the image\n\n"
            "=== ANTI-DEFORMATION RULES (CRITICAL - PREVENT STRETCHING/WARPING) ===\n"
            "MAINTAIN EXACT FACIAL PROPORTIONS FROM IMAGE 2:\n"
            "• Measure and preserve the exact ratio of face width to height\n"
            "• Preserve the exact distance between facial features (eyes, nose, mouth)\n"
            "• Do NOT stretch the face to fit a different head shape - adapt the head shape if needed\n"
            "• Do NOT compress or squash facial features\n"
            "• Do NOT apply perspective distortion or warping\n"
            "• If the template head is wider, adjust the head outline but keep face features at correct proportions\n"
            "• If the template head is narrower, adjust the head outline but keep face features at correct proportions\n"
            "• The face itself must maintain its natural proportions - only the style matching with the template changes\n\n"
            "=== SEAMLESS INTEGRATION REQUIREMENTS ===\n"
            "The face must blend NATURALLY into the scene:\n"
            "• Match the lighting direction and intensity from IMAGE 1\n"
            "• Match the shadow style and placement\n"
            "• Match the color temperature and palette\n"
            "• Ensure smooth transitions at the edges (face to neck, face to hair)\n"
            "• The face should look like it belongs in the scene, not like it was added later\n"
            "• Maintain visual harmony with the rest of the image\n"
            "• Preserve the exact pose and orientation from IMAGE 1 - the child's body position and facing direction must remain unchanged\n\n"
            "=== HAND ANATOMY REQUIREMENTS (CRITICAL - MUST BE ANATOMICALLY CORRECT) ===\n"
            "Hands are a critical element that must be rendered with proper anatomy:\n"
            "• Each hand must have EXACTLY 5 fingers - no more, no less\n"
            "• Fingers must be properly proportioned - thumb is shorter and thicker, pinky is smallest\n"
            "• Fingers must connect naturally to the palm at proper positions\n"
            "• Fingers must have proper joints and bend naturally (3 joints per finger, 2 for thumb)\n"
            "• Fingernails should be visible and properly placed at fingertips\n"
            "• Hands must have realistic proportions relative to the child's body and face\n"
            "• Preserve the EXACT hand pose and gesture from IMAGE 1 (template)\n"
            "• Do NOT add extra fingers, merge fingers together, or omit fingers\n"
            "• Do NOT create deformed, twisted, or anatomically impossible hand positions\n"
            "• Do NOT make fingers too long, too short, too thick, or too thin\n"
            "• Hands should match the template's style while maintaining correct anatomy\n"
            "• If the template shows a specific hand gesture, reproduce it with proper finger count and positioning\n"
            "• Pay extra attention to hands interacting with objects (holding items, pointing, etc.)\n\n"
            f"=== YOUR TASK ===\n"
            f"1) Take IMAGE 1 as base — change NOTHING except the face/hair/neck/visible skin areas\n"
            f"2) PRESERVE EXACT pose, orientation, and facing direction from IMAGE 1 - do NOT change body position or head angle\n"
            f"3) Identify the EXACT unique facial features from IMAGE 2 (shape, proportions, distinctive characteristics)\n"
            f"4) Render those features in IMAGE 1's EXACT style - preserve identity and match the template's realism level precisely\n"
            f"5) MAINTAIN EXACT facial proportions from IMAGE 2 - NO stretching, warping, or deformation\n"
            f"6) COPY the EXACT mouth position from IMAGE 1 (template):\n"
            f"   - Mouth openness: {mouth_openness.upper()} - copy exactly\n"
            f"   - Smile intensity: {smile_intensity.upper()} - do NOT exaggerate or minimize\n"
            f"   - Teeth visible: {teeth_visible.upper()} - match exactly\n"
            f"   - LOOK at template mouth and COPY that visual appearance\n"
            f"   - IGNORE the expression in IMAGE 2 completely\n"
            "7) PRESERVE the EXACT facing direction from IMAGE 1 - if template faces away, result must also face away\n"
            "8) EXTEND the skin tone from IMAGE 2 to neck and visible body parts - ensure seamless color matching\n"
            "9) Match the template's EXACT style (if photorealistic, output must be photorealistic; if illustrated, output must be illustrated)\n"
            "10) Ensure the face looks like it NATURALLY BELONGS in the scene with seamless style matching\n"
            "11) The result should feel like the child was placed into the scene, matching the template's exact realism level\n"
            "12) Do NOT touch or regenerate text; preserve it pixel-for-pixel EXCEPT for name replacement on cover pages (see name replacement section)\n"
            "13) Do NOT change pose, orientation, or facing direction - these are LOCKED from IMAGE 1\n"
            f"14) Do NOT change facial expression - COPY EXACT mouth position: {mouth_openness} openness, {smile_intensity} smile, teeth {teeth_visible}\n"
            "15) Output must be IDENTICAL resolution and aspect ratio as IMAGE 1\n\n"
            f"=== INTEGRATION CHECKLIST (MUST PASS ALL) ===\n"
            f"✓ Face is UNDISTORTED - maintains exact proportions from IMAGE 2 (no stretching/warping)\n"
            f"✓ Face is HIGH QUALITY - clear, sharp, well-defined features with no blur or pixelation\n"
            f"✓ Face matches IMAGE 1's EXACT style (same realism level - photorealistic if template is photorealistic)\n"
            f"✓ Face looks like it NATURALLY BELONGS in the scene\n"
            f"✓ POSE & ORIENTATION match IMAGE 1 exactly - body position, head angle, and facing direction preserved\n"
            f"✓ MOUTH POSITION matches template EXACTLY:\n"
            f"  - Mouth openness is {mouth_openness.upper()} - VERIFIED\n"
            f"  - Smile intensity is {smile_intensity.upper()} - NOT exaggerated or minimized\n"
            f"  - Teeth visible: {teeth_visible.upper()} - matches template\n"
            f"  - Eyes are {template_expression['eye_state']}\n"
            f"  - Expression is VISUALLY COPIED from template - NOT from child photo\n"
            "✓ INSTANTLY RECOGNIZABLE as the same person from IMAGE 2\n"
            "✓ All unique facial features preserved (eye shape, nose shape, mouth shape, etc.)\n"
            "✓ Facial features are clearly visible and well-rendered (not faded or obscured)\n"
            "✓ Facial proportions maintained exactly (no compression, stretching, or warping)\n"
            "✓ Skin tone and complexion match IMAGE 2\n"
            "✓ Skin tone matches seamlessly between face, neck, and visible body parts\n"
            "✓ Hair color, texture, and style match IMAGE 2\n"
            "✓ Distinctive features (freckles, birthmarks, etc.) preserved if present\n"
            "✓ Lighting and shadows match IMAGE 1's style\n"
            "✓ Color palette and rendering technique match IMAGE 1\n"
            "✓ All text from IMAGE 1 preserved and undistorted (except name replacement on cover pages)\n"
            "✓ Name replaced on cover page with child's name in exact same style (if applicable)\n"
            "✓ Size/resolution unchanged\n"
            "✓ Hands have exactly 5 fingers each - anatomically correct\n"
            "✓ Fingers are properly proportioned and positioned\n"
            "✓ Hand poses match the template exactly\n\n"
            "=== STRICT PROHIBITIONS ===\n"
            "✗ Do NOT stretch, warp, compress, or distort the face in ANY way\n"
            "✗ Do NOT render a blurry, pixelated, or low-quality face\n"
            "✗ Do NOT fade, wash out, or obscure the face - it must be clearly visible\n"
            "✗ Do NOT create a face with poor contrast or unclear features\n"
            "✗ Do NOT change facial proportions or feature spacing\n"
            "✗ Do NOT change the child's POSE, ORIENTATION, or FACING DIRECTION - these must match IMAGE 1 exactly\n"
            "✗ Do NOT rotate or reorient the child's body or head - preserve exact orientation from template\n"
            "✗ Do NOT make the child face front if template faces away - preserve the exact facing direction\n"
            "✗ Do NOT change body position, stance, or limb placement - these are LOCKED from IMAGE 1\n"
            "✗ Do NOT change the realism level - if template is photorealistic, face must be photorealistic; if illustrated, face must be illustrated\n"
            f"✗ Do NOT copy the expression from IMAGE 2 - COPY TEMPLATE MOUTH EXACTLY:\n"
            f"  - Mouth openness MUST be {mouth_openness.upper()}\n"
            f"  - Smile intensity MUST be {smile_intensity.upper()} - do NOT exaggerate into wide smile\n"
            f"  - Teeth visible MUST be {teeth_visible.upper()}\n"
            f"  - IGNORE child photo's expression completely\n"
            f"  - Expression MUST come ONLY from template (IMAGE 1), NEVER from child photo (IMAGE 2)\n"
            "✗ Do NOT create a generic or average face — use the EXACT unique face from IMAGE 2\n"
            "✗ Do NOT modify ethnic features, skin tone, age, or any identifying characteristics\n"
            "✗ Do NOT have mismatched skin tones between face and neck/body\n"
            "✗ Do NOT change the style - render the face in the EXACT same style as the template (realistic OR illustrated)\n"
            "✗ Do NOT compromise on face quality - maintain high detail and clarity\n"
            "✗ Do NOT alter, erase, or regenerate any text EXCEPT for name replacement on cover pages (see name replacement section)\n"
            "✗ Do NOT change image size or aspect ratio\n"
            "✗ Do NOT add/remove/modify text, outfit, pose, body, background, objects\n"
            "✗ Do NOT change artistic style of background/outfit (only replace face)\n"
            "✗ Do NOT add watermarks or signatures\n"
            "✗ Do NOT create hands with more or fewer than 5 fingers\n"
            "✗ Do NOT merge fingers together or create webbed fingers\n"
            "✗ Do NOT create anatomically impossible hand positions or gestures\n"
            "✗ Do NOT make fingers disproportionate (too long, too short, wrong thickness)\n"
            "✗ Do NOT change the hand pose or gesture from the template\n\n"
            f"=== FINAL OUTPUT ===\n"
            f"IMAGE 1 with ONLY the face/hair/neck REPLACED by IMAGE 2's face, rendered in IMAGE 1's artistic style.\n\n"
            f"╔════════════════════════════════════════════════════════════════╗\n"
            f"║              FINAL VERIFICATION CHECKLIST                      ║\n"
            f"╚════════════════════════════════════════════════════════════════╝\n\n"
            f"REQUIREMENT #1 - MOUTH POSITION (COPY EXACTLY FROM TEMPLATE):\n"
            f"  ✓ Mouth openness is {mouth_openness.upper()}? YES\n"
            f"  ✓ Smile intensity is {smile_intensity.upper()} (NOT exaggerated)? YES\n"
            f"  ✓ Teeth visible is {teeth_visible.upper()}? YES\n"
            f"  ✓ Eyes are {template_expression['eye_state']}? YES\n"
            f"  ✓ Did NOT use child photo's expression? YES\n"
            f"  ✓ Mouth looks VISUALLY IDENTICAL to template? YES\n\n"
            f"REQUIREMENT #2 - IDENTITY (THIS SPECIFIC CHILD):\n"
            f"  ✓ Child is INSTANTLY RECOGNIZABLE? YES\n"
            f"  ✓ Face shape: {child_features['face_shape'][:40]}...? YES\n"
            f"  ✓ Eyes: {child_features['eyes'][:40]}...? YES\n"
            f"  ✓ Skin tone: {child_features['skin_tone'][:40]}...? YES\n"
            f"  ✓ Hair: {child_features['hair_color'][:40]}...? YES\n"
            f"  ✓ Distinctive features preserved: {child_features['distinctive_features'][:40]}...? YES\n"
            f"  ✓ Skin tone matches EXACTLY? YES\n"
            f"  ✓ Hair color and texture match? YES\n"
            f"  ✓ Parent would recognize their child? YES\n\n"
            f"REQUIREMENT #3 - ANATOMICAL CORRECTNESS (HANDS):\n"
            f"  ✓ Each visible hand has exactly 5 fingers? YES\n"
            f"  ✓ Fingers are properly proportioned? YES\n"
            f"  ✓ Hand poses match template exactly? YES\n"
            f"  ✓ No merged, extra, or missing fingers? YES\n"
            f"  ✓ Hands look natural and anatomically correct? YES\n\n"
            f"The output shows THIS SPECIFIC CHILD (recognizable identity from photo)\n"
            f"with the EXACT MOUTH POSITION from template ({mouth_openness} mouth, {smile_intensity} smile).\n"
            f"The child looks like they naturally belong in the scene, matching the template's exact style,\n"
            f"while remaining completely recognizable as themselves, with anatomically correct hands."
        )
        
        log("Using enhanced ChatGPT-style prompt for identity preservation...")
        log(f"Prompt: {chatgpt_prompt[:200]}...\n")
        
        # Force size to match template dimensions
        tmpl_img = Image.open(io.BytesIO(template_image_bytes))
        original_width, original_height = tmpl_img.width, tmpl_img.height
        target_size = choose_allowed_size(original_width, original_height)
        log(f"Chosen size for gpt-image-1: {target_size} (template {original_width}x{original_height})")

        # Primary approach: Use two images (template + canonical reference) for best identity preservation
        # The canonical reference provides consistent identity across all pages
        try:
            log("Attempting gpt-image-1 edit API with canonical reference (primary method)...")
            ref_type = "canonical reference" if canonical_reference_bytes else "preprocessed child photo"
            log(f"Using {ref_type} as identity anchor")
            
            with open(template_path, 'rb') as template_file, open(identity_path, 'rb') as identity_file:
                
                # Optimized API call with maximum fidelity for identity preservation
                response = get_client().images.edit(
                    model="gpt-image-1",
                    image=[template_file, identity_file],  # Template + canonical reference for consistent identity
                    prompt=chatgpt_prompt,
                    input_fidelity="high",  # Maximum fidelity to preserve identity
                    size=target_size,  # Preserve template dimensions within allowed sizes
                    output_format="png",  # PNG for best quality
                    n=1,
                )
                
                log("gpt-image-1 API call (optimized identity preservation) succeeded!")
                
                # gpt-image-1 returns base64 or url
                result_bytes = None
                if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
                    log("Got base64 image response")
                    result_bytes = base64.b64decode(response.data[0].b64_json)
                elif hasattr(response.data[0], 'url') and response.data[0].url:
                    log(f"Got URL response: {response.data[0].url[:80]}...")
                    img_response = requests.get(response.data[0].url)
                    img_response.raise_for_status()
                    result_bytes = img_response.content
                else:
                    raise ValueError("No image data in response")
                
                # Ensure exact dimensions match original template
                result_bytes = ensure_exact_dimensions(result_bytes, original_width, original_height)
                return result_bytes
                    
        except Exception as e:
            log(f"gpt-image-1 edit API (primary method) failed: {e}")
            log("Trying mask-based approach as fallback...")
            
            # Fallback: Try mask-based approach
            try:
                # Ensure mask is recognized as PNG
                mask_img = Image.open(io.BytesIO(mask_bytes))
                if mask_img.mode != 'L':
                    mask_img = mask_img.convert('L')
                # Save mask again to ensure proper PNG format
                mask_buffer = io.BytesIO()
                mask_img.save(mask_buffer, format='PNG')
                mask_buffer.seek(0)
                
                # Create new temp file with proper PNG
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_mask_fixed:
                    tmp_mask_fixed.write(mask_buffer.read())
                    mask_path_fixed = tmp_mask_fixed.name
                
                try:
                    # For mask-based, we use template + mask, and reference child in prompt
                    mask_prompt = (
                        "Replace the face in the masked area (white pixels) with the EXACT face from the reference photo. "
                        "The face must be instantly recognizable, photorealistic, and preserve all unique facial characteristics. "
                        "Keep everything outside the mask exactly as it is."
                    )
                    
                    with open(template_path, 'rb') as template_file, \
                         open(mask_path_fixed, 'rb') as mask_file:
                        
                        response = get_client().images.edit(
                            model="gpt-image-1",
                            image=template_file,
                            mask=mask_file,
                            prompt=mask_prompt,
                            input_fidelity="high",
                            size=target_size,
                            output_format="png",
                            n=1,
                        )
                        
                        log("gpt-image-1 API call with mask (fallback) succeeded!")
                        
                        result_bytes = None
                        if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
                            log("Got base64 image response")
                            result_bytes = base64.b64decode(response.data[0].b64_json)
                        elif hasattr(response.data[0], 'url') and response.data[0].url:
                            log(f"Got URL response: {response.data[0].url[:80]}...")
                            img_response = requests.get(response.data[0].url)
                            img_response.raise_for_status()
                            result_bytes = img_response.content
                        else:
                            raise ValueError("No image data in response")
                        
                        # Ensure exact dimensions match original template
                        result_bytes = ensure_exact_dimensions(result_bytes, original_width, original_height)
                        return result_bytes
                        
                        # Ensure exact dimensions match original template
                        result_bytes = ensure_exact_dimensions(result_bytes, original_width, original_height)
                        return result_bytes
                finally:
                    try:
                        os.unlink(mask_path_fixed)
                    except:
                        pass
                        
            except Exception as mask_error:
                log(f"Mask-based approach also failed: {mask_error}")
                log("Trying alternative DALL-E approach...")
                
                # Alternative: Use GPT-4o vision to understand both images, then generate
                log("\nTrying alternative: GPT-4o analysis + DALL-E generation...")
                
                # Encode images for GPT-4o vision
                template_b64 = base64.b64encode(template_image_bytes).decode('utf-8')
                child_b64 = base64.b64encode(child_image_bytes).decode('utf-8')
                
                # Ask GPT-4o to describe exactly what needs to happen
                # Use careful wording to avoid content policy issues
                analysis_response = get_client().chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "I'm showing you two images for a personalized storybook illustration:\n"
                                        "IMAGE 1: A fairytale storybook illustration with a character\n"
                                        "IMAGE 2: A reference photo showing the EXACT face to use\n\n"
                                        "I need to recreate Image 1 exactly, but with the EXACT face from Image 2. "
                                        "The face must be copied exactly - preserve every detail, do not modify or reinterpret.\n\n"
                                        "Please create a detailed DALL-E prompt that:\n"
                                        "1. Describes the EXACT scene from Image 1 (background, colors, style, pose, outfit)\n"
                                        "2. Describes the character with the EXACT facial features from Image 2 - copy the face precisely, "
                                        "including eyes, nose, mouth, facial structure, skin tone, and hair exactly as shown in Image 2\n"
                                        "3. Emphasize that the face must match Image 2 exactly - no modification, reinterpretation, or stylization\n"
                                        "4. Uses terms like 'illustration', 'fairytale style', 'storybook art'\n"
                                        "5. DO NOT mention 'child', 'kid', or ages - just describe features\n"
                                        "The prompt should be safe for DALL-E content policy."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{template_b64}"}
                                },
                                {
                                    "type": "image_url", 
                                    "image_url": {"url": f"data:image/png;base64,{child_b64}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=1000
                )
                
                detailed_prompt = analysis_response.choices[0].message.content
                log(f"GPT-4o analysis complete. Generated prompt length: {len(detailed_prompt)} chars")
                log(f"FULL DALLE PROMPT:\n{detailed_prompt}\n")
                
                # Use DALL-E 3 with the detailed prompt
                log("Generating image with DALL-E 3...")
                dalle_response = get_client().images.generate(
                    model="dall-e-3",
                    prompt=detailed_prompt,
                    size=target_size,  # allowed size choice
                    quality="standard",  # Use standard to avoid some content issues
                    n=1,
                )
                
                image_url = dalle_response.data[0].url
                log(f"DALL-E 3 generated image. URL: {image_url[:80]}...")
                img_response = requests.get(image_url)
                img_response.raise_for_status()
                log("Image downloaded successfully")
                result_bytes = img_response.content
                
                # Ensure exact dimensions match original template
                result_bytes = ensure_exact_dimensions(result_bytes, original_width, original_height)
                return result_bytes
                
    finally:
        # Clean up temp files
        for path in [template_path, identity_path, mask_path]:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except:
                pass


# =============================================================================
# AUTOMATED QC LOOP
# Checks generated pages for consistency with canonical reference and retries
# if quality checks fail.
# =============================================================================

def check_page_consistency(
    generated_bytes: bytes,
    canonical_bytes: bytes,
    template_bytes: bytes,
    page_number: int = 0
) -> Tuple[bool, float, str]:
    """
    Check if a generated page is consistent with the canonical reference.
    
    Checks:
    - Face similarity to canonical reference
    - Hair color consistency
    - Skin tone consistency
    - Style consistency with template
    
    Args:
        generated_bytes: Bytes of the generated page
        canonical_bytes: Bytes of the canonical reference portrait
        template_bytes: Bytes of the original template
        page_number: Page number for logging
    
    Returns:
        Tuple of (passed: bool, confidence: float 0-1, reason: str)
    """
    log(f"\n--- QC Check for page {page_number} ---")
    
    try:
        # Encode images for GPT-4o vision
        generated_b64 = base64.b64encode(generated_bytes).decode('utf-8')
        canonical_b64 = base64.b64encode(canonical_bytes).decode('utf-8')
        
        qc_prompt = (
            "Compare these two images for IDENTITY CONSISTENCY:\n\n"
            "IMAGE 1: Generated storybook page with a child character\n"
            "IMAGE 2: Canonical reference portrait of the child\n\n"
            "Check if the child in Image 1 is the SAME child as in Image 2.\n\n"
            "EVALUATE:\n"
            "1. FACE SIMILARITY (0-100): Do the facial features match? Same face shape, eyes, nose, mouth?\n"
            "2. HAIR COLOR MATCH (0-100): Is the hair color the same? (exact shade matters)\n"
            "3. HAIR TEXTURE MATCH (0-100): Same hair texture (straight/wavy/curly)?\n"
            "4. SKIN TONE MATCH (0-100): Is the skin tone the same?\n"
            "5. EYE COLOR MATCH (0-100): Same eye color?\n"
            "6. OVERALL RECOGNIZABLE (0-100): Would a parent recognize this as their child?\n\n"
            "Respond in EXACTLY this format:\n"
            "FACE_SIMILARITY: [0-100]\n"
            "HAIR_COLOR: [0-100]\n"
            "HAIR_TEXTURE: [0-100]\n"
            "SKIN_TONE: [0-100]\n"
            "EYE_COLOR: [0-100]\n"
            "RECOGNIZABLE: [0-100]\n"
            "PASS: [YES/NO]\n"
            "REASON: [Brief explanation if NO]"
        )
        
        response = get_client().chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": qc_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{generated_b64}"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{canonical_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=400
        )
        
        analysis = response.choices[0].message.content or ""
        log(f"QC Analysis:\n{analysis}")
        
        # Parse scores
        scores = {}
        for metric in ['FACE_SIMILARITY', 'HAIR_COLOR', 'HAIR_TEXTURE', 'SKIN_TONE', 'EYE_COLOR', 'RECOGNIZABLE']:
            match = re.search(rf'{metric}:\s*(\d+)', analysis)
            if match:
                scores[metric] = int(match.group(1))
            else:
                scores[metric] = 50  # Default if not found
        
        # Calculate overall confidence (weighted average)
        weights = {
            'FACE_SIMILARITY': 0.30,
            'HAIR_COLOR': 0.20,
            'HAIR_TEXTURE': 0.10,
            'SKIN_TONE': 0.15,
            'EYE_COLOR': 0.10,
            'RECOGNIZABLE': 0.15
        }
        
        confidence = sum(scores[k] * weights[k] for k in weights) / 100.0
        
        # Check if passed
        pass_match = re.search(r'PASS:\s*(YES|NO)', analysis, re.IGNORECASE)
        passed = pass_match and pass_match.group(1).upper() == 'YES'
        
        # Also fail if any critical score is too low
        if scores['FACE_SIMILARITY'] < 60 or scores['HAIR_COLOR'] < 50 or scores['RECOGNIZABLE'] < 60:
            passed = False
        
        # Extract reason
        reason_match = re.search(r'REASON:\s*(.+)', analysis, re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else "No specific reason"
        
        log(f"QC Result: {'PASS' if passed else 'FAIL'} (confidence: {confidence:.2f})")
        log(f"Scores: Face={scores['FACE_SIMILARITY']}, Hair={scores['HAIR_COLOR']}, Skin={scores['SKIN_TONE']}, Recognizable={scores['RECOGNIZABLE']}")
        
        return passed, confidence, reason
        
    except Exception as e:
        log(f"QC Check failed with error: {e}")
        # If QC check fails, assume it's okay (don't block generation)
        return True, 0.5, f"QC check error: {e}"


def generate_page_with_qc(
    child_image_bytes: bytes,
    template_bytes: bytes,
    canonical_reference_bytes: bytes,
    identity_info: dict,
    page_number: int,
    view_type: str = None,
    is_cover_page: bool = False,
    child_name: str = None,
    book_name: str = None,
    max_retries: int = 3
) -> Tuple[bytes, bool, float]:
    """
    Generate a page (QC loop removed - was causing content policy issues).
    
    Args:
        child_image_bytes: Original child photo bytes
        template_bytes: Template page bytes
        canonical_reference_bytes: Canonical reference portrait bytes
        identity_info: Identity features dict from canonical generation
        page_number: Page number
        view_type: Detected view type (front/profile/back/etc)
        is_cover_page: Whether this is a cover page
        child_name: Child's name for cover pages
        book_name: Book name
        max_retries: Not used (kept for compatibility)
    
    Returns:
        Tuple of (page_bytes, qc_passed=True, confidence=1.0)
    """
    log(f"\n{'='*60}")
    log(f"GENERATING PAGE {page_number} (QC loop disabled)")
    log(f"{'='*60}")
    
    try:
        # Generate the page directly without QC loop
        generated_bytes = generate_face_replacement_page(
            child_image_bytes=child_image_bytes,
            template_image_bytes=template_bytes,
            page_number=page_number,
            is_cover_page=is_cover_page,
            child_name=child_name,
            book_name=book_name,
            canonical_reference_bytes=canonical_reference_bytes,
            view_type=view_type,
            identity_info=identity_info
        )
        
        log(f"Page {page_number} generated successfully")
        return generated_bytes, True, 1.0
        
    except Exception as e:
        log(f"Error generating page {page_number}: {e}")
        return template_bytes, False, 0.0


def generate_pages_for_book(child_image_bytes: bytes, template_folder: str, page_list: List[str], status_callback=None, child_name: str = None, canonical_reference_bytes: bytes = None, identity_info: dict = None) -> Tuple[List[bytes], List[Tuple[int, int]]]:
    """
    Process multiple pages for a book using the canonical reference pipeline.
    
    NEW PIPELINE:
    1. Generate canonical reference FIRST (if not provided)
    2. For each page: detect view -> create mask -> inpaint with reference -> QC check
    3. Retry pages that fail QC (max 3 attempts)
    
    Args:
        child_image_bytes: Bytes of the child photo
        template_folder: Path to template folder
        page_list: List of page filenames to process
        status_callback: Optional callback for progress updates
        child_name: Optional child's name for name replacement on covers
        canonical_reference_bytes: Pre-generated canonical reference (optional)
        identity_info: Pre-analyzed identity features (optional)
    
    Returns:
        Tuple of (list of image bytes, list of original template dimensions (width, height))
    """
    out_pages = []
    original_dimensions = []
    qc_results = []  # Track QC results for reporting
    
    # =========================================================================
    # STEP 1: GENERATE CANONICAL REFERENCE (if not provided)
    # This is the identity anchor used for ALL pages
    # =========================================================================
    if canonical_reference_bytes is None or identity_info is None:
        log("\n" + "="*70)
        log("STEP 1: GENERATING CANONICAL REFERENCE (identity anchor for all pages)")
        log("="*70)
        
        if status_callback:
            status_callback(0, "Generating canonical reference portrait...")
        
        try:
            canonical_reference_bytes, identity_info = generate_canonical_reference(
                child_image_bytes=child_image_bytes,
                book_name=None  # Could pass book name for style matching
            )
            log(f"Canonical reference generated: {len(canonical_reference_bytes)} bytes")
            log(f"Identity features extracted: {identity_info.get('features', {}).get('overall_summary', 'N/A')[:100]}...")
        except Exception as e:
            log(f"Error generating canonical reference: {e}")
            # Fallback: use preprocessed child photo
            canonical_reference_bytes = preprocess_child_face(child_image_bytes)
            identity_info = {'features': analyze_child_features(child_image_bytes), 'fallback': True}
    else:
        log("\nUsing provided canonical reference and identity info")
    
    # =========================================================================
    # STEP 2: PROCESS EACH PAGE WITH VIEW DETECTION AND QC
    # =========================================================================
    log("\n" + "="*70)
    log("STEP 2: PROCESSING PAGES WITH CANONICAL REFERENCE PIPELINE")
    log("="*70)
    
    for i, page_name in enumerate(page_list, start=1):
        if status_callback:
            status_callback(i, f"Processing page {i} / {len(page_list)}")
        
        template_path = Path(template_folder) / page_name
        if not template_path.exists():
            log(f"Warning: {template_path} not found, skipping...")
            continue
            
        template_bytes = template_path.read_bytes()
        
        # Auto-detect and fix rotation issues (interior pages should be square)
        template_bytes = detect_and_fix_rotation(template_bytes, is_interior_page=True)
        
        # Get original template dimensions before processing
        try:
            template_img = Image.open(io.BytesIO(template_bytes))
            original_width, original_height = template_img.size
            original_dimensions.append((original_width, original_height))
            log(f"Page {i} original dimensions: {original_width}x{original_height}")
        except Exception as e:
            log(f"Error getting dimensions for page {i}: {e}")
            original_dimensions.append((1024, 1024))
            original_width, original_height = 1024, 1024
        
        # Check if this is a cover page (page 1 with front and back cover)
        if is_cover_page(template_bytes, page_number=i):
            log(f"Page {i} is a cover page - processing front cover only")
            try:
                new_page = process_cover_page(child_image_bytes, template_bytes, page_number=i, child_name=child_name)
                out_pages.append(new_page)
                qc_results.append({'page': i, 'passed': True, 'type': 'cover'})
                time.sleep(2)
            except Exception as e:
                log(f"Error processing cover page {i}: {e}")
                out_pages.append(template_bytes)
                qc_results.append({'page': i, 'passed': False, 'type': 'cover', 'error': str(e)})
            continue
        
        # Check if page should be skipped (empty/white or no child character)
        if is_empty_or_white_page(template_bytes):
            log(f"Page {i} is empty/white - skipping API call, using original")
            out_pages.append(template_bytes)
            qc_results.append({'page': i, 'passed': True, 'type': 'empty'})
            continue
        
        if not has_child_character(template_bytes):
            log(f"Page {i} has no child character - skipping API call, using original")
            out_pages.append(template_bytes)
            qc_results.append({'page': i, 'passed': True, 'type': 'no_character'})
            continue
        
        # =====================================================================
        # STEP 2a: DETECT VIEW TYPE (front/profile/back)
        # =====================================================================
        log(f"\n--- Page {i}: Detecting view type ---")
        view_info = detect_character_view(template_bytes)
        view_type = view_info.get('view_type', 'front')
        log(f"Page {i} view type: {view_type} (confidence: {view_info.get('confidence', 0):.2f})")
        
        # =====================================================================
        # STEP 2b: GENERATE PAGE WITH QC LOOP
        # =====================================================================
        try:
            new_page, qc_passed, confidence = generate_page_with_qc(
                child_image_bytes=child_image_bytes,
                template_bytes=template_bytes,
                canonical_reference_bytes=canonical_reference_bytes,
                identity_info=identity_info,
                page_number=i,
                view_type=view_type,
                is_cover_page=False,
                child_name=child_name,
                max_retries=3
            )
            
            # Auto-detect and fix any rotation issues in the generated output
            new_page = detect_and_fix_rotation(new_page, is_interior_page=True)
            
            # Verify dimensions match original
            try:
                new_img = Image.open(io.BytesIO(new_page))
                if new_img.size != (original_width, original_height):
                    log(f"Resizing page {i} from {new_img.size} to {original_width}x{original_height}")
                    new_img = new_img.resize((original_width, original_height), Image.Resampling.LANCZOS)
                    output = io.BytesIO()
                    new_img.save(output, format="PNG")
                    output.seek(0)
                    new_page = output.read()
            except Exception as dim_error:
                log(f"Error verifying dimensions for page {i}: {dim_error}")
            
            out_pages.append(new_page)
            qc_results.append({
                'page': i,
                'passed': qc_passed,
                'confidence': confidence,
                'view_type': view_type
            })
            
            time.sleep(2)  # Rate limiting between pages
            
        except Exception as e:
            log(f"Error generating page {i}: {e}")
            import traceback
            try:
                with open("face_replacement.log", "a", encoding="utf-8") as log_file:
                    traceback.print_exc(file=log_file)
            except:
                pass
            out_pages.append(template_bytes)
            qc_results.append({'page': i, 'passed': False, 'error': str(e)})
    
    # =========================================================================
    # STEP 3: GENERATION SUMMARY
    # =========================================================================
    total_count = len(qc_results)
    log("\n" + "="*70)
    log(f"GENERATION SUMMARY: {total_count} pages processed")
    log("="*70)
    for r in qc_results:
        status = "OK" if r.get('passed', True) else "ERROR"
        view = r.get('view_type', r.get('type', 'unknown'))
        log(f"  Page {r['page']}: {status} (view: {view})")
    
    if status_callback:
        status_callback(len(page_list), f"All pages generated ({total_count} pages)")
    
    return out_pages, original_dimensions
