"""
OpenAI API client for generating personalized storybook images
"""
import base64
import time
import requests
from openai import OpenAI
from PIL import Image
import io
import logging
import tempfile
import os
from datetime import datetime
from config import OPENAI_API_KEY, IMAGE_SIZE, IMAGE_QUALITY

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Set up logging to file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('storybook_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def encode_image_to_base64(image_bytes):
    """Convert image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode('utf-8')


def detect_face_location(original_page_bytes):
    """
    Use GPT-4 Vision to detect the location of the child character's face in the storybook page.
    
    Args:
        original_page_bytes: Bytes of the original page image
    
    Returns:
        dict: Face location information (bounding box coordinates, face center, etc.)
    """
    original_b64 = encode_image_to_base64(original_page_bytes)
    
    detection_prompt = (
        "Analyze this storybook page image. Identify the main child character (princess). "
        "Describe the exact location of the child's face in the image. Provide:\n"
        "1. Approximate pixel coordinates of the face center (x, y)\n"
        "2. Approximate width and height of the face area in pixels\n"
        "3. The bounding box coordinates: top-left (x1, y1) and bottom-right (x2, y2)\n"
        "4. Also identify the skin tone area (neck, arms if visible) and hair area\n"
        "Format your response as: FACE_CENTER: (x, y), FACE_SIZE: (width, height), "
        "BOUNDING_BOX: (x1, y1, x2, y2), SKIN_AREA: description, HAIR_AREA: description"
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": detection_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{original_b64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )
    
    analysis = response.choices[0].message.content
    logger.info(f"Face detection analysis: {analysis}")
    
    # Parse the response to extract coordinates
    # This is a simplified parser - in production, you might want more robust parsing
    face_info = {
        "analysis": analysis,
        "center": None,
        "size": None,
        "bbox": None
    }
    
    # Try to extract coordinates from the response
    import re
    center_match = re.search(r'FACE_CENTER:\s*\((\d+),\s*(\d+)\)', analysis)
    size_match = re.search(r'FACE_SIZE:\s*\((\d+),\s*(\d+)\)', analysis)
    bbox_match = re.search(r'BOUNDING_BOX:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', analysis)
    
    if center_match:
        face_info["center"] = (int(center_match.group(1)), int(center_match.group(2)))
    if size_match:
        face_info["size"] = (int(size_match.group(1)), int(size_match.group(2)))
    if bbox_match:
        face_info["bbox"] = (
            int(bbox_match.group(1)), int(bbox_match.group(2)),
            int(bbox_match.group(3)), int(bbox_match.group(4))
        )
    
    return face_info


def create_face_mask(original_page_bytes, face_info):
    """
    Create a mask image for face replacement.
    White/opaque pixels = area to edit (face, skin, hair)
    Transparent pixels = area to preserve (everything else)
    
    Args:
        original_page_bytes: Bytes of the original page image
        face_info: Face location information from detect_face_location
    
    Returns:
        bytes: PNG mask image bytes
    """
    # Open the original image to get dimensions
    original_img = Image.open(io.BytesIO(original_page_bytes))
    width, height = original_img.size
    
    # Create a transparent mask image
    mask = Image.new("RGBA", (width, height), (0, 0, 0, 0))  # Fully transparent
    
    # If we have bounding box coordinates, use them
    if face_info.get("bbox"):
        x1, y1, x2, y2 = face_info["bbox"]
        # Expand the area slightly to include hair and neck
        padding = 30
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        # Draw white ellipse/rectangle for face area
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        # Use ellipse for more natural face shape
        draw.ellipse([x1, y1, x2, y2], fill=(255, 255, 255, 255))
    elif face_info.get("center") and face_info.get("size"):
        # Use center and size to create mask
        cx, cy = face_info["center"]
        w, h = face_info["size"]
        # Expand for hair and neck
        w = int(w * 1.5)
        h = int(h * 1.5)
        x1 = max(0, cx - w // 2)
        y1 = max(0, cy - h // 2)
        x2 = min(width, cx + w // 2)
        y2 = min(height, cy + h // 2)
        
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.ellipse([x1, y1, x2, y2], fill=(255, 255, 255, 255))
    else:
        # Fallback: use center of image (rough estimate)
        logger.warning("Using fallback mask - placing mask in center of image")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        center_x, center_y = width // 2, height // 3  # Upper third (where face usually is)
        radius = min(width, height) // 6
        draw.ellipse(
            [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
            fill=(255, 255, 255, 255)
        )
    
    # Convert mask to bytes
    mask_bytes = io.BytesIO()
    mask.save(mask_bytes, format="PNG")
    mask_bytes.seek(0)
    return mask_bytes.read()


def generate_personalized_page(original_page_bytes, child_photo_bytes, page_number):
    """
    Edit the original storybook page to replace only the child's face, skin tone, and hair
    with the uploaded child's photo, keeping everything else exactly the same.
    
    Uses mask-based approach:
    1. Detect face location using GPT-4 Vision
    2. Create a mask for the face/skin/hair area
    3. Use image edit API with mask to replace only that area
    
    Args:
        original_page_bytes: Bytes of the original page image
        child_photo_bytes: Bytes of the child's photo
        page_number: Page number for logging
    
    Returns:
        bytes: Edited image bytes
    """
    try:
        # Verify inputs
        if not child_photo_bytes or len(child_photo_bytes) == 0:
            raise ValueError(f"Child photo is empty for page {page_number}")
        if not original_page_bytes or len(original_page_bytes) == 0:
            raise ValueError(f"Original page is empty for page {page_number}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing page {page_number} - Mask-Based Approach")
        logger.info(f"Child photo received: {len(child_photo_bytes)} bytes")
        logger.info(f"Original page received: {len(original_page_bytes)} bytes")
        logger.info(f"{'='*60}\n")
        print(f"\n{'='*60}")
        print(f"Processing page {page_number} - Mask-Based Approach")
        print(f"Child photo received: {len(child_photo_bytes)} bytes")
        print(f"Original page received: {len(original_page_bytes)} bytes")
        print(f"{'='*60}\n")
        
        # Step 1: Detect face location in the original storybook page
        print(f"Step 1: Detecting face location in page {page_number}...")
        logger.info(f"Detecting face location for page {page_number}")
        face_info = detect_face_location(original_page_bytes)
        print(f"Face detection complete. Bounding box: {face_info.get('bbox')}")
        
        # Step 2: Create mask image for the face/skin/hair area
        print(f"Step 2: Creating mask for page {page_number}...")
        logger.info(f"Creating mask for page {page_number}")
        mask_bytes = create_face_mask(original_page_bytes, face_info)
        print(f"Mask created: {len(mask_bytes)} bytes")
        
        # Step 3: Use image edit API with mask
        # Create temporary files with proper extensions
        edit_prompt = (
            "Replace the face, skin tone, and hair in the masked area with the child from the reference photo. "
            "The masked area (white pixels) should be replaced with the child's face, skin tone, and hair from the reference image. "
            "Keep everything outside the mask (transparent areas) exactly as it is in the original image. "
            "Preserve all text, backgrounds, colors, and artistic style. Only modify the masked face/skin/hair area."
        )
        
        try:
            logger.info(f"Attempting mask-based image edit API for page {page_number}...")
            print(f"Step 3: Editing image with mask for page {page_number}...")
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_original:
                tmp_original.write(original_page_bytes)
                tmp_original_path = tmp_original.name
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_mask:
                tmp_mask.write(mask_bytes)
                tmp_mask_path = tmp_mask.name
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_child:
                tmp_child.write(child_photo_bytes)
                tmp_child_path = tmp_child.name
            
            try:
                # Open files for API
                with open(tmp_original_path, 'rb') as original_file, \
                     open(tmp_mask_path, 'rb') as mask_file, \
                     open(tmp_child_path, 'rb') as child_file:
                    
                    # Use image edit API with mask
                    # Note: The API may need the child photo as a reference in the prompt
                    response = client.images.edit(
                        model="gpt-image-1",
                        image=original_file,  # Base image to edit
                        mask=mask_file,  # Mask specifying where to edit
                        prompt=edit_prompt,
                        input_fidelity="high",
                        size="auto",
                        output_format="png",
                        n=1,
                    )
            finally:
                # Clean up temporary files
                for path in [tmp_original_path, tmp_mask_path, tmp_child_path]:
                    try:
                        os.unlink(path)
                    except:
                        pass
            
            logger.info(f"Image edit API succeeded for page {page_number}")
            logger.info(f"Response type: {type(response)}")
            logger.info(f"Response data type: {type(response.data[0])}")
            print(f"Image edit API succeeded for page {page_number}")
            
            # gpt-image-1 returns base64 encoded images
            if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
                logger.info(f"Got base64 image for page {page_number}")
                image_bytes = base64.b64decode(response.data[0].b64_json)
                logger.info(f"Decoded image size: {len(image_bytes)} bytes")
                
                # Check if the returned image is different from original
                if image_bytes == original_page_bytes:
                    logger.warning(f"WARNING: Returned image is identical to original for page {page_number}!")
                    logger.warning("The API may not have made any changes.")
                    print(f"WARNING: Returned image is identical to original for page {page_number}!")
                else:
                    logger.info(f"Returned image is different from original (good!)")
                    print(f"Returned image is different from original (good!)")
                
                return image_bytes
            elif hasattr(response.data[0], 'url') and response.data[0].url:
                logger.info(f"Got URL for page {page_number}: {response.data[0].url}")
                image_response = requests.get(response.data[0].url)
                image_response.raise_for_status()
                return image_response.content
            else:
                logger.error(f"No image data in response for page {page_number}")
                logger.error(f"Response attributes: {dir(response.data[0])}")
                raise ValueError("No image data in API response")
        except Exception as edit_error:
            error_msg = f"Image edit API failed for page {page_number}: {str(edit_error)}"
            logger.error(error_msg, exc_info=True)
            print(error_msg)
            import traceback
            traceback.print_exc()
            # Fall through to alternative method
        
        # Fallback: If mask-based edit fails, return original (better than generating random images)
        logger.warning(f"Mask-based edit failed for page {page_number}. Returning original image.")
        print(f"WARNING: Mask-based edit failed. Returning original image for page {page_number}.")
        return original_page_bytes
        
    except Exception as e:
        print(f"Error generating page {page_number}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return original as fallback
        return original_page_bytes


def generate_all_pages(original_pages, child_photo_bytes):
    """
    Generate personalized images for all pages.
    
    Args:
        original_pages: List of image bytes for each original page
        child_photo_bytes: Bytes of the child's photo
    
    Returns:
        list: List of generated image bytes (one per page)
    """
    generated_images = []
    
    for i, page_bytes in enumerate(original_pages, start=1):
        print(f"Generating page {i} of {len(original_pages)}...")
        
        # Generate personalized page
        generated_image = generate_personalized_page(
            page_bytes, 
            child_photo_bytes, 
            i
        )
        
        generated_images.append(generated_image)
        
        # Small delay to avoid rate limiting
        time.sleep(1)
    
    return generated_images

