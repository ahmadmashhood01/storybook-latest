"""
PDF generation module for creating the final storybook PDF
"""
import io
import os
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Default PDF target dimensions (in points) - Exact Gelato specifications
# Cover: 1298.27 x 697.323 (200x200mm hardcover photobook with spine)
# Interior: 583.937 x 583.937 (200x200mm / 8x8 inch square pages)
# These values are extracted from the official Gelato template PDF
DEFAULT_PDF_DIMENSIONS = {
    "cover": (1298.27, 697.323),
    "interior": (583.937, 583.937)
}

# Book-specific overrides if any book needs different sizes
BOOK_PDF_DIMENSIONS = {
    "Little Farmer's Big Day": DEFAULT_PDF_DIMENSIONS,
    # Add other books as needed - dimensions should match their original PDFs
    # "A True Princess": {...},
    # "My Animal World": {...},
    # etc.
}


def get_pdf_dimensions_from_original(book_name: str, base_dir: str = None) -> dict:
    """
    Try to read PDF dimensions from the original PDF file if it exists.
    
    Args:
        book_name: Name of the book
        base_dir: Base directory where book folders are located
    
    Returns:
        dict with 'cover' and 'interior' keys containing (width, height) tuples in points,
        or None if original PDF not found
    """
    try:
        from PyPDF2 import PdfReader
        
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
        
        # Try to find the original PDF in the book directory
        # Common patterns: "Book Name.pdf" or "Book Name/Book Name.pdf"
        book_dir = Path(base_dir) / book_name
        pdf_patterns = [
            book_dir / f"{book_name}.pdf",
            book_dir / f"Littler {book_name}.pdf",  # For "Little Farmer's Big Day" -> "Littler Farmer's Big Day.pdf"
            book_dir / f"{book_name}" / f"{book_name}.pdf",
            # Also check for variations with "Littler" prefix
            book_dir / "Littler Farmer's Big Day.pdf",  # Specific case for Little Farmer's Big Day
        ]
        
        for pdf_path in pdf_patterns:
            if pdf_path.exists():
                pdf = PdfReader(str(pdf_path))
                if len(pdf.pages) >= 2:
                    # Get cover page dimensions (page 1)
                    cover_page = pdf.pages[0]
                    cover_width = cover_page.mediabox.width
                    cover_height = cover_page.mediabox.height
                    
                    # Get interior page dimensions (page 2)
                    interior_page = pdf.pages[1]
                    interior_width = interior_page.mediabox.width
                    interior_height = interior_page.mediabox.height
                    
                    return {
                        "cover": (float(cover_width), float(cover_height)),
                        "interior": (float(interior_width), float(interior_height))
                    }
    except Exception as e:
        # If PyPDF2 not available or PDF not found, return None
        pass
    
    return None


def get_target_pdf_dimensions(book_name: str, base_dir: str = None) -> dict:
    """
    Get target PDF dimensions for a book, trying original PDF first, then fallback to mapping.
    
    Args:
        book_name: Name of the book
        base_dir: Base directory where book folders are located
    
    Returns:
        dict with 'cover' and 'interior' keys containing (width, height) tuples in points
    """
    # First try to read from original PDF
    dimensions = get_pdf_dimensions_from_original(book_name, base_dir)
    
    # Fallback to hardcoded mapping
    if dimensions is None:
        dimensions = BOOK_PDF_DIMENSIONS.get(book_name)
    
    # Final fallback to default dimensions
    if dimensions is None:
        dimensions = DEFAULT_PDF_DIMENSIONS
    
    return dimensions


def calculate_target_dimensions(original_dimensions: list, target_dimensions: dict, is_cover_page: callable = None) -> list:
    """
    Calculate target PDF dimensions for each page based on original pixel dimensions.
    
    Args:
        original_dimensions: List of (width, height) tuples in pixels
        target_dimensions: Dict with 'cover' and 'interior' keys containing target (width, height) in points
        is_cover_page: Optional function to determine if a page is a cover (takes index, returns bool)
    
    Returns:
        List of (width, height) tuples in PDF points
    """
    if target_dimensions is None:
        # No target dimensions available, return original (will be wrong but won't crash)
        return original_dimensions
    
    target_list = []
    
    for idx, (pixel_width, pixel_height) in enumerate(original_dimensions):
        # Determine if this is a cover page
        if idx == 0 or (is_cover_page and is_cover_page(idx, pixel_width, pixel_height)):
            # Use cover dimensions
            target_width, target_height = target_dimensions.get("cover", (pixel_width, pixel_height))
        else:
            # Use interior dimensions
            target_width, target_height = target_dimensions.get("interior", (pixel_width, pixel_height))
        
        # Ensure plain floats (ReportLab errors on Decimal)
        target_list.append((float(target_width), float(target_height)))
    
    return target_list


def create_pdf(generated_images, output_path, child_name="Child", original_dimensions=None, book_name=None, base_dir=None):
    """
    Create a PDF from generated images, using target PDF dimensions that match the original PDF.
    Each page uses its own target dimensions (cover can be landscape, interior pages portrait).
    The first page (cover) is displayed at double width to show front and back covers side by side.
    
    Args:
        generated_images: List of image bytes (one per page)
        output_path: Path where PDF should be saved
        child_name: Child's name for filename
        original_dimensions: List of (width, height) tuples for original template dimensions in pixels (required)
        book_name: Name of the book (for dimension lookup)
        base_dir: Base directory where book folders are located
    
    Returns:
        str: Path to the generated PDF file
    """
    if not original_dimensions or len(original_dimensions) != len(generated_images):
        raise ValueError("original_dimensions must be provided and match the number of images")

    # Get target PDF dimensions for this book
    target_dimensions_dict = None
    if book_name:
        target_dimensions_dict = get_target_pdf_dimensions(book_name, base_dir)
    
    # Helper function to determine if a page is a cover
    def is_cover_page_func(idx, pixel_width, pixel_height):
        if idx == 0:
            return True
        aspect_ratio = pixel_width / pixel_height if pixel_height > 0 else 1.0
        return aspect_ratio > 1.5
    
    # Calculate target dimensions for each page (in PDF points)
    if target_dimensions_dict:
        target_dimensions = calculate_target_dimensions(
            original_dimensions, 
            target_dimensions_dict,
            is_cover_page_func
        )
    else:
        # Fallback: use original dimensions (will be wrong but won't crash)
        target_dimensions = original_dimensions

    # Use the target dimensions for the first page as-is (no doubling)
    first_width_pts, first_height_pts = target_dimensions[0]
    c = canvas.Canvas(output_path, pagesize=(first_width_pts, first_height_pts))

    for idx, (image_bytes, pixel_dims, target_dims) in enumerate(zip(generated_images, original_dimensions, target_dimensions)):
        target_width_pts, target_height_pts = target_dims

        # Set page size for this page (after the first page)
        if idx > 0:
            c.showPage()
            c.setPageSize((target_width_pts, target_height_pts))

        # Draw image at (0,0) with target dimensions in points
        img_reader = ImageReader(io.BytesIO(image_bytes))
        c.drawImage(
            img_reader,
            0,
            0,
            width=target_width_pts,  # Target width in points
            height=target_height_pts,  # Target height in points
            preserveAspectRatio=False,
            mask='auto'
        )

    c.save()
    return output_path

