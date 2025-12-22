"""
Service helpers to generate personalized PDFs from Shopify webhook inputs.
"""
import io
import os
import uuid
import tempfile
from pathlib import Path
from typing import Tuple, List, Optional

import requests
from PIL import Image
from config import TOTAL_PAGES, BOOKS_BASE_DIR


def convert_image_to_png(image_bytes: bytes) -> bytes:
    """
    Convert any image format (WebP, JPEG, GIF, etc.) to PNG format.
    
    Shopify returns uploaded images as WebP, but the face replacement system
    requires PNG format for optimal processing.
    
    Args:
        image_bytes: Raw bytes of the image in any supported format
        
    Returns:
        bytes: Image converted to PNG format
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if necessary (handles RGBA, P mode, etc.)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Preserve alpha channel if present
            if img.mode == 'P':
                img = img.convert('RGBA')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        output = io.BytesIO()
        img.save(output, format='PNG', optimize=True)
        output.seek(0)
        return output.read()
    except Exception as e:
        # If conversion fails, return original bytes and let downstream handle it
        print(f"Warning: Image conversion to PNG failed: {e}")
        return image_bytes

OUTPUT_DIR_DEFAULT = os.getenv("OUTPUT_DIR", "/data/output")
from book_utils import get_book_template_path
from openai_client_new import generate_pages_for_book, process_cover_with_new_workflow, generate_canonical_reference
from generate_pdf import create_pdf


class GenerationError(Exception):
    """Raised when generation prerequisites are not met."""


def _safe_filename(name: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in name.strip())
    return keep or "child"


def fetch_child_image(image_url: str, timeout: int = 20) -> bytes:
    """
    Download child image bytes from a URL and convert to PNG format.
    
    Shopify returns uploaded images as WebP format, so we convert to PNG
    for consistent processing by the face replacement system.
    
    Args:
        image_url: URL of the image to download
        timeout: Request timeout in seconds
        
    Returns:
        bytes: Image bytes in PNG format
    """
    resp = requests.get(image_url, timeout=timeout)
    resp.raise_for_status()
    # Convert to PNG format (handles WebP from Shopify)
    return convert_image_to_png(resp.content)


def ensure_output_dir(base_dir: str | Path) -> Path:
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_pdf_output_path(output_dir: Path, child_name: str) -> Path:
    safe = _safe_filename(child_name)
    return output_dir / f"{safe}-{uuid.uuid4().hex}.pdf"


def generate_pdf_for_order(
    *,
    child_name: str,
    child_image_bytes: bytes,
    book_name: str,
    mode: str = "cover",
    output_dir: str | Path = OUTPUT_DIR_DEFAULT,
    base_dir: Optional[str] = None,
) -> Tuple[Path, List[Tuple[int, int]]]:
    """
    Generate a personalized PDF for a given order.

    Args:
        child_name: Name to personalize with.
        child_image_bytes: Bytes of the child's photo.
        book_name: Book title as defined in BOOK_PATHS.
        mode: "cover" for cover-only, "full" for full book.
        output_dir: Directory to write the PDF.
        base_dir: Override for template base directory.
    """
    base_dir = base_dir or BOOKS_BASE_DIR
    mode = (mode or "cover").lower()
    output_dir_path = ensure_output_dir(output_dir)

    template_dir = get_book_template_path(book_name, base_dir)
    if not template_dir.exists():
        raise GenerationError(f"Template directory not found: {template_dir}")

    # Cover workflow files
    front_cover_path = template_dir / "00.png"
    back_cover_path = template_dir / "0.png"
    full_cover_path = template_dir / "1.png"
    use_new_cover_workflow = (
        front_cover_path.exists() and back_cover_path.exists() and full_cover_path.exists()
    )

    # Interior pages list
    interior_page_list: List[str] = []
    if mode == "full":
        for i in range(2, TOTAL_PAGES + 1):
            page_path = template_dir / f"{i}.png"
            if page_path.exists():
                interior_page_list.append(f"{i}.png")

    if mode == "cover" and not use_new_cover_workflow:
        raise GenerationError("Cover-only mode requires 00.png, 0.png, and 1.png in template.")

    if mode == "full" and not interior_page_list and not use_new_cover_workflow:
        raise GenerationError("No template pages found for full-book generation.")

    generated_images: List[bytes] = []
    original_dimensions: List[Tuple[int, int]] = []
    
    # Generate canonical reference for consistent identity (especially important for full book)
    canonical_reference_bytes = None
    identity_info = None
    if mode == "full" and interior_page_list:
        try:
            canonical_reference_bytes, identity_info = generate_canonical_reference(
                child_image_bytes=child_image_bytes,
                book_name=book_name
            )
        except Exception as e:
            print(f"Warning: Could not generate canonical reference: {e}")
            # Will fall back to using child photo directly

    # Process cover
    if use_new_cover_workflow:
        front_cover_bytes = front_cover_path.read_bytes()
        back_cover_bytes = back_cover_path.read_bytes()
        full_cover_bytes = full_cover_path.read_bytes()

        cover_bytes, cover_dims = process_cover_with_new_workflow(
            child_image_bytes,
            front_cover_bytes,
            back_cover_bytes,
            full_cover_bytes,
            child_name=child_name,
            book_name=book_name,
        )
        generated_images.append(cover_bytes)
        original_dimensions.append(cover_dims)

    # Process interiors with canonical reference for consistent identity
    if mode == "full" and interior_page_list:
        interior_images, interior_dims = generate_pages_for_book(
            child_image_bytes,
            str(template_dir),
            interior_page_list,
            status_callback=None,
            child_name=child_name,
            canonical_reference_bytes=canonical_reference_bytes,
            identity_info=identity_info,
        )
        generated_images.extend(interior_images)
        original_dimensions.extend(interior_dims)

    pdf_path = build_pdf_output_path(output_dir_path, child_name)

    create_pdf(
        generated_images,
        str(pdf_path),
        child_name,
        original_dimensions=original_dimensions,
        book_name=book_name,
        base_dir=base_dir,
    )

    return pdf_path, original_dimensions

