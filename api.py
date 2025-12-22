import base64
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse

from services.generator import (
    GenerationError,
    fetch_child_image,
    generate_pdf_for_order,
    convert_image_to_png,
)
from book_utils import BOOK_PATHS

logger = logging.getLogger("webhook")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI()

# Output directory for generated PDFs
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/data/output")

# Gelato API configuration
GELATO_API_KEY = os.getenv("GELATO_API_KEY", "")
GELATO_API_URL = "https://order.gelatoapis.com/v4/orders"
# Hardcover Photo Book 8.5x11 - adjust SKU based on your Gelato product
GELATO_PRODUCT_UID = os.getenv("GELATO_PRODUCT_UID", "photobook_hc_8x11_24")

# Render external URL for PDF access
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "")


def verify_shopify_hmac(body: bytes, hmac_header: str, shared_secret: str) -> bool:
    digest = hmac.new(shared_secret.encode("utf-8"), body, hashlib.sha256).digest()
    computed = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(computed, hmac_header or "")


def load_product_map() -> Dict[str, Dict[str, str]]:
    """
    Load optional product/variant mapping from config_product_map.json.
    Structure:
    {
      "product_id": { "123456": "A True Princess" },
      "variant_id": { "654321": "A True Princess" }
    }
    """
    mapping_path = Path(__file__).parent / "config_product_map.json"
    if mapping_path.exists():
        try:
            return json.loads(mapping_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read config_product_map.json: %s", exc)
    return {"product_id": {}, "variant_id": {}}


def get_property(properties: Any, keys: list[str]) -> Optional[str]:
    if not properties:
        return None
    lower_keys = [k.lower() for k in keys]
    for prop in properties:
        name = str(prop.get("name") or prop.get("key") or "").lower()
        if name in lower_keys:
            return str(prop.get("value") or "").strip()
    return None


def resolve_book_name(item: Dict[str, Any], mapping: Dict[str, Dict[str, str]]) -> Optional[str]:
    product_id = str(item.get("product_id") or "")
    variant_id = str(item.get("variant_id") or "")

    # 1) Explicit mapping by variant, then product
    if variant_id and variant_id in mapping.get("variant_id", {}):
        return mapping["variant_id"][variant_id]
    if product_id and product_id in mapping.get("product_id", {}):
        return mapping["product_id"][product_id]

    # 2) Property override
    prop_book = get_property(item.get("properties"), ["book", "book_name", "storybook"])
    if prop_book and prop_book in BOOK_PATHS:
        return prop_book

    # 3) Title match
    title = item.get("title")
    if title in BOOK_PATHS:
        return title

    return None


def parse_line_item(item: Dict[str, Any], mapping: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    properties = item.get("properties") or []
    child_name = get_property(properties, ["child_name", "child", "name", "Child Name"])
    image_url = get_property(properties, ["image_url", "photo_url", "image", "photo", "upload", "Child Image"])
    mode_raw = get_property(properties, ["mode", "generation_mode", "book_mode"])
    mode = "full"  # Default to full book generation
    if mode_raw:
        mode = "cover" if "cover" in mode_raw.lower() else "full"
    book_name = resolve_book_name(item, mapping)

    if not child_name:
        raise GenerationError("Missing child name in line item properties.")
    if not image_url:
        raise GenerationError("Missing image URL in line item properties.")
    if not book_name:
        raise GenerationError("Unable to resolve book for this line item.")

    return {
        "child_name": child_name,
        "image_url": image_url,
        "mode": mode,
        "book_name": book_name,
    }


def extract_shipping_address(order: Dict[str, Any]) -> Dict[str, str]:
    """Extract shipping address from Shopify order."""
    shipping = order.get("shipping_address") or order.get("billing_address") or {}
    
    return {
        "firstName": shipping.get("first_name", ""),
        "lastName": shipping.get("last_name", ""),
        "addressLine1": shipping.get("address1", ""),
        "addressLine2": shipping.get("address2", ""),
        "city": shipping.get("city", ""),
        "postCode": shipping.get("zip", ""),
        "state": shipping.get("province_code", ""),
        "country": shipping.get("country_code", ""),
        "email": order.get("email", ""),
        "phone": shipping.get("phone", ""),
    }


def send_to_gelato(order_id: str, pdf_path: str, mode: str, order: Dict[str, Any], child_name: str):
    """
    Send order to Gelato for printing.
    
    Args:
        order_id: Shopify order ID
        pdf_path: Path to the generated PDF
        mode: "cover" or "full"
        order: Full Shopify order data (for shipping address)
        child_name: Child's name for the order reference
    """
    if not GELATO_API_KEY:
        logger.warning("GELATO_API_KEY not set - skipping Gelato submission for order %s", order_id)
        return
    
    if not RENDER_EXTERNAL_URL:
        logger.warning("RENDER_EXTERNAL_URL not set - cannot create PDF URL for Gelato")
        return
    
    # Build public PDF URL
    pdf_filename = Path(pdf_path).name
    pdf_url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/pdf/{pdf_filename}"
    
    # Get shipping address
    shipping = extract_shipping_address(order)
    
    # Build Gelato order payload
    gelato_order = {
        "orderType": "order",
        "orderReferenceId": f"shopify-{order_id}",
        "customerReferenceId": order.get("customer", {}).get("id", f"customer-{order_id}"),
        "currency": order.get("currency", "USD"),
        "items": [
            {
                "itemReferenceId": f"book-{order_id}",
                "productUid": GELATO_PRODUCT_UID,
                "files": [
                    {
                        "type": "default",
                        "url": pdf_url
                    }
                ],
                "quantity": 1
            }
        ],
        "shippingAddress": {
            "firstName": shipping["firstName"],
            "lastName": shipping["lastName"],
            "addressLine1": shipping["addressLine1"],
            "addressLine2": shipping["addressLine2"],
            "city": shipping["city"],
            "postCode": shipping["postCode"],
            "state": shipping["state"],
            "country": shipping["country"],
            "email": shipping["email"],
            "phone": shipping["phone"],
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": GELATO_API_KEY,
    }
    
    try:
        logger.info("Submitting order %s to Gelato with PDF: %s", order_id, pdf_url)
        response = requests.post(
            GELATO_API_URL,
            json=gelato_order,
            headers=headers,
            timeout=30
        )
        
        if response.status_code in (200, 201):
            result = response.json()
            gelato_order_id = result.get("id", "unknown")
            logger.info("Gelato order created successfully: %s for Shopify order %s", gelato_order_id, order_id)
        else:
            logger.error(
                "Gelato API error for order %s: %s - %s",
                order_id,
                response.status_code,
                response.text
            )
    except requests.RequestException as exc:
        logger.error("Failed to submit order %s to Gelato: %s", order_id, exc)


def process_order(order: Dict[str, Any], mapping: Dict[str, Dict[str, str]]):
    order_id = str(order.get("id") or "")
    line_items = order.get("line_items") or []

    for item in line_items:
        try:
            parsed = parse_line_item(item, mapping)
        except GenerationError as exc:
            logger.warning("Skipping line item: %s", exc)
            continue

        try:
            child_bytes = fetch_child_image(parsed["image_url"])
            pdf_path, _ = generate_pdf_for_order(
                child_name=parsed["child_name"],
                child_image_bytes=child_bytes,
                book_name=parsed["book_name"],
                mode=parsed["mode"],
            )
            
            # Send to Gelato for printing
            send_to_gelato(
                order_id=order_id,
                pdf_path=str(pdf_path),
                mode=parsed["mode"],
                order=order,
                child_name=parsed["child_name"]
            )
            
            logger.info("Generated PDF for order %s item %s -> %s", order_id, item.get("id"), pdf_path)
            
        except Exception as exc:
            logger.error("Failed to generate PDF for order %s item %s: %s", order_id, item.get("id"), exc)
            continue


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/pdf/{filename}")
async def serve_pdf(filename: str):
    """
    Serve generated PDFs for Gelato to download.
    """
    # Sanitize filename to prevent directory traversal
    safe_filename = Path(filename).name
    pdf_path = Path(OUTPUT_DIR) / safe_filename
    
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    
    if not pdf_path.suffix.lower() == ".pdf":
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=safe_filename
    )


@app.get("/orders")
async def list_orders():
    """
    List generated PDFs (for debugging/admin purposes).
    """
    output_path = Path(OUTPUT_DIR)
    if not output_path.exists():
        return {"pdfs": []}
    
    pdfs = [f.name for f in output_path.glob("*.pdf")]
    return {"pdfs": pdfs, "count": len(pdfs)}


@app.post("/webhook/shopify")
async def shopify_webhook(request: Request, background_tasks: BackgroundTasks):
    shared_secret = os.getenv("SHOPIFY_WEBHOOK_SECRET")
    raw_body = await request.body()
    hmac_header = request.headers.get("X-Shopify-Hmac-Sha256")

    if shared_secret:
        if not hmac_header or not verify_shopify_hmac(raw_body, hmac_header, shared_secret):
            raise HTTPException(status_code=401, detail="HMAC verification failed")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    mapping = load_product_map()
    background_tasks.add_task(process_order, payload, mapping)

    return JSONResponse({"status": "accepted"})


@app.post("/test/generate")
async def test_generate(request: Request, background_tasks: BackgroundTasks):
    """
    Test endpoint to manually trigger generation without Shopify.
    POST body: {
        "child_name": "Emma",
        "image_url": "https://example.com/child.jpg",
        "book_name": "A True Princess",
        "mode": "cover"
    }
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    child_name = data.get("child_name")
    image_url = data.get("image_url")
    book_name = data.get("book_name")
    mode = data.get("mode", "cover")
    
    if not all([child_name, image_url, book_name]):
        raise HTTPException(status_code=400, detail="Missing required fields: child_name, image_url, book_name")
    
    if book_name not in BOOK_PATHS:
        raise HTTPException(status_code=400, detail=f"Unknown book: {book_name}")
    
    def generate_task():
        try:
            child_bytes = fetch_child_image(image_url)
            pdf_path, _ = generate_pdf_for_order(
                child_name=child_name,
                child_image_bytes=child_bytes,
                book_name=book_name,
                mode=mode,
            )
            logger.info("Test generation complete: %s", pdf_path)
        except Exception as exc:
            logger.error("Test generation failed: %s", exc)
    
    background_tasks.add_task(generate_task)
    
    return JSONResponse({"status": "accepted", "message": f"Generating {mode} for {child_name} - {book_name}"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
