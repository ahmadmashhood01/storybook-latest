# Deployment Guide: Shopify → Render → Gelato Pipeline

## Architecture

```
Customer → Shopify Order → Webhook → Render API → OpenAI → PDF → Gelato → Printed Book
```

## Prerequisites

1. **GitHub Account** with LFS enabled
2. **Render Account** (https://render.com)
3. **Gelato Account** with API access (https://dashboard.gelato.com)
4. **Shopify Store** with Admin API access
5. **OpenAI API Key** with gpt-image-1 access

---

## Step 1: Prepare Books for GitHub LFS

Your book templates need to be uploaded via Git LFS due to their size.

```bash
# Install Git LFS if not already installed
git lfs install

# Initialize repository
cd personalized-storybook-generator-main
git init

# LFS will automatically track PNG/JPG files (see .gitattributes)
git add .
git commit -m "Initial commit with storybook generator"

# Create GitHub repository and push
git remote add origin https://github.com/YOUR_USERNAME/storybook-generator.git
git push -u origin main
```

### Book Folder Structure

Each book folder should be in your `Books` directory:
```
Books/
├── A True Princess PNG/
│   ├── 00.png  (front cover)
│   ├── 0.png   (back cover)
│   ├── 1.png   (full wrap)
│   └── 2-33.png (interior pages)
├── My Animal World PNG/
│   └── ...
└── (other books)/
```

---

## Step 2: Deploy to Render

### Option A: Blueprint Deploy (Recommended)

1. Go to https://render.com/deploy
2. Connect your GitHub repository
3. Render will detect `render.yaml` and create services automatically

### Option B: Manual Deploy

1. Create a new **Web Service** on Render
2. Connect your GitHub repository
3. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -k uvicorn.workers.UvicornWorker api:app --bind 0.0.0.0:10000 --timeout 600`
4. Add a **Disk** mounted at `/data` (10GB recommended)

### Environment Variables (Set in Render Dashboard)

| Variable | Value |
|----------|-------|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `SHOPIFY_WEBHOOK_SECRET` | From Shopify webhook settings |
| `GELATO_API_KEY` | `5accf298-9d8c-43fc-85bb-8e6336aeefe1-d69f112d-036c-4e51-a498-3828cdad5cc0:fab844a5-32c1-45b0-82a6-ad45bf12399d` |
| `GELATO_PRODUCT_UID` | Your Gelato product SKU (e.g., `photobook_hc_8x11_24`) |
| `RENDER_EXTERNAL_URL` | Your Render URL (e.g., `https://storybook-webhook.onrender.com`) |

---

## Step 3: Upload Book Templates to Render

After deployment, you need to copy book templates to the Render disk.

### Option A: Include in Repository
Place your book folders in the repository under a `books/` directory and update `BOOKS_BASE_DIR`.

### Option B: Manual Upload via SSH
1. Enable Shell access in Render dashboard
2. SSH into your service
3. Upload books to `/data/books/`

### Option C: Use Render's Disk Snapshot
Create a snapshot with pre-loaded books and restore it.

---

## Step 4: Configure Shopify Webhooks

### Create a Custom App in Shopify

1. Go to **Shopify Admin** → **Settings** → **Apps and sales channels**
2. Click **Develop apps** → **Create an app**
3. Name it "Storybook Generator"
4. Configure **Admin API scopes**:
   - `read_orders`
   - `read_products`
5. Install the app

### Register Webhook

1. In your custom app, go to **Webhooks**
2. Create a new webhook:
   - **Event**: `orders/create`
   - **URL**: `https://YOUR-RENDER-URL.onrender.com/webhook/shopify`
   - **Format**: JSON
3. Copy the **Webhook signing secret** → Set as `SHOPIFY_WEBHOOK_SECRET` in Render

---

## Step 5: Configure Shopify Products

Each product needs custom properties for the child's name and photo.

### Required Line Item Properties

Your Shopify products must collect these properties at checkout:
- `Child Name` - The child's name for personalization
- `Child Image` or `image_url` - URL to the uploaded child photo

### Product ID Mapping

The `config_product_map.json` maps your Shopify product IDs to book names:

```json
{
  "product_id": {
    "15503649145176": "My Little Wonder - Boy",
    "15503595864408": "I Believe in Me",
    ...
  }
}
```

---

## Step 6: Configure Gelato Product

1. Log into **Gelato Dashboard**
2. Go to **Products** and find/create your photobook product
3. Note the **Product UID** (e.g., `photobook_hc_8x11_24`)
4. Set this as `GELATO_PRODUCT_UID` in Render

### Gelato Product Requirements

Your PDF must match Gelato's specifications:
- **Page count**: Check minimum/maximum pages
- **Dimensions**: Match the product (8.5x11" for standard)
- **Bleed**: Include proper bleed margins
- **Color**: CMYK recommended

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Health check |
| `/webhook/shopify` | POST | Shopify order webhook |
| `/pdf/{filename}` | GET | Serve generated PDFs |
| `/orders` | GET | List generated PDFs |
| `/test/generate` | POST | Manual test generation |

### Test Generation Example

```bash
curl -X POST https://YOUR-URL.onrender.com/test/generate \
  -H "Content-Type: application/json" \
  -d '{
    "child_name": "Emma",
    "image_url": "https://example.com/child.jpg",
    "book_name": "A True Princess",
    "mode": "cover"
  }'
```

---

## Troubleshooting

### Check Logs
View logs in Render dashboard or via CLI:
```bash
render logs -s storybook-webhook
```

### Common Issues

1. **HMAC Verification Failed**
   - Ensure `SHOPIFY_WEBHOOK_SECRET` matches Shopify's signing secret

2. **Unable to resolve book**
   - Check `config_product_map.json` has correct product IDs
   - Verify product title matches book name in `BOOK_PATHS`

3. **Gelato Order Failed**
   - Verify `GELATO_API_KEY` is correct
   - Check PDF URL is accessible: `https://YOUR-URL/pdf/filename.pdf`
   - Verify product UID matches your Gelato product

4. **Missing child image**
   - Ensure Shopify product collects `Child Image` property
   - Check image URL is publicly accessible

---

## Order Flow

1. **Customer** places order on Shopify with child name + photo
2. **Shopify** sends webhook to Render
3. **Render API** downloads child photo
4. **OpenAI** generates canonical reference + personalized pages
5. **PDF** created and saved to disk
6. **Gelato** receives order with PDF URL
7. **Gelato** prints and ships book to customer

---

## Cost Estimates

Per book generation:
- **OpenAI**: ~$0.50-2.00 (depending on pages)
- **Render**: $7/month starter plan
- **Gelato**: Based on your product pricing

---

## Support

For issues with:
- **OpenAI API**: https://platform.openai.com/docs
- **Render**: https://render.com/docs
- **Gelato**: https://docs.gelato.com
- **Shopify Webhooks**: https://shopify.dev/docs/apps/webhooks

