# Personalized Storybook Generator

A Streamlit application that generates personalized children's storybooks by replacing the character's face in storybook illustrations with a child's photo.

## Features

- Upload a child's photo to personalize storybook pages
- Uses OpenAI's `gpt-image-1` API for high-fidelity face replacement
- Preserves original storybook illustrations (background, text, style)
- Generates print-ready PDF storybooks
- Currently configured for "A True Princess" storybook (33 pages)

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Momozahaf
```

### 2. Create virtual environment

```bash
cd storybook_app
python -m venv storyenv

# Activate on Windows:
storyenv\Scripts\activate

# Activate on Mac/Linux:
source storyenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API key

```bash
# Copy the example config file
cp config.example.py config.py

# Edit config.py and add your OpenAI API key
# Get your API key from: https://platform.openai.com/api-keys
```

**Important:** You need to verify your OpenAI organization to use `gpt-image-1`:
- Go to: https://platform.openai.com/settings/organization/general
- Click "Verify Organization"
- Wait up to 15 minutes for access to propagate

### 5. Prepare template images

Ensure your storybook template images are in:
```
A True Princess/A True Princess/A True Princess PNG/
  1.png
  2.png
  ...
  33.png
```

## Usage (Streamlit)

### Run the Streamlit app

```bash
cd storybook_app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Generate a storybook

1. Upload a child's photo (PNG, JPG, or JPEG)
2. Enter the child's name
3. Click "Generate Storybook"
4. Wait for generation (about 1-2 minutes per page)
5. Download the personalized PDF

## Headless Shopify webhook service (Render)

- A FastAPI webhook (`api.py`) listens on `/webhook/shopify`, validates Shopify HMAC, and queues generation.
- Book assets are read from `BOOKS_BASE_DIR` (set to `/data/books` on Render disk).
- PDFs are written to `OUTPUT_DIR` (`/data/output` by default).

Deploy on Render:
1. Set env vars: `OPENAI_API_KEY`, `SHOPIFY_WEBHOOK_SECRET`, `BOOKS_BASE_DIR=/data/books`, `OUTPUT_DIR=/data/output`.
2. Mount a persistent disk at `/data/books` and upload your template folders there.
3. Use `render.yaml` (gunicorn + uvicorn worker) to create the web service.
4. Point Shopify order webhooks to `https://<your-render-app>/webhook/shopify`.

## Configuration

Edit `storybook_app/config.py` to:
- Change the number of pages (`TOTAL_PAGES`)
- Adjust image quality settings
- Modify PDF page dimensions

## Project Structure

```
storybook_app/
├── app.py                 # Streamlit UI
├── openai_client_new.py   # OpenAI API integration (face replacement)
├── generate_pdf.py        # PDF generation
├── config.py             # Configuration (not in git - use config.example.py)
├── requirements.txt      # Python dependencies
└── templates/            # Storybook template images (not in git)
```

## Notes

- **Costs:** Each page requires 1 API call to `gpt-image-1`. For 33 pages, this is 33 API calls.
- **Time:** Generation takes approximately 1-2 minutes per page.
- **Organization Verification:** Required to use `gpt-image-1` model.

## Troubleshooting

### "Organization must be verified" error
- Verify your organization at: https://platform.openai.com/settings/organization/general
- Wait up to 15 minutes after verification

### PDF generation errors
- Ensure images are properly sized
- Check that template images exist in the correct directory

### Face replacement not working
- Check logs in `face_replacement.log`
- Ensure child photo is clear and shows the face well
- Verify API key is correct

## License

[Add your license here]

