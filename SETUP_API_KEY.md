# Setting Up Your OpenAI API Key

## Important Security Note
**NEVER commit your API key to GitHub!** The `.env` file is already in `.gitignore` to prevent accidental commits.

## Local Development Setup

### Option 1: Using .env file (Recommended)

1. Copy the example file:
   ```bash
   cp env.example .env
   ```

2. Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. The app will automatically load the API key from `.env` when you run it.

### Option 2: Using Environment Variable

Set the environment variable before running the app:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-your-api-key-here"
streamlit run app.py
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=sk-your-api-key-here
streamlit run app.py
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
streamlit run app.py
```

## For GitHub Deployment (Render, Heroku, etc.)

### Using GitHub Secrets (for GitHub Actions)

If you're using GitHub Actions, add your API key as a secret:

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `OPENAI_API_KEY`
5. Value: Your API key
6. Click **Add secret**

### Using Render/Heroku Environment Variables

1. Go to your service dashboard (Render/Heroku)
2. Navigate to **Environment Variables** or **Config Vars**
3. Add:
   - Key: `OPENAI_API_KEY`
   - Value: Your API key
4. Restart your service

## Verify Your Setup

After setting up, you can verify the API key is loaded by checking the config:

```python
from config import OPENAI_API_KEY
print("API Key loaded:", "Yes" if OPENAI_API_KEY else "No")
```

## Get Your API Key

1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click **Create new secret key**
4. Copy the key (it starts with `sk-`)
5. Store it securely - you won't be able to see it again!

