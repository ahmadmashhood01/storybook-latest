# Streamlit Cloud Setup Guide

## Setting Up OpenAI API Key in Streamlit Cloud

Your app needs the OpenAI API key to be configured as a secret in Streamlit Cloud.

### Steps to Add API Key to Streamlit Cloud:

1. **Go to your Streamlit Cloud Dashboard**
   - Visit: https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Select Your App**
   - Find your app: `storybook-latest`
   - Click on it to open the settings

3. **Go to Settings**
   - Click on the **"Settings"** or **"⚙️"** icon
   - Or click **"Manage app"** → **"Settings"**

4. **Add Secrets**
   - Scroll down to the **"Secrets"** section
   - Click **"Edit secrets"** or the **"+"** button

5. **Add Your API Key**
   - In the secrets editor, add:
   ```toml
   OPENAI_API_KEY = "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"
   ```
   - Or use the format:
   ```toml
   [secrets]
   OPENAI_API_KEY = "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"
   ```

6. **Save and Restart**
   - Click **"Save"**
   - Streamlit Cloud will automatically restart your app
   - Wait for the app to redeploy (usually 1-2 minutes)

### Verify It's Working

After adding the secret and the app restarts:

1. Check the app logs in Streamlit Cloud dashboard
2. Try generating a storybook
3. You should see OpenAI API calls being made

### Alternative: Using Environment Variables

If secrets don't work, you can also set it as an environment variable in Streamlit Cloud:

1. Go to **Settings** → **Advanced settings**
2. Add environment variable:
   - Key: `OPENAI_API_KEY`
   - Value: Your API key

### Troubleshooting

**If API calls still don't work:**

1. **Check the logs** in Streamlit Cloud dashboard for errors
2. **Verify the secret name** is exactly `OPENAI_API_KEY` (case-sensitive)
3. **Check the API key** is valid and has credits/quota
4. **Restart the app** after adding secrets

### Current Configuration

The app now checks for the API key in this order:
1. Streamlit secrets (`st.secrets.OPENAI_API_KEY`)
2. Environment variable (`OPENAI_API_KEY`)
3. `.env` file (local development)
4. Hardcoded default (fallback)

This ensures it works in all environments!

