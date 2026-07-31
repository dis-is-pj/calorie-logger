# Meal Calorie Logger

A personal tool to estimate calories and protein from a meal description (and optional photo), then log it to a Google Sheet via a pre-filled Google Form.

## How it works

1. Describe what you ate + pick meal type (Breakfast / Lunch / Dinner / Snacks / Protein)
2. Optionally attach a photo — snap one in the page with **Take photo**, or pick an existing one with **Upload**
3. A two-stage estimate: `gpt-4.1-nano` splits the meal into separate items, then `gpt-4.1-mini` estimates each item independently and the results are summed
4. Review and edit the values, then click **Open pre-filled form** to log to Google Sheets

A photo on its own is enough to submit, but a short description makes the estimate noticeably better.

### About the camera

**Take photo** captures the frame in the browser (`getUserMedia` → canvas → JPEG) and posts it straight to the API, so the shot is never written to the device — nothing clutters your gallery.

This needs a secure context, so it works on the deployed HTTPS URL and on `http://localhost:8080`, but **not** over a plain `http://<LAN-IP>:8080` address. When the camera can't start, the page says why and **Upload** still works.

## Stack

- **Backend:** FastAPI + OpenAI (`gpt-4.1-nano` splitter, `gpt-4.1-mini` vision estimator)
- **Frontend:** Single-page HTML (no framework)
- **Logging:** Google Forms → Google Sheets

## Run locally

```bash
conda activate calorie_logging
uvicorn main:app --reload --port 8080
```

Open `http://localhost:8080` in your browser.

## Environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

## Deploy to Google Cloud Run

```bash
gcloud run deploy calorie-logger \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your_key_here
```

Requires billing enabled on the GCP project. Redeploys keep the same URL, and env vars persist — only pass `--set-env-vars` again when the key changes.
