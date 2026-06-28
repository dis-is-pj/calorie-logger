# Meal Calorie Logger

A personal tool to estimate calories and protein from a meal description (and optional photo), then log it to a Google Sheet via a pre-filled Google Form.

## How it works

1. Describe what you ate + pick meal type (Breakfast / Lunch / Dinner / Snacks / Protein)
2. Optionally attach a photo
3. GPT-4o estimates calories, protein, and a per-item breakdown
4. Review and edit the values, then click **Open pre-filled form** to log to Google Sheets

## Stack

- **Backend:** FastAPI + OpenAI GPT-4o (vision)
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

Requires billing enabled on the GCP project.
