import asyncio
import base64
import json
import os
from datetime import datetime

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Small/cheap model just isolates the items; the accurate model estimates each one.
SPLITTER_MODEL = "gpt-6-luna"
ITEM_MODEL = "gpt-4.1-mini"

MACROS = ("calories", "protein", "carbs", "fat")

# The Google Form the log is written to. /formResponse is the POST endpoint behind
# the form's own Submit button, so posting to it directly records a real response.
FORM_ID = "1FAIpQLSedsQl0qrze-gL_Nmn1U4lhWVUIkIkufRK_ECkOPar3bfOnNQ"
FORM_POST_URL = f"https://docs.google.com/forms/d/e/{FORM_ID}/formResponse"

ENTRY_DATE = "entry.556709106"
ENTRY_MEAL = "entry.1659704160"
ENTRY_ITEM = "entry.608133542"
ENTRY_CALORIES = "entry.998829307"
ENTRY_PROTEIN = "entry.343920601"

MEAL_TYPES = ("Breakfast", "Lunch", "Dinner", "Snacks", "Protein")


def _load_prompt(filename):
    with open(filename, encoding="utf-8") as f:
        return f.read().strip()


SPLITTER_PROMPT = _load_prompt("splitter_prompt.txt")
ITEM_PROMPT = _load_prompt("item_prompt.txt")


def _num(value):
    """Coerce an LLM-provided value to a number, defaulting to 0."""
    try:
        return round(float(value))
    except (TypeError, ValueError):
        return 0


def _range(value):
    """Coerce a {"min": .., "max": ..} block, defaulting to zeros."""
    value = value or {}
    return {"min": _num(value.get("min")), "max": _num(value.get("max"))}


BREAKDOWN_KEYS = {"calories": "cal", "protein": "prot", "carbs": "carbs", "fat": "fat"}


def aggregate(items):
    """Sum per-item estimates into the top-level response the frontend expects.

    Item totals are recomputed here from each item's breakdown rows rather than
    trusted from the LLM's own top-level fields — the model's arithmetic isn't
    reliable enough, so Python does all the summing.
    """
    totals = {m: 0 for m in MACROS}
    ranges = {m: {"min": 0, "max": 0} for m in MACROS}
    breakdown = []
    names = []
    explanations = []

    for est in items:
        rows = [
            {
                "item": row.get("item", ""),
                "qty": row.get("qty", ""),
                "cal": _num(row.get("cal")),
                "prot": _num(row.get("prot")),
                "carbs": _num(row.get("carbs")),
                "fat": _num(row.get("fat")),
            }
            for row in est.get("breakdown") or []
        ]
        breakdown.extend(rows)

        for m in MACROS:
            key = BREAKDOWN_KEYS[m]
            item_total = sum(row[key] for row in rows) if rows else _num(est.get(m))
            totals[m] += item_total

            r = _range((est.get("ranges") or {}).get(m))
            ranges[m]["min"] += r["min"]
            ranges[m]["max"] += r["max"]

        name = est.get("item_summary") or ""
        if name:
            names.append(name)
        expl = est.get("explanation") or ""
        if expl:
            explanations.append(f"— {name or 'item'} —\n{expl}")

    return {
        "item_summary": ", ".join(names),
        "calories": totals["calories"],
        "protein": totals["protein"],
        "carbs": totals["carbs"],
        "fat": totals["fat"],
        "ranges": ranges,
        "breakdown": breakdown,
        "explanation": "\n--------------------------------\n".join(explanations),
    }


@app.post("/estimate")
async def estimate(
    text: str = Form(""),
    meal_type: str = Form(...),
    image: UploadFile = File(None),
):
    text = (text or "").strip()
    has_image = bool(image and image.filename)
    if not text and not has_image:
        return JSONResponse({"error": "Provide a description or a photo."}, status_code=400)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse({"error": "OPENAI_API_KEY not set"}, status_code=500)

    client = AsyncOpenAI(api_key=api_key)

    # Build the image content block once so it can be reused across all item calls.
    image_block = None
    if has_image:
        raw = await image.read()
        b64 = base64.standard_b64encode(raw).decode("utf-8")
        media_type = image.content_type or "image/jpeg"
        image_block = {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}", "detail": "low"},
        }

    # --- Stage 1: split the meal into isolated items ---
    split_content = []
    if image_block:
        split_content.append(image_block)
    description = text or "(none — identify the items from the photo)"
    split_content.append({"type": "text", "text": f"Meal type: {meal_type}\nDescription: {description}"})

    try:
        split_resp = await client.chat.completions.create(
            model=SPLITTER_MODEL,
            max_completion_tokens=400,
            reasoning_effort="low",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SPLITTER_PROMPT},
                {"role": "user", "content": split_content},
            ],
        )
        items = json.loads(split_resp.choices[0].message.content).get("items") or []
    except Exception as e:
        return JSONResponse({"error": f"Splitter failed: {e}"}, status_code=502)

    items = [i for i in items if isinstance(i, str) and i.strip()]
    if not items:
        # Fall back to treating the whole description as one item.
        items = [text] if text else ["the meal shown in the photo"]

    # --- Stage 2: estimate each item in its own call, concurrently ---
    async def estimate_item(item_text):
        item_content = []
        if image_block:
            item_content.append(image_block)
        item_content.append({"type": "text", "text": f"Meal type: {meal_type}\nItem: {item_text}"})
        resp = await client.chat.completions.create(
            model=ITEM_MODEL,
            max_tokens=700,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": ITEM_PROMPT},
                {"role": "user", "content": item_content},
            ],
        )
        return json.loads(resp.choices[0].message.content)

    results = await asyncio.gather(
        *[estimate_item(i) for i in items], return_exceptions=True
    )

    failed = [items[idx] for idx, r in enumerate(results) if isinstance(r, Exception)]
    if failed:
        return JSONResponse(
            {"error": f"Failed to estimate item(s): {'; '.join(failed)}"},
            status_code=502,
        )

    # --- Stage 3: sum natively and return the existing response contract ---
    return JSONResponse(aggregate(results))


@app.post("/log")
async def log(
    date: str = Form(...),  # ISO "YYYY-MM-DD"
    meal_type: str = Form(...),
    item: str = Form(""),
    calories: str = Form(...),
    protein: str = Form(...),
):
    try:
        day = datetime.strptime(date.strip(), "%Y-%m-%d")
    except ValueError:
        return JSONResponse({"error": f"Bad date: {date!r} (expected YYYY-MM-DD)"}, status_code=400)

    meal_type = (meal_type or "").strip()
    if meal_type not in MEAL_TYPES:
        return JSONResponse({"error": f"Unknown meal type: {meal_type!r}"}, status_code=400)

    for label, raw in (("Calories", calories), ("Protein", protein)):
        try:
            float((raw or "").strip())
        except ValueError:
            return JSONResponse({"error": f"{label} must be a number."}, status_code=400)

    # A date question posts as three separate params — the YYYY-MM-DD form the
    # pre-fill URL uses is not accepted here. The trailing fields are what the
    # form's own page submits alongside the answers.
    payload = {
        f"{ENTRY_DATE}_year": day.year,
        f"{ENTRY_DATE}_month": day.month,
        f"{ENTRY_DATE}_day": day.day,
        ENTRY_MEAL: meal_type,
        ENTRY_ITEM: (item or "").strip(),
        ENTRY_CALORIES: calories.strip(),
        ENTRY_PROTEIN: protein.strip(),
        "fvv": "1",
        "pageHistory": "0",
        "submissionTimestamp": "-1",
    }

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.post(FORM_POST_URL, data=payload)
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"Could not reach Google Forms: {e}"}, status_code=502)

    if resp.status_code != 200:
        return JSONResponse(
            {"error": f"Google Forms returned {resp.status_code}."}, status_code=502
        )

    # Google answers 200 either way: the confirmation page on success, or the form
    # re-rendered with errors on rejection. The form element only exists in the latter.
    if 'id="mG61Hd"' in resp.text or "FormviewerViewNavigationSubmitButton" in resp.text:
        return JSONResponse(
            {"error": "Google rejected the submission — use the manual form link below."},
            status_code=502,
        )

    return JSONResponse({"ok": True})


app.mount("/", StaticFiles(directory="public", html=True), name="static")
