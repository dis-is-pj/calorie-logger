import asyncio
import base64
import json
import os

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
SPLITTER_MODEL = "gpt-4.1-nano"
ITEM_MODEL = "gpt-4.1-mini"

MACROS = ("calories", "protein", "carbs", "fat")


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


def aggregate(items):
    """Sum per-item estimates into the top-level response the frontend expects."""
    totals = {m: 0 for m in MACROS}
    ranges = {m: {"min": 0, "max": 0} for m in MACROS}
    breakdown = []
    names = []
    explanations = []

    for est in items:
        for m in MACROS:
            totals[m] += _num(est.get(m))
            r = _range((est.get("ranges") or {}).get(m))
            ranges[m]["min"] += r["min"]
            ranges[m]["max"] += r["max"]

        for row in est.get("breakdown") or []:
            breakdown.append({
                "item": row.get("item", ""),
                "qty": row.get("qty", ""),
                "cal": _num(row.get("cal")),
                "prot": _num(row.get("prot")),
                "carbs": _num(row.get("carbs")),
                "fat": _num(row.get("fat")),
            })

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
        "explanation": "\n\n".join(explanations),
    }


@app.post("/estimate")
async def estimate(
    text: str = Form(...),
    meal_type: str = Form(...),
    image: UploadFile = File(None),
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse({"error": "OPENAI_API_KEY not set"}, status_code=500)

    client = AsyncOpenAI(api_key=api_key)

    # Build the image content block once so it can be reused across all item calls.
    image_block = None
    if image and image.filename:
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
    split_content.append({"type": "text", "text": f"Meal type: {meal_type}\nDescription: {text}"})

    try:
        split_resp = await client.chat.completions.create(
            model=SPLITTER_MODEL,
            max_tokens=400,
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
        items = [text]

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


app.mount("/", StaticFiles(directory="public", html=True), name="static")
