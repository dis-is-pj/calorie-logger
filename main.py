import base64
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = (
    "You are a nutrition estimator. The user is from Bhopal, India. "
    "Home cooking, maid uses moderate oil. Default roti weight = 38g if not mentioned. "
    "Reply ONLY with a raw JSON object, no markdown, no explanation."
)

JSON_SCHEMA = """
{
  "item_summary": "short comma-separated item list",
  "calories": 0,
  "protein": 0,
  "breakdown": [
    { "item": "", "qty": "", "cal": 0, "prot": 0 }
  ]
}
"""



@app.post("/estimate")
async def estimate(
    text: str = Form(...),
    meal_type: str = Form(...),
    image: UploadFile = File(None),
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse({"error": "OPENAI_API_KEY not set"}, status_code=500)

    client = OpenAI(api_key=api_key)

    user_text = (
        f"Meal type: {meal_type}\n"
        f"Description: {text}\n\n"
        f"Return JSON matching this schema:\n{JSON_SCHEMA}"
    )

    content = []

    if image and image.filename:
        raw = await image.read()
        b64 = base64.standard_b64encode(raw).decode("utf-8")
        media_type = image.content_type or "image/jpeg"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}", "detail": "low"},
        })

    content.append({"type": "text", "text": user_text})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)

    raw_text = response.choices[0].message.content

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return JSONResponse(
            {"error": f"Failed to parse response as JSON: {e}\nRaw: {raw_text}"},
            status_code=502,
        )

    return JSONResponse(data)


app.mount("/", StaticFiles(directory="public", html=True), name="static")
