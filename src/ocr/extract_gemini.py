"""
Stage 2 (alternative): OCR field extraction using Google Gemini's vision model
instead of EasyOCR. A vision-language model understands context (it knows
this is an ID card and what a name/DOB/ID-number typically look like), so it
can correct for blur, skew, and low resolution far better than a traditional
OCR engine that only recognizes character shapes.

Requires: pip install google-genai
Requires: a free Gemini API key from https://aistudio.google.com/apikey
"""

import os
import json
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from a .env file (kept outside this source file,
# in the project root) so the API key never has to be pasted into code.
load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Create a .env file in the project root "
        "with a line: GEMINI_API_KEY=your-actual-key-here"
    )

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-flash-latest"  # alias for the current stable Flash model - avoids breakage when specific dated versions get deprecated

EXTRACTION_PROMPT = """You are looking at a photo of an Indian ID document (Aadhaar/PAN/etc).
Extract exactly these fields and return ONLY a JSON object, no other text:

{
  "name": "full name as printed, corrected for obvious OCR-style misspellings",
  "dob": "date of birth in DD/MM/YYYY format, or null if not visible",
  "id_number": "the ID number exactly as printed (keep spacing as shown), or null if not visible",
  "document_type": "aadhaar / pan / other",
  "confidence": "high / medium / low - your confidence that all fields above are correct"
}

Rules:
- If the image does not appear to be a government ID document at all, set all fields to null and confidence to "low".
- Do not guess or fabricate values you cannot actually read in the image.
- Return ONLY the JSON object, no markdown formatting, no explanation."""


def extract_fields_gemini(image_path, doc_type="aadhaar"):
    """
    Send the ID image to Gemini and get back structured fields.
    Returns a dict in the same shape as the EasyOCR-based extract_fields(),
    so it's a drop-in replacement in the pipeline.
    """
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[EXTRACTION_PROMPT, image_part]
    )
    raw_text = response.text.strip()

    # Gemini sometimes wraps the JSON in ```json ... ``` even when told not to - strip that
    raw_text = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_text.strip())

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return {
            "name": None,
            "dob": None,
            "id_number": None,
            "raw_text": [raw_text],
            "avg_confidence": 0.0,
            "error": "Could not parse Gemini response as JSON"
        }

    confidence_map = {"high": 0.95, "medium": 0.65, "low": 0.3}
    confidence_score = confidence_map.get(str(parsed.get("confidence", "")).lower(), 0.5)

    return {
        "name": parsed.get("name"),
        "dob": parsed.get("dob"),
        "id_number": parsed.get("id_number"),
        "raw_text": [raw_text],  # kept for consistency with the EasyOCR output shape
        "avg_confidence": confidence_score,
        "document_type_detected": parsed.get("document_type")
    }


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "flattened_output.jpg"
    result = extract_fields_gemini(image_path)
    print(json.dumps(result, indent=2))