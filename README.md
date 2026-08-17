# UpKYC — Digital Identity Verification

A prototype end-to-end KYC (Know Your Customer) identity verification system. Combines document detection, OCR field extraction, face matching, liveness detection, and basic spoof-risk heuristics into a single guided verification flow, wrapped in a Streamlit web UI.

> **Prototype only** — not a production KYC/compliance system. See [Honest Scope Notes](#honest-scope-notes) below for what this does and doesn't actually verify.

---

## Features

- **Mobile OTP verification** — confirms phone ownership via OTP (real SMS through Twilio, or simulated on-screen OTP if no SMS gateway is configured)
- **ID document detection & flattening** — detects a card-shaped region in a photo (any angle/background) and perspective-corrects it to a flat, top-down image
- **OCR field extraction** — pulls name, date of birth, and ID number, using either:
  - **Gemini Vision API** (primary — higher accuracy, understands context, needs internet)
  - **EasyOCR** (offline fallback — works without internet, lower accuracy on blurry/angled images)
- **Face detection & matching** — extracts faces from the ID and a live webcam capture, compares them using InsightFace embeddings (cosine similarity → HIGH / MEDIUM / LOW match tiers)
- **Guided liveness capture** — live webcam window with an on-screen oval guide, lighting check, and blink detection before capturing
- **Basic spoof-risk heuristic** — screen-bezel detection + moire-pattern check (flags suspicious captures for manual review, does not auto-reject)
- **Camera selection (Cam 1 / Cam 2)** — toggle between the internal webcam and a genuine external USB webcam, when one is actually connected (virtual/software cameras like AI-effects cams are filtered out automatically)
- **Multi-outcome results** — `VERIFIED`, `REJECTED`, or `REVIEW REQUIRED`, with retry-face-only and manual-review-submission paths for the latter

---

## Project Structure

```
kyc-verification-system/
├── app/
│   └── streamlit_app.py        # Streamlit UI (3-step wizard)
├── data/
│   └── sample_ids/              # Sample test ID images
│       ├── aadhar.jpg
│       └── test1.jpg
├── src/
│   ├── pipeline.py              # Orchestrates all stages, produces final decision
│   ├── auth/
│   │   └── otp_verify.py        # Mobile OTP send/verify (Twilio or simulated)
│   ├── document/
│   │   └── detect.py            # Stage 1: document contour detection + flatten
│   ├── ocr/
│   │   ├── extract.py           # Stage 2: EasyOCR field extraction (offline)
│   │   └── extract_gemini.py    # Stage 2 (alt): Gemini Vision field extraction
│   └── face/
│       ├── detect_face.py       # Face detection/cropping (YuNet)
│       ├── guided_capture.py    # Guided webcam capture with liveness (blink) check
│       ├── liveness.py          # Standalone blink-sequence liveness check
│       ├── verify.py            # Face embedding + similarity comparison (InsightFace)
│       ├── spoof_check.py       # Screen-bezel / moire-pattern spoof heuristic
│       └── webcam_test.py       # Quick standalone webcam sanity check
├── requirements.txt
├── .env                          # API keys (not committed — see Setup)
└── .gitignore
```

> **Note:** `face_detection_yunet_2023mar.onnx` and `requirements.txt` currently exist in more than one place in the repo (project root and inside `src/`). Worth cleaning up to a single copy of each to avoid drift between them.

---

## Setup

### 1. Clone and create a virtual environment

```powershell
cd kyc-verification-system
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
# Required for Gemini OCR (get a free key at https://aistudio.google.com/apikey)
GEMINI_API_KEY=your-gemini-api-key-here

# Optional — only needed for real SMS delivery. Without these, OTP runs in
# simulated mode (OTP shown directly on-screen instead of sent via SMS).
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=
```

### 4. (Optional) Enable accurate external-camera detection

For the Cam 1 / Cam 2 toggle to correctly detect a genuine external USB webcam (as opposed to virtual/software cameras like AI-effects apps or IR cameras):

```powershell
pip install pygrabber
```

Without this, the app assumes a single (internal) camera only.

### 5. Run the app

```powershell
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`.

---

## Usage Flow

1. **Step 1 — Mobile verification:** Enter a phone number, receive/enter the OTP.
2. **Step 2 — Upload ID:** Upload a photo (or take one via camera) of an Aadhaar/PAN/other ID, select the document type, and run verification.
3. **Face capture:** A separate camera window opens outside the browser — position your face in the oval, hold still, then blink naturally when prompted.
4. **Step 3 — Report:** View extracted fields, match score, liveness/spoof results, and the final status:
   - **VERIFIED** — high face match, liveness passed, no spoof flags, mobile verified
   - **REJECTED** — clear mismatch or invalid document
   - **REVIEW REQUIRED** — borderline case; offers **Retry Face Verification** or **Submit for Manual Review**

---

## Honest Scope Notes

This project is built to be transparent about what it actually does versus what a production KYC system would need:

- **Mobile OTP** verifies phone *ownership*, not that the number is registered against the person's Aadhaar with UIDAI (that requires UIDAI-authorized access this project doesn't have).
- **Spoof detection** is a basic heuristic (screen-bezel + moire pattern), not a certified liveness/anti-spoofing system (e.g. FaceTec, iProov). It can be defeated by a screen held close enough that its edges are out of frame, a high-quality printed photo, or a deepfake injected at the software level. Flagged results route to manual review, not automatic rejection.
- **Face matching** uses InsightFace's open-source embeddings — solid for a prototype, but not validated to the standard a regulated financial KYC system would require.

---

## Tech Stack

- **UI:** Streamlit
- **Computer Vision:** OpenCV, MediaPipe (face mesh / blink detection)
- **Face Detection/Recognition:** YuNet (detection), InsightFace `buffalo_l` (embeddings)
- **OCR:** EasyOCR (offline), Google Gemini Vision API (online)
- **SMS:** Twilio (optional)
- **Camera enumeration:** pygrabber (Windows DirectShow, optional)
