import streamlit as st
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import run_kyc_pipeline, retry_face_verification
from auth.otp_verify import send_otp, verify_otp

st.set_page_config(page_title="UpKYC — Identity Verification", layout="centered", page_icon="🛡️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-deep: #071410;
    --bg-mid: #0C2018;
    --glass: rgba(255,255,255,0.07);
    --glass-strong: rgba(255,255,255,0.10);
    --glass-border: rgba(134,239,172,0.22);
    --mint: #4ADE80;
    --mint-soft: #86EFAC;
    --text-primary: #F5FBF7;
    --text-secondary: #A9C2B4;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 12% -5%, rgba(74,222,128,0.16) 0%, transparent 40%),
        radial-gradient(circle at 90% 105%, rgba(52,211,153,0.12) 0%, transparent 45%),
        linear-gradient(180deg, var(--bg-deep) 0%, var(--bg-mid) 100%);
    background-attachment: fixed;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.01em;
}
h1 {
    font-weight: 700 !important;
    background: linear-gradient(135deg, #FFFFFF 0%, var(--mint-soft) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

p, span, label, .stMarkdown, .stCaption, div[data-testid="stCaptionContainer"] {
    color: var(--text-secondary) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(155deg, var(--glass-strong) 0%, var(--glass) 100%) !important;
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    border: 1px solid var(--glass-border) !important;
    border-radius: 20px !important;
    padding: 6px;
    box-shadow:
        0 20px 50px rgba(0,0,0,0.45),
        0 2px 8px rgba(0,0,0,0.3),
        inset 0 1px 0 rgba(255,255,255,0.12),
        inset 0 0 40px rgba(74,222,128,0.03);
}

.stButton > button {
    background: linear-gradient(135deg, rgba(74,222,128,0.22), rgba(74,222,128,0.08));
    border: 1px solid rgba(74,222,128,0.45);
    color: #EAFCF1;
    border-radius: 12px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    padding: 0.55rem 1.2rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(74,222,128,0.35), rgba(74,222,128,0.15));
    border-color: var(--mint);
    box-shadow: 0 0 24px rgba(74,222,128,0.35);
    transform: translateY(-1px);
}

.stTextInput input, .stSelectbox > div > div, div[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}
.stTextInput input::placeholder { color: #6B8577 !important; }

div[data-testid="stAlert"] {
    background: rgba(255,255,255,0.07) !important;
    backdrop-filter: blur(14px);
    border-radius: 14px !important;
    border: 1px solid var(--glass-border) !important;
}

hr { border-color: var(--glass-border) !important; }

.status-badge {
    display: inline-block;
    padding: 10px 24px;
    border-radius: 999px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.4rem;
    letter-spacing: 0.02em;
    margin-top: 6px;
}
.status-verified {
    background: linear-gradient(135deg, rgba(74,222,128,0.26), rgba(74,222,128,0.10));
    border: 1px solid rgba(74,222,128,0.55);
    color: #A7F3D0;
    box-shadow: 0 0 34px rgba(74,222,128,0.3);
}
.status-review {
    background: linear-gradient(135deg, rgba(251,191,36,0.24), rgba(251,191,36,0.08));
    border: 1px solid rgba(251,191,36,0.55);
    color: #FDE68A;
    box-shadow: 0 0 34px rgba(251,191,36,0.2);
}
.status-rejected {
    background: linear-gradient(135deg, rgba(248,113,113,0.24), rgba(248,113,113,0.08));
    border: 1px solid rgba(248,113,113,0.55);
    color: #FECACA;
    box-shadow: 0 0 34px rgba(248,113,113,0.2);
}

.field-row {
    display: flex;
    justify-content: space-between;
    padding: 11px 4px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.field-label { color: var(--text-secondary); font-size: 0.92rem; }
.field-value { color: var(--text-primary); font-weight: 600; font-size: 0.92rem; }

.eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    color: var(--mint-soft);
    font-weight: 700;
    margin-bottom: 4px;
}

.step-track { display: flex; gap: 8px; margin-bottom: 22px; }
.step-dot {
    height: 4px;
    flex: 1;
    border-radius: 4px;
    background: rgba(255,255,255,0.10);
    transition: background 0.3s ease;
}
.step-dot.done { background: linear-gradient(90deg, #4ADE80, #86EFAC); }
.step-dot.active { background: linear-gradient(90deg, rgba(74,222,128,0.5), rgba(134,239,172,0.5)); }
</style>
""", unsafe_allow_html=True)


def field_row(label, value):
    st.markdown(
        f"<div class='field-row'><span class='field-label'>{label}</span>"
        f"<span class='field-value'>{value}</span></div>",
        unsafe_allow_html=True,
    )



_VIRTUAL_CAMERA_NAME_FRAGMENTS = [
    "avatarmaster",      
    "virtual camera",
    "obs virtual",
    "obs-camera",
    "snap camera",
    "iriun",
    "droidcam",
    "epoccam",
    "manycam",
    "nvidia broadcast",
    "streamlabs",
    "xsplit",
    "camtwist",
    "ir camera",         
    "infrared",
]


def _is_genuine_external_camera(device_name):
    name_lower = device_name.lower()
    return not any(fragment in name_lower for fragment in _VIRTUAL_CAMERA_NAME_FRAGMENTS)


@st.cache_resource
def list_camera_devices():
  
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        devices = graph.get_input_devices()
        return devices
    except Exception:
        import cv2
        cap = cv2.VideoCapture(0)
        has_cam0 = cap.isOpened()
        if has_cam0:
            has_cam0 = cap.read()[0]
        cap.release()
        return ["Camera 0"] if has_cam0 else []


@st.cache_resource
def detect_external_camera():

    devices = list_camera_devices()
    if len(devices) < 2:
        return False, None

    for idx, name in enumerate(devices):
        if idx == 0:
            continue  # index 0 assumed to be the internal/built-in camera
        if _is_genuine_external_camera(name):
            return True, idx

    return False, None


defaults = {
    "wizard_step": 1,
    "mobile_verified": None,
    "otp_sent_to": None,
    "otp_send_result": None,
    "kyc_result": None,
    "pending_doc_type": None,
    "manual_review_submitted": False,
    "camera_index": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown("<div class='eyebrow'>Digital Identity Verification</div>", unsafe_allow_html=True)
st.title("UpKYC")
st.caption("Prototype only — not a production KYC/compliance system.")


external_available, external_index = detect_external_camera()

if external_available:
    cam_col1, cam_col2 = st.columns([5, 1])
    with cam_col2:
        cam_label = "📷 Cam 1 (internal)" if st.session_state.camera_index == 0 else "📷 Cam 2 (external)"
        if st.button(f"🔄 {cam_label}", help="Switch between internal and external webcam"):
            st.session_state.camera_index = external_index if st.session_state.camera_index == 0 else 0
            st.rerun()
else:

    st.session_state.camera_index = 0

dots_html = "<div class='step-track'>"
for i in (1, 2, 3):
    cls = "done" if st.session_state.wizard_step > i else ("active" if st.session_state.wizard_step == i else "")
    dots_html += f"<div class='step-dot {cls}'></div>"
dots_html += "</div>"
st.markdown(dots_html, unsafe_allow_html=True)


if st.session_state.wizard_step == 1:
    with st.container(border=True):
        st.markdown("<div class='eyebrow'>Step 1 of 3</div>", unsafe_allow_html=True)
        st.subheader("Verify mobile number")
        st.caption(
            "Confirms you own this phone number via OTP. This does not check that "
            "the number is registered with your Aadhaar/ID (that requires UIDAI "
            "access this prototype doesn't have)."
        )

        if st.session_state.mobile_verified and st.session_state.mobile_verified.get("verified"):
            st.success(f"✅ Mobile number verified: {st.session_state.otp_sent_to}")
            if st.button("Continue →"):
                st.session_state.wizard_step = 2
                st.rerun()
        else:
            phone_number = st.text_input("Phone number (E.164 format, e.g. +919876543210)")

            if st.button("Send OTP"):
                if phone_number:
                    result = send_otp(phone_number)
                    st.session_state.otp_sent_to = phone_number
                    st.session_state.otp_send_result = result
                else:
                    st.warning("Enter a phone number first.")

            if st.session_state.otp_send_result:
                result = st.session_state.otp_send_result
                if result["mode"] == "simulated":
                    st.info(f"📱 SIMULATED — no real SMS sent. Your OTP is: **{result['otp']}**")
                else:
                    st.info(result["message"])

                entered_otp = st.text_input("Enter the OTP")
                if st.button("Verify OTP"):
                    verify_result = verify_otp(st.session_state.otp_sent_to, entered_otp)
                    if verify_result["verified"]:
                        st.session_state.mobile_verified = verify_result
                        st.rerun()
                    else:
                        st.error(verify_result["reason"])

# ---------------------------------------------------------------------------
# STEP 2: Upload ID and run verification
# ---------------------------------------------------------------------------
elif st.session_state.wizard_step == 2:
    with st.container(border=True):
        st.markdown("<div class='eyebrow'>Step 2 of 3</div>", unsafe_allow_html=True)
        st.subheader("Upload ID document")
        input_method = st.radio("How do you want to provide your ID?", ["Upload photo", "Use camera"], horizontal=True)

        id_image_bytes = None

        if input_method == "Upload photo":
            uploaded_id = st.file_uploader("Upload ID document image", type=["jpg", "jpeg", "png"])
            if uploaded_id:
                id_image_bytes = uploaded_id.getbuffer()
        else:
            camera_id = st.camera_input("Take a photo of your ID document")
            if camera_id:
                id_image_bytes = camera_id.getbuffer()

        doc_type = st.selectbox("Document type", ["aadhaar", "pan", "other"])

        col_back, col_run = st.columns([1, 2])
        with col_back:
            if st.button("← Back"):
                st.session_state.wizard_step = 1
                st.rerun()
        with col_run:
            if id_image_bytes is not None and st.button("Run KYC Verification"):
                temp_path = "_uploaded_id.jpg"
                with open(temp_path, "wb") as f:
                    f.write(id_image_bytes)

                st.session_state.pending_doc_type = doc_type
                st.session_state.manual_review_submitted = False

                active_cam = "Cam 1 (internal)" if st.session_state.camera_index == 0 else "Cam 2 (external)"
                st.info(f"A camera window will pop up outside the browser, using {active_cam}. Position your face in the oval, hold still, then blink when prompted. Press 'q' in that window to cancel.")

                with st.spinner("Running pipeline... check the camera window that just opened."):
                    result = run_kyc_pipeline(
                        temp_path,
                        doc_type,
                        mobile_verified=st.session_state.mobile_verified,
                        camera_index=st.session_state.camera_index,
                    )
                st.session_state.kyc_result = result
                st.session_state.wizard_step = 3
                st.rerun()


elif st.session_state.wizard_step == 3:
    result = st.session_state.kyc_result

    # --- Gemini OCR fallback decision ---
    if result and result.get("final_status") == "NEEDS_OCR_FALLBACK_DECISION":
        with st.container(border=True):
            st.markdown("<div class='eyebrow'>Step 3 of 3</div>", unsafe_allow_html=True)
            st.warning(f"⚠️ {result['reason']}")
            st.write("Try offline OCR instead? Accuracy may be lower on blurry/angled images.")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, use offline OCR"):
                    with st.spinner("Retrying with offline OCR..."):
                        new_result = run_kyc_pipeline(
                            "_uploaded_id.jpg",
                            st.session_state.pending_doc_type,
                            mobile_verified=st.session_state.mobile_verified,
                            ocr_fallback=True,
                            camera_index=st.session_state.camera_index,
                        )
                    st.session_state.kyc_result = new_result
                    st.rerun()
            with col2:
                if st.button("No, cancel"):
                    with st.spinner("Cancelling..."):
                        new_result = run_kyc_pipeline(
                            "_uploaded_id.jpg",
                            st.session_state.pending_doc_type,
                            mobile_verified=st.session_state.mobile_verified,
                            ocr_fallback=False,
                            camera_index=st.session_state.camera_index,
                        )
                    st.session_state.kyc_result = new_result
                    st.rerun()

    # --- Manual review already submitted: show confirmation, stop here ---
    elif st.session_state.manual_review_submitted:
        with st.container(border=True):
            st.markdown("<div class='eyebrow'>Step 3 of 3</div>", unsafe_allow_html=True)
            st.markdown(
                "<span class='status-badge status-review'>QUEUED FOR MANUAL REVIEW</span>",
                unsafe_allow_html=True,
            )
            st.write("")
            st.caption(
                "Your submission has been queued for a human reviewer. In this "
                "prototype no real queue exists - this screen represents what a "
                "production system would show at this point in the flow."
            )
            st.write("")
            if st.button("Start New Verification"):
                st.session_state.wizard_step = 1
                st.session_state.mobile_verified = None
                st.session_state.otp_sent_to = None
                st.session_state.otp_send_result = None
                st.session_state.kyc_result = None
                st.session_state.manual_review_submitted = False
                st.rerun()

    # --- Final report ---
    elif result:
        with st.container(border=True):
            st.markdown("<div class='eyebrow'>Step 3 of 3 — Report</div>", unsafe_allow_html=True)
            st.subheader("KYC Report")

            if result.get("doc_type_mismatch_warning"):
                st.warning(f"⚠️ {result['doc_type_mismatch_warning']}")

            field_row("Document Detected", "✅ Yes" if result.get("document_detected") else "❌ No")

            if "ocr" in result:
                ocr = result["ocr"]
                field_row("Name", ocr.get("name") or "Not detected")
                field_row("Date of Birth", ocr.get("dob") or "Not detected")
                field_row("ID Number", ocr.get("id_number") or "Not detected")
                field_row("OCR Confidence", f"{ocr.get('avg_confidence', 0) * 100:.1f}%")
                if ocr.get("document_type_detected"):
                    field_row("Detected Document Type", ocr["document_type_detected"])
                if ocr.get("ocr_method"):
                    method_labels = {
                        "gemini": "Gemini Vision (online)",
                        "easyocr": "EasyOCR (offline)",
                        "easyocr_fallback": "EasyOCR (offline fallback — Gemini failed)",
                    }
                    field_row("OCR Method", method_labels.get(ocr["ocr_method"], ocr["ocr_method"]))
                if ocr.get("accuracy_note"):
                    st.caption(ocr["accuracy_note"])

            mv = result.get("mobile_verification")
            if mv:
                field_row("Mobile Verification", "✅ Verified" if mv.get("verified") else f"❌ {mv.get('reason')}")

            if "face_verification" in result:
                fv = result["face_verification"]
                if "error" not in fv:
                    field_row("Face Similarity Score", fv["similarity_score"])
                    field_row("Match Tier", f"{fv['tier']} — {fv['message']}")

            if "liveness" in result:
                lv = result["liveness"]
                field_row("Liveness Check", "✅ Passed" if lv.get("liveness_passed") else "❌ Failed")

            if "spoof_check" in result:
                sc = result["spoof_check"]
                if sc.get("spoof_risk_flagged"):
                    st.warning(f"⚠️ Basic spoof-risk heuristic flagged this capture. {sc.get('note', '')}")
                else:
                    field_row("Spoof Risk Check", "✅ No signals detected (basic heuristic)")

            status = result.get("final_status", "UNKNOWN")
            badge_class = {
                "VERIFIED": "status-verified",
                "REVIEW REQUIRED": "status-review",
                "REJECTED": "status-rejected",
            }.get(status, "status-review")

            st.write("")
            st.markdown("<div class='eyebrow'>Final Status</div>", unsafe_allow_html=True)
            st.markdown(f"<span class='status-badge {badge_class}'>{status}</span>", unsafe_allow_html=True)

            if "reason" in result:
                st.write("")
                st.caption(f"Reason: {result['reason']}")

            st.write("")

            # --- REVIEW REQUIRED: offer retry-face-only or manual-review paths ---
            if status == "REVIEW REQUIRED":
                col_retry, col_manual = st.columns(2)
                with col_retry:
                    if st.button("🔄 Retry Face Verification"):
                        active_cam = "Cam 1 (internal)" if st.session_state.camera_index == 0 else "Cam 2 (external)"
                        st.info(f"A camera window will pop up, using {active_cam}. Position your face in the oval, hold still, then blink when prompted.")
                        with st.spinner("Re-running face verification..."):
                            new_result = retry_face_verification(
                                st.session_state.pending_doc_type,
                                result.get("ocr", {}),
                                mobile_verified=st.session_state.mobile_verified,
                                camera_index=st.session_state.camera_index,
                            )
                        st.session_state.kyc_result = new_result
                        st.rerun()
                with col_manual:
                    if st.button("📋 Submit for Manual Review"):
                        st.session_state.manual_review_submitted = True
                        st.rerun()
            else:
                if st.button("Start New Verification"):
                    st.session_state.wizard_step = 1
                    st.session_state.mobile_verified = None
                    st.session_state.otp_sent_to = None
                    st.session_state.otp_send_result = None
                    st.session_state.kyc_result = None
                    st.session_state.manual_review_submitted = False
                    st.rerun()