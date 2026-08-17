"""
Orchestrates all stages and produces the final KYC decision.
"""

from document.detect import detect_and_flatten
from ocr.extract import extract_fields
from ocr.extract_gemini import extract_fields_gemini
from face.detect_face import detect_and_crop_face
from face.verify import verify_faces_multiframe
from face.guided_capture import guided_capture
from face.spoof_check import check_spoof_risk

import cv2

_FLATTENED_CACHE_PATH = "_flattened_temp.jpg"


def capture_burst_frames(camera_index=0, num_frames=6, delay_frames=2):
    """
    Grab a short burst of frames from the webcam, a couple of camera frames
    apart, right after the guided liveness capture succeeds. Averaging
    embeddings across these frames smooths out noise from any single
    frame's angle/lighting/motion-blur.

    camera_index: 0 = Cam 1 (built-in), 1 = Cam 2 (external).
    """
    cap = cv2.VideoCapture(camera_index)
    frames = []
    if not cap.isOpened():
        cap.release()
        return frames

    for i in range(num_frames * delay_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        if i % delay_frames == 0:
            frames.append(frame)

    cap.release()
    return frames


def run_kyc_pipeline(id_image_path, doc_type="aadhaar", use_gemini=True,
                      mobile_verified=None, ocr_fallback=None, camera_index=0):
    """
    use_gemini: True  -> use Gemini Vision API for OCR (higher accuracy, needs internet)
                False -> use EasyOCR (works offline, lower accuracy on poor-quality images)
    mobile_verified: dict returned by auth.otp_verify.verify_otp(), or None if
                      the OTP step was skipped/not completed.
    ocr_fallback: how to handle a Gemini OCR failure (see docstring below in
                  the None/True/False branches) - avoids blocking on
                  terminal input() inside a web UI.
    camera_index: 0 = Cam 1 (built-in webcam), 1 = Cam 2 (external webcam).
                  Passed through to the guided capture + burst-frame stages.
    """
    report = {}
    report["mobile_verification"] = mobile_verified or {
        "verified": False,
        "reason": "OTP step not completed"
    }

    # Stage 1: document detection
    try:
        flattened = detect_and_flatten(id_image_path)
        report["document_detected"] = True
    except Exception as e:
        report["document_detected"] = False
        report["final_status"] = "REJECTED"
        report["reason"] = f"Document detection failed: {e}"
        return report

    cv2.imwrite(_FLATTENED_CACHE_PATH, flattened)

    # Stage 2: OCR
    if use_gemini:
        try:
            ocr_fields = extract_fields_gemini(_FLATTENED_CACHE_PATH, doc_type)
            ocr_fields["ocr_method"] = "gemini"
        except Exception as e:
            if ocr_fallback is None:
                report["final_status"] = "NEEDS_OCR_FALLBACK_DECISION"
                report["reason"] = f"Gemini OCR failed: {e}"
                report["_flattened_path"] = _FLATTENED_CACHE_PATH
                return report
            elif ocr_fallback is True:
                ocr_fields = extract_fields(_FLATTENED_CACHE_PATH, doc_type)
                ocr_fields["ocr_method"] = "easyocr_fallback"
                ocr_fields["accuracy_note"] = "Extracted using offline OCR - results may be less accurate than Gemini, especially on blurry, angled, or low-resolution images. Please verify fields manually."
            else:
                report["final_status"] = "REJECTED"
                report["reason"] = "OCR unavailable (Gemini failed, offline OCR declined)"
                return report
    else:
        ocr_fields = extract_fields(_FLATTENED_CACHE_PATH, doc_type)
        ocr_fields["ocr_method"] = "easyocr"
        ocr_fields["accuracy_note"] = "Extracted using offline OCR - results may be less accurate than Gemini, especially on blurry, angled, or low-resolution images. Please verify fields manually."
    report["ocr"] = ocr_fields

    if not ocr_fields.get("name") and not ocr_fields.get("dob") and not ocr_fields.get("id_number"):
        report["final_status"] = "REJECTED"
        report["reason"] = (
            "No ID fields (name, date of birth, or ID number) could be read "
            "from this image. This does not appear to be a valid ID document."
        )
        return report

    detected_type = ocr_fields.get("document_type_detected")
    if detected_type and detected_type.lower() != "other" and detected_type.lower() != doc_type.lower():
        report["final_status"] = "REJECTED"
        report["reason"] = (
            f"Selected document type was '{doc_type}', but the uploaded "
            f"document appears to be '{detected_type}'. Please re-upload "
            f"with the correct document type selected."
        )
        return report

    # Stage 3 onward: shared with retry_face_verification()
    return _run_face_stages(flattened, ocr_fields, mobile_verified, report, camera_index)


def _run_face_stages(flattened, ocr_fields, mobile_verified, report, camera_index=0):
    """
    Stage 3 (face extraction on ID) through Stage 4 (verification) and the
    final decision. Factored out so a REVIEW REQUIRED result caused by a
    weak face match or a failed liveness/spoof check can be retried on its
    own - re-running the webcam capture and comparison - without redoing
    document detection or OCR.

    camera_index: 0 = Cam 1 (built-in), 1 = Cam 2 (external). Passed down
                  to guided_capture() and capture_burst_frames().
    """
    id_face = detect_and_crop_face(flattened)
    if id_face is None:
        report["final_status"] = "REJECTED"
        report["reason"] = "No face found on ID document"
        return report

    webcam_frame, liveness_result = guided_capture(camera_index=camera_index)
    report["liveness"] = liveness_result

    if webcam_frame is None or not liveness_result.get("liveness_passed"):
        report["final_status"] = "REVIEW REQUIRED"
        report["reason"] = liveness_result.get("reason", "Webcam capture failed")
        return report

    webcam_face = detect_and_crop_face(webcam_frame)
    if webcam_face is None:
        report["final_status"] = "REVIEW REQUIRED"
        report["reason"] = "Could not extract face from captured frame"
        return report

    spoof_result = check_spoof_risk(webcam_frame)
    report["spoof_check"] = spoof_result

    burst_frames = capture_burst_frames(camera_index=camera_index)
    all_frames = [webcam_frame] + burst_frames

    face_result = verify_faces_multiframe(flattened, all_frames)
    report["face_verification"] = face_result

    report["final_status"] = decide_final_status(
        ocr_fields, face_result, liveness_result, spoof_result, mobile_verified
    )
    return report


def retry_face_verification(doc_type, ocr_fields, mobile_verified=None, camera_index=0):
    """
    Re-run just the face/liveness/spoof/verification stages, reusing the
    cached flattened ID image from the most recent run_kyc_pipeline() call
    instead of re-uploading and re-running document detection + OCR. Used
    when a result comes back REVIEW REQUIRED due to a weak face match or a
    failed liveness/spoof check - the user can retry the camera step alone.

    camera_index: 0 = Cam 1 (built-in), 1 = Cam 2 (external).
    """
    import cv2
    flattened = cv2.imread(_FLATTENED_CACHE_PATH)
    if flattened is None:
        return {
            "final_status": "REJECTED",
            "reason": "Cached document image not found - please re-upload your ID document.",
        }

    report = {"ocr": ocr_fields}
    report["mobile_verification"] = mobile_verified or {
        "verified": False,
        "reason": "OTP step not completed"
    }
    report["document_detected"] = True
    return _run_face_stages(flattened, ocr_fields, mobile_verified, report, camera_index)


def decide_final_status(ocr_fields, face_result, liveness_result, spoof_result=None, mobile_verified=None):
    if not ocr_fields.get("name") or not ocr_fields.get("id_number"):
        return "REVIEW REQUIRED"

    if "error" in face_result:
        return "REVIEW REQUIRED"

    tier = face_result.get("tier")
    liveness_passed = liveness_result.get("liveness_passed", False)
    spoof_flagged = bool(spoof_result and spoof_result.get("spoof_risk_flagged"))
    mobile_ok = bool(mobile_verified and mobile_verified.get("verified"))

    if tier == "HIGH" and liveness_passed and not spoof_flagged and mobile_ok:
        return "VERIFIED"
    elif tier == "LOW":
        return "REJECTED"
    elif not liveness_passed:
        return "REVIEW REQUIRED"
    elif spoof_flagged:
        return "REVIEW REQUIRED"
    elif not mobile_ok:
        return "REVIEW REQUIRED"
    else:  # MEDIUM tier
        return "REVIEW REQUIRED"


if __name__ == "__main__":
    import sys
    import json
    result = run_kyc_pipeline(sys.argv[1], ocr_fallback=True)
    print(json.dumps(result, indent=2, default=str))