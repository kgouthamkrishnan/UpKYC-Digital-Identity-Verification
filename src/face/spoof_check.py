
import cv2
import numpy as np


def detect_screen_bezel(frame, min_area_ratio=0.15):

    h, w = frame.shape[:2]
    img_area = h * w
    min_area = img_area * min_area_ratio

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, None, iterations=2)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
  
        if 4 <= len(approx) <= 6:
            x, y, bw, bh = cv2.boundingRect(approx)
           
            if bw > w * 0.5 and bh > h * 0.5:
                return True

    return False


def detect_moire_pattern(frame, threshold=8.0):
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256)) 

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)

   
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    outer_ring = magnitude.copy()
    outer_ring[cy - 30:cy + 30, cx - 30:cx + 30] = 0 

    high_freq_energy = np.mean(outer_ring)
    return high_freq_energy > threshold


def check_spoof_risk(frame):
   
    bezel_detected = False
    moire_detected = False

    try:
        bezel_detected = detect_screen_bezel(frame)
    except Exception:
        pass

    try:
        moire_detected = detect_moire_pattern(frame)
    except Exception:
        pass

    risk_flagged = bezel_detected or moire_detected

    return {
        "spoof_risk_flagged": risk_flagged,
        "screen_bezel_detected": bezel_detected,
        "moire_pattern_detected": moire_detected,
        "note": (
            "Basic heuristic only - not a certified liveness/anti-spoofing "
            "system. A flagged result should route to manual review, not "
            "automatic rejection, due to false-positive risk."
        )
    }


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        result = check_spoof_risk(frame)
        print(result)