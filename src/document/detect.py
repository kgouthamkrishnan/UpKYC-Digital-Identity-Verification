
import cv2
import numpy as np
import os
import urllib.request

CARD_ASPECT_RATIO = 1.6  
ASPECT_TOLERANCE = 0.35   

_YUNET_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "face", "face_detection_yunet_2023mar.onnx")
_YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def order_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      
    rect[2] = pts[np.argmax(s)]      
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   
    rect[3] = pts[np.argmax(diff)]   
    return rect


def _aspect_ratio_ok(w, h):
    if h == 0:
        return False
    ratio = w / h if w >= h else h / w
    return abs(ratio - CARD_ASPECT_RATIO) <= ASPECT_TOLERANCE


def _find_face_bbox(image):

    if not os.path.exists(_YUNET_MODEL_PATH):
        try:
            urllib.request.urlretrieve(_YUNET_URL, _YUNET_MODEL_PATH)
        except Exception:
            return None

    h, w = image.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        _YUNET_MODEL_PATH, "", (w, h),
        score_threshold=0.5, nms_threshold=0.3, top_k=5000
    )
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, fw, fh = faces[0][:4].astype(int)
    return max(0, x), max(0, y), fw, fh


def _card_region_from_face(image, face_bbox):

    img_h, img_w = image.shape[:2]
    fx, fy, fw, fh = face_bbox

    card_h = int(fh / 0.22)
    card_w = int(card_h * CARD_ASPECT_RATIO)

    face_cx = fx + fw / 2
    face_cy = fy + fh / 2

    left = int(face_cx - card_w * 0.18)
    top = int(face_cy - card_h * 0.25)
    right = left + card_w
    bottom = top + card_h

    left = max(0, left)
    top = max(0, top)
    right = min(img_w, right)
    bottom = min(img_h, bottom)

    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]])


def find_document_contour(image, min_area_ratio=0.3):

    img_area = image.shape[0] * image.shape[1]
    min_area = img_area * min_area_ratio

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=1)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            x, y, w, h = cv2.boundingRect(pts)
            if _aspect_ratio_ok(w, h):
                return pts

    face_bbox = _find_face_bbox(image)
    if face_bbox is not None:
        return _card_region_from_face(image, face_bbox)

    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)

    for c in contours:
        if cv2.contourArea(c) >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

   
    return None


def warp_perspective(image, pts):
    """Flatten the detected card region into a top-down rectangle."""
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def detect_and_flatten(image_path, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    contour = find_document_contour(image)
    if contour is None:
        raise ValueError(
            "No ID document detected in the uploaded image. Please upload "
            "a clear photo of a government ID card."
        )

    flattened = warp_perspective(image, contour)

    if output_path:
        cv2.imwrite(output_path, flattened)

    return flattened


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "../../data/sample_ids/test1.jpg"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "flattened_output.jpg"

    result = detect_and_flatten(input_path, output_path)
    print(f"Saved flattened document to {output_path}, shape: {result.shape}")