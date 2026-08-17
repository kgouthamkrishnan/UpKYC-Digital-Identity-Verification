import cv2
import numpy as np
import urllib.request
import os

MODEL_PATH = "face_detection_yunet_2023mar.onnx"
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading YuNet model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def get_detector(input_size=(320, 320)):
    ensure_model()
    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH, "", input_size,
        score_threshold=0.5, nms_threshold=0.3, top_k=5000
    )
    return detector


def detect_and_crop_face(image):
    """Given a BGR image, return the cropped largest face, or None."""
    h, w = image.shape[:2]
    detector = get_detector((w, h))
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return None

    # pick the largest detected face by box area
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, fw, fh = faces[0][:4].astype(int)
    x, y = max(0, x), max(0, y)
    crop = image[y:y + fh, x:x + fw]
    return crop


def crop_face_from_path(image_path, output_path=None):
    image = cv2.imread(image_path)
    face = detect_and_crop_face(image)
    if face is None:
        raise ValueError(f"No face detected in {image_path}")
    if output_path:
        cv2.imwrite(output_path, face)
    return face


def crop_face_from_webcam():
    """Capture a single frame from webcam and return cropped face."""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not access webcam")
    face = detect_and_crop_face(frame)
    if face is None:
        raise ValueError("No face detected in webcam frame")
    return face


if __name__ == "__main__":
    import sys
    id_face = crop_face_from_path(sys.argv[1], "id_face.jpg")
    print(f"ID face saved: id_face.jpg, shape={id_face.shape}")

    webcam_face = crop_face_from_webcam()
    cv2.imwrite("webcam_face.jpg", webcam_face)
    print("Webcam face saved: webcam_face.jpg")