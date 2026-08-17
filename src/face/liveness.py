import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# landmark indices for left/right eye (MediaPipe 468-point mesh)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    pts = [(landmarks[i].x * image_w, landmarks[i].y * image_h) for i in eye_indices]
    pts = np.array(pts)
    vertical1 = np.linalg.norm(pts[1] - pts[5])
    vertical2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear


def check_blink_sequence(num_frames=30, ear_threshold=0.21):
    """
    Capture frames from webcam for ~a few seconds, track EAR,
    and check whether a blink (EAR dip below threshold) occurred.
    """
    cap = cv2.VideoCapture(0)
    ear_values = []

    with mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:

        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0
                ear_values.append(avg_ear)

    cap.release()

    if not ear_values:
        return {"liveness_passed": False, "reason": "No face detected during capture"}

    min_ear = min(ear_values)
    blink_detected = min_ear < ear_threshold

    return {
        "liveness_passed": blink_detected,
        "min_ear": round(min_ear, 3),
        "reason": "Blink detected" if blink_detected else "No blink detected in sequence"
    }


if __name__ == "__main__":
    print("Look at the camera and blink naturally within the next few seconds...")
    result = check_blink_sequence()
    print(result)
