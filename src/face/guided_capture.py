

import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_BLINK_THRESHOLD = 0.21
DARKNESS_THRESHOLD = 60  
HOLD_FRAMES_REQUIRED = 10  
BLINK_TIMEOUT_FRAMES = 300 
EAR_ROLLING_WINDOW = 5  


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices])
    vertical1 = np.linalg.norm(pts[1] - pts[5])
    vertical2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)


def face_inside_oval(landmarks, w, h, oval_center, oval_axes):
    """Check what fraction of key face landmarks fall inside the guide oval."""
    check_points = [10, 152, 234, 454, 127, 356, 132, 361]
    cx, cy = oval_center
    ax, ay = oval_axes

    inside_count = 0
    for idx in check_points:
        px = landmarks[idx].x * w
        py = landmarks[idx].y * h
        val = ((px - cx) ** 2) / (ax ** 2) + ((py - cy) ** 2) / (ay ** 2)
        if val <= 1.15: 
            inside_count += 1

    return inside_count / len(check_points)


def check_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def _open_camera(camera_index):
    
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        cap.release()
        return None, {
            "error": "camera_unavailable",
            "reason": (
                f"Could not access camera index {camera_index}. "
                f"{'No webcam detected on this device.' if camera_index == 0 else 'No second/external camera detected - try switching back to Cam 1.'}"
            )
        }

    
    ret, _ = cap.read()
    if not ret:
        cap.release()
        return None, {
            "error": "camera_read_failed",
            "reason": (
                f"Camera index {camera_index} opened but returned no frame. "
                "It may be in use by another application, or disconnected."
            )
        }

    return cap, None


def guided_capture(window_name="KYC Face Verification - Position your face in the oval", camera_index=0):

    cap, open_error = _open_camera(camera_index)
    if open_error is not None:
        return None, {"liveness_passed": False, "reason": open_error["reason"]}

    hold_counter = 0
    waiting_for_blink = False
    ear_values_during_wait = []
    blink_wait_counter = 0
    captured_frame = None
    liveness_result = {"liveness_passed": False, "reason": "Cancelled"}

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as face_mesh:

            consecutive_read_failures = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    consecutive_read_failures += 1
                    
                    if consecutive_read_failures >= 30:
                        liveness_result = {
                            "liveness_passed": False,
                            "reason": "Lost connection to the camera during capture."
                        }
                        break
                    continue
                consecutive_read_failures = 0

                frame = cv2.flip(frame, 1)  
                h, w = frame.shape[:2]
                display = frame.copy()

                oval_center = (w // 2, h // 2)
                oval_axes = (int(w * 0.18), int(h * 0.32))

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                brightness = check_brightness(frame)
                too_dark = brightness < DARKNESS_THRESHOLD

                positioned_correctly = False
                status_text = "Position your face in the oval"
                oval_color = (0, 0, 255)  # red (BGR)

                if too_dark:
                    status_text = "Too dark - move to better lighting"
                    oval_color = (0, 0, 255)
                elif results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    fit_ratio = face_inside_oval(landmarks, w, h, oval_center, oval_axes)

                    if fit_ratio >= 0.85:
                        positioned_correctly = True
                        oval_color = (0, 200, 0)  # green
                        status_text = "Good - hold still"
                    else:
                        status_text = "Align your face inside the oval"
                        oval_color = (0, 0, 255)
                else:
                    status_text = "No face detected"
                    oval_color = (0, 0, 255)

                cv2.ellipse(display, oval_center, oval_axes, 0, 0, 360, oval_color, 4)

                if not waiting_for_blink:
                    if positioned_correctly:
                        hold_counter += 1
                    else:
                        hold_counter = 0

                    if hold_counter >= HOLD_FRAMES_REQUIRED:
                        waiting_for_blink = True
                        ear_values_during_wait = []
                        blink_wait_counter = 0

                else:
                    status_text = "Now blink naturally..."
                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0].landmark
                        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
                        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
                        avg_ear = (left_ear + right_ear) / 2.0
                        ear_values_during_wait.append(avg_ear)

                        
                        if len(ear_values_during_wait) >= EAR_ROLLING_WINDOW:
                            recent = ear_values_during_wait[-EAR_ROLLING_WINDOW:]
                            min_ear = min(recent)
                            if min_ear < EAR_BLINK_THRESHOLD:
                                captured_frame = frame.copy()
                                liveness_result = {
                                    "liveness_passed": True,
                                    "min_ear": round(min_ear, 3),
                                    "reason": "Blink detected"
                                }
                                break

                    blink_wait_counter += 1
                    if blink_wait_counter >= BLINK_TIMEOUT_FRAMES:
                        liveness_result = {
                            "liveness_passed": False,
                            "reason": "No blink detected within time limit"
                        }
                        break

                
                cam_label = "Cam 1 (built-in)" if camera_index == 0 else "Cam 2 (external)"
                cv2.putText(display, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2)
                cv2.putText(display, f"{cam_label} - Press 'q' to cancel", (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    liveness_result = {"liveness_passed": False, "reason": "Cancelled by user"}
                    break
    except Exception as e:
        
        liveness_result = {"liveness_passed": False, "reason": f"Camera capture failed: {e}"}
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captured_frame, liveness_result


if __name__ == "__main__":
    import sys
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    frame, result = guided_capture(camera_index=cam_idx)
    if frame is not None:
        cv2.imwrite("guided_capture_result.jpg", frame)
        print("Captured successfully:", result)
    else:
        print("Capture failed/cancelled:", result)