import numpy as np
import insightface
from insightface.app import FaceAnalysis

_app = None


def get_app():
    global _app
    if _app is None:
        _app = FaceAnalysis(name="buffalo_l")
        _app.prepare(ctx_id=0, det_size=(320, 320))  # ctx_id=-1 to force CPU
    return _app


def get_embedding(face_image):
    """
    face_image: BGR numpy array. Should be a loosely-cropped region containing
    the face (not a tight pre-cropped face) - InsightFace runs its own
    detection + 5-point landmark alignment internally, which needs some
    margin around the face to work accurately.
    """
    app = get_app()
    faces = app.get(face_image)
    if not faces:
        return None
    faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)
    return faces[0].embedding


def get_averaged_embedding(face_images):
    """
    Given a list of BGR frames (e.g. several webcam frames captured a few
    hundred ms apart), compute an embedding for each and return the
    normalized average. Averaging over multiple frames smooths out noise
    from a single frame's angle/lighting/motion-blur, giving a more stable
    embedding than relying on one shot. Frames where no face is detected
    are skipped.
    """
    embeddings = []
    for frame in face_images:
        emb = get_embedding(frame)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        return None

    avg = np.mean(embeddings, axis=0)
    avg = avg / np.linalg.norm(avg)  # re-normalize after averaging
    return avg


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def classify_similarity(score):
    if score > 0.6:
        return "HIGH", "Strong match"
    elif score >= 0.4:
        return "MEDIUM", "Possible aging / appearance change - additional verification required"
    else:
        return "LOW", "Likely mismatch"


def verify_faces(id_face_image, webcam_face_image):
    """
    Original single-frame interface - kept for backward compatibility
    (e.g. command-line testing with two static images).
    """
    emb1 = get_embedding(id_face_image)
    emb2 = get_embedding(webcam_face_image)

    if emb1 is None or emb2 is None:
        return {"error": "Could not generate embedding for one or both faces"}

    score = cosine_similarity(emb1, emb2)
    tier, message = classify_similarity(score)

    return {
        "similarity_score": round(score, 3),
        "tier": tier,
        "message": message
    }


def verify_faces_multiframe(id_face_image, webcam_frames):
    """
    id_face_image: single BGR image (the ID document / flattened crop).
    webcam_frames: list of BGR frames captured from the webcam (a short burst).
    Averages the webcam embeddings across frames before comparing against
    the ID embedding - reduces noise from any single frame's angle/blur/lighting.
    """
    emb_id = get_embedding(id_face_image)
    if emb_id is None:
        return {"error": "Could not generate embedding for ID face"}

    emb_webcam = get_averaged_embedding(webcam_frames)
    if emb_webcam is None:
        return {"error": "Could not generate embedding from any webcam frame"}

    score = cosine_similarity(emb_id, emb_webcam)
    tier, message = classify_similarity(score)

    return {
        "similarity_score": round(score, 3),
        "tier": tier,
        "message": message,
        "frames_used": len(webcam_frames)
    }


if __name__ == "__main__":
    import cv2
    id_face = cv2.imread("id_face.jpg")
    webcam_face = cv2.imread("webcam_face.jpg")
    result = verify_faces(id_face, webcam_face)
    print(result)