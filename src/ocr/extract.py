#extract.py


import re
import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['en']) 


def preprocess_for_ocr(image_path):

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]
    scale = 2.0 if max(h, w) < 1200 else 1.0
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    image = cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)

    return image


def run_ocr(image_path):
    """Run OCR (with preprocessing) and return list of (bbox, text, confidence)."""
    processed = preprocess_for_ocr(image_path)
    results = reader.readtext(processed)
    return results


def extract_dob(text_lines):
    pattern_strict = r'\b(\d{2}[/-]\d{2}[/-]\d{4})\b'
    pattern_loose = r'\b(\d{2})\D?(\d{2})\D?(\d{4})\b'

    for line in text_lines:
        match = re.search(pattern_strict, line)
        if match:
            return match.group(1)

    for line in text_lines:
        if "DOB" not in line.upper() and not re.search(r'\d{6,8}', line):
            continue
        match = re.search(pattern_loose, line)
        if match:
            dd, mm, yyyy = match.groups()
            if 1 <= int(mm) <= 12 and 1 <= int(dd) <= 31:
                return f"{dd}/{mm}/{yyyy}"
    return None


def extract_id_number(text_lines, doc_type="aadhaar"):
    if doc_type == "aadhaar":
        pattern = r'\b(\d{4}\s?\d{4}\s?\d{4})\b'
    elif doc_type == "pan":
        pattern = r'\b([A-Z]{5}\d{4}[A-Z])\b'
    else:
        pattern = r'\b([A-Z0-9]{6,12})\b'

    for line in text_lines:
        match = re.search(pattern, line)
        if match:
            digits = re.sub(r'\s', '', match.group(1))
            if doc_type == "aadhaar" and len(digits) == 12:
                return f"{digits[0:4]} {digits[4:8]} {digits[8:12]}"
            return match.group(1)

    if doc_type == "aadhaar":
        for i in range(len(text_lines) - 2):
            joined = " ".join(text_lines[i:i + 3])
            match = re.search(pattern, joined)
            if match:
                digits = re.sub(r'\s', '', match.group(1))
                if len(digits) == 12:
                    return f"{digits[0:4]} {digits[4:8]} {digits[8:12]}"

    return None


def extract_name(text_lines):
 
    candidates = []
    dob_index = None

    for i, line in enumerate(text_lines):
        clean = line.strip()
        if "DOB" in clean.upper() or re.search(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b', clean):
            if dob_index is None:
                dob_index = i

        if re.match(r'^[A-Za-z\s.]{4,40}$', clean) and len(clean.split()) <= 4:
            if clean.upper() not in ("GOVERNMENT OF INDIA", "INCOME TAX DEPARTMENT",
                                       "PERMANENT ACCOUNT NUMBER", "UNIQUE IDENTIFICATION",
                                       "MALE", "FEMALE"):
                candidates.append((i, clean))

    if not candidates:
        return None

    if dob_index is not None:
        # prefer the candidate closest to (and before) the DOB line
        before_dob = [c for c in candidates if c[0] < dob_index]
        if before_dob:
            return min(before_dob, key=lambda c: dob_index - c[0])[1]

    return candidates[0][1]


def extract_fields(image_path, doc_type="aadhaar"):
    ocr_results = run_ocr(image_path)
    text_lines = [text for (_, text, _) in ocr_results]
    confidences = [conf for (_, _, conf) in ocr_results]

    fields = {
        "name": extract_name(text_lines),
        "dob": extract_dob(text_lines),
        "id_number": extract_id_number(text_lines, doc_type),
        "raw_text": text_lines,
        "avg_confidence": round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    }
    return fields


if __name__ == "__main__":
    import sys
    import json
    image_path = sys.argv[1] if len(sys.argv) > 1 else "flattened_output.jpg"
    result = extract_fields(image_path)
    print(json.dumps(result, indent=2))