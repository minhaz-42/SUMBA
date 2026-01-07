"""Simple landmark-based lip decoder prototype.

This is a lightweight, deterministic heuristic decoder for quick demos.
It uses the vertical spread (max_y - min_y) of provided face landmarks (per-frame)
as a proxy for mouth openness and maps averaged openness across frames to a
small set of syllable-like tokens as a stand-in for a trained lip decoder.

Replace this with a trained model in future work.
"""
from typing import List, Dict

# Thresholds tuned heuristically for demo purposes
LOW_THRESH = 0.01
MID_THRESH = 0.02
HIGH_THRESH = 0.04


def _frame_openness(face_landmarks: List[Dict]) -> float:
    """Compute a simple vertical span (y-range) of the provided landmarks.

    face_landmarks: list of {'x': float, 'y': float, 'z': float} dicts
    Returns: float vertical span (max_y - min_y)
    """
    if not face_landmarks:
        return 0.0
    ys = [lm.get('y', 0.0) for lm in face_landmarks]
    return max(ys) - min(ys)


# Common mouth landmark indices from MediaPipe FaceMesh (outer lip)
MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]


def _extract_mouth_landmarks(frame_landmarks: List[Dict]) -> List[Dict]:
    """Given a face landmarks list (by index), return only mouth-related landmarks.

    Accepts either list of landmark dicts or a dict with key 'face'.
    """
    if not frame_landmarks:
        return []

    # If it's a dict/frame with 'face' key
    if isinstance(frame_landmarks, dict):
        face = frame_landmarks.get('face', [])
    else:
        face = frame_landmarks

    # If face is still not an indexed list, return empty
    if not face or not isinstance(face, list):
        return []

    mouth = []
    for idx in MOUTH_INDICES:
        if idx < len(face):
            mouth.append(face[idx])
    return mouth


def _frame_mouth_features(face_landmarks: List[Dict]) -> Dict:
    """Compute simple mouth features for a single frame.

    Returns dict with 'openness' (y-range), 'width' (x-range) and 'ratio'.
    """
    mouth = _extract_mouth_landmarks(face_landmarks)
    if not mouth:
        return {'openness': 0.0, 'width': 0.0, 'ratio': 0.0}

    xs = [lm.get('x', 0.0) for lm in mouth]
    ys = [lm.get('y', 0.0) for lm in mouth]
    openness = max(ys) - min(ys)
    width = max(xs) - min(xs) if xs else 0.0
    ratio = openness / width if width > 0.0 else 0.0
    return {'openness': openness, 'width': width, 'ratio': ratio}


# A small demo phrase list for mapping heuristic outputs
DEMO_PHRASES = [
    "Hello, how are you?",
    "Thank you very much",
    "Nice to meet you",
    "My name is...",
    "Good morning",
    "I love you",
]


def predict_from_face_frames(face_frames: List[Dict]) -> str:
    """Predict a short phrase from a sequence of face frames.

    Accepts either:
      - list of frames [{ 'face': [landmarks] }, ...]
      - list of landmark lists [[{x,y,z}, ...], ...]

    Returns a short string prediction (phrase). Uses simple heuristics.
    """
    if not face_frames:
        return ''

    # Normalize input: convert list of landmark-lists or frames to a list of landmark-lists
    frames = []
    for f in face_frames:
        if isinstance(f, dict) and 'face' in f:
            frames.append(f.get('face', []))
        elif isinstance(f, list):
            frames.append(f)
        else:
            # Unknown format: skip
            continue

    if not frames:
        return ''

    features = [_frame_mouth_features(fr) for fr in frames]

    # Aggregate statistics
    avg_openness = sum([fv['openness'] for fv in features]) / len(features)
    avg_ratio = sum([fv['ratio'] for fv in features]) / len(features)
    var_openness = max([fv['openness'] for fv in features]) - min([fv['openness'] for fv in features])

    # Heuristic mapping to demo phrases
    # Preference ordering picked to roughly reflect mouth activity
    if avg_openness < 0.005:
        idx = 0  # quiet short phrase
    elif avg_openness < 0.01:
        idx = 1
    elif avg_openness < 0.02:
        idx = 2
    elif avg_openness < 0.035:
        idx = 3
    else:
        idx = 4

    # Use variance to pick between similar options
    if var_openness > 0.02:
        idx = min(idx + 1, len(DEMO_PHRASES) - 1)

    return DEMO_PHRASES[idx]
