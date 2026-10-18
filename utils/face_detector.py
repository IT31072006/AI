"""
Face Detection and Auto-Cropping Utility using OpenCV.
Detects human faces in uploaded or camera images and crops them cleanly
before passing to the deep learning recognition model.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
from PIL import Image


# Load OpenCV Haar Cascade for Frontal Face Detection
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)


def detect_and_crop_face(
    image: Image.Image,
    margin_ratio: float = 0.25,
    min_size: int = 40
) -> Tuple[Image.Image, bool, Optional[Tuple[int, int, int, int]]]:
    """
    Detects the primary face in a PIL image and crops it with a surrounding margin.
    
    Args:
        image: Input PIL Image (RGB)
        margin_ratio: Extra padding around the bounding box (e.g., 0.25 = 25% extra on all sides)
        min_size: Minimum face bounding box width/height in pixels
        
    Returns:
        cropped_image: PIL Image of the cropped face (or original image if no face detected)
        detected: Boolean indicating whether a face was found
        bbox: (x, y, w, h) bounding box coordinates in original image or None
    """
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Run face detection with multiple scale steps
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(min_size, min_size)
    )
    
    if len(faces) == 0:
        # Fallback to slightly more sensitive detection
        faces = _face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(min_size, min_size)
        )
        
    if len(faces) == 0:
        return image, False, None
        
    # Pick the largest face (assumed to be the main subject)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    
    img_h, img_w = img_np.shape[:2]
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)
    
    cropped_np = img_np[y1:y2, x1:x2]
    cropped_pil = Image.fromarray(cropped_np)
    
    return cropped_pil, True, (x, y, w, h)


def draw_face_box(image: Image.Image, bbox: Tuple[int, int, int, int], label: str = "Face Detected") -> Image.Image:
    """Draw a modern rounded bounding box around the detected face."""
    img_np = np.array(image.convert("RGB")).copy()
    x, y, w, h = bbox
    # Draw green bounding box (Emerald #10b981)
    cv2.rectangle(img_np, (x, y), (x + w, y + h), (16, 185, 129), 3)
    cv2.putText(img_np, label, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (16, 185, 129), 2)
    return Image.fromarray(img_np)
