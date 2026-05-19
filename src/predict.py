import os
import re
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from model import GTSRBModel

# GTSRB has 43 classes.
CLASSES: Dict[int, str] = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

NORMALIZE_MEAN = np.array([0.3403, 0.3121, 0.3214], dtype=np.float32)
NORMALIZE_STD = np.array([0.1340, 0.1295, 0.1386], dtype=np.float32)

# Model cache prevents loading the same .pth file for every detected sign.
_MODEL_CACHE = {}
BBox = Tuple[int, int, int, int]


def _resolve_model_path(model_path: str) -> str:
    """If the specified model_path does not exist, try to find the latest gtsrb_model_{number}.pth."""
    if os.path.exists(model_path):
        return model_path

    directory = os.path.dirname(model_path) or 'models'
    if not os.path.exists(directory):
        return model_path

    pattern = re.compile(r"^gtsrb_model_(\d+)\.pth$")
    candidates = []
    for f in os.listdir(directory):
        match = pattern.match(f)
        if match:
            try:
                candidates.append((int(match.group(1)), f))
            except ValueError:
                continue

    if candidates:
        # Sort by number descending and take the largest
        latest_file = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
        resolved = os.path.join(directory, latest_file)
        print(f"Model '{model_path}' not found. Using latest model: '{resolved}'")
        return resolved

    return model_path


def _load_model(model_path: str = 'models/gtsrb_model_1.pth'):
    """Load the trained model once and reuse it."""
    model_path = _resolve_model_path(model_path)
    abs_path = os.path.abspath(model_path)

    if abs_path in _MODEL_CACHE:
        return _MODEL_CACHE[abs_path]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please train the model first or check the path."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GTSRBModel(num_classes=43).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _MODEL_CACHE[abs_path] = (model, device)
    return model, device


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a normalized CHW tensor without requiring torchvision."""
    resized = image.convert('RGB').resize((48, 48))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - NORMALIZE_MEAN) / NORMALIZE_STD
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def _classify_crop(image: Image.Image, model, device) -> Dict[str, Union[int, str, float]]:
    """Classify one cropped traffic sign image."""
    tensor = _image_to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        class_id = int(predicted.item())

    return {
        'class_id': class_id,
        'class_name': CLASSES.get(class_id, 'Unknown'),
        'confidence': float(confidence.item())
    }


def _iou(box_a: BBox, box_b: BBox) -> float:
    """Intersection over Union for two bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - intersection

    return intersection / union if union else 0.0


def _non_max_suppression(boxes: List[BBox], iou_threshold: float = 0.35) -> List[BBox]:
    """Remove duplicate boxes around the same sign."""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    selected = []

    for box in boxes:
        if all(_iou(box, selected_box) < iou_threshold for selected_box in selected):
            selected.append(box)

    return selected


def detect_sign_candidates(image: Image.Image, max_candidates: int = 20) -> List[BBox]:
    """
    Detect several possible traffic signs in one image.

    This is a lightweight color/contour detector, not a separate YOLO model.
    It works best for typical red, blue and yellow road signs. Each detected
    candidate is later classified by the trained GTSRB CNN.
    """
    rgb = np.array(image.convert('RGB'))
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # Masks for the most common traffic-sign colors: red, blue and yellow.
    red_low_1 = cv2.inRange(hsv, np.array([0, 50, 45]), np.array([12, 255, 255]))
    red_low_2 = cv2.inRange(hsv, np.array([165, 50, 45]), np.array([180, 255, 255]))
    blue = cv2.inRange(hsv, np.array([85, 45, 40]), np.array([135, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([15, 45, 60]), np.array([40, 255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(red_low_1, red_low_2), cv2.bitwise_or(blue, yellow))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = width * height
    min_area = max(80, image_area * 0.00035)
    max_area = image_area * 0.70
    boxes: List[BBox] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        extent = area / float(max(1, w * h))

        # Traffic signs are usually compact. This filters long strips, text, sky, etc.
        if not 0.40 <= aspect_ratio <= 2.50:
            continue
        if extent < 0.12:
            continue

        padding = int(max(w, h) * 0.20) + 4
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)

        # Ignore boxes that are still too tiny after padding.
        if (x2 - x1) < 12 or (y2 - y1) < 12:
            continue

        boxes.append((x1, y1, x2, y2))

    boxes = _non_max_suppression(boxes)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes[:max_candidates]


def predict(image_path: str, model_path: str = 'models/gtsrb_model_1.pth') -> Union[str, None]:
    """
    Backward-compatible single-sign prediction.

    It classifies the whole image as one sign, just like the original version.
    """
    try:
        model, device = _load_model(model_path)
    except FileNotFoundError as error:
        print(error)
        return None

    image = Image.open(image_path).convert('RGB')
    result = _classify_crop(image, model, device)
    return str(result['class_name'])


def predict_many(
    image_path: str,
    model_path: str = 'models/gtsrb_model_1.pth',
    max_results: int = 10,
    min_confidence: float = 0.0
) -> List[Dict[str, Union[int, str, float, BBox]]]:
    """
    Detect and classify more than one traffic sign in a single image.

    Returns a list of dictionaries:
    {
        'bbox': (x1, y1, x2, y2),
        'class_id': int,
        'class_name': str,
        'confidence': float
    }
    """
    try:
        model, device = _load_model(model_path)
    except FileNotFoundError as error:
        print(error)
        return []

    image = Image.open(image_path).convert('RGB')
    boxes = detect_sign_candidates(image, max_candidates=max_results * 3)

    # Fallback: if no separate colored candidates were found, classify the full image.
    if not boxes:
        boxes = [(0, 0, image.width, image.height)]

    predictions = []
    for box in boxes:
        crop = image.crop(box)
        prediction = _classify_crop(crop, model, device)
        prediction['bbox'] = box

        if prediction['confidence'] >= min_confidence:
            predictions.append(prediction)

    # Keep the strongest results, then display them in natural reading order.
    predictions = sorted(predictions, key=lambda p: p['confidence'], reverse=True)[:max_results]
    predictions = sorted(predictions, key=lambda p: (p['bbox'][1], p['bbox'][0]))
    return predictions


def annotate_image(
    image_path: str,
    predictions: List[Dict[str, Union[int, str, float, BBox]]],
    output_path: str = None
) -> Image.Image:
    """Draw bounding boxes and labels for predictions."""
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype('arial.ttf', 14)
    except OSError:
        font = ImageFont.load_default()

    for index, prediction in enumerate(predictions, start=1):
        x1, y1, x2, y2 = prediction['bbox']
        label = f"{index}. {prediction['class_name']} ({prediction['confidence']:.0%})"

        draw.rectangle((x1, y1, x2, y2), outline='red', width=3)

        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_y = max(0, y1 - text_height - 6)

        draw.rectangle((x1, label_y, x1 + text_width + 6, label_y + text_height + 6), fill='red')
        draw.text((x1 + 3, label_y + 3), label, fill='white', font=font)

    if output_path:
        image.save(output_path)

    return image


