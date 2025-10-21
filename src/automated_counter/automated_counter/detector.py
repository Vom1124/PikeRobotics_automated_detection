import cv2
import numpy as np
from scipy.optimize import differential_evolution
from automated_counter import image_morph
from automated_counter import p_tile_algo

# Optional YOLO import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] Ultralytics YOLO not installed. YOLO detector will be skipped.")


# -------------------------------------------------------------
# --- 1. SIMPLE THRESHOLD DETECTOR
# -------------------------------------------------------------
def threshold_detector(img, threshold=50, min_area=25):
    """Detect blobs or bolts using binary thresholding."""
    if img is None:
        return 0, []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = [
        cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area
    ]
    return len(bboxes), bboxes


# -------------------------------------------------------------
# --- 2. ADAPTIVE THRESHOLD DETECTOR
# -------------------------------------------------------------
def adaptive_detector(img, min_area=25):
    """Adaptive thresholding works better under variable lighting."""
    if img is None:
        return 0, []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = [
        cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area
    ]
    return len(bboxes), bboxes


# -------------------------------------------------------------
# --- 3. CIRCLE DETECTOR (HOUGH TRANSFORM)
# -------------------------------------------------------------

def circle_detector(img, dp=1.5, min_dist=20, param1=80, param2=15, min_radius=2, max_radius=10):
    """
    Detect small circular bolts using HoughCircles with pre-processing.
    
    Parameters:
    - img : np.ndarray
        Input image (BGR or grayscale)
    - dp : float
        Inverse ratio of accumulator resolution to image resolution.
    - min_dist : int
        Minimum distance between circle centers.
    - param1 : int
        Upper threshold for internal Canny edge detector.
    - param2 : int
        Threshold for center detection — lower = more sensitive.
    - min_radius, max_radius : int
        Range of possible circle radii (tune for bolt size).

    Returns:
    - count : int
        Number of detected bolts
    - bboxes : list of tuples
        Bounding boxes around detected circles
    """
    if img is None:
        return 0, []

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Convert to 8-bit if not already
    if gray.dtype != np.uint8:
        gray_8u = cv2.convertScaleAbs(gray, alpha=(255.0/np.max(gray)))
    else:
        gray_8u = gray.copy()

    # STEP 1: Enhance local contrast (metal bolts on reflective surface)
    gray_eq = cv2.equalizeHist(gray_8u)

    # STEP 2: Reduce noise and highlight circular features
    blurred = cv2.medianBlur(gray_eq, 5)

    # STEP 3: Optional edge emphasis (helps if bolts are faint)
    edges = cv2.Laplacian(blurred, cv2.CV_8U, ksize=3)
    gray_enhanced = cv2.addWeighted(blurred, 0.8, edges, 0.2, 0)

    # STEP 4: Run Hough Circle Transform
    circles = cv2.HoughCircles(
        gray_enhanced,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    bboxes = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            # Create a mask for the circle
            mask = np.zeros(gray_enhanced.shape, np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)

            # Compute area and perimeter for circularity
            area = cv2.countNonZero(cv2.bitwise_and(mask, gray_enhanced))
            perimeter = 2 * np.pi * r
            if perimeter == 0:
                continue  # avoid division by zero

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Keep only near-circular candidates
            if circularity > 0.4:  # adjust threshold as needed
                bboxes.append((x - r, y - r, 2 * r, 2 * r))

    count = len(bboxes)
    return count, bboxes



# -------------------------------------------------------------
# --- 4. TEMPLATE MATCHING DETECTOR
# -------------------------------------------------------------
def template_detector(img, template_path="bolt_template.png", threshold=0.6):
    """Match a predefined bolt template."""
    if img is None:
        return 0, []

    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"[WARN] Template image not found: {template_path}")
        return 0, []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    bboxes = []
    h, w = template.shape
    for pt in zip(*loc[::-1]):
        bboxes.append((pt[0], pt[1], w, h))

    return len(bboxes), bboxes


# -------------------------------------------------------------
# --- 5. YOLO DETECTOR (RGB ONLY)
# -------------------------------------------------------------
def yolo_detector(img, model_path="yolov8n.pt", conf=0.05):
    """Use YOLOv8 for bolt detection. Always expects RGB images."""
    if not YOLO_AVAILABLE:
        print("[WARN] YOLO not available. Please install ultralytics.")
        return 0, []

    if img is None:
        return 0, []

    # Ensure the image is 3-channel RGB
    if len(img.shape) == 2:  # grayscale or depth
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert from OpenCV BGR → RGB

    # Load YOLO model
    model = YOLO(model_path)
    results = model(img_rgb, conf=conf, verbose=False)

    bboxes = []
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            bboxes.append((x1, y1, x2 - x1, y2 - y1))

    return len(bboxes), bboxes



# -------------------------------------------------------------
# --- 6. DE Saliency
# -------------------------------------------------------------
def DE_Detector(img, fitness_func, bounds=[(250,300),(200,250),(5,15),(5,15)],
                maxiter=10, popsize=25, fitness_threshold=1e4):
    """
    Differential Evolution detector for multiple salient regions in an image (RGB or grayscale).
    
    Parameters:
    - img: np.ndarray
        Input RGB or grayscale image
    - fitness_func: callable
        Function: fitness(individual, gray_map)
    - bounds: list of tuples
        DE bounds for [x, y, w, h]
    - maxiter: int
        Maximum DE iterations per region
    - popsize: int
        Population size
    - max_regions: int
        Maximum number of bounding boxes to detect
    - fitness_threshold: float
        Stop adding regions if fitness is above this
    
    Returns:
    - count: int
        Number of detected regions
    - bboxes: list of tuples
        Bounding boxes [(x,y,w,h), ...]
    """
    if img is None:
        return 0, []

    # Convert to grayscale
    if len(img.shape) == 3:
        gray_map = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_map = img.copy()

    # Enhance image with segmentation + morphology
    thresh_value = p_tile_algo.p_tile_thresh(gray_map, AOI=0.35)
    ret, gray_map_seg = cv2.threshold(gray_map, thresh_value, np.max(gray_map), cv2.THRESH_TOZERO_INV)
    gray_map = image_morph.image_morph_algo(gray_map_seg)
    cv2.imshow("Gray Map (Morph + P-tile)", gray_map)

    # Run DE once
    result = differential_evolution(fitness_func, bounds, args=(gray_map,), maxiter=maxiter, popsize=popsize)
    best_fitness = result.fun
    print(result.fun)

    # Only return bounding box if region is salient
    if best_fitness < fitness_threshold:
        x, y, w, h = map(int, result.x)
        x = np.clip(x, 0, gray_map.shape[1]-1)
        y = np.clip(y, 0, gray_map.shape[0]-1)
        w = np.clip(w, 5, gray_map.shape[1]-x)
        h = np.clip(h, 5, gray_map.shape[0]-y)
        return 1, [(x, y, w, h)]
    else:
        # No salient region found
        return 0, []

def fitness_func(individual, img):
    """
    Fitness function for Differential Evolution.
    individual = [x, y, w, h]
    """
    x, y, w, h = map(int, individual)
    x = np.clip(x, 0, img.shape[1]-1)
    y = np.clip(y, 0, img.shape[0]-1)
    w = np.clip(w, 5, img.shape[1]-x)
    h = np.clip(h, 5, img.shape[0]-y)

    region = img[y:y+h, x:x+w]
    if region.size == 0 or np.isnan(region).any():
        return 1e6

    # Evaluate saliency of region
    mean_val = np.mean(region)
    std_val = np.std(region)

    # Simple heuristics: higher contrast regions are more likely to be bolts
    distance_penalty = 1e6 if mean_val > 5 else 0.0  # very bright regions penalized
    proximity = 1e3 / (mean_val + 1e-6)
    _lambda_ = 1
    penalty = _lambda_ * max(0, abs(w*h - 200))

    total_score = proximity + std_val - penalty + distance_penalty
    return total_score

# -------------------------------------------------------------
# --- MAIN DISPATCHER
# -------------------------------------------------------------
def detect(img, method="threshold", **kwargs):
    """
    Generic detector dispatcher.
    Supported methods:
        - "threshold"
        - "adaptive"
        - "circle"
        - "template"
        - "yolo"

    Returns:
        count (int), bboxes (list)
    """
    if method == "threshold":
        return threshold_detector(img, **kwargs)
    elif method == "adaptive":
        return adaptive_detector(img, **kwargs)
    elif method == "circle":
        return circle_detector(img, **kwargs)
    elif method == "template":
        return template_detector(img, **kwargs)
    elif method == "yolo":
        return yolo_detector(img, **kwargs)
    elif method == "DE":
        return DE_Detector(img, fitness_func, **kwargs)
    else:
        raise ValueError(f"Unknown detection method: {method}")
