import cv2
import numpy as np
from scipy.optimize import differential_evolution
from automated_counter import image_morph
from automated_counter import p_tile_algo
import os
import time

# Optional YOLO import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] Ultralytics YOLO not installed. YOLO detector will be skipped.")

current_directory = os.getcwd()

# ------------------------------
# Crop ROI
#-----------------------
def crop_roi(img, roi, shift_up=0, shift_left=0):
    """
    Crop image to ROI; returns full coordinates offset as well.
    
    Parameters:
        img: input image
        roi: tuple (x, y, w, h)
        shift_up: number of pixels to move ROI upward (positive moves up)
        shift_left: number of pixels to move ROI left (positive moves left)
    """
    if roi is None:
        return img, (0, 0)
    
    x, y, w, h = roi
    
    # Shift ROI upward and left, ensuring it stays inside image
    x_new = max(x - shift_left, 0)
    y_new = max(y - shift_up, 0)
    
    w_new = min(w , img.shape[1] - x_new)   # adjust width if near right edge
    h_new = min(h , img.shape[0] - y_new)     # adjust height if near bottom edge
    
    return img[y_new:y_new+h_new, x_new:x_new+w_new], (x_new, y_new)


# -----------------------------
# Global tracker parameters
# -----------------------------
bolt_count = 0
last_detect_time = 0
cooldown_period = 1.5    # seconds to ignore duplicate bolts
min_frames = 2         # minimum consecutive frames a bolt must appear
consec_detections = 0     # consecutive frames seen for current bolt

# -----------------------------
# Object Tracking
# -----------------------------
def update_tracked_bolts(detections):
    """
    Count new bolts if they appear after cooldown and minimum frames.
    
    Args:
        detections: list of detected bboxes in current frame
    """
    global bolt_count, last_detect_time, consec_detections

    if not detections:
        consec_detections = 0  # reset if nothing detected
        return bolt_count

    # Increment consecutive detection count
    consec_detections += 1
    now = time.time()

    # Only count a new bolt if cooldown expired AND minimum frames seen
    if (now - last_detect_time > cooldown_period) and (consec_detections >= min_frames):
        bolt_count += 1
        last_detect_time = now
        consec_detections = 0  # reset for next bolt

    return bolt_count

# ------------------------------
# Adaptive threshold detector
# ------------------------------
def adaptive_detector(img, roi=None, min_area=25):
    img_roi, offset = crop_roi(img, roi)
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY) if len(img_roi.shape) == 3 else img_roi.copy()
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = [(x+offset[0], y+offset[1], w, h)
              for (x, y, w, h) in [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]]
    return len(bboxes), bboxes

# ------------------------------
# Circle detector
# ------------------------------
def circle_detector(img, roi=None, dp=1.5, min_dist=20, param1=80, param2=15, min_radius=2, max_radius=10):
    img_roi, offset = crop_roi(img, roi)
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY) if len(img_roi.shape) == 3 else img_roi.copy()
    gray_8u = cv2.convertScaleAbs(gray, alpha=(255.0/np.max(gray))) if gray.dtype != np.uint8 else gray.copy()
    gray_eq = cv2.equalizeHist(gray_8u)
    blurred = cv2.medianBlur(gray_eq, 5)
    edges = cv2.Laplacian(blurred, cv2.CV_8U, ksize=3)
    gray_enhanced = cv2.addWeighted(blurred, 0.8, edges, 0.2, 0)

    circles = cv2.HoughCircles(gray_enhanced, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                               param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    bboxes = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            area = cv2.countNonZero(cv2.bitwise_and(
                cv2.circle(np.zeros_like(gray_enhanced), (x, y), r, 255, -1), gray_enhanced))
            perimeter = 2 * np.pi * r
            if perimeter == 0: continue
            circularity = 4 * np.pi * area / (perimeter*perimeter)
            if circularity > 0.4:
                bboxes.append((x+offset[0]-r, y+offset[1]-r, 2*r, 2*r))
    return len(bboxes), bboxes

# ------------------------------
# Template detector
# ------------------------------
def template_detector(img, method_type="cv", roi=None,
                      template_path=f"{current_directory}/SUNCOR/SUNCOR_24/template_ir_enhanced.png",
                      threshold=0.5):
    img_roi, offset = crop_roi(img, roi, shift_up=25, shift_left=125)
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(11,11))
    gray = clahe.apply(gray)
    # cv2.imshow("CLAHE gray", gray)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"[WARN] Template not found: {template_path}")
        return 0, []

    if method_type == "cv":
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        bboxes = [(pt[0]+offset[0], pt[1]+offset[1], template.shape[1], template.shape[0])
                  for pt in zip(*loc[::-1])]
        return len(bboxes), bboxes
    elif method_type == "orb":
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(gray, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        if des1 is None or des2 is None: return 0, []
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 4: return 0, []
        points = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(points)
        return 1, [(x+offset[0], y+offset[1], w, h)]
    return 0, []

# ------------------------------
# YOLO detector
# ------------------------------
def yolo_detector(img, roi=None, model_path="yolov8n.pt", conf=0.05):
    if not YOLO_AVAILABLE:
        print("[WARN] YOLO not available.")
        return 0, []
    img_roi, offset = crop_roi(img, roi)
    if len(img_roi.shape) == 2:
        img_rgb = cv2.cvtColor(img_roi, cv2.COLOR_GRAY2BGR)
    else:
        img_rgb = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
    model = YOLO(model_path)
    results = model(img_rgb, conf=conf, verbose=False)

    bboxes = []
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            bboxes.append((x1+offset[0], y1+offset[1], x2-x1, y2-y1))
    return len(bboxes), bboxes

# ------------------------------
# DE Saliency detector
# ------------------------------
def DE_Detector(img, fitness_func, roi=None, bounds=[(250,300),(200,250),(5,15),(5,15)],
                maxiter=10, popsize=25, fitness_threshold=1e4):
    img_roi, offset = crop_roi(img, roi)
    gray_map = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY) if len(img_roi.shape)==3 else img_roi.copy()
    # thresh_value = p_tile_algo.p_tile_thresh(gray_map, AOI=0.35)
    # ret, gray_map_seg = cv2.threshold(gray_map, thresh_value, np.max(gray_map), cv2.THRESH_TOZERO_INV)
    gray_map = image_morph.image_morph_algo(gray_map_seg)

    result = differential_evolution(fitness_func, bounds, args=(gray_map,), maxiter=maxiter, popsize=popsize)
    best_fitness = result.fun

    if best_fitness < fitness_threshold:
        x, y, w, h = map(int, result.x)
        x = np.clip(x+offset[0], 0, gray_map.shape[1]-1)
        y = np.clip(y+offset[1], 0, gray_map.shape[0]-1)
        w = np.clip(w, 5, gray_map.shape[1]-x)
        h = np.clip(h, 5, gray_map.shape[0]-y)
        return 1, [(x, y, w, h)]
    return 0, []

# ------------------------------
# Main dispatcher
# ------------------------------
def detect(img, method="threshold", method_type="cv", roi=None, fitness_func=None, **kwargs):
    if method == "threshold":
        count, bboxes = threshold_detector(img, roi, **kwargs)
    elif method == "adaptive":
        count, bboxes = adaptive_detector(img, roi, **kwargs)
    elif method == "circle":
        count, bboxes = circle_detector(img, roi, **kwargs)
    elif method == "template":
        count, bboxes = template_detector(img, method_type, roi, **kwargs)
    elif method == "yolo":
        count, bboxes = yolo_detector(img, roi, **kwargs)
    elif method == "DE":
        count, bboxes = DE_Detector(img, fitness_func, roi, **kwargs)
    else:
        raise ValueError(f"Unknown detection method: {method}")

    cumulative_count = update_tracked_bolts(bboxes)
    return cumulative_count, bboxes