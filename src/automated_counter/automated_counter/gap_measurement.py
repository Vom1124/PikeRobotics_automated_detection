import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import time
from datetime import datetime
import sys

from automated_counter import detector
from automated_counter import image_morph_algo
from automated_counter import p_tile_algo

# ==========================
# CONFIGURATION
# ==========================
use_camera = False          # True = live ROS2, False = load from folder

# H, W = rgb_frame.shape[:2]
H, W = (172, 224) # Dimension of the Flexx Camera depth stream :(172.224,3)
roi_w, roi_h = 100, 50
roi_x = (W - roi_w) // 2
roi_y = (H - roi_h) // 2 + 25
roi = (roi_x, roi_y, roi_w, roi_h)

# ==========================
# DATASET CONFIGURATION
# ==========================
use_path_1 = False  # Choose dataset CPCHem or SUNCOR

# Dataset paths
working_dir= "/home/vom/ros2_ws/PikeRobotics_automated_detection"

# Dynamically set dataset path and name
if use_path_1:
    __dataname__ = "CPC"
    path = f"{working_dir}/{__dataname__}"
    # <-- set this to whichever CPC subfolder you’re using
    current_dataset = "CPC_9"
    # current_dataset = "CPC_10"
    # current_dataset = "CPC_19"
    # current_dataset = "CPC_43"
    # current_dataset = "CPC_61"   
else:
    __dataname__ = "SUNCOR"
    path = f"{working_dir}/{__dataname__}"
    current_dataset = "SUNCOR_24"  # <-- set this to whichever SUNCOR subfolder you’re using

#------ Current dataset 
dataset_path = current_dataset
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'MJPG', 'MP4V'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"{path}/{dataset_path}/gap_automated_meas_{timestamp}.avi"
output_depth_vis = cv2.VideoWriter(output_file, fourcc, 15, (224,172), isColor=True)
# Define video paths dynamically 
#-- CPC Files 
# rgb_video_path = f"{path}/{dataset_path}/{dataset_path}__wombot_gen3proto_seal_cameras_realsense_color_image_raw_compressed.mp4" 
# ir_video_path = f"{path}/{dataset_path}/CPC_9__wombot_gen3proto_seal_cameras_realsense_infra1_image_rect_raw_compressed.mp4" 
# gray_video_path = f"{path}/{dataset_path}/CPC_9__wombot_gen3proto_seal_cameras_flexx_gray_image_raw_compressed.mp4" 
# depth_video_path = f"{path}/{dataset_path}/8_FlexxDepth.mp4" 
# #--SUNCOR Files #
# rgb_video_path = f"{path}/{dataset_path}/4_RealsenseColor.mp4" # 
# ir_video_path = f"{path}/{dataset_path}/5_realsense_infra1_image_rect_raw_compressed.mp4" # 
# gray_video_path = f"{path}/{dataset_path}/5_RealsenseGray.mp4" # 
depth_video_path = f"{path}/{dataset_path}/8_FlexxDepth.mp4" 
# print("Using dataset:", path) # 
# print("gray video path:", gray_video_path)


# ==========================
# MODE 1: VIDEO FILES
# ==========================

#----Selecting the stream...
use_depth = True  # Enable if depth video exists


# Open videos
depth_cap = cv2.VideoCapture(depth_video_path) if use_depth else None

# Check if videos opened
if use_depth and not depth_cap.isOpened(): raise IOError(f"Depth video cannot be opened: {depth_video_path}")

print("Playing videos from folder...")

# -----------------------------
# Camera intrinsics (from user-provided camera_info)
# -----------------------------
# Flexx Depth camera intrinsics
fx = 206.13072204589844
fy = 206.13072204589844
cx = 110.7488021850586
cy = 86.2164077758789

# -----------------------------
# Helper functions for gap isolation & measurement
# -----------------------------
def kmeans_two_clusters_depth(depth_vals):
    """
    depth_vals: (N,1) float32 ndarray
    returns: (labels (N,), centers (2,))
    Uses OpenCV kmeans to avoid sklearn dependency.
    """
    # cv2.kmeans needs float32 samples in shape (N,1)
    samples = depth_vals.astype(np.float32)
    # criteria, attempts, flags
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    K = 2
    attempts = 10
    flags = cv2.KMEANS_PP_CENTERS
    compactness, labels, centers = cv2.kmeans(samples, K, None, criteria, attempts, flags)
    labels = labels.flatten()
    centers = centers.flatten()
    return labels, centers

def find_nearest_valid_in_column(depth_img, col_idx, row_idx, search_limit=50):
    """
    For a given (row_idx, col_idx) that is NaN/zero, search upward and downward in the same column
    for the nearest valid depth pixel. Returns (z_top, row_top, z_bottom, row_bottom).
    If not found within search_limit, returns (None, None, None, None).
    """
    H = depth_img.shape[0]
    z_top = None; row_top = None
    z_bottom = None; row_bottom = None
    # search upwards (decreasing row)
    for r in range(row_idx-1, max(row_idx - search_limit - 1, -1), -1):
        val = depth_img[r, col_idx]
        if not (np.isnan(val) or val == 0):
            z_top = float(val); row_top = r
            break
    # search downwards (increasing row)
    for r in range(row_idx+1, min(row_idx + search_limit + 1, H)):
        val = depth_img[r, col_idx]
        if not (np.isnan(val) or val == 0):
            z_bottom = float(val); row_bottom = r
            break
    return z_top, row_top, z_bottom, row_bottom

def isolate_and_measure_gap_from_roi(depth_roi, fx, fy, cx, cy,
                                     min_component_area=2, search_limit=50):
    """
    Robust gap detection using NaN-driven column clustering.
    Only considers NaN regions sandwiched between valid top and bottom surfaces.

    Args:
        depth_roi: (H, W) depth in mm. NaNs or 0 for missing points.
        fx, fy, cx, cy: camera intrinsics
        min_component_area: minimum number of pixels to consider a gap
        search_limit: vertical search limit for nearest valid top/bottom pixels

    Returns:
        dict with gap_mask (bool), width_mm, height_mm, bbox_roi, points_3D
    """
    depth = depth_roi.copy().astype(np.float32)
    H, W = depth.shape

    # 1) Create NaN mask (gap candidates)
    nan_mask = (np.isnan(depth)) | (depth == 0)

    # 2) For each column, find contiguous NaN regions
    gap_mask = np.zeros_like(depth, dtype=bool)
    for c in range(W):
        col = nan_mask[:, c]
        r = 0
        while r < H:
            if col[r]:
                # start of NaN segment
                start_r = r
                while r < H and col[r]:
                    r += 1
                end_r = r - 1

                # Check if this NaN segment is sandwiched by valid top/bottom pixels
                z_top = None
                z_bottom = None

                # Search upwards
                for ru in range(start_r-1, max(start_r - search_limit - 1, -1), -1):
                    val = depth[ru, c]
                    if not (np.isnan(val) or val == 0):
                        z_top = val
                        break
                # Search downwards
                for rd in range(end_r+1, min(end_r + search_limit + 1, H)):
                    val = depth[rd, c]
                    if not (np.isnan(val) or val == 0):
                        z_bottom = val
                        break

                if z_top is not None and z_bottom is not None:
                    # This segment is a valid gap
                    gap_mask[start_r:end_r+1, c] = True
            else:
                r += 1

    if np.count_nonzero(gap_mask) < min_component_area:
        return None

    # 3) Connected components to pick largest gap
    gap_components_img = (gap_mask.astype(np.uint8) * 255).astype(np.uint8)
    num_labels, labels_cc = cv2.connectedComponents(gap_components_img)
    if num_labels <= 1:
        return None

    # compute areas and pick the largest component (excluding background label 0)
    areas = []
    for lab in range(1, num_labels):
        areas.append(np.sum(labels_cc == lab))
    largest_idx = int(np.argmax(areas)) + 1
    if areas[largest_idx - 1] < min_component_area:
        return None

    gap_mask_final = (labels_cc == largest_idx)

    # Optional: smooth/fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    gap_mask_final = cv2.morphologyEx(gap_mask_final.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    # 4) Back-project gap pixels to 3D camera frame
    gap_indices = np.argwhere(gap_mask_final)
    if gap_indices.size == 0:
        return None

    Xs, Ys, Zs = [], [], []
    for (r, c) in gap_indices:
        # Find nearest valid top/bottom depths
        z_top, _, z_bottom, _ = find_nearest_valid_in_column(depth, c, r, search_limit)
        if z_top is None or z_bottom is None:
            continue
        z_mid = 0.5 * (z_top + z_bottom)
        X = (c - cx) * z_mid / fx
        Y = (r - cy) * z_mid / fy
        Xs.append(X); Ys.append(Y); Zs.append(z_mid)

    if len(Xs) == 0:
        return None

    Xs = np.array(Xs)
    Ys = np.array(Ys)
    Zs = np.array(Zs)

    width_mm = float(np.max(Xs) - np.min(Xs))
    height_mm = float(np.max(Ys) - np.min(Ys))

    # Bounding box in ROI coordinates
    cmin = int(np.min(gap_indices[:,1]))
    cmax = int(np.max(gap_indices[:,1]))
    rmin = int(np.min(gap_indices[:,0]))
    rmax = int(np.max(gap_indices[:,0]))

    return {
        "gap_mask": gap_mask_final,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "bbox_roi": (cmin, rmin, cmax, rmax),
        "points_3D": np.stack((Xs, Ys, Zs), axis=1)
    }

# -----------------------------
# Main processing loop (video)
# -----------------------------
while True:
    ret_depth, depth_frame = (depth_cap.read() if use_depth else (False, None))

    if ( not ret_depth):
        print("End of one of the videos reached.")
        break
    ##########Depth
    if use_depth: 
        # ####################
        # GAP MEASUREMENT
        #######################
        # count_depth, bboxes_depth = detector.detect(depth_frame, method="template", method_type="cv",  roi =roi)
        # cv2.putText(depth_frame, f"Bolts: {count_depth}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        # for (x, y, w, h) in bboxes_depth:
        #     cv2.rectangle(depth_frame, (x, y), (x+w, y+h), (0,0,255), 2)
        
        # Crop ROI
        depth_frame = depth_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # Convert depth to single channel
        if depth_frame.ndim == 3 and depth_frame.shape[2] == 3:
            depth_frame = depth_frame[:, :, 0]
                
        thresh_value = p_tile_algo.p_tile_thresh(depth_frame, 0.995)
        ret, thresh_seg = cv2.threshold(depth_frame, thresh_value, np.max(depth_frame), cv2.THRESH_TOZERO_INV)
        depth_out = image_morph_algo.image_morph(thresh_seg)

    
        # Visualization
        depth_vis = cv2.cvtColor(depth_out, cv2.COLOR_GRAY2BGR)

        # -----------------------------
        # NEW: robust detection + measurement using kmeans-adjacency approach
        # -----------------------------
        # Use depth_out as already preprocessed (output of your p-tile + morphology steps)
        measurement = isolate_and_measure_gap_from_roi(depth_out, fx, fy, cx, cy,
                                                       min_component_area=2,
                                                       search_limit=50)

        if measurement is not None:
            gap_mask = measurement["gap_mask"]
            width_mm = measurement["width_mm"]
            height_mm = measurement["height_mm"]
            bbox = measurement["bbox_roi"]  # (cmin, rmin, cmax, rmax)
            pts3d = measurement["points_3D"]

            # draw ROI coordinate bbox onto depth_vis
            cmin, rmin, cmax, rmax = bbox
            # draw rectangle in ROI coords
            cv2.rectangle(depth_vis, (cmin, rmin), (cmax, rmax), (0, 0, 255), 2)

            # overlay the gap mask (translucent)
            overlay = depth_vis.copy()
            overlay[gap_mask] = (0, 0, 200)  # dark red for gap
            cv2.addWeighted(overlay, 0.5, depth_vis, 0.5, 0, depth_vis)

            # Put text on frame
            text = f"Gap: W={width_mm:.1f} mm H={height_mm:.1f} mm"
            depth_vis_scaled = cv2.resize(depth_vis, (224, 172), interpolation=cv2.INTER_NEAREST)
            cv2.putText(depth_vis_scaled, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            print(text)
        else:
            # optional: show no-gap message
            depth_vis_scaled = cv2.resize(depth_vis, (224, 172), interpolation=cv2.INTER_NEAREST)
            cv2.putText(depth_vis_scaled, "No valid gap found", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,255), 1, cv2.LINE_AA)

        # -----------------------------
        # Display frames
        # -----------------------------
        cv2.imshow("Depth ROI Raw", depth_frame)
        cv2.imshow("Depth ROI Processed", depth_out)
        output_depth_vis.write(depth_vis_scaled)
        cv2.imshow("Gap Detection Scaled Image", depth_vis_scaled)

    # max_delay = max(rgb_delay, ir_delay, gray_delay, depth_delay)
    if cv2.waitKey(150) & 0xFF == ord("q"):  # ESC to quit
        # self.get_logger().info("Q pressed. Exiting...")
        # sys.exit()
        print("Q pressed. Exiting...")
        sys.exit()

# Cleanup
if use_depth: depth_cap.release()
output_depth_vis.release()
cv2.destroyAllWindows()
