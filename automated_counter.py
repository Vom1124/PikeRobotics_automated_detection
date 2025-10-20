import os
import sys
import io
import csv 
import time
import re
#===ROS packages
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# #========Sourcing the virtual environment to load adn execute Ultralytics (YOLO).
# os.system("source ~/venv_yolo/bin/activate")
from ultralytics import YOLO
import glob



# ==========================
# CONFIGURATION
# ==========================
use_camera = True          # True = live IntelRealSense via ROS2, False = load from folder
use_depth_only = False      # True = YOLO on depth-only, False = YOLO on RGB
rgb_folder = "/home/husarion/ros2_ws/PikeRobotics_automated_counter/CPC_9_primary/RawRGBStream.png"
depth_folder = "/home/husarion/ros2_ws/PikeRobotics_automated_counter/CPC_9_primaryRawDepthMap.png"

model_path = "yolov8n.pt"
# ==========================

# Load YOLO model
model = YOLO(model_path)
#Loading CSV file for saving the computational latency
CSV_FILE_PATH = "/home/husarion/ros2_ws/PikeRobotics_automated_counter/CPC_9_primary/yolo_latency.csv"

total_frames = 0
total_latency_sum = 0
total_pre = 0
total_inf = 0
total_post = 0

def csv_logger(pre_ms, inf_ms, post_ms):
    """Log YOLO latency components for a single frame."""
    global total_latency_sum, total_frames, total_pre, total_inf, total_post
    total_ms = pre_ms + inf_ms + post_ms

    # Update running sum and count
    total_pre+= pre_ms
    total_inf+= inf_ms
    total_post+= post_ms
    total_latency_sum += total_ms
    total_frames += 1
        
    # Save to CSV
    file_exists = os.path.exists(CSV_FILE_PATH)
    with open(CSV_FILE_PATH, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'preprocess_ms', 'inference_ms', 'postprocess_ms', 'total_ms']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp': time.time(),
            'preprocess_ms': pre_ms,
            'inference_ms': inf_ms,
            'postprocess_ms': post_ms,
            'total_ms': total_ms
        })
    avg_inf = total_inf / total_frames
    avg_pre = total_pre / total_frames
    avg_post = total_post / total_frames
    avg_total = total_latency_sum / total_frames
    print(f"YOLO Latency -> Pre: {pre_ms:.2f} ms, Inference: {inf_ms:.2f} ms, "
          f"Post: {post_ms:.2f} ms, Total: {total_ms:.2f} ms")
    print(f"Running Averages after {total_frames} frames -> "
          f"Pre: {avg_pre:.2f}, Inference: {avg_inf:.2f}, Post: {avg_post:.2f}, Total: {avg_total:.2f} ms\n")
        

def preprocess_depth(depth_img):
    """
    Ensure depth image is uint8 where pixel value 0–255 corresponds to 0–8000 mm.
    If raw input is still 16-bit (0–8000 mm directly), then normalize.
    """
    if depth_img.dtype == np.uint16 or depth_img.dtype == np.int32: 
        # Convert 0–8000 mm range into 0–255 pixel range
        depth_gray = cv2.convertScaleAbs(depth_img, alpha=255.0 / 8000.0)
    elif depth_img.dtype == np.uint8:
        # Already in 0–255 format — representing scaled 0–8000 mm
        depth_gray = depth_img.copy()
    else:
        raise ValueError(f"Unsupported depth image format: {depth_img.dtype}")
    # depth_gray = cv2.equalizeHist(depth_gray)
    return depth_gray


def draw_detections(base_img, depth_img, results):
    """
    Draw YOLO detections with depth info.
    depth_img is assumed to be uint8 where:
        depth_mm = pixel_value * (8000 / 255)
    """
    vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Extract depth values inside the detection box
            roi_depth = depth_img[y1:y2, x1:x2]
            roi_valid = roi_depth[roi_depth > 0]

            if roi_valid.size > 0:
                median_pixel = np.median(roi_valid)
                depth_mm = median_pixel * (8000.0 / 255.0)
                depth_m = depth_mm / 1000.0
            else:
                depth_mm, depth_m = -1, -1

            print(f"Depth at bounding box: {depth_mm:.1f} mm")

            # Draw bounding box in RED
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # Optional: show depth value on image
            # cv2.putText(vis, f"{depth_m:.2f} m", (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return vis
# ==========================
# MODE 1: LOAD FROM FOLDER
# ==========================
if not use_camera:
    rgb_files = sorted(glob.glob(rgb_folder)) # glob used for batch-processing but not necessary
    depth_files = sorted(glob.glob(depth_folder))
    for rgb_path, depth_path in zip(rgb_files, depth_files):
        rgb_img = cv2.imread(rgb_path)
        # cv2.imshow("raw rgb", rgb_img)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth_gray = preprocess_depth(depth_img)
        input_img = cv2.merge([depth_gray]*3) if use_depth_only else rgb_img
        # results = model(input_img, verbose=False)
        results = model(input_img, conf=0.175, verbose=True)
        print("Detections found:", len(results[0].boxes))
        # vis = draw_detections(depth_gray if use_depth_only else rgb_img, depth_img, results) # Drawing the bounding box on either rgb or depth based on selected option in "use_depth_only"
        vis = draw_detections(depth_gray, depth_img, results) # Drawing the bounding box only on depth map 
        cv2.imshow("YOLO-Depth (Folder Mode)", vis)
        if cv2.waitKey(0) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

# ==========================
# MODE 2: LIVE CAMERA (ROS topic form rosbag)
# ==========================
else:
    class IntelRealSenseubscriber(Node):
        def __init__(self):
            super().__init__('Detection_node')
            self.bridge = CvBridge()
            self.rgb_img = None
            self.depth_img = None
            self.depth_gray = None
            self.rgb_sub = self.create_subscription(
                Image, '/camera/color/image_raw', self.rgb_callback, 10
            )
            self.depth_sub = self.create_subscription(
                Image, '/camera/depth/image_raw', self.depth_callback, 10
            )

        def rgb_callback(self, msg):
            self.rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        def depth_callback(self, msg):
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            self.depth_gray = preprocess_depth(self.depth_img)

        def process_frame(self):
            if self.depth_img is None:
                return
            base_img = self.rgb_img if not use_depth_only else preprocess_depth(self.depth_img)
            input_img = cv2.merge([base_img]*3) if use_depth_only else base_img           
            
            # ======= Measure YOLO latency =======
            results = model(input_img, conf=0.25, verbose=True)
            for r in results:
                # Access the speed attribute
                inf_ms = r.speed['inference']
                pre_ms = r.speed['preprocess']
                post_ms = r.speed['postprocess']

            # Log to CSV or update running average
            csv_logger(pre_ms, inf_ms, post_ms)

            # =====================================
            # vis = draw_detections(depth_gray if use_depth_only else rgb_img, depth_img, results) # Drawing the bounding box on either rgb or depth based on selected option in "use_depth_only"
            vis = draw_detections(self.depth_gray, self.depth_img, results)# Drawing the bounding box only on depth map         
            cv2.imshow("YOLO-Depth", vis)
            if cv2.waitKey(1) & 0xFF == "q":
                self.get_logger().info("Q pressed, shutting down node...")
                rclpy.shutdown()  # exits rclpy.spin_once()

    rclpy.init()
    node = IntelRealSenseSubscriber()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.process_frame()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()