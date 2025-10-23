import os
import sys
import cv2
import numpy as np
import time

from automated_counter import detector

# ==========================
# USER TOGGLE: DETECT OR TRAIN
# ==========================
generate_detections = False   # <<< SET TO False TO TRAIN MODEL

# # ==========================
# # AUTO-RESTART INSIDE YOLO VENV
# # # ==========================
# venv_path = "/home/vom/ros2_ws/venv_yolo"
# venv_python = os.path.join(venv_path, "bin", "python")
# # Check if we’re already inside the YOLO venv
# if venv_path not in sys.prefix:
#     if os.path.exists(venv_python):
#         print(f"[INFO] Restarting inside YOLO venv: {venv_path}")
#         os.execvp(venv_python, [venv_python] + sys.argv)
#     else:
#         raise FileNotFoundError(f"[ERROR] No python executable found at {venv_python}")

# ==========================
# CHECK YOLO AVAILABILITY
# ==========================
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] Ultralytics YOLO not installed or venv not active. Training will be skipped.")

# ==========================
# CONFIGURATION
# ==========================
prev_tracked_count = 0  # store count from previous frame
frame_id = 0            # for saved frames

H, W = (480, 640)
roi_w, roi_h = 100, 150
roi_x = (W - roi_w) // 2
roi_y = (H - roi_h) // 2
roi = (roi_x, roi_y, roi_w, roi_h)

# ==========================
# DATASET CONFIGURATION
# ==========================
use_path_1 = True  # True = use CPC dataset, False = use SUNCOR

# Dataset paths
path_1 = "/home/vom/ros2_ws/PikeRobotics_automated_counting/CPC"
path_2 = "/home/vom/ros2_ws/PikeRobotics_automated_counting/SUNCOR"

# Dynamically set dataset path and name
if use_path_1:
    path = path_1
    __dataname__ = "CPC"
    # <-- set this to whichever CPC subfolder you’re using
    current_dataset = "CPC_9"
    # current_dataset = "CPC_10"
    # current_dataset = "CPC_19"
    # current_dataset = "CPC_43"
    # current_dataset = "CPC_61"   
else:
    path = path_2
    __dataname__ = "SUNCOR"
    current_dataset = "SUNCOR_24"  # <-- set this to whichever SUNCOR subfolder you’re using

dataset_path = current_dataset

# Define video paths dynamically 
#-- CPC Files 
if use_path_1:
    rgb_video_path = f"{path}/{dataset_path}/{dataset_path}__wombot_gen3proto_seal_cameras_realsense_color_image_raw_compressed.mp4"
else:
    #-- SUNCOR Files #
    rgb_video_path = f"{path}/{dataset_path}/4_RealsenseColor.mp4"

# Output directory for frames and labels (separate folders per dataset)
output_dir = os.path.join("yolo_datasets", current_dataset)
os.makedirs(output_dir, exist_ok=True)
image_dir = os.path.join(output_dir, "images")
label_dir = os.path.join(output_dir, "labels")
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# Detection parameters
method = "template"
method_type = "cv"
class_id = 0  # YOLO class for bolts

# -------------------------
# YOLO TRAINING FUNCTION
# -------------------------
def train_yolo(dataset_dir, epochs=50, imgsz=640, device="cpu", model="yolov8n.pt"):
    """
    Train YOLOv8 on the collected frames.
    Args:
        dataset_dir: str, path to the dataset containing 'images/' and 'labels/'.
        epochs: int, number of training epochs
        imgsz: int, image size
        device: CPU/GPU
        model: str, base YOLOv8 model
    """
    if not YOLO_AVAILABLE:
        print("[WARN] YOLO not available. Skipping training.")
        return

    # Create dataset yaml
    dataset_yaml = os.path.join(dataset_dir, "dataset.yaml")
    with open(dataset_yaml, "w") as f:
        f.write(f"path: {dataset_dir}\n")
        f.write("train: images\n")
        f.write("val: images\n")  # if you have separate val set, change this
        f.write("nc: 1\n")
        f.write("names: ['bolt']\n")

    print(f"[INFO] Starting YOLO training for dataset: {dataset_dir}") 
    model_yolo = YOLO(model)

    # Unique model name for each dataset
    model_name = f"bolt_detection_{__dataname__}_{os.path.basename(dataset_dir)}"

    model_yolo.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project="yolo_train",
        name=model_name
    )

    print(f"[INFO] YOLO training completed for {dataset_dir}. Model saved as '{model_name}' in 'yolo_train/' directory.")

# -------------------------
# VIDEO PROCESSING LOOP
# -------------------------
def generate_training_data():
    global prev_tracked_count, frame_id

    cap = cv2.VideoCapture(rgb_video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {rgb_video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        # DETECTION
        count, bboxes = detector.detect(frame, method=method, method_type=method_type, roi=roi)

        # Draw detections for visualization
        cv2.putText(frame, f"Bolts: {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        for (x, y, w, h) in bboxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

        # SAVE FRAME ONLY IF NEW DETECTION
        if count > prev_tracked_count:
            img_filename = f"frame_{frame_id:05d}.jpg"
            img_path = os.path.join(image_dir, img_filename)
            cv2.imwrite(img_path, frame)

            # Save YOLO labels
            h_img, w_img = frame.shape[:2]
            label_path = os.path.join(label_dir, img_filename.replace(".jpg", ".txt"))
            with open(label_path, "w") as f:
                for (x, y, bw, bh) in bboxes:
                    x_center = (x + bw/2) / w_img
                    y_center = (y + bh/2) / h_img
                    w_norm = bw / w_img
                    h_norm = bh / h_img
                    f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")

            print(f"[INFO] Saved frame {frame_id} with {len(bboxes)} detections.")
            frame_id += 1

        prev_tracked_count = count

        # Display
        cv2.imshow("RGB Video", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==========================
# MAIN EXECUTION
# ==========================
if __name__ == "__main__":
    if generate_detections:
        print(f"[MODE] GENERATE DATA for dataset '{current_dataset}'")
        generate_training_data()
    else:
        print(f"[MODE] TRAIN YOLO for dataset '{current_dataset}'")
        sys.exit()
        train_yolo(output_dir, epochs=100)
