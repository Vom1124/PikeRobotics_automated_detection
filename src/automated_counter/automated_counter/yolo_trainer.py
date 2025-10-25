import os
import sys
import cv2
import numpy as np
import time

from automated_counter import detector

# import torch

# torch.set_num_threads(1)

# os.environ["USE_NNPACK"] = "0"

# ==========================
# USER TOGGLE: DETECT OR TRAIN
# ==========================
generate_detections = True   # <<< SET TO False TO TRAIN MODEL
# Selecting the YOLO model
use_base_model = False  # True = start from yolov8n.pt, False = continue training from last checkpoint

# ==========================
# AUTO-RESTART INSIDE YOLO VENV
# ==========================
venv_path = "/home/vom/venv_avx_free_yolo" # <---- Change this to reflect your python venv where ultralytics is installed
venv_python = os.path.join(venv_path, "bin", "python")
# Check if we’re already inside the YOLO venv
if venv_path not in sys.prefix:
    if os.path.exists(venv_python):
        print(f"\033[33;1m [INFO] Restarting inside YOLO venv: {venv_path}\033[0m")
        os.execvp(venv_python, [venv_python] + sys.argv)
    else:
        raise FileNotFoundError(f"[ERROR] No python executable found at {venv_python}")

# ==========================
# CHECK YOLO AVAILABILITY
# ==========================
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("\033[31;0m [WARN] Ultralytics YOLO not installed or venv not active. Training will be skipped.\033[0m")

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
working_dir= "/home/vom/ros2_ws/PikeRobotics_automated_counting"

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
def train_yolo(dataset_dir, epochs=50, imgsz=640, device="cpu", model_path=None):
    """
    Train YOLOv8 on the collected frames.
    Args:
        dataset_dir: str, path to the dataset containing 'images/' and 'labels/'.
        epochs: int, number of training epochs
        imgsz: int, image size
        device: CPU/GPU
        model_path: str, path to YOLO model weights to start training from
                    If None, starts from base yolov8n.pt
    """
    if not YOLO_AVAILABLE:
        print("\033[31;1m[WARN] YOLO not available. Skipping training.\033[0m")
        return

    # Create dataset yaml
    dataset_yaml = os.path.join(dataset_dir, "dataset.yaml")
    with open(dataset_yaml, "w") as f:
        f.write(f"path: {dataset_dir}\n")
        f.write("train: images\n")
        f.write("val: images\n")  # if you have separate val set, change this
        f.write("nc: 1\n")
        f.write("names: ['bolt']\n")

    # Determine model to use
    if model_path is None:
        model_path = "yolov8n.pt"
    else:
        # If existing model doesn't exist, fallback to base model
        if not os.path.exists(model_path):
            print(f"\033[31;1m[WARN] Model not found at {model_path}. Falling back to base model.\033[0m")
            model_path = "yolov8n.pt"

    print(f"\033[33;1m [INFO] Starting YOLO training with model: {model_path}\033[0m")
    model_yolo = YOLO(model_path)

    # Unique model name for each dataset
    model_name = f"bolt_detection_{__dataname__}_{os.path.basename(dataset_dir)}"

    model_yolo.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=min(imgsz, 640),  # reduce image size for CPU
        # batch=2,                 # small batch size to avoid CPU instruction issues
        project="yolo_train",
        name=model_name,
        device=device,
        # half=False,              # force FP32
        # compile=False            # disable torch.compile if used in newer Ultralytics
    )

    print(f"\033[32;1m [INFO] YOLO training completed for {dataset_dir}. Model saved in 'yolo_train/{model_name}/weights/'.\033[0m")

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
            print("*************************End of video reached.************************")
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

            print(f"\033[33;1m[INFO] Saved frame {frame_id} with {len(bboxes)} detections.\033[0m")
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
        print(f"\033[35;1m[MODE] GENERATE DATA for dataset '{current_dataset}'\033[0m")
        generate_training_data()
    else:
        print(f"\033[35;1m[MODE] TRAIN YOLO for dataset '{current_dataset}'\033[0m")

        # Dynamically set existing model path if use_base_model is False
        existing_model_path = os.path.join(
            "yolo_train",
            f"bolt_detection_{__dataname__}_{os.path.basename(output_dir)}",
            "weights",
            "best.pt"
        )

        model_to_train = "yolov8n.pt" if use_base_model else existing_model_path

        # -------------------------
        # CPU-SAFE TRAINING PARAMETERS
        # -------------------------
        cpu_safe_imgsz = 640  # reduce image size for CPU

        train_yolo(
            dataset_dir=output_dir,
            epochs=100,
            imgsz=cpu_safe_imgsz,
            device="cpu",
            model_path=model_to_train
        )
