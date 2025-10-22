# Pike Robotics – Automated Counting

Automated counting pipeline for internal evaluation of **bolts** and **secondary rubber seals** using **RGB-D sensor data** captured in ROS bag files.

<b> <u> Note: </b></u> This repo was generated and tested for Ubuntu 22.04 running ROS 2 Humble OS. In order for this repo to be cloned and executed seamlessly, a fresh colcon build --symlink-install needs to be done within the workspace. If error still persists for sourcing, try deleting the install, log, and build files within the workpsace and re-build it to reflect the correct directory for sourcing. Re-building the workspace from scratch should resolve the sourcing errors. 

The system reprocesses recorded `.bag` files and applies multiple techniques for **object isolation and count estimation**, comparing classical and deep-learning-based approaches.
#### Convert the rosbag to compatible ROS version before replaying the recorded bag files. For example, .bag (ROS 1) to .db3 (ROS 2) file or vice versa as needed. For converting in the ROS 2 Humble OS, do the following:
    pip3 install rosbags && \
    rosbags-convert --src <path_to_ros1_bag_file.bag> --dst <path_to_ros2_bag_directory>
  Now, the converted folder should have the bag file comptabile with ROS 2 with .db3 extension and its corresponding metadata.yaml file.
  <u>Exception:</u> Use the "fix_metadata.py" file in the main directory if the converted metadata.yaml file is not parsable by the ROS 2 environment.  

---

## Project Overview

The goal is to determine the most reliable method for **counting small mechanical components** under varying background and lighting conditions. 

#### Strategy Evaluation Status

All techniques listed below are **experimental** and may be added, removed, or modified as results come in.

| Strategy Type       | Example Methods                                  | Status          |
|---------------------|--------------------------------------------------|-----------------|
| Detection           | YOLO / template matching                              | Under evaluation |
| Segmentation        | Classical global segmentation models         | Preliminary test done |
| Thresholding        | Depth / color isolation                         | Preliminary test done |
| Circle Detector using Hough Transform        | Depth / color isolation                         | Preliminary test done |
| Morphological Ops   | Erosion / dilation for cleanup                  | Preliminary test done |

The goal at this stage is **not optimization**, but **validation of feasibility and consistency** across varying recording conditions. RGB and depth frames are extracted from **ROS bag replays**, processed offline, and analyzed to chose the best method based on the ground truth manual counting. 

---

## Pipeline (Experimental Draft)

1. Replay ROS bag file (`.bag`)

## Pipeline

1. Replay ROS bag file (`.bag`)
2. Extract RGB + Depth frames
3. Apply selected detection / segmentation / thresholding technique
4. Process masks to isolate target object
5. Estimate object count

---

## Environment Notes

- --- NumPy should remain **below version 2.0**. Higher versions may introduce conflicts with **ROS 2** or **Ultralytics** (this step is just if YOLO is being used), depending on the globally installed package set.
- --- To install YOLOv8n, refer to the public repo: https://github.com/Vom1124/venv_yolo.git

---

## TODO / Progress

- [x] Implement baseline thresholding-based detection:\
      A simple histogram based and/or global thresholding can not delineate the bolts/rims (skid plates) with the  background vessel.
      
- [x] Benchmark segmentation-based approaches:\
      All segmentation methods, including morphology applied segemtnation without learning-based model did not perform well.
      
- [x] Non-learnign based detectors:\
      Implemented Hough Transform to detect the shape of the bolt. Not much success here...
      
- [ ] Integrate template-matching detection:\
      A simple yet effective template matching using SSD/NCC functions and/or with feature/keypoint matching using ORB, SIFT, etc. Might be robust if pre-processing and post-processing is done correctly. Post-processing example: RANSAC to detect inliers after featrues are extracted and matched. Currently, in testing ...
      
- [ ] Integrate YOLO-based detection:\
      Is very robust if pre-trained with templates. Still need to access the integrability due to hardware/resource constraints for real-time execution.Might work for a specific ROI.

- [ ] Record accuracy and consistency metrics:\
      In the process of validating different models for robustness across different dataset and be able to perform well under different lighting condiitons in real-time.
      
- [ ] Develop the tracking algorithm for counting:\
    This is final phase of this project to execute the counting/iding. Label detected features (bolts, skid plates, etc.) uniquely frame-to-frame. Can fuse odometry reading to assign unqiue labels so that the same detected feature is not counted more than once. 

---

## Repository Status

This is a **private repository** intended for **internal research and experimentation**.  
Results and methods are subject to change as techniques evolve.

