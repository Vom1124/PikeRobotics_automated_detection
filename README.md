# Pike Robotics – Automated Counting

Automated counting pipeline for internal evaluation of **bolts** and **secondary rubber seals** using **RGB-D sensor data** captured in ROS bag files.

The system reprocesses recorded `.bag` files and applies multiple techniques for **object isolation and count estimation**, comparing classical and deep-learning-based approaches.
#### Convert the rosbag to compatible ROS version before replaying the recorded bag files. For example, .bag (ROS 1) to .db3 (ROS 2) file or vice versa as needed. 

---

## Project Overview

The goal is to determine the most reliable method for **counting small mechanical components** under varying background and lighting conditions. The following strategies are evaluated:

| Technique Type      | Example Methods                                  |
|---------------------|--------------------------------------------------|
| Detection           | YOLO / Ultralytics                              |
| Segmentation        | Instance or semantic segmentation models         |
| Thresholding        | Color / depth-based isolation                    |
| Morphological Ops   | Erosion / dilation for cleanup                   |


---

## Strategy Evaluation Status

All techniques listed below are **experimental** and may be added, removed, or modified as results come in.

| Strategy Type       | Example Methods                                  | Status          |
|---------------------|--------------------------------------------------|-----------------|
| Detection           | YOLO / Ultralytics                              | Under evaluation |
| Segmentation        | Instance or semantic segmentation models         | Under evaluation |
| Thresholding        | Depth / color isolation                         | Under evaluation |
| Morphological Ops   | Erosion / dilation for cleanup                  | Under evaluation |

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

- NumPy should remain **below version 2.0**. Higher versions may introduce conflicts with **ROS 2** or **Ultralytics** (this step is just if YOLO is being used), depending on the globally installed package set.
- --- To instal YOL)V8n, refer to the public repo: https://github.com/Vom1124/venv_yolo.git

---

## TODO

- [ ] Implement baseline thresholding-based counting
- [ ] Integrate YOLO-based detection
- [ ] Benchmark segmentation-based approaches
- [ ] Record accuracy and consistency metrics

---

## Repository Status

This is a **private repository** intended for **internal research and experimentation**.  
Results and methods are subject to change as techniques evolve.

