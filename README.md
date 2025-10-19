# Pike Robotics – Automated Counting

Automated counting pipeline for internal evaluation of **bolts** and **secondary rubber seals** using **RGB-D sensor data** captured in ROS bag files.

The system reprocesses recorded `.bag` files and applies multiple techniques for **object isolation and count estimation**, comparing classical and deep-learning-based approaches.

---

## Project Overview

The goal is to determine the most reliable method for **counting small mechanical components** under varying background and lighting conditions. The following strategies are evaluated:

| Technique Type      | Example Methods                                  |
|---------------------|--------------------------------------------------|
| Detection           | YOLO / Ultralytics                              |
| Segmentation        | Instance or semantic segmentation models         |
| Thresholding        | Color / depth-based isolation                    |
| Morphological Ops   | Erosion / dilation for cleanup                   |

RGB and depth frames are extracted from **ROS bag replays**, processed offline, and analyzed for consistency and accuracy.

---

## Pipeline

1. Replay ROS bag file (`.bag`)
2. Extract RGB + Depth frames
3. Apply selected detection / segmentation / thresholding technique
4. Process masks to isolate target object
5. Estimate object count

---

## Environment Notes

- NumPy should remain **below version 2.0**. Higher versions may introduce conflicts with **ROS 2** or **Ultralytics**, depending on the globally installed package set.

---

## TODO

- [ ] Add setup and dependency installation script
- [ ] Automate ROS bag replay with topic selection
- [ ] Implement baseline thresholding-based counting
- [ ] Integrate YOLO-based detection
- [ ] Benchmark segmentation-based approaches
- [ ] Record accuracy and consistency metrics

---

## Repository Status

This is a **private repository** intended for **internal research and experimentation**.  
Results and methods are subject to change as techniques evolve.

