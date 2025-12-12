# online-cam-lidar-calib

This project performs real-time refinement of camera–LiDAR extrinsic parameters using projected LiDAR edges, online optimization, and metric tracking.

This repository contains:

A configurable camera–LiDAR calibration pipeline

- Real-time projection and rendering of LiDAR points into the camera frame
- Online update of extrinsic parameters (rotation + translation)
- Metrics tracking (edge cost, error, recovery, confidence, update magnitudes)
- Plot and JSON export of calibration progress

