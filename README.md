# Azure Kinect — Ground Truth

This repository contains code to extract the 3D centerline (ground truth curve) of a slender object captured by an Azure Kinect. The project fits a surface to the camera point cloud, projects the points to the XY plane to fit a spline, and then lifts that spline back up to the surface to estimate the object centerline.


## Table of contents

- [Overview](#overview).  
- [Prerequisites](#prerequisites).  
- [Recommended reading and tutorials](#recommended-reading-and-tutorials).  
- [How it works (high level)](#how-it-works-high-level).  
- [Nodes and scripts (details)](#nodes-and-scripts-details).  
- [Running the code](#running-the-code).  
- [Notes and known issues](#notes-and-known-issues).  
- [Helpful links](#helpful-links).  

---

## Overview

This code extracts the shape of a curve in 3D space from an Azure Kinect point cloud and estimates the center of the cross-section along that curve. The pipeline:

- captures point clouds from the Azure Kinect.  
- filters points by color to isolate a green slender object.  
- fits a surface to those points.  
- projects the surface points to the XY plane and fits a spline to that projection.  
- projects the spline back up to the fitted surface.  
- computes surface normals at spline points and offsets points by the joint radius to estimate the centerline.  

---

## Prerequisites

- Ubuntu 22.04.  
- Azure Kinect SDK (see this repo: `!!!!!!!!`).  

---

## Recommended reading and tutorials

These resources are helpful for understanding the ROS2 driver and the point cloud messages used by this project:

- Azure Kinect camera settings and overview (video, watch the first ~31 minutes).  
  `https://www.youtube.com/watch?v=HzeYb00eQRI`.  
- Helpful explanation of the `sensor_msgs/PointCloud2` message.  
  `https://medium.com/@tonyjacob_/pointcloud2-message-explained-853bd9907743`.  

---

## How it works (high level)

1. The camera captures a point cloud containing a green slender object.  
2. The code fits a smooth surface to those points.  
3. Points are projected onto the XY plane to determine the object outline from the camera perspective.  
4. A spline is fitted to the projected points in the XY plane.  
5. The spline is projected up to the previously fitted surface.  
6. (Optional) To convert from the measured outer surface to the object centerline:  
   - compute surface normals at every point on the spline.  
   - scale the unit normals by the estimated joint radius.  
   - add the scaled normals to the spline points to estimate the centerline points.  

---

## Nodes and scripts (details)

- `azure_kinect_node` (from the `azure_kinect_ros2_driver` package).  
  - Publishes raw Azure Kinect data as a point cloud and other sensor topics.  
- `filtered_pc` (from the `azure_listener` package).  
  - Subscribes to the raw point cloud topic, filters out non-green colors, and republishes the filtered point cloud on `filtered_pc`.  
- `PC_extractor` (from the `azure_listener` package).  
  - Subscribes to `filtered_pc` and writes the point cloud to a CSV file, continuously overwriting the file with new data.  
- `Finding_arm_shape.py`.  
  - Loads the CSV, fits the surface, projects and fits the spline, computes normals, offsets points to the centerline, and displays an interactive 3D plot that updates periodically.  

> Implementation note: The current implementation is functional but not fully optimized for efficiency.

---

## Running the code

There is no ROS launch file provided yet, so run each node in a separate terminal window:

```bash
# Terminal 1: start the Azure Kinect ROS2 driver node.
ros2 run azure_kinect_ros2_driver azure_kinect_node
````

```bash
# Terminal 2: start the filtering node.
ros2 run azure_listener filtered_pc
```

```bash
# Terminal 3: start the CSV extractor that writes the filtered point cloud.
ros2 run azure_listener PC_extractor
```

Then, in a fourth terminal, navigate to the package folder and run the analysis script:

```bash
cd ~/ros2_ws/src/azure_listener/azure_listener
python3 Finding_arm_shape.py
```

---

## Notes and known issues

* The interactive 3D plot updates automatically every second or so, but in practice it lags behind the camera by roughly 10 seconds.
* The pipeline measures the outside surface of the joint and then offsets by the radius to estimate the centerline; depending on your use case, that step may be unnecessary or may require tuning.
* Color filtering is used to isolate the object; lighting and camera exposure can affect filtering performance.
* The current implementation trades performance for simplicity; profiling and optimization (for example, reducing data copies, using a launch file, or performing computation in C++ nodes) would be beneficial for real-time use.

---

## Helpful links

* Azure Kinect SDK: `!!!!!!!!`.
* Orientation / visualization for coordinate conventions: `!!!!!!!!`.
* Personal notes with screenshots and statistics for the ground truth code (includes a discussion of drawbacks):
  `https://docs.google.com/document/d/19dlQiKceWllj79F0mKil7e75QcpLl_2TVpSsGnVbJ_E/edit?usp=sharing`.
