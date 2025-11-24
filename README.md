# Toronto-3D Dataset MCAP Simulation

A 15-second sensor simulation using the Toronto-3D LiDAR dataset, exported to MCAP format for visualization in Foxglove Studio.

## 🎥 Demo

[Link to video demonstration]

## 📋 Overview

This project creates a realistic autonomous vehicle sensor simulation with:
- **Pinhole Camera Model**: Renders RGB images from point cloud data
- **64-Beam LiDAR Simulation**: Simulates realistic LiDAR scanning
- **Linear Trajectory**: Camera flies through Toronto street scene
- **MCAP Export**: Industry-standard format for robotics data

## Features

- ✅ 15-second simulation with 150 frames
- ✅ Aerial view of Toronto intersection
- ✅ Real-time sensor data synchronization
- ✅ Foxglove Studio compatible
- ✅ Configurable point cloud density

## Requirements

```bash
pip install open3d numpy mcap opencv-python matplotlib
- Python 3.8+
- open3d
- numpy
- mcap
- opencv-python (cv2)
- matplotlib

## Dataset

Download the Toronto-3D dataset:
- Source: [Toronto-3D Official](https://github.com/WeikaiTan/Toronto-3D)
- File used: `L001.ply`

## Quick Start

### 3. Run Simulation
run toronto.ipynb


### 4. View in Foxglove
1. Download [Foxglove Studio](https://foxglove.dev/download)
2. Open the generated `toronto_3d_simulation_linear.mcap` file
3. Add panels:
   - **Image panel** → Select `/camera/image_compressed`
   - **3D panel** → Select `/lidar/points`
4. Press **Play** ▶️

## 📊 Output
- **Duration**: 15 seconds
- **Frame rate**: 10 Hz (150 frames)
- **Camera resolution**: 1280×720
- **LiDAR points/frame**: ~500-1000 points

Graph of actual trajectory:

<img width="1390" height="590" alt="image" src="https://github.com/user-attachments/assets/f987dfe4-7a00-42ca-b5fa-2010337259de" />



