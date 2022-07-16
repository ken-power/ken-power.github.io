# Sensor Fusion Projects

* [Obstacle Detection with LiDAR Sensors](#obstacle-detection-with-LiDAR-sensors)
* [Feature Tracking with Camera Sensors](#feature-tracking-with-camera-sensors)
* [3D Object Tracking](#3d-object-tracking)
* [RADAR Target Generation and Detection System](#radar-target-generation-and-detection-system)
* [Unscented Kalman Filter](#unscented-kalman-filter)

## Obstacle Detection with LiDAR Sensors
The goal of this project is to use Lidar to detect traffic, including cars and trucks, and other obstacles (e.g., poles, traffic signs) on a narrow street. The detection pipeline implements filtering, segmentation, clustering, and bounding boxes. Also the segmentation and clustering methods are created from scratch, rather than using Point Cloud Library’s built-in functions. The code places bounding boxes around all obstacles on the road.

See [the code](https://github.com/ken-power/SensorFusionND-Lidar-ObstacleDetection) for this project.

## Feature Tracking with Camera Sensors
The goal of this project is to build the feature tracking part of a collision detection system, and test
various combinations of keypoint detectors and descriptors to see which combinations perform best.

See [the code](https://github.com/ken-power/SensorFusionND-Camera-FeatureTracking) for this project.

## 3D Object Tracking
The goal of this project is to compute time-to-collision (TTC) using Lidar and Camera sensors. Identify suitable keypoint detector-descriptor combinations for TTC estimation. To accomplish this, there are four major tasks to complete: 

* First, develop a way to match 3D objects over time by using keypoint correspondences.
* Second, compute the TTC based on Lidar measurements.
* Then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches.
* And lastly, conduct various tests with the framework. 

The goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor.

See [the code](https://github.com/ken-power/SensorFusionND-3D-Object-Tracking) for this project.

## RADAR Target Generation and Detection System
The goal of this project is to use MATLAB to implement a Radar target generation and detection system. This involves a number of steps, including:

* Configure the FMCW (frequency modulated continuous wave) waveform based on the system requirements.
* Define the range and velocity of target and simulate its displacement.
* For the same simulation loop process the transmit and receive signal to determine the beat signal
* Perform Range FFT (Fast Fourier Transform) on the received signal to determine the Range
* Towards the end, perform the CFAR (constant false alarm rate) processing on the output of 2nd FFT to display the target.

See [the code](https://github.com/ken-power/SensorFusionND-Radar) for this project.

## Unscented Kalman Filter
This project implements an Unscented Kalman Filter (UKF) to estimate the state of multiple cars on a highway using noisy lidar and radar measurements. The project obtains RMSE (root-mean-square error) values that are lower than a specified tolerance.

See [the code](https://github.com/ken-power/SensorFusionND-UnscentedKalmanFilter) for this project.

