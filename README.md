# Ken Power

AI software engineer, technical leader, and researcher developing machine learning software systems for autonomous vehicles.

[Self-Driving Car Projects](#self-driving-car-projects)
[Computer Vision Projects](#computer-vision-projects)
[Sensor Fusion Projects](#sensorfusion-projects)


# Self-Driving Car Projects

## Lane Line Detection
<img src="images/pipeline-scene-3.gif" alt="">

The goal of this proejct is to write a software pipeline to identify the road lane boundaries in a video. The
steps to achieve this goal include the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image (“birds-eye view”).
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

See [the code](https://github.com/ken-power/SelfDrivingCarND-AdvancedLaneLines) for this project.

## Traffic Sign Classification
<img src="images/traffic_signs.png" alt="A row of road traffic signs">

The goal of this proejct is to build a Convolutional Neural Network (CNN) that recognizes traffic signs. The
steps to achieve this goal include the following:

* Load the data set
* Explore, summarize, and visualize the data set
* Design, train, and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

See [the code](https://github.com/ken-power/SelfDrivingCarND-TrafficSignClassifier) for this project.

## Behavioral Cloning
The goal of this proejct is to use convolutional neural networks (CNNs) to clone driving behavior and train a
self-driving car to autonomously navigate a track. This project has the following requirements:

* Use a simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

See [the code](https://github.com/ken-power/SelfDrivingCarND-BehavioralCloning) for this project.

## Extended Kalman Filter
Implement an Extended Kalman Filter (EKF) and use the EFK with noisy LiDAR and RADAR measurements to estimate
the state of a moving object of interest. An extended Kalman filter (EKF) is the nonlinear version of the
Kalman filter which linearizes about an estimate of the current mean and covariance. In the case of well
defined transition models, the EKF has been considered the de facto standard in the theory of nonlinear
state estimation, navigation systems, and GPS (EKF). The Extended Kalman Filter is also used widely in
self-driving cars and sensor fusion.

See [the code](https://github.com/ken-power/SelfDrivingCarND-ExtendedKalmanFilter) for this project.

## 2D Particle Filter
The goal of this project is to implement a 2-dimensional particle filter in C++. The particle filter will be
given a map and some initial localization information (analogous to what a GPS would provide). At each time
step the filter will also get observation and control data. This is a sparse localization problem, i.e., we
are building an end-to-end localizer where we are localizing relative to a sparse set of landmarks using
particle filters.

See [the code](https://github.com/ken-power/SelfDrivingCarND-KidnappedVehicle) for this project.

## Path Planning
A Path Planner that creates smooth, safe trajectories for a self-driving car to follow, enabling the car to
safely navigate around a virtual highway with other traffic.

Project notes:

* The goal of this project is to build a path planner that creates smooth, safe trajectories for the car
to follow. The highway track has other vehicles, all going different speeds, but approximately obeying
the 50 MPH speed limit.
* objective is to safely navigate around a virtual highway with other traffic that is driving +/-10 MPH of
the 50 MPH speed limit.
* We are provided with the car’s localization and sensor fusion data. There is also a sparse map list of
waypoints around the highway. The car should try to go as close as possible to the 50 MPH speed limit,
which means passing slower traffic when possible.
* Note that other cars will try to change lanes too. The car should avoid hitting other cars at all cost
as well as driving inside of the marked road lanes at all times, unless going from one lane to another.
* The car should be able to make one complete loop around the 6946m highway. Since the car is trying to go
50 MPH, it should take a little over 5 minutes to complete 1 loop. Also the car should not experience
total acceleration over 10 m/s^2 and jerk that is greater than 10 m/s^3.

See [the code](https://github.com/ken-power/SelfDrivingCarND-PathPlanning) for this project.

## PID Controller

The goal of this project is to implement a PID controller to enable a self-driving car to manoeuvre around a
track. _Control_ in this context refers to how we use the steering, throttle, and breaks to move a car
where we want it to go. Control is a trickier problem than it might first seem. When a human comes to an
intersection, we use our intuition to determine how hard to steer, when to accelerate, or whether we need to
step on the brakes. Teaching a computer how to do this is difficult. Control algorithms are often called
*controllers*. One of the most common and fundamental of controllers is called the _PID controller_. The goal of this project is to implement a PID controller in C++, and tune the PID hyperparameters, to enable a self-driving car to manoeuvre around a track. The simulator provides the cross track error (CTE) and the velocity (mph) in order to compute the appropriate steering angle. The speed limit is 100 mph. The goal is to drive SAFELY as fast as possible. There is no specified minimum speed.

See [the code](https://github.com/ken-power/SelfDrivingCarND-PID-Controller) for this project.

## Autonomous Vehicle Control
Program a real Self-Driving Car by writing ROS nodes to implement core functionality of the autonomous
vehicle system. For this project, I wrote ROS nodes to implement core functionality of the autonomous
vehicle system, including traffic light detection, control, and waypoint following.

See [the code](https://github.com/ken-power/SelfDrivingCarND-Capstone) for this project.

# Computer Vision Projects

## Facial Keypoint Detection
The goal of this project is to combine computer vision techniques and deep learning architectures to build a
facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face
and are used in many applications. These applications include facial tracking, facial pose recognition,
facial filters, and emotion recognition. The facial keypoint detector is able to look at any image, detect
faces, and predict the locations of facial keypoints on each face.

See [the code](https://github.com/ken-power/CVND-FacialKeypointDetection) for this project.

## Automatic Image Captioning
The goal of this project is to create a neural network architecture to automatically generate captions from
images. We use the <a href="https://cocodataset.org/#home">Common Objects in COntext (MS COCO) dataset</a>
to train the network, and then test the network on novel images.

See [the code](https://github.com/ken-power/CVND-AutomaticImageCaptioning) for this project.

## Simultaneous Localization and Mapping (SLAM)
The goal of this project is to implement SLAM (Simultaneous Localization and Mapping) for a 2 dimensional
world and combine knowledge of robot sensor measurements and movement to create a map of an environment from
only sensor and motion data gathered by a robot, over time. Combining knowledge of robot sensor measurements
and movement, we create a map of an environment from only sensor and motion data gathered by a robot, over
time. SLAM provides a way to track the location of a robot in the world in real-time and identify the
locations of landmarks such as buildings, trees, rocks, and other world features.

See [the code](https://github.com/ken-power/CVND-SLAM) for this project.

## Object Motion and Localization
A set of Object Motion and Localization projects, focused on localising robots, including self-driving cars.
The sub-projects in this project demonstrate various computer vision applications and techniques related to
object motion and localization, including:

* <a href="https://github.com/ken-power/CVND-ObjectMotionAndLocalization/blob/main/Optical_Flow">Optical Flow</a> 
* <a href="https://github.com/ken-power/CVND-ObjectMotionAndLocalization/blob/main/4_2_Robot_Localization">Robot Localization</a> 
* <a href="https://github.com/ken-power/CVND-ObjectMotionAndLocalization/blob/main/4_3_2D_Histogram_Filter">2D Histogram Filter</a> 
* <a href="https://github.com/ken-power/CVND-ObjectMotionAndLocalization/blob/main/4_4_Kalman_Filters">Kalman Filters</a> 
* <a href="https://github.com/ken-power/CVND-ObjectMotionAndLocalization/blob/main/4_5_State_and_Motion">State and Motion</a> 
* <a href="https://github.com/ken-power/CVND-ObjectMotionAndLocalization/blob/main/4_6_Matrices_and_Transformation_of_State">Matrices
and Transformation of State</a> 
* <a href="https://github.com/ken-power/CVND-ObjectMotionAndLocalization/blob/main/4_7_SLAM">Simultaneous Location and Mapping (SLAM)<a> 
* <a href="https://github.com/ken-power/CVND-ObjectMotionAndLocalization/blob/main/4_8_Vehicle_Motion_and_Calculus">Vehicle Motion and Calculus</a> 
* <a href="https://github.com/ken-power/CVND-ObjectMotionAndLocalization/blob/main/Project_Landmark%20Detection">Landmark Detection</a> 

See [the code](https://github.com/ken-power/CVND-ObjectMotionAndLocalization) for this project.

# Sensor Fusion Projects

## Obstacle Detection with LiDAR Sensors
The goal of this project is to use Lidar to detect traffic, including cars and trucks, and other obstacles
(e.g., poles, traffic signs) on a narrow street. The detection pipeline implements filtering, segmentation,
clustering, and bounding boxes. Also the segmentation and clustering methods are created from scratch,
rather than using PCL’s built-in functions. The code places bounding boxes around all obstacles on the road.

See [the code](https://github.com/ken-power/SensorFusionND-Lidar-ObstacleDetection) for this project.

## Feature Tracking with Camera Sensors
The goal of this project is to build the feature tracking part of a collision detection system, and test
various combinations of keypoint detectors and descriptors to see which combinations perform best.

See [the code](https://github.com/ken-power/SensorFusionND-Camera-FeatureTracking) for this project.

## 3D Object Tracking
Compute time-to-collision (TTC) using Lidar and Camera sensors. Identify suitable keypoint
detector-descriptor combinations for TTC estimation. To accomplish this, there are four major tasks to complete: 

* First, develop a way to match 3D objects over time by using keypoint correspondences.
* Second, compute the TTC based on Lidar measurements.
* Then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches.
* And lastly, conduct various tests with the framework. The goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor.

See [the code](https://github.com/ken-power/SensorFusionND-3D-Object-Tracking) for this project.

## RADAR
The goal of this project is to use MATLAB to implement a Radar target generation and detection system. This involves a number of steps, including:

* Configure the FMCW waveform based on the system requirements.
* Define the range and velocity of target and simulate its displacement.
* For the same simulation loop process the transmit and receive signal to determine the beat signal
* Perform Range FFT on the received signal to determine the Range
* Towards the end, perform the CFAR processing on the output of 2nd FFT to display the target.

See [the code](https://github.com/ken-power/SensorFusionND-Radar) for this project.

## Unscented Kalman Filter
This project implements an Unscented Kalman Filter to estimate the state of multiple cars on a highway using
noisy lidar and radar measurements. The project obtains RMSE values that are lower than a specified
tolerance.

See [the code](https://github.com/ken-power/SensorFusionND-UnscentedKalmanFilter) for this project.
