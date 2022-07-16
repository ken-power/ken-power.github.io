 [Home](../README.md) | [Projects](README.md)

# Computer Vision Projects

* [Facial Keypoint Detection](#facial-keypoint-detection)
* [Automatic Image Captioning](#automatic-image-captioning)
* [Simultaneous Localization and Mapping (SLAM)](#simultaneous-localization-and-mapping-slam)
* [Object Motion and Localization](#object-motion-and-localization)
* [Optical Flow](#optical-flow)
* [Analysis of 3D Magnetic Resonance (MR) Images](#analysis-of-3d-magnetic-resonance-mr-images)


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
SLAM (Simultaneous Localization and Mapping) provides a way to track the location of a robot (e.g., a self-driving car) in the world in real-time and identify the locations of landmarks such as buildings, trees, rocks, and other world features. The goal of this project is to implement SLAM for a 2-dimensional world and combine knowledge of robot sensor measurements and movement to create a map of an environment from only sensor and motion data gathered over time by a robot. 

See [the code](https://github.com/ken-power/CVND-SLAM) for this project.

## Object Motion and Localization
The goal of this project is to implement a set of Object Motion and Localization sub-projects, focused on localising robots, including self-driving cars. The sub-projects in this project demonstrate various computer vision applications and techniques related to
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

## Optical Flow
![](https://github.com/ken-power/ComputerVision-OpticalFlow/blob/main/OpticalFlow/images/skateboard_dense_optical_flow.gif)
A set of projects that illustrate different approches to Optical Flow.
* **Optical Flow** explores three techniques to tackle the tracking problem, Feature Tracking, Sparse Optical Flow, and Dense Optical Flow. 
* **FlowNet** illustrates Deep Learning for Optical Flow by implementing the FlowNet algorithm using PyTorch and training the models on the KITTI dataset. The goal is to output the optical flow of two images.
*  **RAFT** explores the RAFT deep network architecture for optical flow.
*  **Visual SLAM** shows an example of Visual SLAM (Simultaneous Localization and Mapping) using visual features.

See [the code](https://github.com/ken-power/ComputerVision-OpticalFlow) for this project.

## Analysis of 3D Magnetic Resonance (MR) Images
![](https://github.com/ken-power/MR-Image-Analysis/blob/main/images/brain-image.png)

The goal of this project is to process and analyze magnetic resonance (MR) images of the brain. Unlike more traditional images, structural MR images are actually 3D image volumes, i.e., 3D arrays of numbers. Structural MR images show the anatomy of a patient, as opposed to functional MR images, which highlight areas of blood flow. 

See [the code](https://github.com/ken-power/MR-Image-Analysis) for this project.

