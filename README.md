A comprehensive tutorial series on Image Formation and Camera Calibration in Python with example
## Setting up
Assuming you've anaconda installed, create a virtual environment and install dependencies. 

### Create Virtual Environment
```
conda create -n camera-calibration-python python=3.6 anaconda
conda activate camera-calibration-python
```
### Clone and Install dependencies
```
git clone https://github.com/wingedrasengan927/Image-formation-and-camera-calibration.git
cd Image-formation-and-camera-calibration
pip install -r requirements.txt
```
There are two main libraries we'll be using:
<br>[**pytransform3d**](https://github.com/rock-learning/pytransform3d): This library has great functions for visualizations and transformations in the 3D space.
<br>[**ipympl**](https://github.com/matplotlib/ipympl): It makes the matplotlib plot interactive allowing us to perform pan, zoom, and rotation in real time within the notebook which is really helpful when working with 3D plots.

**Note:** If you're using Jupyter Lab, please install the Jupyter Lab extension of ipympl from [here](https://github.com/matplotlib/ipympl)

ipympl can be accessed in the notebook by including the magic command `%matplotlib widget`

## Contents
[**Part 1: Image Formation and Pinhole Model of the Camera**](https://medium.com/p/53872ee4ee92)
<br>
Here we discuss pinhole model of the camera and image formation. We also give an overview of camera extrinsics, camera intrinsics, and camera calibration.
<br>
<br>
[**Part 2: Camera Extrinsics in Python**](https://medium.com/p/cfe80acab8dd)
<br>
Here we discuss camera extrinsic matrix in depth including change of basis and linear transformation in rotation and translation.
<br>
<br>
[**Part 3: Camera Intrinsics in Python**](https://medium.com/p/d79bf2478c12)
<br>
Here we discuss camera intrinsic matrix, and the projection transformation of the points from camera coordinate system to the image plane of the camera.
<br>
<br>
[**Part 4: Positive Definite Matrices and Ellipsoids**](https://medium.com/p/79c2a3b397fc)
<br>
Here we discuss the properties of positive definite matrices which we'll later use in camera calibration.
<br>
<br>
[**Part 5: Camera Calibration in Python**](https://medium.com/p/5147e945cdeb)
<br>
Here we discuss the different methods of camera calibration in python with examples.
