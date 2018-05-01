## Search and Sample Return

My solution of RoboND-Rover-Project assignment from Udacity Robotics Nanodegree
course, Term 1. See project assignment starter code in
https://github.com/udacity/RoboND-Rover-Project

[rover_image]: ./misc/rover_image.jpg
![alt text][rover_image]

---


**The goals / steps of this project are the following:**

**Training / Calibration**

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing
steps (perspective transform, color threshold etc.) to get from raw images to a
map.  The `output_image` you create in this step should demonstrate that your
mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the
`process_image()` function.  Include the video you produce as part of your
submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script
with the appropriate image processing functions to create a map and update
`Rover()` data (similar to what you did with `process_image()` in the notebook).
* Fill in the `decision_step()` function within the `decision.py` script with
conditional statements that take into consideration the outputs of the
`perception_step()` in deciding how to issue throttle, brake and steering
commands.
* Iterate on your perception and decision function until your rover does a
reasonable (need to define metric) job of navigating and mapping.

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points

Here I will consider the rubric points individually and describe how I addressed
each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

Instead of providing explicit thresholds for color selection, I have trained
machine learning classifiers from sklearn library against labelled images:

[example_rock1]: ./calibration_images/example_rock1.jpg
[example_rock1_mask]: ./calibration_images/example_rock1_mask.png

![example_rock1]
![example_rock1_mask]

`GaussianMixture` classifier is trained to detect rock pixels. `GaussianNB` is
trained to detect navigable pixels.

Instead of boolean values, all the classifiers return confidence values
expressed as logarithms of positive probability over negative probability
ratios to smoothly update confidence maps. Details are explained in additional
cells and code comments inside the
[notebook](./code/Rover_Project_Test_Notebook.ipynb).

If a confidence value is close to zero, that means the class of the pixel is
unknown. The higher a positive value is, the more confidence there is that the
pixel belongs to the positive class (e.g., navigable terrain or a rock sample to
be collected). The lower a negative value is, the more confidence there is  that
the pixel belongs to the negative class (e.g., an obstacle or 'a lack of rock
object'). 

#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.

Confidence values, represented by log-probability ratios, are extracted from the
original image, and then projected into the top view. This allows to deal more
naturally with pixels in the top view, not covered by the original
view: they just hold zeros, which means their status is unknown. Also,
confidence values, corresponding to distant areas, are mixed together with
billinear filtering. Therefore, if pixels of two classes in the original view
are located cloase together, their mixed confidence values will have smaller
magnitude in the top view, meaning that they are less likely belong to a
particular class.

Confidence values of the local map are transformed into the global confidence
map space with `cv2.warpAffine()` function. In Python code, only the entries
of the transformation matrix are calculated, and all per-pixel operations are
dedicated to fast C++ implementation of `cv2.warpAffine()`.

After the transformation, local map confidence values are added to the
corresponding global confidence map values. Therefore, zeros from the local map
leave corresponding values of the global confidence map unchanged.

After each update, global confidence map pixels are clipped to be in the region
of [-255, 255] so as to prevent overconfidence and numerical issues.

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.


#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.


