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

Instead of comparing colors against explicit thresholds, I have tought
a few machine learning classifiers from sklearn library with labelled images.
Such an approach should scale better to realistic environment, where color
classification can also be substituted with fully convolutional deep neural
network. Also, machine learning relieved me from tedious work of manually
adjusting the thresholds, which includes more of a black art rather than
engineering

[example_rock1]: ./calibration_images/example_rock1.jpg
[example_rock1_mask]: ./calibration_images/example_rock1_mask.png

![example_rock1]
![example_rock1_mask]

In my labelled images, like the one above, blue values of 255 highlight
obstacles; green values of 255 highlight navigable terrain; and red values of
255 highlight rock pixels. Additionally, I mark sky areas and areas that are
boundaries between objects with black color, meaning that their class cannot be
determined robustly. I let machine learning algorithms smooth these areas in the
output the way they prefer.
 
`GaussianMixture` classifier is trained to detect rock pixels. `GaussianNB` is
trained to detect navigable pixels.

Instead of boolean values, all the classifiers return confidence values
expressed as logarithms of positive probability over negative probability
ratios to smoothly update confidence maps. Details are explained in additional
cells and code comments inside the
[notebook](./code/Rover_Project_Test_Notebook.ipynb).

A confidence value close to zero means that the class of a pixel is unknown.
The higher a positive value is, the more confidence there is that the pixel
belongs to a positive class (e.g., navigable terrain or a rock sample to be
collected). The lower a negative value is, the more confidence there is that the
pixel belongs to a negative class (e.g., an obstacle or 'a lack of rock
object').

[classification]: ./misc/classification.png
![classification]

#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.

Confidence values are first extracted from the original image by sklearn
algorithms. Then, they are projected into the top view by
`cv2.warpPerspective()`. This allows me to deal more naturally with pixels in
the top view, not covered by the original view: they initially hold zeros, which
means their class is unknown.

Confidence values, corresponding to distant areas, are mixed together with
billinear filtering implemented inside `cv2.warpPerspective()`. Therefore,
if two pixels of two opposite classes in the original view are projected into
the same pixel in the top view, the magnitude of their bilinear mixture will be
smaller, meaning that the pixel is less likely belong to any particular class.
This also leads to natural processing of higher uncertainty for distant values.

Confidence values of the local map are transformed into the global confidence
map space with `cv2.warpAffine()` function. In Python code, only the entries
of the transformation matrix are calculated, and all per-pixel operations are
dedicated to fast C++ implementation of `cv2.warpAffine()`.

[classification_top]: ./misc/classification_top.png
![classification_top]

In the right picture above, gray pixels hold zero values, black pixels are
negative and white pixels are positive.

After the transformation, local map confidence values are added to the
corresponding global confidence map values. As a result, zeros from the local
map leave corresponding values of the global confidence map unchanged.

After each update, global confidence map pixels are clipped to the range 
[-255, 255] so as to prevent overconfidence and numerical issues.

All the code of the notebook is cleaned up and refactored to follow PEP-8
standard.

The result of the lab can be seen in the picture below:

[lab_result]: ./misc/lab_result.png
![lab_result]


### Autonomous Navigation and Mapping

In order to learn better how the starter code works, I have refactored it. All
pylint warnings have been resolved or disabled. Code duplications are removed by
extracting corresponding methods and modules.

Before adding any new code, I prepared unit tests. I am a big fan of Test Driven
Development.

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

`perception_step()` is based on machine learning classifiers, similar to
notebook. So, I have just copy-pasted the notebook code and applied
recommendations, encountered in the manual and youtube videos:

* The global confidence map is only updated when both the roll and the pitch
rover angles are close to zero

* Only the bottom half of the original top view is used in order to exclude
distant pixels, which are subject to noise

In the notebook lab, I took an attempt to deal with noise more naturally by
applying machine learning algorithms with probabilistic filters. However, it
turned out not to be enough due to the following reasons:

1. Mountains do not lay in the ground plane. Therefore, they may be projected
into the top view incorrectly, occluding navigable terrain and rock pixels
behind them.

2. Probabilistic filters are built on the assumption that input measurements
adjacent in time are independent. However, in our case, measurements coming
from the same camera position from the synthetic environment are very correlated

Besides enhancements recommended by other people, I have added a `cost_map`
array, which implements a basic version of Value Iteration algorithm. `cost_map`
is used instead of navigable map to control the steering angle of the rover. The
`cost_map` is created to be the size of the global map.

The purpose of the `cost_map` is to have the rover explore the environment
rather than aimlessly follow the navigable terrain. Inside the `cost_map`,
values, which have low absolute confidence in the global confidence map,
have the highest rank. All the other values are blurred with a
`cv2.boxFilter()`. 

For the given location of the rover, its patch is transformed into the top view
area of the local rover reference frame, and is masked with obstacles.  

In my current implementation, I haven't changed the logic of `decision_step()`,
adjusting only stop and go thresholds.

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.

During my testing, I launched the simulator in 1024x768 with Good Graphics
Quality on my MacBook Pro, with 2.6 Ghz Intel Core i7 processor.

The rover was able to cover the area of more than 40% map with fidelity of
higher than 60%, finding a few rock samples a long the way, which meets the
passing submission criteria for this project.

[final_result]: ./misc/final_result.png
![final_result]

The rover may sometimes get stuck in the mountaneous rocks, or roll loops,
never exploring the rest of the map. Without the `cost_map`, the area of
exploration would be much smaller, though. 

[got_stuck]: ./misc/got_stuck.png
![got_stuck]

Here are the techniques that could be used to further improve the quality of the
rover:

* Implement more advanced behaviors with a Behavior Tree
    * Detect by analyzing position of the rover if it is stuck somewhere so as
    to initiate unstuck behavior with a set of random actions

    * Add a goal to search for rocks or initial position to `cost_map`. Let the
    rover collect rocks and return home with a more complex Behavior Tree
    
* Detect rocks in the original view with a blob detector. Calculate distances 
and directions to them from positions and sizes of the blobs. Project restored
3D coordinates in the top view to more accurately locate the rocks. Since
rock pixels do not fully belong to the ground plane, they are not projected
correctly by `warpPerspective()` transformation

* Project only obstacle boundaries into the top view rather than the whole area.
Most obstacle pixels are also not part of the ground plane, which is the source
of errors. After this fix, the top view area of the navigable terrain could be
extended. Unfortunately, my naive implementation of this approach failed:
obstacle boundaries turned out to be very thin so that they were quickly washed
out by misdetected navigable pixels

* Apply more advanced control of the rover so that it better follows the
`cost_map` in the direction of the maximum cost. Consider delays in between
steering command and the actual change in position with a PD or MPC controllers

* Since the time the picture from the rover camera is taken may not exactly
correspond to roll and pitch times, analyze how roll and pitch values change.
Update global confidence map one only if roll and pitch values remain constant
for a while. That is test that pitch and roll derivatives are also close to zero

* Apply more advanced transformation matrix in `cv2.warpPerspective` to deal
with arbitrary pitch and roll angle values

* Do not update global confidence map with similar measurements that come from
stationary rover position. Apply changes only when the rover sufficiently
changes its existing position or orientation