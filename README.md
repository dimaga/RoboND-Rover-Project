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
adjusting the thresholds, which is more a black art than
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

The purpose of the `cost_map` is to have the rover reach some goal rather than
aimlessly follow navigable terrain. For example, for environment exploration,
pixels, which have low absolute confidence in the global confidence map,
have the highest value in `cost_map`. All the other values are blurred with
`cv2.filter2D()` to let the cost propagate across neighbouring pixels. 

For the given location of the rover, `cost_map` patch is transformed into the
top view area of the local rover reference frame, and is masked with obstacles.
The patch consists of both forward and backward pixels.

The patch is made of a circular shape so that all the directions are treated
equally. Additionally, I apply a forward direction gradient so that the rover
tends to move forward rather than rotate. All the masks and gradients are
implemented in `prepare_forward_mask()` function of `perception.py`

In `control.py`, `navi_direction()` calculates a forward direction vector.
This function differs from its notebook version. Instead of a weighted vector
summation, I fill the histogram of orientations, taking the vector with the
maximum histogram value. Such an approach overcomes a problem with direction
distributions that have multiple peaks. Instead of taking a global average
direction which may lead to forward collision, I take the the direction
corresponding to one of the peaks.

`decision_step()` method in `decision.py` runs a Behavior Tree to control the
rover behavior. The theory behind the Behavior Tree structure is well explained
in Artificial Intelligence for Games by Ian Millington, Secon Edition
(Chapter 5.4, pg 334). The structure of the rover behavior tree is shown below:

```
[Selection("Root")]
 [Sequence("Take All Rocks")]
  [IsAnyRockLeft]
  [Selection("Explore, Unstuck, Take")]
   [UntilFail]
    [Sequence("Unstuck")]
     [IsStuck]
     [GetUnstuck]
   [Selection("Take Rock")]
    [Sequence("Follow Rock Loop")]
     [AreRocksRevealed]
     [Not]
      [IsRockPickable]
     [Selection("Approach or Follow Rock")]
      [SlowlyFollowRock]
      [Sequence("Follow Rock")]
       [SetGoal(Goal.Rock)]
       [Selection("Follow Goal or Rotate")]
        [FollowGoal]
        [Sequence("Rotate To Goal")]
         [Stop]
         [Rotate]
    [Sequence("Pick Up Rock")]
     [IsRockPickable]
     [Stop]
     [PickRock]
   [Sequence("Explore")]
    [Not]
     [AreRocksRevealed]
    [SetGoal(Goal.Explore)]
    [Selection("Follow Goal or Rotate")]
     [FollowGoal]
     [Sequence("Rotate To Goal")]
      [Stop]
      [Rotate]
 [Selection("Follow Home")]
  [UntilFail]
   [Sequence("Unstuck")]
    [IsStuck]
    [GetUnstuck]
  [Sequence("Follow Home Loop")]
   [Not]
    [IsStuck]
   [Not]
    [IsAnyRockLeft]
   [SetGoal(Goal.Home)]
   [Selection("Follow Goal or Rotate")]
    [FollowGoal]
    [Sequence("Rotate To Goal")]
     [Stop]
     [Rotate]
   [UntilFail]
    [Sequence("Stay Home Forever")]
     [IsAtHomePoint]
     [Stop]
```

The rover with the behavior tree, should explore the environment, picking rocks
as it encounters them, and return to the home, which is the middle of the map.
If the rover is stuck, its `GetUnstuck` behavior node is activated, which
sends random steering and throttle to the rover until it gets unstuck.

Basic building blocks of the behavior tree are implemented in
`behavior_tree_basic.py` and covered with unit tests. Rover specific behavior
nodes are implemented in `behavior_tree_rover.py`. Each class, representing a
Node, has a comment, explaining its purpose.

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.

During my testing, I launched the simulator in 1024x768 with Good Graphics
Quality on my MacBook Pro, with 2.6 Ghz Intel Core i7 processor.

Sometimes, the rover was able to pick all six rocks and return home. Though, it
could take quite a long (more than 2000 seconds).

[final_result]: ./misc/final_result.png
![final_result]

The rover may sometimes get stuck forever in collisions, continue forever
exploring only half of the map, or be unable to pick some of the rocks, which
are placed in narrow corridors

In this project, I tried techniques that are too complex for this simple rover
environment. As a result, I ended up with "variance" trap. Much more time should
be spent on various local adjustments and probable bug fixing of all the code.

If would be great if the simulator allowed quick retrace of its physics with
final statistics output, so that we could compare different parameters, without
waiting too long for the final outcome. Also, it would be great if it were
possible to retrace the history of motion starting from some intermediate state
with up-dated code.

Here are the techniques that could be used to further improve the quality of the
rover:
   
* Apply Model Predictive Controller (MPC) sampling rather than simple steering
with a constant throttle. Calculate the sum of `cost_map` values across the
sampled trajectories, evaluating distances to potential obstacles. MPC would
allow to take into account current dynamic state of the rover, and to generate
more complex maneuvers. 
  
* Apply more advanced transformation matrix in `cv2.warpPerspective` to deal
with arbitrary pitch and roll angle values

* Detect rocks in the original view with a blob detector. Calculate distances 
and directions to them from positions and sizes of the blobs. Project restored
3D coordinates in the top view to more accurately locate the rocks. Since
rock pixels do not fully belong to the ground plane, they are not projected
correctly by `warpPerspective()` transformation

* Do not update global confidence map with similar measurements that come from
stationary rover position. Apply changes only when the rover sufficiently
changes its existing position or orientation

* Project only obstacle boundaries into the top view rather than the whole area.
Most obstacle pixels are also not part of the ground plane, which is the source
of errors. After this fix, the top view area of the navigable terrain could be
extended. Unfortunately, my naive implementation of this approach failed:
obstacle boundaries turned out to be very thin so that they were quickly washed
out by misdetected navigable pixels
