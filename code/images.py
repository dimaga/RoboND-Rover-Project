#!python
"""Keeps calibration images loaded in memory for convenience"""

#pylint: disable=import-error
import matplotlib.image as mpimg
import numpy as np

GRID = mpimg.imread('../calibration_images/example_grid1.jpg')
HEIGHT, WIDTH = GRID.shape[:2]

ROCK1 = mpimg.imread('../calibration_images/example_rock1.jpg')
ROCK1_LABEL = mpimg.imread('../calibration_images/example_rock1_mask.png')

ROCK2 = mpimg.imread('../calibration_images/example_rock2.jpg')
ROCK2_LABEL = mpimg.imread('../calibration_images/example_rock2_mask.png')

GROUND_TRUTH = mpimg.imread('../calibration_images/map_bw.png')

GROUND_TRUTH_3D = np.dstack(
    (GROUND_TRUTH * 0, GROUND_TRUTH * 255, GROUND_TRUTH * 0)).astype(np.float)