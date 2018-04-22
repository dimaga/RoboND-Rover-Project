#!python
"""Keeps calibration images loaded in memory for convenience"""

#pylint: disable=import-error
import matplotlib.image as mpimg

GRID = mpimg.imread('../calibration_images/example_grid1.jpg')
HEIGHT, WIDTH = GRID.shape[:2]

ROCK1 = mpimg.imread('../calibration_images/example_rock1.jpg')
ROCK1_LABEL = mpimg.imread('../calibration_images/example_rock1_mask.png')

ROCK2 = mpimg.imread('../calibration_images/example_rock2.jpg')
ROCK2_LABEL = mpimg.imread('../calibration_images/example_rock2_mask.png')
