#!python
"""Transformations from projective view into a top view"""

import numpy as np

#pylint: disable=import-error
import matplotlib.pyplot as plt
import cv2

from images import WIDTH, HEIGHT, GRID

PIXELS_PER_METER = 10.0
BOTTOM_OFFSET = 6

POINTS_PERSPECTIVE = np.float32([
    [14, 140],
    [301, 140],
    [200, 96],
    [118, 96]
])

POINTS_TOP = np.float32([
    [0.5 * (WIDTH - PIXELS_PER_METER), HEIGHT - BOTTOM_OFFSET],
    [0.5 * (WIDTH + PIXELS_PER_METER), HEIGHT - BOTTOM_OFFSET],
    [0.5 * (WIDTH + PIXELS_PER_METER), HEIGHT - PIXELS_PER_METER - BOTTOM_OFFSET],
    [0.5 * (WIDTH - PIXELS_PER_METER), HEIGHT - PIXELS_PER_METER - BOTTOM_OFFSET]
])

PERSPECTIVE_2_TOP = cv2.getPerspectiveTransform(POINTS_PERSPECTIVE, POINTS_TOP)


def perspective_2_top(img):
    """Transforms from perspective view of the rover into the top view"""

    warped = cv2.warpPerspective(
        img,
        PERSPECTIVE_2_TOP,
        (img.shape[1], img.shape[0]))  # keep same size as input image

    return warped


def main():
    """Shows results of what the module does if run as a separate application"""

    plt.figure(figsize=(6, 6))

    plt.subplot(211)
    plt.imshow(GRID)

    plt.subplot(212)
    plt.imshow(perspective_2_top(GRID))

    plt.show()
    return


if __name__ == '__main__':
    main()
