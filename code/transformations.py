#!python
"""Transformations from projective view into a top view"""

import numpy as np

# pylint: disable=import-error
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

TOP_WIDTH = WIDTH
TOP_HEIGHT = HEIGHT // 2

POINTS_TOP = np.float32([
    [0.5 * (TOP_WIDTH - PIXELS_PER_METER),
     TOP_HEIGHT - BOTTOM_OFFSET],

    [0.5 * (TOP_WIDTH + PIXELS_PER_METER),
     TOP_HEIGHT - BOTTOM_OFFSET],

    [0.5 * (TOP_WIDTH + PIXELS_PER_METER),
     TOP_HEIGHT - PIXELS_PER_METER - BOTTOM_OFFSET],

    [0.5 * (TOP_WIDTH - PIXELS_PER_METER),
     TOP_HEIGHT - PIXELS_PER_METER - BOTTOM_OFFSET]
])

PERSPECTIVE_2_TOP = cv2.getPerspectiveTransform(POINTS_PERSPECTIVE, POINTS_TOP)


def perspective_2_top(img):
    """Transforms from perspective view of the rover into the top view"""

    warped = cv2.warpPerspective(
        img,
        PERSPECTIVE_2_TOP,
        (TOP_WIDTH, TOP_HEIGHT))

    return warped


LOCAL_2_ROVER = np.array([
    [0.0, -1.0, TOP_HEIGHT],
    [-1.0, 0.0, TOP_WIDTH * 0.5]], np.float32)


def local_2_global(xpos, ypos, yaw_deg):
    """Calculate transformation 2x3 affine transformation matrix from
    local confidence map into global confidence map reference frames"""

    yaw_rad = yaw_deg * np.pi / 180
    c_scaled = np.cos(yaw_rad) / PIXELS_PER_METER
    s_scaled = np.sin(yaw_rad) / PIXELS_PER_METER

    local_2_rover_3x3 = np.vstack([LOCAL_2_ROVER, [0.0, 0.0, 1.0]])

    rover_2_global = np.array([
        [c_scaled, -s_scaled, xpos],
        [s_scaled, c_scaled, ypos]], np.float32)

    return rover_2_global.dot(local_2_rover_3x3)


IMG_ROWS, IMG_COLS = np.indices((TOP_HEIGHT, TOP_WIDTH))
IMG_ROWS = IMG_ROWS.ravel()
IMG_COLS = IMG_COLS.ravel()
IMG_ONES = np.ones_like(IMG_COLS)

LOCAL_CONF_POINTS = np.vstack([
    IMG_COLS,
    IMG_ROWS,
    IMG_ONES]).astype(np.float32).T

ROVER_CONF_POINTS = LOCAL_2_ROVER.dot(LOCAL_CONF_POINTS.T).T

ROVER_CONF_DIRS = ROVER_CONF_POINTS / np.linalg.norm(
    ROVER_CONF_POINTS,
    axis=1).reshape(-1, 1)

np.nan_to_num(ROVER_CONF_DIRS, False)


def warp_angle180(angle_deg):
    """Warps an angle to be in a range of [-180, 180]"""

    angle_deg = angle_deg % 360.0

    if angle_deg > -180:
        angle_deg -= 360.0

    if angle_deg < -180:
        angle_deg += 360.0

    return angle_deg


def main():
    """Shows results of what the module does if run as a separate application"""

    plt.figure(figsize=(6, 6))

    plt.subplot(211)
    plt.imshow(GRID)

    plt.subplot(212)

    plt.ylim(-160, 160)
    plt.xlim(-160, 160)

    top_view = perspective_2_top(GRID)
    top_view_gray = cv2.cvtColor(top_view, cv2.COLOR_RGB2GRAY)

    plt.pcolor(
        ROVER_CONF_POINTS[:, 0].reshape(top_view_gray.shape),
        ROVER_CONF_POINTS[:, 1].reshape(top_view_gray.shape),
        top_view_gray,
        cmap="gray")

    plt.show()
    return


if __name__ == '__main__':
    main()
