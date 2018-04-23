#!python
"""Implements map classification and rock detection algorithms"""

import numpy as np
import cv2
import transformations
import classifiers
import control

def perception_step(rover):
    """Perform perception steps to update rover()"""

    img = rover.img

    loc_2_glob = transformations.local_2_global(
        rover.pos[0],
        rover.pos[1],
        rover.yaw)

    rocks = classifiers.ROCKS.predict(img)
    rocks_top = transformations.perspective_2_top(rocks)
    rover.global_conf_cur.fill(0)

    cv2.warpAffine(
        rocks_top,
        loc_2_glob,
        (rover.global_conf_cur.shape[1], rover.global_conf_cur.shape[0]),
        rover.global_conf_cur)

    # Clipping to prevent the map from being overconfident
    rover.global_conf_rocks = np.clip(
        rover.global_conf_rocks + rover.global_conf_cur,
        -10.0,
        10.0)

    navi = classifiers.NAVI.predict(img)
    navi_top = transformations.perspective_2_top(navi)

    rover.global_conf_cur.fill(0)

    cv2.warpAffine(
        navi_top,
        loc_2_glob,
        (rover.global_conf_cur.shape[1], rover.global_conf_cur.shape[0]),
        rover.global_conf_cur)

    # Clipping to prevent the map from being overconfident
    rover.global_conf_navi = np.clip(
        rover.global_conf_navi + rover.global_conf_cur,
        -255.0,
        255.0)

    rover.nav_dir = control.navi_direction(navi_top, False)
    rover.nav_pixels = np.sum((navi_top > 0).ravel())

    rover.vision_image[:, :, 0] = 255 * (navi_top < 0)
    rover.vision_image[:, :, 1] = 255 * (rocks_top > 0)
    rover.vision_image[:, :, 2] = 255 * (navi_top > 0)

    rocks_mask = rover.global_conf_rocks > 0

    rover.worldmap[:, :, 0] = np.maximum(
        255 * rocks_mask,
        -rover.global_conf_navi * (rover.global_conf_navi < 0))

    rover.worldmap[:, :, 1] = 255 * rocks_mask

    rover.worldmap[:, :, 2] = np.maximum(
        255 * rocks_mask,
        rover.global_conf_navi * (rover.global_conf_navi > 0))

    return rover
