#!python
"""Implements map classification and rock detection algorithms"""

import numpy as np
import cv2
import transformations
import classifiers
import control


def perception_step(rover):
    """Perform perception steps to update rover"""

    perception = rover.perception

    aligned_to_ground = (
        abs(transformations.warp_angle180(perception.pitch_deg) < 0.5
            and transformations.warp_angle180(perception.roll_deg)) < 0.5)

    img = perception.img

    loc_2_glob = transformations.local_2_global(
        perception.pos[0],
        perception.pos[1],
        perception.yaw_deg)

    rocks = classifiers.ROCKS.predict(img)
    rocks_top = transformations.perspective_2_top(rocks)

    navi = classifiers.NAVI.predict(img)
    navi_top = transformations.perspective_2_top(navi)

    r_map = rover.map

    if aligned_to_ground:
        r_map.global_conf_cur.fill(0)

        cv2.warpAffine(
            navi_top,
            loc_2_glob,
            (r_map.global_conf_cur.shape[1], r_map.global_conf_cur.shape[0]),
            r_map.global_conf_cur)

        # Clipping to prevent the map from being overconfident
        r_map.global_conf_navi = np.clip(
            r_map.global_conf_navi + r_map.global_conf_cur,
            -255.0,
            255.0)

        r_map.global_conf_cur.fill(0)

        cv2.warpAffine(
            rocks_top,
            loc_2_glob,
            (r_map.global_conf_cur.shape[1], r_map.global_conf_cur.shape[0]),
            r_map.global_conf_cur)

        # Clipping to prevent the map from being overconfident
        r_map.global_conf_rocks = np.clip(
            r_map.global_conf_rocks + r_map.global_conf_cur,
            -255.0,
            255.0)

        rocks_mask = r_map.global_conf_rocks > 0

        r_map.worldmap[:, :, 0] = np.maximum(
            255 * rocks_mask,
            -r_map.global_conf_navi * (r_map.global_conf_navi < 0))

        r_map.worldmap[:, :, 1] = 255 * rocks_mask

        r_map.worldmap[:, :, 2] = np.maximum(
            255 * rocks_mask,
            r_map.global_conf_navi * (r_map.global_conf_navi > 0))

    decision = rover.decision

    decision.nav_pixels = np.sum((navi_top > 0).ravel())
    decision.nav_dir = control.navi_direction(navi_top, False)

    r_map.vision_image[:, :, 0] = 255 * (navi_top < 0)
    r_map.vision_image[:, :, 1] = 255 * (rocks_top > 0)
    r_map.vision_image[:, :, 2] = 255 * (navi_top > 0)

    return rover
