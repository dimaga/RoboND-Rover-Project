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
        update_global(loc_2_glob, r_map, navi_top, r_map.global_conf_navi)
        update_global(loc_2_glob, r_map, rocks_top, r_map.global_conf_rocks)

        rocks_mask = r_map.global_conf_rocks > 0

        r_map.worldmap[:, :, 0] = np.maximum(
            255 * rocks_mask,
            -r_map.global_conf_navi * (r_map.global_conf_navi < 0))

        r_map.worldmap[:, :, 1] = 255 * rocks_mask

        r_map.worldmap[:, :, 2] = np.maximum(
            255 * rocks_mask,
            r_map.global_conf_navi * (r_map.global_conf_navi > 0))

    decision = rover.decision

    goals_mask = create_goal_to_explore(r_map)
    update_cost_map(decision, goals_mask)

    direction_map = extract_local_cost_map(decision, loc_2_glob)
    direction_map[:] *= (navi_top > 0)
    direction_map[:] += (navi_top < 0) * -255.0

    decision.nav_pixels = np.sum((navi_top > 0).ravel())
    decision.nav_dir = control.navi_direction(direction_map, False)

    min_y = navi_top.shape[0] - r_map.vision_image.shape[0]

    r_map.vision_image[min_y:, :, 0] = -direction_map * (direction_map < 0)
    r_map.vision_image[min_y:, :, 1] = 255 * (rocks_top > 0)
    r_map.vision_image[min_y:, :, 2] = direction_map * (direction_map > 0)

    return rover


def extract_local_cost_map(decision, loc_2_glob):
    """Returns a patch of the cost map in local coordinates"""

    glob_2_loc = np.linalg.inv(np.vstack([loc_2_glob, [0.0, 0.0, 1.0]]))[:2, :]

    local_cost_map = cv2.warpAffine(
        decision.cost_map,
        glob_2_loc,
        (transformations.TOP_WIDTH, transformations.TOP_HEIGHT))

    return local_cost_map


def update_cost_map(decision, goals_mask):
    """Recalculate the state of the cost_map, using value iteration algorithm"""

    decision.cost_map = (
        goals_mask * 255.0
        + (~goals_mask * 0.99 * cv2.boxFilter(decision.cost_map, -1, (3, 3))))

    decision.cost_map[:] = np.maximum(decision.cost_map[:], 0.1)


def create_goal_to_explore(r_map):
    """Masks unexplored areas in the confidence map that need to be explored"""

    return np.abs(r_map.global_conf_navi) <= 1.0


def update_global(loc_2_glob, r_map, local_map, global_map):
    """Updates global confidence map from local map"""

    r_map.global_conf_cur.fill(0)

    cv2.warpAffine(
        local_map,
        loc_2_glob,
        (r_map.global_conf_cur.shape[1], r_map.global_conf_cur.shape[0]),
        r_map.global_conf_cur)

    global_map += r_map.global_conf_cur

    # Clipping to prevent the map from being overconfident
    np.clip(global_map, -255.0, 255.0, out=global_map)
