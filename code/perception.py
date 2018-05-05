#!python
"""Implements map classification and rock detection algorithms"""

import numpy as np
import cv2

# pylint: disable=import-error
import matplotlib.pyplot as plt

import transformations
import classifiers
import control


def prepare_forward_mask():
    """Build circular weighted mask that prefers forward regions of the cost map
    over backward regions in order to break a tie in decision process"""

    distances = np.linalg.norm(transformations.ROVER_CONF_POINTS, axis=1)
    circle_mask = distances < transformations.TOP_HEIGHT // 2

    weights = 0.5 * transformations.ROVER_CONF_DIRS.dot([1.0, 0.0]) + 0.5

    return (circle_mask * weights).reshape(
        transformations.TOP_HEIGHT,
        transformations.TOP_WIDTH)


FORWARD_MASK = prepare_forward_mask()


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
    nav_top = transformations.perspective_2_top(navi)

    r_map = rover.map
    statistics = rover.statistics

    if aligned_to_ground:
        update_global(loc_2_glob, r_map, nav_top, r_map.global_conf_navi)
        update_global(loc_2_glob, r_map, rocks_top, r_map.global_conf_rocks)

        # Slowly forget rocks map to make the rover re-explore the map rather
        # than get stuck after the exploration somewhere. Note that only
        # rocks map is being forgotten, since it is used for exploration. We
        # should not forget navigable map, as mapping percent is one of the
        # passing criteria for this project
        r_map.global_conf_rocks *= 0.99

        rocks_mask = r_map.global_conf_rocks > 0

        statistics.worldmap[:, :, 0] = np.maximum(
            255 * rocks_mask,
            -r_map.global_conf_navi * (r_map.global_conf_navi < 0))

        statistics.worldmap[:, :, 1] = 255 * rocks_mask

        statistics.worldmap[:, :, 2] = np.maximum(
            255 * rocks_mask,
            r_map.global_conf_navi * (r_map.global_conf_navi > 0))

    decision = rover.decision

    update_cost_map(decision, r_map.global_conf_navi)

    glob_2_loc = np.linalg.inv(np.vstack([loc_2_glob, [0.0, 0.0, 1.0]]))[:2, :]

    direction_map = prepare_direction_map(decision, r_map, glob_2_loc)
    r_map.local_rocks = to_local_map(r_map.global_conf_rocks, glob_2_loc)
    r_map.local_navi = to_local_map(r_map.global_conf_navi, glob_2_loc)

    choose_best_direction(decision, direction_map, r_map.local_navi)

    statistics.vision_image[:, :, 0] = -direction_map * (direction_map < 0)
    statistics.vision_image[:, :, 1] = 255 * (rocks_top > 0)
    statistics.vision_image[:, :, 2] = direction_map * (direction_map > 0)

    return rover


def choose_best_direction(decision, direction_map, nav_top):
    """Find best direction for the rover motion"""

    decision.nav_dir, score = control.navi_direction(direction_map)
    decision.nav_pixels = calc_nav_pixels(decision.nav_dir, nav_top)

    # Try perpendicular decisions in case there is an obstacle in front
    if np.linalg.norm(decision.nav_dir) > 1e-1:
        left_dir = np.array([-decision.nav_dir[1], decision.nav_dir[0]])
        try_adjacent_dir(decision, score, direction_map, nav_top, left_dir)
        try_adjacent_dir(decision, score, direction_map, nav_top, -left_dir)


def try_adjacent_dir(decision, score, direction_map, nav_top, adjacent_dir):
    """Tries adjacent direction to see if it produces more navigable pixels"""

    mask_inliers = (transformations.ROVER_CONF_DIRS.dot(
        adjacent_dir) > 0.5).reshape(direction_map.shape)

    refined_dir, refined_score = control.navi_direction(
        direction_map * mask_inliers)

    if refined_score > score:
        decision.nav_dir = refined_dir
        decision.nav_pixels = calc_nav_pixels(refined_dir, nav_top)


def calc_nav_pixels(nav_dir, nav_top):
    """Calculates number of pixels along the selected direction"""

    similar_dirs = (transformations.ROVER_CONF_DIRS.dot(nav_dir) > 0.8).ravel()
    navigatable = (nav_top > 0).ravel()

    return np.sum(similar_dirs * navigatable)


def prepare_direction_map(decision, r_map, glob_2_loc):
    """Prepares a local direction map out of cost_map and navigable map to
    make decisions about steering directions to reach a distant goal"""

    direction_map = to_local_map(decision.cost_map, glob_2_loc)
    navi_map = to_local_map(r_map.global_conf_navi, glob_2_loc)

    navigable = navi_map > 0
    obstacles = navi_map < 0

    direction_map *= navigable
    direction_map += obstacles * -255.0
    direction_map *= FORWARD_MASK

    return direction_map


def to_local_map(global_map, glob_2_loc):
    """Returns a patch of the cost map in local coordinates"""

    local_cost_map = cv2.warpAffine(
        global_map,
        glob_2_loc,
        (transformations.TOP_WIDTH, transformations.TOP_HEIGHT))

    return local_cost_map


def update_cost_map(decision, global_navi_map):
    """Recalculate the state of the cost_map, using value iteration algorithm"""
    decision.cost_map *= global_navi_map > -1.0
    decision.cost_map = 0.998 * cv2.boxFilter(decision.cost_map, -1, (3, 3))
    decision.cost_map[:] = np.maximum(decision.cost_map[:], 0.1)


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


def main():
    """Shows results of what the module does if run as a separate application"""

    plt.imshow(FORWARD_MASK, cmap="gray")
    plt.show()
    return


if __name__ == '__main__':
    main()
