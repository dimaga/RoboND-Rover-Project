#!python
"""Specific rover components of the behavior tree to control its decisions"""

import math
from enum import Enum
import random
import numpy as np
import transformations
from behavior_tree_basic import Node, Result

ROCKS_THRESHOLD = 7


class IsStuck(Node):
    """Returns true if the rover remains immobile for quiate a long period
    of time despite steering or throttle commands"""

    def _run(self, rover):
        if rover.decision.stuck_pos is None:
            return Result.Failure

        if rover.decision.stuck_time is None:
            return Result.Failure

        if rover.time.total is None:
            return Result.Failure

        if rover.time.total < rover.decision.stuck_time + 20:
            return Result.Failure

        return Result.Success


IS_STUCK = IsStuck()


class GetUnstuck(Node):
    """Performs random actions to unstuck from the collision"""

    def __init__(self):
        super().__init__()
        self.__stage = 0
        self.__next_time = 0
        self.__steer = 0
        self.__throttle = 0
        self.__brake = 0


    def _run(self, rover):
        if rover.time.total is None:
            self.__stage = 0
            self.__next_time = 0
            return Result.Failure

        if rover.time.total < self.__next_time:
            rover.control.steer = self.__steer
            rover.control.throttle = self.__throttle
            rover.control.brake = self.__brake
            return Result.Continue

        if 0 == self.__stage:
            self.__next_time = rover.time.total + 3.0 * random.random()
            self.__steer = 20 if random.random() > 0.5 else -20
            self.__throttle = 0.0
            self.__brake = 0.0
        elif 1 == self.__stage:
            self.__next_time = rover.time.total + 2.0
            self.__steer = 0.0
            self.__throttle = 5.0
            self.__brake = 0.0
        elif 2 == self.__stage:
            self.__next_time = rover.time.total + 1.0
            self.__steer = 0.0
            self.__throttle = 0.0
            self.__brake = 10.0
        else:
            self.__stage = 0
            self.__next_time = 0
            return Result.Success

        self.__stage += 1
        return Result.Continue


GET_UNSTUCK = GetUnstuck()


class AreRocksRevealed(Node):
    """Returns true if some rocks are detected in the map"""

    def _run(self, rover):
        if np.max(rover.map.global_conf_rocks) > ROCKS_THRESHOLD:
            return Result.Success

        return Result.Failure


ARE_ROCKS_REVEALED = AreRocksRevealed()


class IsRockPickable(Node):
    """Returns true if there is a close rock that can be picked"""

    def _run(self, rover):
        if rover.perception.near_sample:
            return Result.Success

        return Result.Failure


IS_ROCK_PICKABLE = IsRockPickable()


class IsAnyRockLeft(Node):
    """Returns true if some rock samples are still on the ground"""

    def _run(self, rover):

        statistics = rover.statistics
        if statistics.samples_to_find > statistics.samples_collected:
            return Result.Success

        return Result.Failure


IS_ANY_ROCK_LEFT = IsAnyRockLeft()


class Goal(Enum):
    """Possible goal values for SetGoal node"""
    Explore = 0
    Rock = 1
    Home = 2


class SetGoal(Node):
    """Sets a goal for the cost map"""


    def __init__(self, goal):
        super().__init__()
        self.__goal = goal


    @property
    def name(self):
        """Returns the name of the node for debugging output"""
        return "[" + type(self).__name__ + "(" + str(self.__goal) + ")]"


    def _run(self, rover):
        if Goal.Explore == self.__goal:
            sigma = 3
            value = rover.map.global_conf_rocks / sigma
            v_sq = value * value
            goals = 255 * np.exp(-v_sq)

        elif Goal.Rock == self.__goal:
            goals = 255 * (rover.map.global_conf_rocks > ROCKS_THRESHOLD)

        elif Goal.Home == self.__goal:
            goals = np.zeros((rover.map.global_conf_rocks.shape), np.bool)
            goals[87, 98] = 255

        rover.decision.cost_map += goals
        np.minimum(255, rover.decision.cost_map, out=rover.decision.cost_map)

        return Result.Success


SET_GOAL_EXPLORE = SetGoal(Goal.Explore)
SET_GOAL_ROCK = SetGoal(Goal.Rock)
SET_GOAL_HOME = SetGoal(Goal.Home)


def nav_angle(nav_dir):
    """Steering angle dependently on navi direction"""

    angle_rad = math.atan2(nav_dir[1], nav_dir[0])
    return 180 * angle_rad / np.pi


def is_valid_nav_angle(angle_deg):
    """Checks if the given navigation angle is valid for forward motion"""

    return 35 > abs(angle_deg)


class FollowGoal(Node):
    """Follows the rover along the cost map"""

    def _run(self, rover):
        decision = rover.decision
        control = rover.control

        nav_dir_valid = np.linalg.norm(decision.nav_dir) >= 1e-1
        nav_pixels = decision.nav_pixels

        if not nav_dir_valid or nav_pixels < 500:
            return Result.Failure

        angle_deg = nav_angle(decision.nav_dir)
        if not is_valid_nav_angle(angle_deg):
            return Result.Failure

        control.steer = np.clip(angle_deg, -15, 15)
        control.brake = 0.0
        control.throttle = 0.2

        return Result.Success


FOLLOW_GOAL = FollowGoal()


class SlowlyFollowRock(Node):
    """Slowly approaches very close rock"""

    def _run(self, rover):
        r_map = rover.map

        rocks = (r_map.local_rocks > ROCKS_THRESHOLD).ravel()
        if 0 == np.sum(rocks):
            return Result.Failure

        all_distances = np.linalg.norm(
            transformations.ROVER_CONF_POINTS,
            axis=1)

        rock_distances = all_distances[rocks]
        closest_idx = np.argmin(rock_distances)
        closest_rock_dist = rock_distances[closest_idx]

        nav_dir = transformations.ROVER_CONF_DIRS[rocks][closest_idx]

        closer_pts = all_distances < min(closest_rock_dist - 7, 30)
        similar_dirs = transformations.ROVER_CONF_DIRS.dot(nav_dir) > 0.99
        pts_on_the_way = np.logical_and(closer_pts, similar_dirs)

        if np.sum(pts_on_the_way) > 0:
            obstacles = np.sum(r_map.local_navi.ravel()[pts_on_the_way] < -10)
            if obstacles > 20:
                return Result.Failure

        angle_deg = nav_angle(nav_dir)
        if is_valid_nav_angle(angle_deg):
            if rover.perception.vel > 2.0:
                rover.control.brake = 10.0
                rover.control.throttle = 0.0
            elif rover.perception.vel > 0.2:
                rover.control.brake = 0.0
                rover.control.throttle = 0.05
            else:
                rover.control.brake = 0.0
                rover.control.throttle = 0.2
        else:
            rover.control.throttle = 0.0
            if rover.perception.vel > 0.2:
                rover.control.brake = 10.0
            else:
                rover.control.brake = 0.0

        rover.control.steer = np.clip(angle_deg, -15, 15)
        return Result.Success


SLOWLY_FOLLOW_ROCK = SlowlyFollowRock()


class Rotate(Node):
    """Rotates the rover to target navigable pixels"""

    def _run(self, rover):
        decision = rover.decision
        control = rover.control

        nav_dir_valid = np.linalg.norm(decision.nav_dir) >= 1e-1

        if nav_dir_valid:
            angle_deg = nav_angle(decision.nav_dir)
            if not is_valid_nav_angle(angle_deg):
                control.throttle = 0.0
                control.brake = 0.0
                control.steer = -5 if angle_deg < 0 else 5
                return Result.Success

        control.throttle = 0.0
        control.brake = 0.0
        control.steer = -10

        return Result.Success


ROTATE = Rotate()


class Stop(Node):
    """Stops the rover"""

    def __init__(self):
        super().__init__()
        self.__time = 0


    def _run(self, rover):
        if rover.time.total is None:
            return Result.Failure

        if rover.perception.vel > 0.2:
            rover.control.throttle = 0
            rover.control.brake = 10.0
            rover.control.steer = 0.0

            self.__time = rover.time.total
            return Result.Continue

        if rover.time.total < self.__time + 3:
            # Wait for a while until the rover stops shaking to treat
            # perception correctly
            return Result.Continue

        return Result.Success


STOP = Stop()


class PickRock(Node):
    """Picks a rock sample from the ground"""

    def _run(self, rover):
        if rover.control.picking_up:
            return Result.Continue

        if rover.perception.near_sample and abs(rover.perception.vel) < 1e-4:
            rover.control.send_pickup = True
            return Result.Continue

        return Result.Success


PICK_ROCK = PickRock()
