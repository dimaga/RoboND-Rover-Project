#!python
"""Specific rover components of the behavior tree to control its decisions"""

import math
import numpy as np
from behavior_tree_basic import Node, Result

ROCKS_THRESHOLD = 10


class IsStuck(Node):
    """Returns true if the rover remains immobile for quiate a long period
    of time despite steering or throttle commands"""

    def _run(self, rover):
        return Result.Failure


IS_STUCK = IsStuck()


class GetUnstuck(Node):
    """Performs random actions to unstuck from the collision"""

    def _run(self, rover):
        return Result.Failure


GET_UNSTUCK = GetUnstuck()


class AreRocksRevealed(Node):
    """Returns true if some rocks are detected in the map"""

    def _run(self, rover):
        if np.max(rover.map.global_conf_rocks) > ROCKS_THRESHOLD:
            return Result.Success

        return Result.Failure


ARE_ROCKS_REVEALED = AreRocksRevealed()


class AnyRockClose(Node):
    """Returns true if there is some rock in the neighbourhood"""

    def _run(self, rover):
        if rover.perception.near_sample:
            return Result.Success

        return Result.Failure


ANY_ROCK_CLOSE = AnyRockClose()


class IsAnyRockLeft(Node):
    """Returns true if some rock samples are still on the ground"""

    def _run(self, rover):

        statistics = rover.statistics
        if statistics.samples_to_find > statistics.samples_collected:
            return Result.Success

        return Result.Failure


IS_ANY_ROCK_LEFT = IsAnyRockLeft()


class SetGoal(Node):
    """Sets a goal for the cost map"""

    Explore = 0
    Rock = 1
    Home = 2

    def __init__(self, goal):
        super().__init__()

        self.__goal = goal


    def _run(self, rover):
        if SetGoal.Explore == self.__goal:
            goals_mask = np.abs(rover.map.global_conf_navi) <= 1.0
        elif SetGoal.Rock == self.__goal:
            goals_mask = rover.map.global_conf_rocks > ROCKS_THRESHOLD
        elif SetGoal.Home == self.__goal:
            goals_mask = np.zeros((rover.map.global_conf_rocks.shape), np.bool)
            goals_mask[100, 100] = True

        rover.decision.cost_map = (
            goals_mask * 255
            + ~goals_mask * rover.decision.cost_map)

        return Result.Success


SET_GOAL_EXPLORE = SetGoal(SetGoal.Explore)
SET_GOAL_ROCK = SetGoal(SetGoal.Rock)
SET_GOAL_HOME = SetGoal(SetGoal.Home)


class FollowGoal(Node):
    """Follows the rover along the cost map"""

    def _run(self, rover):
        decision = rover.decision
        control = rover.control

        nav_dir_valid = np.linalg.norm(decision.nav_dir) >= 1e-1
        nav_pixels = decision.nav_pixels

        if nav_dir_valid and nav_pixels > 2000:
            angle_rad = math.atan2(decision.nav_dir[1], decision.nav_dir[0])
            control.steer = np.clip(180 * angle_rad / np.pi, -15, 15)
            control.brake = 0.0
            control.throttle = 0.2
        else:
            control.throttle = 0.0
            control.brake = 0.0
            control.steer = -15

        return Result.Success


FOLLOW_GOAL = FollowGoal()


class Stop(Node):
    """Stops the rover"""

    def _run(self, rover):
        return Result.Failure


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
