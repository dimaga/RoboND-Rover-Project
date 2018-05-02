#!python
"""Specific rover components of the behavior tree to control its decisions"""

from behavior_tree_basic import Node, Result

class IsStuck(Node):
    """Returns true if the rover remains immobile for quiate a long period
    of time despite steering or throttle commands"""

    def run(self, rover):
        return Result.Failure

IS_STUCK = IsStuck()


class GetUnstuck(Node):
    """Performs random actions to unstuck from the collision"""

    def run(self, rover):
        return Result.Failure

GET_UNSTUCK = GetUnstuck()


class AreRocksRevealed(Node):
    """Returns true if some rocks are detected in the map"""

    def run(self, rover):
        return Result.Failure

ARE_ROCKS_REVEALED = AreRocksRevealed()


class AnyRockClose(Node):
    """Returns true if there is some rock in the neighbourhood"""

    def run(self, rover):
        return Result.Failure

ANY_ROCK_CLOSE = AnyRockClose()


class IsAnyRockLeft(Node):
    """Returns true if some rock samples are still on the ground"""

    def run(self, rover):
        return Result.Failure

IS_ANY_ROCK_LEFT = IsAnyRockLeft()


class SetGoal(Node):
    """Sets a goal for the cost map"""

    Explore = 0
    Rock = 1
    Home = 2

    def __init__(self, goal):
        self.__goal = goal


    def run(self, rover):
        return Result.Failure


SET_GOAL_EXPLORE = SetGoal(SetGoal.Explore)
SET_GOAL_ROCK = SetGoal(SetGoal.Rock)
SET_GOAL_HOME = SetGoal(SetGoal.Home)


class FollowGoal(Node):
    """Follows the rover along the cost map"""

    def run(self, rover):
        return Result.Failure

FOLLOW_GOAL = FollowGoal()


class Stop(Node):
    """Stops the rover"""

    def run(self, rover):
        return Result.Failure

STOP = Stop()


class PickRock(Node):
    """Picks a rock sample from the ground"""

    def run(self, rover):
        return Result.Failure

PICK_ROCK = PickRock()
