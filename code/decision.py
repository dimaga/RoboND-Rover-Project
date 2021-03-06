#!python
"""Implements decision making algorithms, defining controls of the robot"""

from behavior_tree_basic import UntilFail, Not, Sequence, Selection

from behavior_tree_rover import \
    IS_STUCK, \
    GET_UNSTUCK, \
    ARE_ROCKS_REVEALED, \
    IS_ROCK_PICKABLE, \
    IS_ANY_ROCK_LEFT, \
    SLOWLY_FOLLOW_ROCK, \
    SET_GOAL_EXPLORE, \
    SET_GOAL_ROCK, \
    SET_GOAL_HOME, \
    FOLLOW_GOAL, \
    ROTATE, \
    STOP, \
    PICK_ROCK, \
    IS_AT_HOME_POINT


def create_behavior_tree():
    """Create a Behavior Tree to control complex rover behavior"""

    sequence = Selection("Root")
    sequence.append(take_all())
    sequence.append(follow_home())

    return sequence


def loop_unstuck():
    """Creates unstuck behavior"""

    sequence = Sequence("Unstuck")
    sequence.append(IS_STUCK)
    sequence.append(GET_UNSTUCK)

    return UntilFail(sequence)


LOOP_UNSTUCK = loop_unstuck()


def follow_goal_or_rotate():
    """Follows the goal and if no navigable pixels are available,
    turns around"""

    result = Selection("Follow Goal or Rotate")
    result.append(FOLLOW_GOAL)
    result.append(rotate_to_goal())
    return result


def rotate_to_goal():
    """Rotates the rover in the direction of goal"""

    result = Sequence("Rotate To Goal")
    result.append(STOP)
    result.append(ROTATE)

    return result


FOLLOW_GOAL_OR_ROTATE = follow_goal_or_rotate()


def take_all():
    """Create a subtree to search and collect rock samples"""
    sequence = Sequence("Take All Rocks")

    sequence.append(IS_ANY_ROCK_LEFT)
    sequence.append(explore_unstuck_take())

    return sequence


def explore_unstuck_take():
    """Create a subtree to select between map exploration, unstucking and taking
    rocks"""

    result = Selection("Explore, Unstuck, Take")
    result.append(LOOP_UNSTUCK)
    result.append(take())
    result.append(loop_explorer())

    return result


def loop_explorer():
    """Create a subtree to explore the map until any rock is found"""

    sequence = Sequence("Explore")

    sequence.append(Not(ARE_ROCKS_REVEALED))
    sequence.append(SET_GOAL_EXPLORE)
    sequence.append(FOLLOW_GOAL_OR_ROTATE)

    return sequence


def take():
    """Create a subtree to approach and take a rock"""

    result = Selection("Take Rock")
    result.append(follow_rock_loop())
    result.append(pick_up_rock())

    return result


def pick_up_rock():
    """Picks up a rock if it is possible to do so"""

    result = Sequence("Pick Up Rock")

    result.append(IS_ROCK_PICKABLE)
    result.append(STOP)
    result.append(PICK_ROCK)

    return result


def follow_rock_loop():
    """Create a subtree to run the loop, approaching a rock"""

    sequence = Sequence("Follow Rock Loop")

    sequence.append(ARE_ROCKS_REVEALED)
    sequence.append(Not(IS_ROCK_PICKABLE))
    sequence.append(approach_or_follow_rock())

    return sequence


def approach_or_follow_rock():
    """Makes a decision whether to approach or follow the rock"""

    selection = Selection("Approach or Follow Rock")
    selection.append(SLOWLY_FOLLOW_ROCK)
    selection.append(follow_rock())

    return selection


def follow_rock():
    """Follows the distant rock"""

    sequence = Sequence("Follow Rock")

    sequence.append(SET_GOAL_ROCK)
    sequence.append(FOLLOW_GOAL_OR_ROTATE)

    return sequence


def follow_home():
    """Create a subtree to return home and get unstuck if the rover is stuck
    along the way"""

    result = Selection("Follow Home")
    result.append(LOOP_UNSTUCK)
    result.append(follow_home_loop())

    return result


def follow_home_loop():
    """Create a subtree to run the loop, returning home"""

    sequence = Sequence("Follow Home Loop")

    sequence.append(Not(IS_STUCK))
    sequence.append(Not(IS_ANY_ROCK_LEFT))
    sequence.append(SET_GOAL_HOME)
    sequence.append(FOLLOW_GOAL_OR_ROTATE)
    sequence.append(stay_home_forever())

    return sequence


def stay_home_forever():
    """If rover reaches the destination point, stay there forever"""

    sequence = Sequence("Stay Home Forever")
    sequence.append(IS_AT_HOME_POINT)
    sequence.append(STOP)

    return UntilFail(sequence)


ROOT = create_behavior_tree()
print(ROOT.dump())


def decision_step(rover):
    """Run decision, determining throttle, brake and steer commands based on
    the output of the perception_step() function"""
    ROOT.run(rover)
    return rover
