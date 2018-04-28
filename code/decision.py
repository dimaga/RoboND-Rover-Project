#!python
"""Implements decision making algorithms, defining controls of the robot"""

import math

import numpy as np


def decision_step(rover):
    """Decision tree, determining throttle, brake and steer commands based on
    the output of the perception_step() function"""

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if rover.decision.nav_dir is not None:

        nav_dir_valid = np.linalg.norm(rover.decision.nav_dir) >= 1e-1
        nav_pixels = rover.decision.nav_pixels

        # Check for rover.decision.rover status
        if rover.decision.mode == 'forward':
            # Check the extent of navigable terrain
            if nav_dir_valid and nav_pixels >= rover.constants.stop_forward:
                # If mode is forward, navigable terrain looks good
                # and velocity is below max, then throttle
                if rover.perception.vel < rover.constants.max_vel:
                    # Set throttle value to throttle setting
                    rover.control.throttle = rover.constants.throttle_set
                else:  # Else coast
                    rover.control.throttle = 0
                rover.control.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                rover.control.steer = nav_dir_2_steer(rover)
            # If there's a lack of navigable terrain pixels then go to 'stop'
            # mode
            else:
                # Set mode to "stop" and hit the brakes!
                rover.control.throttle = 0
                # Set brake to stored brake value
                rover.control.brake = rover.constants.brake_set
                rover.control.steer = 0
                rover.decision.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif rover.decision.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if rover.perception.vel > 0.2:
                rover.control.throttle = 0
                rover.control.brake = rover.constants.brake_set
                rover.control.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            else:

                # If we're stopped but see sufficient navigable terrain in front
                # then go!
                if nav_dir_valid and nav_pixels >= rover.constants.go_forward:
                    # Set throttle back to stored value
                    rover.control.throttle = rover.constants.throttle_set
                    # Release the brake
                    rover.control.brake = 0
                    # Set steer to mean angle
                    rover.control.steer = nav_dir_2_steer(rover)
                    rover.decision.mode = 'forward'

                else:
                    # Now we're stopped and we have vision data to
                    # see if there's a path forward
                    rover.control.throttle = 0
                    # Release the brake to allow turning
                    rover.control.brake = 0

                    # Turn range is +/- 15 degrees, when stopped the
                    # next line will induce 4-wheel turning

                    # Could be more clever here about which way to turn
                    rover.control.steer = -15

    # Just to make the rover do something
    # even if no modifications have been made to the code
    else:
        rover.control.throttle = rover.constants.throttle_set
        rover.control.steer = 0
        rover.control.brake = 0

    # If in a state where want to pickup a rock send pickup command
    if rover.perception.near_sample and \
            abs(rover.perception.vel) < 1e-4 and \
            not rover.control.picking_up:
        rover.control.send_pickup = True

    return rover


def nav_dir_2_steer(rover):
    """Converts nav_dir direction vector in the recommended steer command"""

    angle_rad = math.atan2(rover.decision.nav_dir[1], rover.decision.nav_dir[0])
    return np.clip(180 * angle_rad / np.pi, -15, 15)
