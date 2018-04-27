#!python
"""Main file to launch the rover brain and communicate with simulator"""

import argparse
import os
import shutil
import time
from datetime import datetime

#pylint: disable=import-error
import eventlet
import eventlet.wsgi
import numpy as np
import socketio
from flask import Flask

from decision import decision_step
# Import functions for perception and decision making
from perception import perception_step
from supporting_functions import update_rover, create_output_images
from images import GROUND_TRUTH_3D

# Initialize socketio server and Flask application
# (learn more at: https://python-socketio.readthedocs.io/en/latest/)
SIO = socketio.Server()

class RoverState():
    """Define RoverState() class to retain rover state parameters"""

    #pylint: disable=too-few-public-methods

    def __init__(self):
        self.start_time = None  # To record the start time of navigation
        self.total_time = None  # To record total duration of naviagation
        self.img = None  # Current camera image
        self.pos = None  # Current position (x, y)
        self.yaw = None  # Current yaw angle
        self.pitch = None  # Current pitch angle
        self.roll = None  # Current roll angle
        self.vel = None  # Current velocity
        self.steer = 0  # Current steering angle
        self.throttle = 0  # Current throttle value
        self.brake = 0  # Current brake value
        self.nav_dir = None  # Angles of navigable terrain pixels
        self.nav_pixels = None  # Number of navigatable pixels
        self.ground_truth = GROUND_TRUTH_3D  # Ground truth worldmap
        self.mode = 'forward'  # Current mode (can be forward or stop)
        self.throttle_set = 0.2  # Throttle setting when accelerating
        self.brake_set = 10  # Brake setting when braking
        # The stop_forward and go_forward fields below represent total count
        # of navigable terrain pixels.  This is a very crude form of knowing
        # when you can keep going and when you should stop.  Feel free to
        # get creative in adding new fields or modifying these!
        self.stop_forward = 50  # Threshold to initiate stopping
        self.go_forward = 500  # Threshold to go forward again
        self.max_vel = 2  # Maximum velocity (meters/second)
        # Image output from perception step
        # Update this image to display your intermediate analysis steps
        # on screen in autonomous mode
        self.vision_image = np.zeros((160, 320, 3), dtype=np.float)
        # Worldmap
        # Update this image with the positions of navigable terrain
        # obstacles and rock samples
        self.worldmap = np.zeros((200, 200, 3), dtype=np.float)
        self.global_conf_rocks = np.zeros((200, 200)).astype(np.float)
        self.global_conf_navi = np.zeros((200, 200)).astype(np.float)
        self.global_conf_cur = np.zeros((200, 200)).astype(np.float)
        self.samples_pos = None  # To store the actual sample positions
        self.samples_to_find = 0  # To store the initial count of samples
        self.samples_located = 0  # To store number of samples located on map
        self.samples_collected = 0  # To count the number of samples collected
        self.near_sample = 0  # Will be set to telemetry value data["near_sample"]
        self.picking_up = 0  # Will be set to telemetry value data["picking_up"]
        self.send_pickup = False  # Set to True to trigger rock pickup


# Initialize our rover
ROVER = RoverState()

# Variables to track frames per second (FPS)
# Intitialize frame counter
FRAME_COUNTER = 0
# Initalize second counter
SECOND_COUNTER = time.time()
FPS = None
IMAGE_FOLDER = ''


@SIO.on('telemetry')
def telemetry(_, data):
    """Defines telemetry function for what to do with incoming data"""

    # pylint: disable=global-statement

    global FRAME_COUNTER, SECOND_COUNTER, FPS
    FRAME_COUNTER += 1
    # Do a rough calculation of frames per second (FPS)
    if (time.time() - SECOND_COUNTER) > 1:
        FPS = FRAME_COUNTER
        FRAME_COUNTER = 0
        SECOND_COUNTER = time.time()
    print("Current FPS: {}".format(FPS))

    if data:
        global ROVER
        # Initialize / update rover with current telemetry
        ROVER, image = update_rover(ROVER, data)

        if np.isfinite(ROVER.vel):

            # Execute the perception and decision steps to update the rover's
            # state
            ROVER = perception_step(ROVER)
            ROVER = decision_step(ROVER)

            # Create output images to send to server
            out_image_string1, out_image_string2 = create_output_images(ROVER)

            # The action step!  Send commands to the ROVER!

            # Don't send both of these, they both trigger the simulator
            # to send back new telemetry so we must only send one
            # back in respose to the current telemetry data.

            # If in a state where want to pickup a rock send pickup command
            if ROVER.send_pickup and not ROVER.picking_up:
                send_pickup()
                # Reset ROVER flags
                ROVER.send_pickup = False
            else:
                # Send commands to the ROVER!
                commands = (ROVER.throttle, ROVER.brake, ROVER.steer)
                send_control(commands, out_image_string1, out_image_string2)

        # In case of invalid telemetry, send null commands
        else:

            # Send zeros for throttle, brake and steer and empty images
            send_control((0, 0, 0), '', '')

        # To save camera images from autonomous driving, specify a path
        # Example: $ python drive_rover.py image_folder_path
        # Conditional to save image frame if folder was specified

        global IMAGE_FOLDER
        if IMAGE_FOLDER != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(IMAGE_FOLDER, timestamp)
            image.save('{}.jpg'.format(image_filename))

    else:
        SIO.emit('manual', data={}, skip_sid=True)


@SIO.on('connect')
def connect(sid, _):
    """Connects to simulator"""

    print("connect ", sid)
    send_control((0, 0, 0), '', '')
    sample_data = {}

    SIO.emit(
        "get_samples",
        sample_data,
        skip_sid=True)


def send_control(commands, image_string1, image_string2):
    """Sends commands to the Rover"""

    # Define commands to be sent to the rover
    data = {
        'throttle': commands[0].__str__(),
        'brake': commands[1].__str__(),
        'steering_angle': commands[2].__str__(),
        'inset_image1': image_string1,
        'inset_image2': image_string2,
    }

    # Send commands via socketIO server
    SIO.emit(
        "data",
        data,
        skip_sid=True)

    eventlet.sleep(0)


def send_pickup():
    """Sends the "pickup" command to the Rover"""

    print("Picking up")
    pickup = {}
    SIO.emit(
        "pickup",
        pickup,
        skip_sid=True)

    eventlet.sleep(0)


def main():
    """The method is called upon the module launch"""

    parser = argparse.ArgumentParser(description='Remote Driving')

    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder, where the images from the run are saved.'
    )
    args = parser.parse_args()

    # pylint: disable=global-statement
    global IMAGE_FOLDER
    IMAGE_FOLDER = args.image_folder

    # os.system('rm -rf IMG_stream/*')
    if IMAGE_FOLDER != '':
        print("Creating image folder at {}".format(IMAGE_FOLDER))
        if not os.path.exists(IMAGE_FOLDER):
            os.makedirs(IMAGE_FOLDER)
        else:
            shutil.rmtree(IMAGE_FOLDER)
            os.makedirs(IMAGE_FOLDER)
        print("Recording this run ...")
    else:
        print("NOT recording this run ...")

    # wrap Flask application with socketio's middleware
    app = socketio.Middleware(SIO, Flask(__name__))

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


if __name__ == '__main__':
    main()
