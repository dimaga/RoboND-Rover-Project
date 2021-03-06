#!python
"""Functions, copying received data into RoverState and providing debug output
"""

import base64
import time
from io import BytesIO

import cv2
import numpy as np

# pylint: disable=import-error
from PIL import Image

from decision import ROOT


def convert_to_float(string_to_convert):
    """Converts telemetry strings to float independent of decimal convention"""

    if ',' in string_to_convert:
        float_value = np.float(string_to_convert.replace(',', '.'))
    else:
        float_value = np.float(string_to_convert)
    return float_value


PREV_TRACE = ""

def update_rover(rover, data):
    """Read received data from simulator and write information into rover"""

    # Initialize start time and sample positions
    if rover.time.start is None:
        rover.time.start = time.time()
        rover.time.total = 0

        samples_xpos = np.int_([convert_to_float(pos.strip())
                                for pos in data["samples_x"].split(';')])

        samples_ypos = np.int_([convert_to_float(pos.strip())
                                for pos in data["samples_y"].split(';')])

        rover.statistics.samples_pos = (samples_xpos, samples_ypos)
        rover.statistics.samples_to_find = np.int(data["sample_count"])
    # Or just update elapsed time
    else:
        tot_time = time.time() - rover.time.start
        if np.isfinite(tot_time):
            rover.time.total = tot_time

    # Print out the fields in the telemetry data dictionary
    # print(data.keys())

    # The current speed of the rover in m/s
    rover.perception.vel = convert_to_float(data["speed"])

    # The current position of the rover
    rover.perception.pos = [convert_to_float(pos.strip())
                            for pos in data["position"].split(';')]

    # The current yaw angle of the rover
    rover.perception.yaw_deg = convert_to_float(data["yaw"])
    # The current yaw angle of the rover
    rover.perception.pitch_deg = convert_to_float(data["pitch"])
    # The current yaw angle of the rover
    rover.perception.roll_deg = convert_to_float(data["roll"])
    # The current throttle setting
    rover.control.throttle = convert_to_float(data["throttle"])
    # The current steering angle
    rover.control.steer = convert_to_float(data["steering_angle"])
    # Near sample flag
    rover.perception.near_sample = np.int(data["near_sample"])
    # Picking up flag
    rover.control.picking_up = np.int(data["picking_up"])

    # Update number of rocks collected
    rover.statistics.samples_collected = (
        rover.statistics.samples_to_find - np.int(data["sample_count"]))

    #print(
    #    'speed =', rover.perception.vel,
    #    'position =', rover.perception.pos,
    #    'throttle =', rover.control.throttle,
    #    'steer_angle =', rover.control.steer,
    #    'near_sample:', rover.perception.near_sample,
    #    'picking_up:', data["picking_up"],
    #    'sending pickup:', rover.control.send_pickup,
    #    'total time:', rover.time.total,
    #    'samples remaining:', data["sample_count"],
    #    'samples collected:', rover.statistics.samples_collected)

    #pylint: disable=global-statement
    global PREV_TRACE

    cur_trace = ROOT.trace()
    if PREV_TRACE != cur_trace:
        print(rover.time.total, cur_trace)
        PREV_TRACE = cur_trace

    # Get the current image from the center camera of the rover
    img_string = data["image"]
    image = Image.open(BytesIO(base64.b64decode(img_string)))
    rover.perception.img = np.asarray(image)

    # Return updated rover and separate image for optional saving
    return rover, image


def create_output_images(rover):
    """Creates display output given worldmap results"""

    map_add, plotmap, samples_located = create_output_map(rover)

    output_statistics(map_add, rover, samples_located, plotmap)

    return pack_to_strings(map_add, rover)


def create_output_map(rover):
    """Create a scaled map for plotting and clean up obs/nav pixels a bit"""

    navigable = rover.statistics.worldmap[:, :, 2]
    obstacle = rover.statistics.worldmap[:, :, 0]

    likely_nav = navigable >= obstacle

    obstacle[likely_nav] = 0

    plotmap = np.zeros_like(rover.statistics.worldmap)
    plotmap[:, :, 0] = obstacle
    plotmap[:, :, 2] = navigable
    plotmap = plotmap.clip(0, 255)

    # Overlay obstacle and navigable terrain map with ground truth map
    map_add = cv2.addWeighted(plotmap, 1, rover.statistics.ground_truth, 0.5, 0)

    # Check whether any rock detections are present in worldmap
    rock_world_pos = rover.statistics.worldmap[:, :, 1].nonzero()

    # If there are, we'll step through the known sample positions
    # to confirm whether detections are real
    samples_located = 0

    if rock_world_pos[0].any():
        rock_size = 2

        for idx in range(len(rover.statistics.samples_pos[0])):
            test_rock_x = rover.statistics.samples_pos[0][idx]
            test_rock_y = rover.statistics.samples_pos[1][idx]

            rock_sample_dists = np.sqrt(
                (test_rock_x - rock_world_pos[1]) ** 2 + \
                (test_rock_y - rock_world_pos[0]) ** 2)

            # If rocks were detected within 3 meters of known sample positions
            # consider it a success and plot the location of the known
            # sample on the map
            if np.min(rock_sample_dists) < 3:
                samples_located += 1

                map_add[
                    test_rock_y - rock_size:test_rock_y + rock_size,
                    test_rock_x - rock_size:test_rock_x + rock_size,
                    :] = 255

    # Flip the map for plotting so that the y-axis points upward in the display
    map_add = np.flipud(map_add).astype(np.float32)

    return map_add, plotmap, samples_located


def output_statistics(map_add, rover, samples_located, plotmap):
    """Output some statistics on the map results"""

    # Calculate some statistics on the map results
    # First get the total number of pixels in the navigable terrain map
    tot_nav_pix = np.float(len((plotmap[:, :, 2].nonzero()[0])))

    # Next figure out how many of those correspond to ground truth pixels
    good_nav_pix = np.float(
        len(((plotmap[:, :, 2] > 0)
             & (rover.statistics.ground_truth[:, :, 1] > 0)).nonzero()[0]))

    # Grab the total number of map pixels
    tot_map_pix = np.float(
        len((rover.statistics.ground_truth[:, :, 1].nonzero()[0])))

    # Calculate the percentage of ground truth map that has been successfully
    # #found
    perc_mapped = round(100 * good_nav_pix / tot_map_pix, 1)

    # Calculate the number of good map pixel detections divided by total pixels
    # found to be navigable terrain
    if tot_nav_pix > 0:
        fidelity = round(100 * good_nav_pix / (tot_nav_pix), 1)
    else:
        fidelity = 0

    # Add some text about map and rock sample detection results
    font_params = (
        cv2.FONT_HERSHEY_COMPLEX,
        0.4,
        (255, 255, 255),
        1)

    cv2.putText(
        map_add,
        "Time: " + str(np.round(rover.time.total, 1)) + ' s',
        (0, 10),
        *font_params)

    cv2.putText(
        map_add,
        "Mapped: " + str(perc_mapped) + '%',
        (0, 25),
        *font_params)

    cv2.putText(
        map_add,
        "Fidelity: " + str(fidelity) + '%',
        (0, 40),
        *font_params)

    cv2.putText(
        map_add,
        "Rocks",
        (0, 55),
        *font_params)

    cv2.putText(
        map_add,
        "  Located: " + str(samples_located),
        (0, 70),
        *font_params)

    cv2.putText(
        map_add,
        "  Collected: " + str(rover.statistics.samples_collected),
        (0, 85),
        *font_params)


def pack_to_strings(map_add, rover):
    """Convert map and vision image to base64 strings for sending to server"""

    pil_img = Image.fromarray(map_add.astype(np.uint8))
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    encoded_string1 = base64.b64encode(buff.getvalue()).decode("utf-8")

    pil_img = Image.fromarray(rover.statistics.vision_image.astype(np.uint8))
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    encoded_string2 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return encoded_string1, encoded_string2
