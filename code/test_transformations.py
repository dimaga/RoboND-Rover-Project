#!python
"""Unit tests for transformations"""

import unittest
import cv2
import numpy as np

#pylint: disable=import-error
import pandas as pd
import matplotlib.image as mpimg

import transformations


class TestTransformations(unittest.TestCase):
    """Test cases to verify transformations module"""

    def test_perspective_2_top(self):
        """Perspective points are correctly transformed into a top view"""

        buf = np.zeros([transformations.HEIGHT, transformations.WIDTH], np.float32)

        pts_perspective = transformations.POINTS_PERSPECTIVE.astype(np.int32)
        pts_top = transformations.POINTS_TOP.astype(np.int32)

        # T inverts the order of rows and columns to follow Cartesian order
        buf[[pts_perspective[:, 1], pts_perspective[:, 0]]] = 1.0
        buf_top = transformations.perspective_2_top(buf)

        self.assertTrue(np.allclose(
            buf_top[[pts_top[:, 1]], [pts_top[:, 0]]],
            [1.0, 1.0, 1.0, 1.0]))


    def test_local_2_global(self):
        """Points are correctly transformed from the local into
        map reference frames"""

        robot_log = pd.read_csv(
            '../test_dataset/robot_log.csv',
            delimiter=';',
            decimal='.')

        idx = 0

        xpos = robot_log["X_Position"].values[idx]
        ypos = robot_log["Y_Position"].values[idx]
        yaw_deg = robot_log["Yaw"].values[idx]
        loc_2_glob = transformations.local_2_global(xpos, ypos, yaw_deg)

        img = mpimg.imread(robot_log["Path"].tolist()[idx])
        img_top = transformations.perspective_2_top(img)

        ground_truth_bw = 255 * mpimg.imread(
            '../calibration_images/map_bw.png').astype(np.uint8)

        map_piece = cv2.warpAffine(
            img_top,
            loc_2_glob,
            (ground_truth_bw.shape[1], ground_truth_bw.shape[0]))

        map_piece_bw = cv2.cvtColor(map_piece, cv2.COLOR_RGB2GRAY)
        expected_map_piece = ground_truth_bw * (map_piece_bw != 0)

        correlation = cv2.matchTemplate(
            map_piece_bw,
            expected_map_piece,
            cv2.TM_CCORR_NORMED)[0][0]

        self.assertGreater(correlation, 0.6)


if __name__ == '__main__':
    unittest.main()
