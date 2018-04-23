#!python
"""Unit tests for control routines"""

import unittest
import control
import images
import numpy as np


class TestConstrol(unittest.TestCase):
    # pylint: disable=no-self-use
    """Test cases to verify control module"""


    def test_navi_direction_distract(self):
        """Show clear direction along the navigatable pixels"""

        nav = np.ones((images.HEIGHT, images.WIDTH), np.float32)
        nav *= -1.0
        nav[:, 160] = 1.0

        direction = control.navi_direction(nav, False)

        # dir is returned expected in rover space
        np.testing.assert_almost_equal(np.array([1.0, 0.0]), direction)


    def test_navi_direction_draw(self):
        """If there is a draw, returns zero vector"""

        nav = np.zeros((images.HEIGHT, images.WIDTH), np.float32)
        direction = control.navi_direction(nav, False)
        np.testing.assert_almost_equal(np.array([0.0, 0.0]), direction)


    def test_navi_direction_no_distract(self):
        """Distract from obstacles in the opposite directions"""
        nav = np.ones((images.HEIGHT, images.WIDTH), np.float32)
        nav *= -1.0
        nav[:, 160] = 1.0

        direction = control.navi_direction(nav, True)

        # dir is returned expected in rover space
        np.testing.assert_almost_equal(
            np.array([-1.0, 0.0]),
            direction,
            decimal=2)


if __name__ == '__main__':
    unittest.main()
