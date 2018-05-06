#!python
"""Unit tests for control routines"""

import unittest
import control
from transformations import TOP_HEIGHT, TOP_WIDTH, TOP_CENTER_X, TOP_CENTER_Y
import numpy as np


class TestConstrol(unittest.TestCase):
    # pylint: disable=no-self-use
    """Test cases to verify control module"""


    def test_navi_direction_distract(self):
        """Show clear direction along the navigatable pixels"""

        nav = np.ones((TOP_HEIGHT, TOP_WIDTH), np.float32)
        nav *= -1.0
        nav[:TOP_CENTER_Y, TOP_CENTER_X] = 1.0

        direction = control.navi_direction(nav)

        # dir is returned in rover space
        np.testing.assert_almost_equal(
            np.array([1.0, 0.0]),
            direction,
            decimal=1)


if __name__ == '__main__':
    unittest.main()
