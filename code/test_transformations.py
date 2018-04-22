#!python
"""Unit tests for transformations"""

import unittest
import numpy as np
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


if __name__ == '__main__':
    unittest.main()
