#!python
"""Unit tests for classifiers"""

import unittest
import numpy as np
import classifiers
import images


def get_accuracy(cls, img, expected):
    """Calculate accuracy of the output for the given classifier cls"""

    local_confidence = cls.predict(img)
    given = (local_confidence > 0).ravel()
    expected = (0 != expected).ravel()

    return np.sum(given == expected) / float(len(expected))


class TestClassifiers(unittest.TestCase):
    """Test cases to evaluate the quality of color classifiers"""

    def test_classifier_rocks(self):
        """Test that ROCKS classifier can detect rocks reliably on
        a sample from the training set"""

        accuracy = get_accuracy(
            classifiers.ROCKS,
            images.ROCK1,
            images.ROCK1_LABEL[:, :, 0])

        self.assertGreater(accuracy, 0.95)


    def test_classifier_navi(self):
        """Test that ROCKS classifier can detect rocks reliably on
        a sample from the training set"""

        accuracy = get_accuracy(
            classifiers.NAVI,
            images.ROCK1,
            images.ROCK1_LABEL[:, :, 1])

        self.assertGreater(accuracy, 0.95)


if __name__ == '__main__':
    unittest.main()
