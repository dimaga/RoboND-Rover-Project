#!python
"""Machine learning classifiers of pixel colors"""
import glob
import numpy as np

# pylint: disable=import-error
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
import images


def training_samples():
    """Load all available training samples"""
    masks = glob.glob('../calibration_images/*_mask.png')

    result = []
    for mask in masks:
        img_path = mask.replace("_mask.png", ".jpg")
        img = mpimg.imread(img_path)
        label_img = mpimg.imread(mask)

        result.append((img, label_img))

    return result


TRAINING_SAMPLES = training_samples()

NON_EMPTY = np.concatenate([
    np.logical_or(
        y[:, :, 0] != 0,
        np.logical_or(
            y[:, :, 1] != 0,
            y[:, :, 2] != 0)).ravel() for _, y in TRAINING_SAMPLES])

TRAINING_X = np.concatenate([
    x.reshape(-1, 3) for x, _ in TRAINING_SAMPLES])

TRAINING_ROCKS = np.concatenate([
    (y[:, :, 0] != 0).reshape(-1) for _, y in TRAINING_SAMPLES])

TRAINING_NAVI = np.concatenate([
    (y[:, :, 1] != 0).reshape(-1) for _, y in TRAINING_SAMPLES])

TRAINING_X, TRAINING_ROCKS, TRAINING_NAVI = shuffle(
    TRAINING_X[NON_EMPTY],
    TRAINING_ROCKS[NON_EMPTY],
    TRAINING_NAVI[NON_EMPTY],
    random_state=0)


class ClassifierNavi:
    """Naive Bayesian classifier to determine if pixels are navigatable"""

    # pylint: disable=too-few-public-methods

    def __init__(self):
        """Construct the navigatable pixels classifier"""
        self.__cls = GaussianNB()
        self.__cls.fit(TRAINING_X, TRAINING_NAVI)

    def predict(self, img):
        """Returns ln p(color | navigatable) - ln p(color | obstacle)"""
        scores = self.__cls.predict_log_proba(img.reshape(-1, 3))
        return (scores[:, 1] - scores[:, 0]).reshape(img.shape[:2])


class ClassifierRocks:
    """Gaussian Expected Maximization classifier to determine if pixels are
    rocks"""

    # pylint: disable=too-few-public-methods

    def __init__(self):
        """Construct the rock pixels classifier"""
        self.__cls = GaussianMixture(random_state=0).fit(
            TRAINING_X[TRAINING_ROCKS])

        means_init = np.array([[0.0, 0.0, 0.0], [255.0, 255.0, 255.0]])

        self.__not_cls = GaussianMixture(
            2,
            random_state=0,
            means_init=means_init)

        self.__not_cls.fit(TRAINING_X[~TRAINING_ROCKS])


    def predict(self, img):
        """Returns ln p(color | rock) - ln p(color | not rock)"""
        input_x = img.reshape(-1, 3)

        score_diff = (
            self.__cls.score_samples(input_x)
            - self.__not_cls.score_samples(input_x))

        return score_diff.reshape(img.shape[:2])


ROCKS = ClassifierRocks()
NAVI = ClassifierNavi()


def main():
    """Shows results of what the module does if run as a separate application"""

    plt.figure(figsize=(12, 6))
    plt.subplot(221)
    plt.imshow(images.ROCK1)

    plt.subplot(222)
    plt.imshow(ROCKS.predict(images.ROCK1), cmap='gray')

    plt.subplot(223)
    plt.imshow(images.ROCK2)

    plt.subplot(224)
    plt.imshow(NAVI.predict(images.ROCK2), cmap='gray')

    plt.show()


if __name__ == '__main__':
    main()
