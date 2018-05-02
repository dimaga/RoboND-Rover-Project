#!python
"""Routines to calculate some aspects of the robot control"""

# pylint: disable=import-error
import matplotlib.pyplot as plt
import numpy as np
import transformations
import images
import classifiers


def navi_direction(navi_top_view):
    """Calculate recommended direction of motion given navigatable
    local confidence map"""

    """
    values = navi_top_view.reshape(-1, 1)
    dirs = transformations.ROVER_CONF_DIRS

    idx = np.argmax(values)
    return dirs[idx]
    """

    values = navi_top_view.reshape(-1, 1)
    dirs = transformations.ROVER_CONF_DIRS

    # Ignore walls
    values = np.copy(values)
    values[values < 0] = 0

    result = np.mean(dirs * values, axis=0)

    with np.errstate(all='ignore'):
        result /= np.linalg.norm(result)

    np.nan_to_num(result, False)
    
    return result



def main():
    """Shows results of what the module does if run as a separate application"""

    img = images.ROCK1
    img_top = transformations.perspective_2_top(img)
    img_navi = classifiers.NAVI.predict(img_top)
    img_navi_dir = navi_direction(img_navi)

    plt.figure(figsize=(12, 9))

    plt.subplot(221)
    plt.imshow(img)

    plt.subplot(222)
    plt.imshow(img_top)

    plt.subplot(223)
    plt.imshow(img_navi, cmap='gray')

    plt.subplot(224)

    plt.ylim(-160, 160)
    plt.xlim(-160, 160)

    plt.pcolor(
        transformations.ROVER_CONF_POINTS[:, 0].reshape(img_navi.shape),
        transformations.ROVER_CONF_POINTS[:, 1].reshape(img_navi.shape),
        img_navi,
        cmap='gray')

    plt.arrow(
        0,
        0,
        img_navi_dir[0] * 100,
        img_navi_dir[1] * 100,
        color='red',
        zorder=2,
        head_width=10,
        width=2)

    plt.show()


if __name__ == '__main__':
    main()
