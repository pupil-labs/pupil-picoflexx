import cv2
import numpy as np

import cython_methods
from picoflexx.royale.extension import roypycy
from ..utils import get_hue_color_map, MICRO_TO_SECONDS


class DepthFrame(object):
    """
    Contains data relating to a given depth frame, essentially providing a
    wrapper around the information inside of a roypy DepthData object.
    """

    def __init__(self, depth_data):
        """
        :param depth_data: either a DepthData object from roypy, or a dict
        containing equivalent information.
        """

        if type(depth_data) is dict:
            self.timestamp = depth_data["timestamp"]
            self._data = depth_data["_data"]
            self.exposure_times = depth_data["exposure_times"]
            self.width, self.height = depth_data["width"], depth_data["height"]
        else:
            self.timestamp = roypycy.get_depth_data_ts(depth_data)  # microseconds
            self.timestamp *= MICRO_TO_SECONDS  # seconds
            self._data = roypycy.get_backend_data(depth_data)
            self.exposure_times = depth_data.exposureTimes

            self.height = depth_data.height
            self.width = depth_data.width

        self.shape = self.height, self.width, 3

        # indicate that the frame does not have a native yuv or jpeg buffer
        self.yuv_buffer = None
        self.jpeg_buffer = None

        self._depth_img = None
        self._gray = None

    @property
    def bgr(self):
        """
        Notably, this may be modified by other plugins

        :return: The colored depth image
        """

        if self._depth_img is None:
            depth_values = self.true_depth.reshape(self.height, self.width)
            depth_values = (2 ** 16) * depth_values / depth_values.max()
            depth_values = depth_values.astype(np.uint16)
            self._depth_img = cython_methods.cumhist_color_map16(depth_values)
        return self._depth_img

    def get_color_mapped(self, hue_near: float, hue_far: float, dist_near: float, dist_far: float, use_true_depth: bool):
        """
        Colorizes the depth image using interpolated hue values against the
        depth data, optionally using the "true" depth instead of the z-value.

        :param hue_near: Hue value [0-1] to use for near distances
        :param hue_far: Hue value [0-1] to use for far distances
        :param dist_near: Distance to consider near
        :param dist_far: Distance to consider far
        :param use_true_depth: Whether to use the "true" depth (point to point)
         instead of the z-value (plane to plane)
        :return: The hue color mapped image using the given parameters
        """

        if use_true_depth:
            original_depth = self.true_depth.reshape(self.height, self.width)
        else:
            original_depth = self._data.z.reshape(self.height, self.width)

        return get_hue_color_map(original_depth, hue_near, hue_far, dist_near, dist_far)

    @property
    def gray(self):
        """
        :return: The depth image converted to gray scale.
        """

        if self._gray is None:
            self._gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return self._gray

    @property
    def img(self):
        """
        Notably, this may be modified by other plugins

        :return: The colored depth image
        """

        return self.bgr

    @property
    def confidence(self):
        return self._data.depthConfidence.reshape(self.height, self.width)

    @property
    def noise(self):
        return self._data.noise.reshape(self.height, self.width)

    @property
    def true_depth(self):
        xyz = np.column_stack((self._data.x, self._data.y, self._data.z))
        return np.linalg.norm(xyz, axis=1)

    @property
    def dense_pointcloud(self):
        return self._data[["x", "y", "z"]]
