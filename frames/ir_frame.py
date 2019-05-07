import cv2

from ..utils import MICRO_TO_SECONDS


class IRFrame(object):
    def __init__(self, ir_data):
        if type(ir_data) is dict:
            self.timestamp = ir_data["timestamp"]
            self._ir_data = None
            self.data = ir_data["data"]
            self.exposure_times = ir_data["exposure_times"]
            self.width, self.height = ir_data["width"], ir_data["height"]
        else:
            self._ir_data = ir_data
            self.data = ir_data.data
            self.timestamp = ir_data.timestamp * MICRO_TO_SECONDS
            self.width = ir_data.width
            self.height = ir_data.height

        self.shape = self.height, self.width
        self.index = None
        self._ir_img = None
        self._ir_img_bgr = None

        # indicate that the frame does not have a native yuv or jpeg buffer
        self.yuv_buffer = None
        self.jpeg_buffer = None

    @property
    def img(self):
        return self.bgr

    @property
    def gray(self):
        if self._ir_img is None:
            self._ir_img = self.data.reshape(self.shape)
        return self._ir_img

    @property
    def bgr(self):
        if self._ir_img_bgr is None:
            self._ir_img_bgr = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        return self._ir_img_bgr
