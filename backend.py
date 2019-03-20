"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import itertools
import logging
import queue
from time import sleep, time

import numpy as np
from pyglui import ui

import cv2
import cython_methods
from camera_models import load_intrinsics, Radial_Dist_Camera
from video_capture import manager_classes
from video_capture.base_backend import Base_Manager, Base_Source, Playback_Source

logger = logging.getLogger(__name__)

try:
    import picoflexx.roypy as roypy
    from picoflexx.roypy_platform_utils import PlatformHelper
except ImportError:
    import traceback

    logger.debug(traceback.format_exc())
    logger.info("Pico Flexx backend requirements (roypy) not installed properly")
    raise

try:
    from . import roypycy
except ImportError:
    import traceback

    logger.debug(traceback.format_exc())
    logger.warning("Pico Flexx backend requirements (roypycy) not installed properly")
    raise


class Frame(object):
    """docstring of Frame"""

    current_index = 0

    def __init__(self, depth_data):
        # self.timestamp = depth_data.timeStamp  # Not memory safe!
        self.timestamp = None
        self._data = roypycy.get_backend_data(depth_data)

        self.height = depth_data.height
        self.width = depth_data.width
        self.shape = depth_data.height, depth_data.width, 3
        self.index = self.current_index
        self.current_index += 1

        # indicate that the frame does not have a native yuv or jpeg buffer
        self.yuv_buffer = None
        self.jpeg_buffer = None

        self._img = None
        self._gray = None

    @property
    def img(self):
        if self._img is None:
            depth_values = self._data.z.reshape(self.height, self.width)
            depth_values = (2 ** 16) * depth_values / depth_values.max()
            depth_values = depth_values.astype(np.uint16)
            self._img = cython_methods.cumhist_color_map16(depth_values)
        return self._img

    @property
    def gray(self):
        if self._gray is None:
            self._gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return self._gray

    @property
    def bgr(self):
        return self.img

    @property
    def confidence(self):
        return self._data.depthConfidence.reshape(self.height, self.width)

    @property
    def noise(self):
        return self._data.noise.reshape(self.height, self.width)

    @property
    def dense_pointcloud(self):
        return self._data[["x", "y", "z"]]


class DepthDataListener(roypy.IDepthDataListener):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def onNewData(self, data):
        try:
            self.queue.put(Frame(data), block=False)
        except queue.Full:
            pass  # dropping frame
        except Exception:
            import traceback

            traceback.print_exc()


class Picoflexx_Source(Playback_Source, Base_Source):
    name = "Picoflexx"

    def __init__(self, g_pool, color_z_min=0.4, color_z_max=1.0, *args, **kwargs):
        super().__init__(g_pool, *args, **kwargs)
        self.color_z_min = color_z_min
        self.color_z_max = color_z_max

        self.camera = None
        self.queue = queue.Queue(maxsize=1)
        self.data_listener = DepthDataListener(self.queue)
        self.init_device()

        self.fps = 30
        self.frame_count = 0

        self._ui_exposure = None
        self._current_exposure = 0  # TODO obtain current exposure from most recent DepthData event
        self._current_exposure_mode = False

    def init_device(self):
        cam_manager = roypy.CameraManager()
        try:
            cam_id = cam_manager.getConnectedCameraList()[0]
        except IndexError:
            logger.error("No Pico Flexx camera connected")
            return

        self.camera_id = cam_id
        self.camera = cam_manager.createCamera(cam_id)
        self.camera.initialize()
        self.camera.registerDataListener(self.data_listener)
        self.camera.startCapture()
        roypycy.set_exposure_mode(self.camera, 1)
        self._current_exposure_mode = self.get_exposure_mode()
        self._online = True

    def init_ui(self):  # was gui
        self.add_menu()
        self.menu.label = "Pico Flexx"
        self.update_ui()

    def update_ui(self):
        del self.menu[:]

        text = ui.Info_Text("Pico Flexx Options")
        self.menu.append(text)

        if self.online:
            use_cases = self.camera.getUseCases()
            use_cases = [
                use_cases[uc]
                for uc in range(use_cases.size())
                if "MIXED" not in use_cases[uc]
            ]
            default = "Select to activate"
            use_cases.insert(0, default)

            self.menu.append(
                ui.Selector(
                    "selected_usecase",
                    selection=use_cases,
                    getter=lambda: default,
                    setter=self.set_usecase,
                    label="Activate usecase",
                )
            )

            exposure_limits = self.camera.getExposureLimits()
            self.menu.append(
                ui.Slider(
                    "selected_exposure",
                    min=exposure_limits.first,
                    max=exposure_limits.second,
                    getter=lambda: self._current_exposure,
                    setter=self.set_exposure_delayed,
                    label="Exposure",
                )
            )
            self._ui_exposure = self.menu[-1]

            self.menu.append(
                ui.Switch(
                    "selected_exposure_mode",
                    getter=lambda: self._current_exposure_mode,
                    setter=self.set_exposure_mode,
                    label="Auto Exposure",
                )
            )
        else:
            text = ui.Info_Text("Pico Flexx needs to be reactivated")
            self.menu.append(text)

    def deinit_ui(self):
        self.remove_menu()

    def cleanup(self):
        if self.camera:
            if self.camera.isConnected() and self.camera.isCapturing():
                self.camera.stopCapture()
            self.camera.unregisterDataListener()
            self.camera = None

    def on_notify(self, notification):
        if notification["subject"] == "picoflexx.set_exposure":
            self.set_exposure(notification["exposure"])

    def set_usecase(self, usecase):
        if self.camera.isCapturing():
            self.camera.stopCapture()
        self.camera.setUseCase(usecase)

        # Update UI with expsoure limits of this use case
        exposure_limits = self.camera.getExposureLimits()
        self._ui_exposure.minimum = exposure_limits.first
        self._ui_exposure.maximum = exposure_limits.second
        if self._current_exposure > exposure_limits.second:
            # Exposure is implicitly clamped to new max
            self._current_exposure = exposure_limits.second

        self.camera.startCapture()

    def set_exposure_delayed(self, exposure):
        self.notify_all(
            {"subject": "picoflexx.set_exposure", "delay": 0.3, "exposure": exposure}
        )

    def set_exposure(self, exposure):
        status = self.camera.setExposureTime(exposure)
        if status != 0:
            logger.warning(
                "setExposureTime: Non-zero return: {} - {}".format(
                    status, roypy.getStatusString(status)
                )
            )

        self._current_exposure = exposure

    def get_exposure_mode(self):
        return roypycy.get_exposure_mode(self.camera) == 1

    def set_exposure_mode(self, exposure_mode):
        roypycy.set_exposure_mode(self.camera, 1 if exposure_mode else 0)
        self._current_exposure_mode = exposure_mode

    def recent_events(self, events):
        frame = self.get_frame()
        if frame:
            events["frame"] = frame
            self._recent_frame = frame

    def get_frame(self):
        try:
            frame = self.queue.get(True, 0.02)
        except queue.Empty:
            return

        # Given: timestamp in microseconds precision (time since epoch 1970)
        # Overwrite with Capture timestamp
        frame.timestamp = self.g_pool.get_timestamp()

        return frame

    @property
    def frame_size(self):
        return (
            (self._recent_frame.width, self._recent_frame.height)
            if self._recent_frame
            else (1280, 720)
        )

    # @frame_size.setter
    # def frame_size(self, new_size):
    #     # closest match for size
    #     sizes = [abs(r[0] - new_size[0]) for r in self.frame_sizes]
    #     best_size_idx = sizes.index(min(sizes))
    #     size = self.frame_sizes[best_size_idx]
    #     if size != new_size:
    #         logger.warning(
    #             "%s resolution capture mode not available. Selected %s."
    #             % (new_size, size)
    #         )
    #     self.make_img(size)

    @property
    def frame_rates(self):
        return (30, 30)

    @property
    def frame_sizes(self):
        return ((640, 480), (1280, 720), (1920, 1080))

    @property
    def frame_rate(self):
        return self.fps

    @frame_rate.setter
    def frame_rate(self, new_rate):
        rates = [abs(r - new_rate) for r in self.frame_rates]
        best_rate_idx = rates.index(min(rates))
        rate = self.frame_rates[best_rate_idx]
        if rate != new_rate:
            logger.warning(
                "%sfps capture mode not available at (%s) on 'Fake Source'. Selected %sfps. "
                % (new_rate, self.frame_size, rate)
            )
        self.fps = rate

    @property
    def jpeg_support(self):
        return False

    @property
    def online(self):
        return self.camera and self.camera.isConnected() and self.camera.isCapturing()

    @property
    def intrinsics(self):
        return Radial_Dist_Camera(
            [[212.924133, 0, 117.875443], [0.0, 212.924133, 87.564507], [0, 0, 1]],
            [0.288401, -3.919852, 0, 0, 6.981279],
            self.frame_size,
            self.name,
        )
        if self._intrinsics is None or self._intrinsics.resolution != self.frame_size:
            self._intrinsics = load_intrinsics(
                self.g_pool.user_dir, self.name, self.frame_size
            )
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, model):
        self._intrinsics = model


class Picoflexx_Manager(Base_Manager):
    """Simple manager to explicitly activate a fake source"""

    gui_name = "Pico Flexx"

    def __init__(self, g_pool):
        super().__init__(g_pool)

    # Initiates the UI for starting the webcam.
    def init_ui(self):
        self.add_menu()
        from pyglui import ui

        self.menu.append(ui.Info_Text("Backend for https://pmdtec.com/picofamily/"))
        self.menu.append(ui.Button("Activate Pico Flexx", self.activate_source))

    def activate_source(self):
        settings = {}
        settings["name"] = "Picoflexx_Source"
        # if the user set fake capture, we dont want it to auto jump back to the old capture.
        if self.g_pool.process == "world":
            self.notify_all(
                {
                    "subject": "start_plugin",
                    "name": "Picoflexx_Source",
                    "args": settings,
                }
            )
        else:
            logger.warning("Pico Flexx backend is not supported in the eye process.")

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self, events):
        pass

    def get_init_dict(self):
        return {}


manager_classes.append(Picoflexx_Manager)
