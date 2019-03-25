"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import collections
import itertools
import logging
import queue
from time import sleep, time

import numpy as np
from pyglui import ui

import cv2
import cython_methods
import gl_utils
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

FramePair = collections.namedtuple("FramePair", ["ir", "depth"])


class IRFrame(object):
    def __init__(self, ir_data):
        self._ir_data = ir_data
        self.timestamp = ir_data.timestamp
        self.width = ir_data.width
        self.height = ir_data.height
        self.shape = self.height, self.width
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
            self._ir_img = self._ir_data.data
            self._ir_img.shape = self.shape
        return self._ir_img

    @property
    def bgr(self):
        if self._ir_img_bgr is None:
            self._ir_img_bgr = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        return self._ir_img_bgr


class DepthFrame(object):
    def __init__(self, depth_data):
        # self.timestamp = depth_data.timeStamp  # Not memory safe!
        self.timestamp = None
        self._data = roypycy.get_backend_data(depth_data)
        self.exposure_times = depth_data.exposureTimes

        self.height = depth_data.height
        self.width = depth_data.width
        self.shape = depth_data.height, depth_data.width, 3

        # indicate that the frame does not have a native yuv or jpeg buffer
        self.yuv_buffer = None
        self.jpeg_buffer = None

        self._depth_img = None
        self._gray = None

    @property
    def bgr(self):
        if self._depth_img is None:
            depth_values = self._data.z.reshape(self.height, self.width)
            depth_values = (2 ** 16) * depth_values / depth_values.max()
            depth_values = depth_values.astype(np.uint16)
            self._depth_img = cython_methods.cumhist_color_map16(depth_values)
        return self._depth_img

    @property
    def gray(self):
        if self._gray is None:
            self._gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return self._gray

    @property
    def img(self):
        return self.bgr

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
    current_index = 0

    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self._ir_ref = None

        self._data_depth = None
        self._data_ir = None  # type: roypycy.PyIrImage

    def _check_frame(self):
        if self._data_depth is None or self._data_ir is None:
            return

        try:
            frame_depth = DepthFrame(self._data_depth)
            frame_ir = IRFrame(self._data_ir)
            frame_depth.index = frame_ir.index = self.current_index
            self.current_index += 1

            self.queue.put(FramePair(ir=frame_ir, depth=frame_depth), block=False)
        except queue.Full:
            pass  # dropping frame pair
        except Exception:
            import traceback

            traceback.print_exc()

        self._data_depth = None
        self._data_ir = None

    def onNewData(self, data):
        self._data_depth = data
        self._check_frame()

    def onNewIrData(self, data: roypycy.PyIrImage):
        self._data_ir = data
        self._check_frame()

    def registerIrListener(self, camera):
        self._ir_ref = roypycy.register_ir_image_listener(camera, self.onNewIrData)

    def unregisterIrListener(self, camera):
        if self._ir_ref:
            roypycy.unregister_ir_image_listener(camera, self._ir_ref, self.onNewData)
            self._ir_ref = None


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
        self._current_exposure = 0
        self._current_exposure_mode = False
        self._preview_depth = True

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
        self.data_listener.registerIrListener(self.camera)
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

            self._current_exposure_mode = self.get_exposure_mode()
            self._ui_exposure.read_only = self._current_exposure_mode

            self.menu.append(
                ui.Switch(
                    "selected_exposure_mode",
                    getter=lambda: self._current_exposure_mode,
                    setter=self.set_exposure_mode,
                    label="Auto Exposure",
                )
            )

            text = ui.Info_Text(
                "Enabling Preview Depth will display a cumulative histogram colored "
                "version of the depth data. Disabling the option will display the "
                "according IR image. Independent of which option is selected, the IR "
                "image stream will be stored to `world.mp4` during a recording."
            )
            self.menu.append(text)
            self.menu.append(ui.Switch("_preview_depth", self, label="Preview Depth"))
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
            self.data_listener.unregisterIrListener(self.camera)
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
        # set displayed exposure early, to reduce jankiness while dragging slider
        self._current_exposure = exposure

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

    def get_exposure_mode(self):
        return roypycy.get_exposure_mode(self.camera) == 1

    def set_exposure_mode(self, exposure_mode):
        roypycy.set_exposure_mode(self.camera, 1 if exposure_mode else 0)
        self._current_exposure_mode = exposure_mode
        self._ui_exposure.read_only = exposure_mode

    def recent_events(self, events):
        frames = self.get_frames()
        if frames:
            events["frame"] = frames.ir
            events["depth_frame"] = frames.depth
            self._recent_frame = frames.ir
            self._recent_depth_frame = frames.depth

            if self._current_exposure_mode:  # auto exposure
                self._current_exposure = frames.depth.exposure_times[1]

    def get_frames(self):
        try:
            frames = self.queue.get(True, 0.02)
        except queue.Empty:
            return

        # Given: timestamp in microseconds precision (time since epoch 1970)
        # Overwrite with Capture timestamp
        recv_ts = self.g_pool.get_timestamp()
        frames.ir.timestamp = recv_ts
        frames.depth.timestamp = recv_ts

        return frames

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
        if self._intrinsics is None or self._intrinsics.resolution != self.frame_size:
            lens_params = roypycy.get_lens_parameters(self.camera)
            c_x, c_y = lens_params["principalPoint"]
            f_x, f_y = lens_params["focalLength"]
            p_1, p_2 = lens_params["distortionTangential"]
            k_1, k_2, *k_other = lens_params["distortionRadial"]
            K = [[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]]
            D = k_1, k_2, p_1, p_2, *k_other
            self._intrinsics = Radial_Dist_Camera(K, D, self.frame_size, self.name)
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, model):
        logger.error("Picoflexx backend does not support setting intrinsics manually")

    def gl_display(self):
        if self.online:
            if self._preview_depth and self._recent_depth_frame is not None:
                self.g_pool.image_tex.update_from_ndarray(self._recent_depth_frame.bgr)
            elif self._recent_frame is not None:
                self.g_pool.image_tex.update_from_ndarray(self._recent_frame.gray)
            gl_utils.glFlush()
            gl_utils.make_coord_system_norm_based()
            self.g_pool.image_tex.draw()
        else:
            super().gl_display()

        gl_utils.make_coord_system_pixel_based(
            (self.frame_size[1], self.frame_size[0], 3)
        )


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
