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
from time import time

import cv2
import logging
import numpy as np
import os
import queue
from pyglui import ui
from typing import Tuple, Optional

import csv_utils
import cython_methods
import gl_utils
from version_utils import VersionFormat
from camera_models import Radial_Dist_Camera, Dummy_Camera
from video_capture import manager_classes
from video_capture.base_backend import Base_Manager, Base_Source, Playback_Source

logger = logging.getLogger(__name__)

try:
    from . import roypy as roypy
    from .roypy_platform_utils import PlatformHelper
except ImportError:
    import traceback

    logger.debug(traceback.format_exc())
    logger.warning("Pico Flexx backend requirements (roypy) not installed properly")
    raise

assert VersionFormat(roypy.getVersionString()) >= VersionFormat(
    "3.21.1.70"
), "roypy out of date, please upgrade to newest version. Have: {}, Want: {}".format(
    roypy.getVersionString(), "3.21.1.70"
)

try:
    from . import roypycy
except ImportError:
    import traceback

    logger.debug(traceback.format_exc())
    logger.warning("Pico Flexx backend requirements (roypycy) not installed properly")
    raise

FramePair = collections.namedtuple("FramePair", ["ir", "depth"])

MICRO_TO_SECONDS = 1e-6


def roypy_wrap(
        func,
        *args,
        check_status: bool = True,
        tag: str = None,
        reraise: bool = False,
        level: int = logging.WARNING,
        **kwargs
) -> Tuple[Optional[RuntimeError], Optional[int]]:
    func_name = tag or getattr(func, '__name__', None) or 'Unknown function'

    try:
        status = func(*args, **kwargs)
    except RuntimeError as e:
        if e.args:
            logger.log(level, "{}: {}".format(func_name, e.args[0]))
        else:
            logger.log(level, "{}: RuntimeError".format(func_name))

        if reraise:
            raise

        return e, None

    if check_status and status != 0:
        logger.log(
            level,
            "{}: Non-zero return: {} - {}".format(func_name, status, roypy.getStatusString(status))
        )

        return None, status


class IRFrame(object):
    def __init__(self, ir_data):
        self._ir_data = ir_data
        self.timestamp = ir_data.timestamp * MICRO_TO_SECONDS
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
        self.timestamp = roypycy.get_depth_data_ts(depth_data)  # microseconds
        self.timestamp *= MICRO_TO_SECONDS  # seconds
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
            depth_values = self.true_depth.reshape(self.height, self.width)
            depth_values = (2 ** 16) * depth_values / depth_values.max()
            depth_values = depth_values.astype(np.uint16)
            self._depth_img = cython_methods.cumhist_color_map16(depth_values)
        return self._depth_img

    def get_color_mapped(self, hue_near: float, hue_far: float, dist_near: float, dist_far: float, use_true_depth: bool):
        if use_true_depth:
            original_depth = self.true_depth.reshape(self.height, self.width)
        else:
            original_depth = self._data.z.reshape(self.height, self.width)

        depth_values = (original_depth - dist_near) / (dist_far - dist_near)
        depth_values = np.clip(depth_values, 0, 1)
        depth_values = (hue_near + (depth_values * (hue_far - hue_near))) * 255
        depth_values = depth_values.astype(np.uint8)

        hsv = depth_values[:, :, np.newaxis]

        # set saturation and value to 255
        a = np.zeros(hsv.shape, dtype=np.uint8)
        a[:] = 255
        b = np.concatenate((hsv, a, a), axis=2)

        dest = cv2.cvtColor(b, cv2.COLOR_HSV2BGR)

        # blank out missing data
        dest[original_depth == 0] = (0, 0, 0)

        return dest

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
    def true_depth(self):
        xyz = np.column_stack((self._data.x, self._data.y, self._data.z))
        return np.linalg.norm(xyz, axis=1)

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
        if roypycy.get_depth_data_ts(self._data_depth) != self._data_ir.timestamp:
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

    def __init__(
        self,
        g_pool,
        auto_exposure=False,
        preview_depth=True,
        record_pointcloud=False,
        current_exposure=0,
        selected_usecase=None,
        hue_near=0.0,
        hue_far=0.55,
        dist_near=0.14,
        dist_far=0.8,
        preview_true_depth=False,
        *args,
        **kwargs,
    ):
        super().__init__(g_pool, *args, **kwargs)
        self.camera = None
        self.queue = queue.Queue(maxsize=1)
        self.data_listener = DepthDataListener(self.queue)

        self.selected_usecase = selected_usecase
        self.frame_count = 0
        self.record_pointcloud = record_pointcloud
        self.royale_timestamp_offset = None

        self._recent_frame = None  # type: Optional[IRFrame]
        self._recent_depth_frame = None  # type: Optional[DepthFrame]

        self._ui_exposure = None
        self._current_exposure = current_exposure
        self._current_exposure_mode = auto_exposure
        self._preview_depth = preview_depth
        self._hue_near = hue_near
        self._hue_far = hue_far
        self._dist_near = dist_near
        self._dist_far = dist_far
        self._preview_true_depth = preview_true_depth

        self.init_device()

    def get_init_dict(self):
        return {
            "preview_depth": self._preview_depth,
            "record_pointcloud": self.record_pointcloud,
            "auto_exposure": self._current_exposure_mode,
            "current_exposure": self._current_exposure,
            "selected_usecase": self.selected_usecase,
            "hue_near": self._hue_near,
            "hue_far": self._hue_far,
            "dist_near": self._dist_near,
            "dist_far": self._dist_far,
            "preview_true_depth": self._preview_true_depth,
        }

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
        try:
            # can sporadically claim "Camera is disconnected"
            roypy_wrap(self.camera.startCapture, tag='Failed to start camera', reraise=True, level=logging.ERROR)
        except RuntimeError as e:
            return

        # Apply settings
        self.set_exposure_mode(self._current_exposure_mode)

        if self.selected_usecase is not None:
            self.set_usecase(self.selected_usecase)

        if not self._current_exposure_mode and self._current_exposure != 0:
            self.set_exposure(self._current_exposure)

        self._online = True

        self.load_camera_state()

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

            self.menu.append(
                ui.Selector(
                    "selected_usecase",
                    selection=use_cases,
                    getter=lambda: self.selected_usecase,
                    setter=self.set_usecase,
                    label="Activate usecase",
                )
            )

            self._ui_exposure = ui.Slider(
                "_current_exposure",
                self,
                min=0,
                max=0,
                setter=self.set_exposure_delayed,
                label="Exposure",
            )
            self.menu.append(self._ui_exposure)

            self.menu.append(
                ui.Switch(
                    "_current_exposure_mode",
                    self,
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

            depth_preview_menu = ui.Growing_Menu("Depth preview settings")
            depth_preview_menu.collapsed = True
            depth_preview_menu.append(
                ui.Info_Text("Set hue and distance ranges for the depth preview.")
            )
            depth_preview_menu.append(
                ui.Slider("_hue_near", self, min=0.0, max=1.0, label="Near Hue")
            )
            depth_preview_menu.append(
                ui.Slider("_hue_far", self, min=0.0, max=1.0, label="Far Hue")
            )
            depth_preview_menu.append(ui.Button("Fit distance (15th and 85th percentile)", self._fit_distance))
            depth_preview_menu.append(
                ui.Slider("_dist_near", self, min=0.0, max=4.8, label="Near Distance (m)")
            )
            depth_preview_menu.append(
                ui.Slider("_dist_far", self, min=0.2, max=5.0, label="Far Distance (m)")
            )
            depth_preview_menu.append(ui.Switch("_preview_true_depth", self, label="Preview using linalg distance"))
            self.menu.append(depth_preview_menu)

            self._switch_record_pointcloud = ui.Switch("record_pointcloud", self, label="Include 3D pointcloud in recording")
            self.menu.append(self._switch_record_pointcloud)

            self.load_camera_state()
        else:
            text = ui.Info_Text("Pico Flexx needs to be reactivated")
            self.menu.append(text)

    def _fit_distance(self):
        if not self._recent_depth_frame:
            logger.warning("No recent frame, can't fit hue.")
            return

        if self._preview_true_depth:
            depth_data = self._recent_depth_frame.true_depth
        else:
            depth_data = self._recent_depth_frame._data.z

        self._dist_near, self._dist_far = np.quantile(depth_data, (0.15, 0.85))

    def load_camera_state(self):
        if not self.online:
            logger.error("Can't get state, not online")
            return

        self.selected_usecase = self.camera.getCurrentUseCase()
        self._current_exposure_mode = self.get_exposure_mode()
        exposure_limits = self.camera.getExposureLimits()
        if self._current_exposure > exposure_limits.second:
            # Exposure is implicitly clamped to new max
            self._current_exposure = exposure_limits.second

        if getattr(self, 'menu', None) is not None:  # UI is initialized
            # load exposure mode
            self._ui_exposure.read_only = self._current_exposure_mode

            # Update UI with exposure limits of this use case
            self._ui_exposure.minimum = exposure_limits.first
            self._ui_exposure.maximum = exposure_limits.second

    def deinit_ui(self):
        self.remove_menu()

    def cleanup(self):
        if self.camera:
            if self.camera.isConnected() and self.camera.isCapturing():
                roypy_wrap(self.camera.stopCapture)
            self.camera.unregisterDataListener()
            self.data_listener.unregisterIrListener(self.camera)
            self.camera = None

    def on_notify(self, notification):
        if not self.menu:  # we've never been online
            return

        if notification["subject"] == "picoflexx.set_exposure":
            self.set_exposure(notification["exposure"])
        elif notification["subject"] == "recording.started":
            self._switch_record_pointcloud.read_only = True

            self.start_pointcloud_recording(notification["rec_path"])
        elif notification["subject"] == "recording.stopped":
            self._switch_record_pointcloud.read_only = False

            self.stop_pointcloud_recording()
            self.append_recording_metadata(notification["rec_path"])

    def start_pointcloud_recording(self, rec_loc):
        if not self.record_pointcloud:
            return

        video_path = os.path.join(rec_loc, "pointcloud.rrf")
        roypy_wrap(self.camera.startRecording, video_path, 0, 0, 0)

    def stop_pointcloud_recording(self):
        if not self.record_pointcloud:
            return

        roypy_wrap(self.camera.stopRecording)

    def set_usecase(self, usecase):
        roypy_wrap(self.camera.setUseCase, usecase)

        if not self.camera.isCapturing():
            roypy_wrap(self.camera.startCapture)

        self.load_camera_state()

    def set_exposure_delayed(self, exposure):
        # set displayed exposure early, to reduce jankiness while dragging slider
        self._current_exposure = exposure

        self.notify_all(
            {"subject": "picoflexx.set_exposure", "delay": 0.3, "exposure": exposure}
        )

    def set_exposure(self, exposure):
        roypy_wrap(self.camera.setExposureTime, exposure)

    def get_exposure_mode(self):
        return self.camera.getExposureMode() == roypy.ExposureMode_AUTOMATIC

    def set_exposure_mode(self, exposure_mode):
        roypy_wrap(
            self.camera.setExposureMode,
            roypy.ExposureMode_AUTOMATIC if exposure_mode else roypy.ExposureMode_MANUAL
        )
        self._current_exposure_mode = exposure_mode
        if self._ui_exposure is not None:
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

        if self.royale_timestamp_offset is None:
            # use a constant offset so timestamps from the RRF can be matched
            self.royale_timestamp_offset = self.g_pool.get_timestamp() - time()

        # picoflexx time epoch is unix time, readjust timestamps to pupil time
        frames.ir.timestamp += self.royale_timestamp_offset
        frames.depth.timestamp += self.royale_timestamp_offset

        # To calculate picoflexx camera delay:
        # self.g_pool.get_timestamp() - frames.ir.timestamp
        # Result: ~2-6ms delay depending on selected usecase

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
        return 1, self.camera.getMaxFrameRate() if self.online else 30

    @property
    def frame_sizes(self):
        return ((640, 480), (1280, 720), (1920, 1080))

    @property
    def frame_rate(self):
        return self.camera.getFrameRate() if self.online else 30

    @frame_rate.setter
    def frame_rate(self, new_rate):
        rates = [abs(r - new_rate) for r in self.frame_rates]
        best_rate_idx = rates.index(min(rates))
        rate = self.frame_rates[best_rate_idx]
        if rate != new_rate:
            logger.warning(
                "%sfps capture mode not available at (%s) on 'PicoFlexx Source'. Selected %sfps. "
                % (new_rate, self.frame_size, rate)
            )
        roypy_wrap(self.camera.setFrameRate, rate)

    @property
    def jpeg_support(self):
        return False

    @property
    def online(self):
        return self.camera and self.camera.isConnected() and self.camera.isCapturing()

    @property
    def intrinsics(self):
        if not self.online:
            return self._intrinsics or Dummy_Camera(self.frame_size, self.name)

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
                self.g_pool.image_tex.update_from_ndarray(self._recent_depth_frame.get_color_mapped(
                    self._hue_near, self._hue_far, self._dist_near, self._dist_far, self._preview_true_depth
                ))
            elif self._recent_frame is not None:
                self.g_pool.image_tex.update_from_ndarray(self._recent_frame.img)
            gl_utils.glFlush()
            gl_utils.make_coord_system_norm_based()
            self.g_pool.image_tex.draw()
        else:
            super().gl_display()

        gl_utils.make_coord_system_pixel_based(
            (self.frame_size[1], self.frame_size[0], 3)
        )

    def append_recording_metadata(self, rec_path):
        meta_info_path = os.path.join(rec_path, "info.csv")

        with open(meta_info_path, "a", newline="") as csvfile:
            csv_utils.write_key_value_file(
                csvfile,
                {
                    "Royale Timestamp Offset": self.royale_timestamp_offset,
                },
                append=True,
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
