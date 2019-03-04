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
from camera_models import Dummy_Camera
from video_capture import manager_classes
from video_capture.base_backend import Base_Manager, Base_Source, Playback_Source

logger = logging.getLogger(__name__)

try:
    import picoflexx.roypy
    from picoflexx.roypy_platform_utils import PlatformHelper
except ImportError:
    import traceback

    logger.debug(traceback.format_exc())
    logger.info("Pico Flexx backend requirements not installed properly")
    raise


class Frame(object):
    """docstring of Frame"""

    def __init__(self, timestamp, img, index):
        self.timestamp = timestamp
        self._img = img
        self.bgr = img
        self.height, self.width, _ = img.shape
        self._gray = None
        self.index = index
        # indicate that the frame does not have a native yuv or jpeg buffer
        self.yuv_buffer = None
        self.jpeg_buffer = None

    @property
    def img(self):
        return self._img

    @property
    def gray(self):
        if self._gray is None:
            self._gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return self._gray

    @gray.setter
    def gray(self, value):
        raise Exception("Read only.")


class DepthDataListener(roypy.IDepthDataListener):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def onNewData(self, data):
        try:
            zvalues = []
            for i in range(data.getNumPoints()):
                zvalues.append(data.getZ(i))
            zarray = np.asarray(zvalues)
            p = zarray.reshape(-1, data.width)
            p = 255 * p / p.max()
            p = cv2.applyColorMap(p.astype(np.uint8), cv2.COLORMAP_JET)
            self.queue.put(p)
        except Exception:
            import traceback

            traceback.print_exc()


class Picoflexx_Source(Playback_Source, Base_Source):
    def __init__(self, g_pool, cam_id=None, *args, **kwargs):
        super().__init__(g_pool, *args, **kwargs)

        self._online = False
        self.queue = queue.Queue(maxsize=1)
        self.data_listener = DepthDataListener(self.queue)
        self.init_device(cam_id)

        self.fps = 30
        self.frame_count = 0

    def init_device(self, cam_id):
        cam_manager = roypy.CameraManager()
        if not cam_id:
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
        self._online = True

    def init_ui(self):  # was gui
        self.add_menu()
        self.menu.label = "Pico Flexx"

        text = ui.Info_Text("Pico Flexx Options")
        self.menu.append(text)

    def deinit_ui(self):
        self.remove_menu()

    def cleanup(self):
        if self.camera.isConnected() and self.camera.isCapturing():
            self.camera.stopCapture()
        self.camera.unregisterDataListener()
        self.camera = None

    def recent_events(self, events):
        frame = self.get_frame()
        if frame:
            events["frame"] = frame
            self._recent_frame = frame

    def get_frame(self):
        try:
            data = self.queue.get(True, 0.02)
        except queue.Empty:
            return

        frame_count = self.frame_count
        self.frame_count += 1
        timestamp = self.g_pool.get_timestamp()

        return Frame(timestamp, data, frame_count)

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
        return self._online and self.camera.isConnected() and self.camera.isCapturing()


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
        cam_manager = roypy.CameraManager()
        devices = cam_manager.getConnectedCameraList()
        if not devices:
            logger.error("No Pico Flexx camera connected")
            return

        settings = {}
        settings["name"] = "Picoflexx_Source"
        settings["cam_id"] = devices[0]
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
            # self.notify_all(
            #     {
            #         "subject": "start_eye_capture",
            #         "target": self.g_pool.process,
            #         "name": "Picoflexx_Source",
            #         "args": settings,
            #     }
            # )

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self, events):
        pass

    def get_init_dict(self):
        return {}


manager_classes.append(Picoflexx_Manager)
