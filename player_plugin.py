import logging
import os
import queue
from typing import Optional

import gl_utils
from plugin import Visualizer_Plugin_Base
from video_capture import File_Source
from . import roypy, roypycy
from .backend import DepthFrame, IRFrame, roypy_wrap, DepthDataListener
from .roypycy import PyIReplay
from .utils import append_depth_preview_menu

logger = logging.getLogger(__name__)


class Picoflexx_Player_Plugin(Visualizer_Plugin_Base):
    uniqueness = "by_class"
    icon_chr = chr(0xE886)
    icon_font = "pupil_icons"

    def __init__(self, g_pool, **kwargs):
        super().__init__(g_pool)
        self.order = 0.001  # Ensure we're after FileSource but before anything else
        self.menu = None

        self._hue_near = kwargs.get('hue_near', 0.0)
        self._hue_far = kwargs.get('hue_far', 0.75)
        self._dist_near = kwargs.get('dist_near', 0.14)
        self._dist_far = kwargs.get('dist_far', 5.0)
        self._preview_true_depth = kwargs.get('preview_true_depth', False)
        self._preview_depth = kwargs.get('preview_depth', True)

        self._recent_depth_frame = None  # type: Optional[DepthFrame]
        self._recent_ir_frame = None  # type: Optional[IRFrame]
        self.recording_camera = None  # type: Optional[roypy.ICameraDevice]
        self.recording_replay = None  # type: Optional[PyIReplay]

        cloud_path = os.path.join(self.g_pool.rec_dir, 'pointcloud.rrf')
        if not os.path.exists(cloud_path):
            self.recent_events = self._abort
            logger.error("There is no pointcloud in this recording.")
            return

        cam_manager = roypy.CameraManager()
        self.recording_camera = cam_manager.createCamera(cloud_path)
        roypy_wrap(self.recording_camera.initialize)

        # As we're accessing a recording, we need to cast the ICameraDevice
        # to IReplay to access extra functionality
        self.recording_replay = roypycy.toReplay(self.recording_camera)

        self.queue = queue.Queue(maxsize=1)
        self.data_listener = DepthDataListener(self.queue)
        self.recording_camera.registerDataListener(self.data_listener)
        self.data_listener.registerIrListener(self.recording_camera)

    def _abort(self, _):
        self.alive = False
        self.g_pool.plugins.clean()

    def recent_events(self, events):
        frame = events.get("frame")
        if not frame:
            return

        capture = self.g_pool.capture  # type: File_Source

        if capture.current_frame_idx < self.recording_replay.frame_count():
            if capture.current_frame_idx != self.recording_replay.current_frame()\
                    or self._recent_depth_frame is None:
                self.recording_replay.seek(capture.current_frame_idx)

                # depth data appears to arrive within ~9-12 microseconds
                self._recent_ir_frame, self._recent_depth_frame = self.queue.get()

            events["depth_frame"] = self._recent_depth_frame
        elif capture.current_frame_idx > self.recording_replay.frame_count():
            # The last frame is fake/nil
            logger.warning("More frames in world.mp4 than pointcloud.rrf?")

        if self._preview_depth and self._recent_depth_frame is not None:
            frame.img[:] = self._recent_depth_frame.get_color_mapped(
                self._hue_near, self._hue_far, self._dist_near, self._dist_far, self._preview_true_depth
            )

    def init_ui(self):
        self.add_menu()
        self.menu.label = self.pretty_class_name

        append_depth_preview_menu(self)

    def deinit_ui(self):
        if self.menu is not None:
            self.remove_menu()

    def cleanup(self):
        if self.recording_camera is not None:
            self.recording_camera.unregisterDataListener()
            self.data_listener.unregisterIrListener(self.recording_camera)
            del self.recording_camera

    def get_init_dict(self):
        return {
            "preview_depth": self._preview_depth,
            "hue_near": self._hue_near,
            "hue_far": self._hue_far,
            "dist_near": self._dist_near,
            "dist_far": self._dist_far,
            "preview_true_depth": self._preview_true_depth,
        }
