import logging
import os
from decimal import Decimal
from typing import Optional

from pyglui import ui

from video_capture import File_Source
from . import roypy, roypycy
from .common import PicoflexxCommon
from .roypycy import PyIReplay
from .utils import append_depth_preview_menu, roypy_wrap

logger = logging.getLogger(__name__)


class Picoflexx_Player_Plugin(PicoflexxCommon):
    uniqueness = "by_class"
    icon_chr = chr(0xE886)
    icon_font = "pupil_icons"

    def __init__(self, g_pool, **kwargs):
        super().__init__(g_pool)
        self.order = 0.001  # Ensure we're after FileSource but before anything else
        self.menu = None

        self.recording_camera = None  # type: Optional[roypy.ICameraDevice]
        self.recording_replay = None  # type: Optional[PyIReplay]
        self.frame_offset = 0  # type: int
        self._found_frame_offset = False

        cloud_path = os.path.join(self.g_pool.rec_dir, 'pointcloud.rrf')
        if not os.path.exists(cloud_path):
            self.recent_events = self._abort
            logger.error("There is no pointcloud in this recording.")
            return

        cam_manager = roypy.CameraManager()
        self.recording_camera = cam_manager.createCamera(cloud_path)  # type: roypy.ICameraDevice
        roypy_wrap(self.recording_camera.initialize)

        # As we're accessing a recording, we need to cast the ICameraDevice
        # to IReplay to access extra functionality
        self.recording_replay = roypycy.toReplay(self.recording_camera)  # type: roypycy.PyIReplay
        self.recording_camera.registerDataListener(self.data_listener)
        self.data_listener.registerIrListener(self.recording_camera)

    def _abort(self, _):
        self.alive = False
        self.g_pool.plugins.clean()

    def _find_frame_offset(self) -> int:
        meta_info = self.g_pool.meta_info
        offset = meta_info.get('Royale Timestamp Offset', None)
        capture = self.g_pool.capture  # type: File_Source

        if offset is None:
            return 0

        dec_offset = Decimal(offset)

        def compare_offsets(av, rrf):
            target_entry = capture.videoset.lookup[av]
            _, av_frame_idx, av_frame_ts = target_entry

            self.recording_replay.seek(rrf)
            rrf_ts = Decimal(self.queue.get()[0].timestamp)
            av_in_unix = Decimal(av_frame_ts) - dec_offset

            return abs(av_in_unix - rrf_ts)

        best_diff = compare_offsets(0, 0)

        rrf_offset = 0
        for i in range(1, min(45 * 5, self.recording_replay.frame_count())):
            diff = compare_offsets(0, i)

            if diff < best_diff:
                best_diff = diff
                rrf_offset = i
            else:
                break

        if rrf_offset != 0:
            return -rrf_offset

        # check if RRF started earlier than world.mp4
        av_offset = 0
        for i in range(1, min(45 * 5, capture.get_frame_count())):
            diff = compare_offsets(i, 0)

            if diff < best_diff:
                best_diff = diff
                av_offset = i
            else:
                break

        return av_offset

    def recent_events(self, events):
        frame = events.get("frame")
        if not frame:
            return

        if not self._found_frame_offset:
            self.frame_offset = self._find_frame_offset()
            self._found_frame_offset = True
            print('Frame offset: {}'.format(self.frame_offset))

        capture = self.g_pool.capture  # type: File_Source
        target_entry = capture.videoset.lookup[capture.current_frame_idx]
        true_frame = target_entry[1] + self.frame_offset

        if 0 <= true_frame < self.recording_replay.frame_count():
            if true_frame != self.recording_replay.current_frame() \
                    or self._recent_depth_frame is None:
                self.recording_replay.seek(true_frame)

                # depth data appears to arrive within ~9-12 microseconds
                self._recent_frame, self._recent_depth_frame = self.queue.get()
                self.current_exposure = self._recent_depth_frame.exposure_times[1]

            events["depth_frame"] = self._recent_depth_frame

        if self.preview_depth and self._recent_depth_frame is not None:
            frame.img[:] = self._recent_depth_frame.get_color_mapped(
                self.hue_near, self.hue_far, self.dist_near, self.dist_far, self.preview_true_depth
            )

    def init_ui(self):
        self.add_menu()
        self.menu.label = self.pretty_class_name

        self.menu.append(ui.Slider("frame_offset", self, min=-15, max=15, label="Frame offset"))
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
        return super(PicoflexxCommon, self).get_init_dict()
