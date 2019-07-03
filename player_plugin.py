import bisect
import logging
import os

import gl_utils
from picoflexx.royale import RoyaleReplayDevice
from picoflexx.royale.rrf_utils import RrfHelper
from video_capture import File_Source
from .common import PicoflexxCommon

logger = logging.getLogger(__name__)


class Picoflexx_Player_Plugin(PicoflexxCommon):
    uniqueness = "by_class"
    icon_chr = chr(0xE886)
    icon_font = "pupil_icons"
    expected_app = {"player", "exporter"}

    def __init__(self, g_pool, **kwargs):
        super().__init__(g_pool, **kwargs)
        self.order = 0.001  # Ensure we're after FileSource but before anything else
        self.menu = None

        self.recording_replay = RoyaleReplayDevice()
        self._current_recording_idx = None

        # Abort if the plugin is enabled in an unexpected app (Capture)
        if self.g_pool.app not in self.expected_app:
            self.gl_display = self._abort
            logger.error("Expected app {!r} instead of {!r}!.".format(
                self.expected_app,
                self.g_pool.app,
            ))
            return

        meta_info = self.g_pool.meta_info
        self.offset = float(meta_info.get('Royale Timestamp Offset', 0))

        cloud_path = os.path.join(self.g_pool.rec_dir, 'pointcloud.rrf')
        if not os.path.exists(cloud_path):
            self.recent_events = self._abort
            logger.error("There is no pointcloud in this recording.")
            return

        self.rrf_entries = list(self._rrf_container_times(0, cloud_path))

        # enumerate all pointclouds in the recording
        p_idx = 1
        while True:
            fn = os.path.join(self.g_pool.rec_dir, 'pointcloud_{}.rrf'.format(p_idx))
            if not os.path.exists(fn):
                break

            self.rrf_entries += self._rrf_container_times(p_idx, fn)

            p_idx += 1

        self.rrf_timestamps = list(map(lambda x: x[2], self.rrf_entries))

        logger.info("{} pointclouds present".format(p_idx))

        self.load_recording_container(0)

    def load_recording_container(self, container_idx: int):
        if self._current_recording_idx == container_idx:
            return

        logger.info("Switching to container {}".format(container_idx))
        self._current_recording_idx = container_idx

        if container_idx == 0:
            filename = "pointcloud.rrf"
        else:
            filename = "pointcloud_{}.rrf".format(container_idx)

        cloud_path = os.path.join(self.g_pool.rec_dir, filename)
        self.recording_replay.initialize(cloud_path)

    def _rrf_container_times(self, container_idx: int, path: str):
        return map(
            lambda x: (container_idx, x[0], x[1]),
            enumerate(RrfHelper(path).frame_timestamps)
        )

    def _abort(self, _=None):
        self.alive = False
        self.g_pool.plugins.clean()

    def recent_events(self, events):
        frame = events.get("frame")
        if not frame:
            return

        capture = self.g_pool.capture  # type: File_Source
        target_entry = capture.videoset.lookup[capture.current_frame_idx]

        # sometimes the entries representing missing data are `[[entry]]`
        # instead of just `entry`?
        if len(target_entry) == 1 or target_entry[0] == -1:
            self._recent_frame = None
            self._recent_depth_frame = None
            return

        # Find the index of the rrf frame with closest timestamp (best frame
        # should be i_right or i_right-1).
        target_ts = target_entry[2] - self.offset
        i_right = bisect.bisect_right(self.rrf_timestamps, target_ts)

        # Calculate timestamp differences for the surrounding frames so we can
        # select the optimal frame.
        diffs = [
            (di,
             target_ts - self.rrf_timestamps[
                 max(0, min(i_right + di, len(self.rrf_entries) - 1))])
            for di in range(-2, 3)
        ]
        # print(i_right, diffs)
        best_idx = max(0, min(diffs, key=lambda x: abs(x[1]))[0] + i_right)
        # print(i_right, best_idx)
        rrf_entry = self.rrf_entries[best_idx]

        # print(target_entry, rrf_entry)

        # Ensure the rrf frame we've selected falls within the bounds of the
        # recording.
        if 0 <= rrf_entry[1] < self.recording_replay.frame_count():
            if rrf_entry[1] != self.recording_replay.current_frame() \
                    or self._recent_depth_frame is None \
                    or self._current_recording_idx != rrf_entry[0]:
                self.load_recording_container(rrf_entry[0])
                self.recording_replay.seek(rrf_entry[1])

                # depth data appears to arrive within ~9-12 microseconds
                self._recent_frame, self._recent_depth_frame = self.recording_replay.get_frame()
                self.current_exposure = self._recent_depth_frame.exposure_times[1]

                self._recent_frame.timestamp += self.offset
                self._recent_depth_frame.timestamp += self.offset

            events["depth_frame"] = self._recent_depth_frame

        if self.preview_depth and self._recent_depth_frame is not None:
            frame.img[:] = self._recent_depth_frame.get_color_mapped(
                self.hue_near, self.hue_far, self.dist_near, self.dist_far, self.preview_true_depth
            )

    def gl_display(self):
        if self.recording_replay.is_connected() and self.preview_depth:
            gl_utils.glPushMatrix()
            gl_utils.adjust_gl_view(*self.g_pool.camera_render_size)
            self._render_color_bar()
            gl_utils.glPopMatrix()

    @property
    def online(self):
        return True

    def init_ui(self):
        self.add_menu()
        self.menu.label = self.pretty_class_name

        self.append_depth_preview_menu()

    def cleanup(self):
        if self.recording_replay is not None:
            self.recording_replay.close()
            self.recording_replay = None

    def get_init_dict(self):
        return super(PicoflexxCommon, self).get_init_dict()
