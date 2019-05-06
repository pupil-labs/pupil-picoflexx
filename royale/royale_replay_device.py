import logging

from . import RoyaleCameraDevice, roypy_wrap
from .extension import roypycy
from .. import roypy

logger = logging.getLogger(__name__)


class RoyaleReplayDevice(RoyaleCameraDevice):
    def __init__(self):
        super().__init__()

        self._replay = None

    def initialize(self, file_path: str):
        cam_manager = roypy.CameraManager()
        self._camera = cam_manager.createCamera(file_path)  # type: roypy.ICameraDevice
        roypy_wrap(self._camera.initialize)

        # As we're accessing a recording, we need to cast the ICameraDevice
        # to IReplay to access extra functionality
        self._replay = roypycy.toReplay(self._camera)  # type: roypycy.PyIReplay

        self.register_ir_listener(self._data_listener.onNewIrData)
        self.register_data_listener(self._data_listener)

    def seek(self, frame_number: int):
        return self._replay.seek(frame_number)

    def loop(self, restart: int):
        return self._replay.loop(restart)

    def use_timestamps(self, timestamps_used: int):
        return self._replay.use_timestamps(timestamps_used)

    def frame_count(self) -> int:
        return self._replay.frame_count()

    def current_frame(self) -> int:
        return self._replay.current_frame()

    def pause(self):
        return self._replay.pause()

    def resume(self):
        return self._replay.resume()

    def get_file_version(self) -> int:
        return self._replay.get_file_version()
