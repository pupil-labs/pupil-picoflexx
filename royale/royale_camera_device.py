import logging
import queue
from typing import Optional, Tuple

from picoflexx.royale.extension import roypycy
from . import roypy_wrap, roypy
from .lens_paramaters import LensParamaters
from ..frames import DepthDataListener
from ..frames.depth_data_listener import FramePair

logger = logging.getLogger(__name__)


class RoyaleCameraDevice:
    def __init__(self):
        self._camera = None
        self._queue = queue.Queue(maxsize=1)
        self._data_listener = DepthDataListener(self._queue)
        self._ir_ref = None
        self.camera_id = None

        self.cached_exposure_mode = None

    def initialize(self):
        cam_manager = roypy.CameraManager()
        try:
            cam_id = cam_manager.getConnectedCameraList()[0]
        except IndexError:
            logger.error("No Pico Flexx camera connected")
            return

        self.camera_id = cam_id
        self._camera = cam_manager.createCamera(cam_id)
        self._camera.initialize()

        self.register_ir_listener(self._data_listener.onNewIrData)
        self.register_data_listener(self._data_listener)

        try:
            # can sporadically claim "Camera is disconnected"
            self.start_capture()
        except RuntimeError as e:
            return

    def close(self):
        if self.is_connected() and self.is_capturing():
            self.stop_capture()

        self.unregister_data_listener()
        self.unregister_ir_listener()
        self._camera = None

    def start_capture(self, **kwargs):
        return roypy_wrap(
            self._camera.startCapture,
            tag='Failed to start camera',
            reraise=True,
            level=logging.ERROR,
            **kwargs,
        )

    def stop_capture(self, **kwargs):
        return roypy_wrap(
            self._camera.stopCapture,
            **kwargs,
        )

    def set_exposure_mode(self, exposure_mode: bool, **kwargs) -> bool:
        roypy_wrap(
            self._camera.setExposureMode,
            roypy.ExposureMode_AUTOMATIC if exposure_mode else roypy.ExposureMode_MANUAL,
            **kwargs,
        )

        return self.get_exposure_mode()

    def get_exposure_mode(self) -> bool:
        self.cached_exposure_mode = self._camera.getExposureMode() == roypy.ExposureMode_AUTOMATIC

        return self.cached_exposure_mode

    def set_exposure(self, exposure: int, **kwargs):
        return roypy_wrap(
            self._camera.setExposureTime,
            exposure,
            **kwargs,
        )

    def set_usecase(self, usecase: str, start_capturing: bool = True, **kwargs):
        roypy_wrap(
            self._camera.setUseCase,
            usecase,
            **kwargs,
        )

        if start_capturing and not self.is_capturing():
            roypy_wrap(self._camera.startCapture)

    def get_usecases(self):
        return self._camera.getUseCases()

    def get_current_usecase(self):
        return self._camera.getCurrentUseCase()

    def get_exposure_limits(self):
        limits = self._camera.getExposureLimits()

        return limits.first, limits.second

    def is_connected(self):
        return self._camera.isConnected()

    def is_capturing(self):
        return self._camera.isCapturing()

    def start_recording(self, file_name: str, number_of_frames: int = 0, frame_skip: int = 0, ms_skip: int = 0, **kwargs):
        return roypy_wrap(
            self._camera.startRecording,
            file_name,
            number_of_frames,
            frame_skip,
            ms_skip,
            **kwargs,
        )

    def stop_recording(self, **kwargs):
        return roypy_wrap(
            self._camera.stopRecording,
            **kwargs,
        )

    def get_max_frame_rate(self) -> int:
        return self._camera.getMaxFrameRate()

    def get_frame_rate(self) -> int:
        return self._camera.getFrameRate()

    def set_frame_rate(self, frame_rate: int, **kwargs):
        return roypy_wrap(
            self._camera.setFrameRate,
            frame_rate,
            **kwargs,
        )

    def get_lens_parameters(self) -> LensParamaters:
        params = roypycy.get_lens_parameters(self._camera)

        return LensParamaters(params)

    @property
    def online(self):
        return self._camera and self.is_connected() and self.is_capturing()

    def register_ir_listener(self, ir_callback):
        if self._ir_ref is not None:
            self.unregister_ir_listener()

        self._ir_ref = roypycy.register_ir_image_listener(self._camera, ir_callback), ir_callback

    def unregister_ir_listener(self):
        if self._ir_ref is None:
            return

        roypycy.unregister_ir_image_listener(self._camera, self._ir_ref[0], self._ir_ref[1])
        self._ir_ref = None

    def register_data_listener(self, data_listener: roypy.IDepthDataListener, **kwargs):
        return roypy_wrap(
            self._camera.registerDataListener,
            data_listener,
            **kwargs,
        )

    def unregister_data_listener(self, **kwargs):
        return roypy_wrap(
            self._camera.unregisterDataListener,
            **kwargs,
        )

    def get_frame(self, block: bool = True, timeout: Optional[float] = 0.02) -> Optional[FramePair]:
        try:
            return self._queue.get(block, timeout)
        except queue.Empty:
            return None

    def get_max_sensor_width(self) -> int:
        return self._camera.getMaxSensorWidth()

    def get_max_sensor_height(self) -> int:
        return self._camera.getMaxSensorHeight()

    def get_max_sensor_dimensions(self) -> Tuple[int, int]:
        return self.get_max_sensor_width(), self.get_max_sensor_height()

    def get_camera_name(self) -> str:
        return self._camera.getCameraName()
