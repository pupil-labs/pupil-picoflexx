import logging

from version_utils import VersionFormat
from .utils import roypy_wrap

logger = logging.getLogger(__name__)

try:
    from .. import roypy
    from ..roypy_platform_utils import PlatformHelper
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

from .royale_camera_device import RoyaleCameraDevice
from .royale_replay_device import RoyaleReplayDevice
