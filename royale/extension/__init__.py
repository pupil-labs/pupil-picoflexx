import logging

logger = logging.getLogger(__name__)

try:
    from . import roypycy
except ImportError:
    import traceback

    logger.debug(traceback.format_exc())
    logger.warning("Pico Flexx backend requirements (roypycy) not installed properly")
    raise
