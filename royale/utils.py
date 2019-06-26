import logging
from typing import Tuple, Optional

from picoflexx import roypy
from picoflexx.utils import logger


def roypy_wrap(
        func,
        *args,
        check_status: bool = True,
        tag: str = None,
        reraise: bool = False,
        level: int = logging.WARNING,
        **kwargs
) -> Tuple[Optional[RuntimeError], Optional[int]]:
    """
    Wrap a call to roypy catching errors from roypy and (optionally) reraising
    them after logging.

    By default, will ensure the call returned a successful status.

    :param func: roypy function to execute
    :param args: Arguments to pass to the function
    :param check_status: Whether to check the status returned by the function
    (default: True)
    :param tag: (Optional) Name of the function to be used when logging
    :param reraise: Whether to reraise any errors from roypy (default: False)
    :param level: The logging level to log messages on
    :param kwargs: Any kwargs to pass to the roypy function
    :return:
    """

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
