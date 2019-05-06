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
