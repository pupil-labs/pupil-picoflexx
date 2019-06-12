"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

try:
    from picoflexx.royale.extension import roypycy
    from .utils import get_version as _get_version

    __version__ = _get_version(__file__) or "Unknown"

    from .backend import Picoflexx_Source, Picoflexx_Manager
    from .example_plugin import Example_Picoflexx_Plugin
    from .player_plugin import Picoflexx_Player_Plugin

    from .mobile.full_remote_rrf import Full_Remote_RRF_Manager
    from .mobile.full_remote_rrf import Full_Remote_RRF_Source
except (ImportError, AssertionError):
    import traceback
    traceback.print_exc()

