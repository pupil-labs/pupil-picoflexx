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
    from .backend import Picoflexx_Source, Picoflexx_Manager
    from .example_plugin import Example_Picoflexx_Plugin
    from .player_plugin import Picoflexx_Player_Plugin
except (ImportError, AssertionError):
    import traceback
    traceback.print_exc()

