import logging
import os
import subprocess
from typing import Optional

import cv2
import numpy as np
from pyglui import ui

from plugin import Plugin

MICRO_TO_SECONDS = 1e-6

logger = logging.getLogger(__name__)


def append_depth_preview_menu(plugin: Plugin):
    text = ui.Info_Text(
        "Enabling Preview Depth will display a colourised version of the data "
        "based on the depth. Disabling the option will display the "
        "according IR image." +
        ("Independent of which option is selected, the IR image stream will "
         "be stored to `world.mp4` during a recording."
         if plugin.g_pool.app == 'capture' else "")
    )
    plugin.menu.append(text)
    plugin.menu.append(ui.Switch("preview_depth", plugin, label="Preview Depth"))

    depth_preview_menu = ui.Growing_Menu("Depth preview settings")
    depth_preview_menu.collapsed = True
    depth_preview_menu.append(
        ui.Info_Text("Set hue and distance ranges for the depth preview.")
    )
    depth_preview_menu.append(
        ui.Slider("hue_near", plugin, min=0.0, max=1.0, label="Near Hue")
    )
    depth_preview_menu.append(
        ui.Slider("hue_far", plugin, min=0.0, max=1.0, label="Far Hue")
    )
    depth_preview_menu.append(ui.Button("Fit distance (15th and 85th percentile)", lambda: _fit_distance(plugin)))
    depth_preview_menu.append(
        ui.Slider("dist_near", plugin, min=0.0, max=4.8, label="Near Distance (m)")
    )
    depth_preview_menu.append(
        ui.Slider("dist_far", plugin, min=0.2, max=5.0, label="Far Distance (m)")
    )
    depth_preview_menu.append(ui.Switch("preview_true_depth", plugin, label="Preview using linalg distance"))

    plugin.menu.append(depth_preview_menu)


def _fit_distance(plugin: Plugin):
    if not plugin.recent_depth_frame:
        logger.warning("No recent frame, can't fit hue.")
        return

    if plugin.preview_true_depth:
        depth_data = plugin.recent_depth_frame.true_depth
    else:
        depth_data = plugin.recent_depth_frame._data.z

    near, far = np.percentile(depth_data, (15, 85))
    plugin.dist_near, plugin.dist_far = float(near), float(far)


def find_setting_source(g_pool):
    for plugin in g_pool.plugins:
        if plugin.alive and hasattr(plugin, 'preview_true_depth'):
            return plugin

    return None


def get_hue_color_map(original_depth, hue_near: float, hue_far: float, dist_near: float, dist_far: float):
    depth_values = (original_depth - dist_near) / (dist_far - dist_near)
    depth_values = np.clip(depth_values, 0, 1)
    depth_values = (hue_near + (depth_values * (hue_far - hue_near))) * 360
    depth_values = depth_values.astype(np.float32)

    hsv = depth_values[:, :, np.newaxis]

    # set saturation and value to 255
    a = np.zeros(hsv.shape, dtype=np.float32)
    a[:] = 1
    b = np.concatenate((hsv, a, a), axis=2)

    dest = cv2.cvtColor(b, cv2.COLOR_HSV2BGR)
    dest *= 255
    dest = dest.astype(np.uint8)

    # blank out missing data
    dest[original_depth == 0] = (0, 0, 0)

    return dest


def clamp(a, value, b):
    return max(a, min(value, b))


def get_version(directory) -> Optional[str]:
    if not os.path.isdir(directory):
        directory = os.path.dirname(directory)

    try:
        process = subprocess.Popen(["git", "describe", "--tags", "--always", "--dirty"], cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if process.wait() == 0:
            return process.stdout.readline().decode('utf-8').strip()
    except FileNotFoundError:
        pass  # git not on path

    return None


def monkeypatch_zre_msg_number1():
    import struct

    def patched_put(self, nr):
        d = struct.pack('>B', nr % 256)
        self.struct_data += d

    def patched_get(self):
        num = struct.unpack_from('>B', self.struct_data, offset=self._needle)
        self._needle += struct.calcsize('>B')
        return num[0]

    from pyre import zre_msg
    zre_msg.ZreMsg._put_number1 = patched_put
    zre_msg.ZreMsg._get_number1 = patched_get


def monkeypatch_pyre_peer_status():
    def patched(self):
        return self.status & 0xFF

    from pyre.pyre_peer import PyrePeer
    PyrePeer.get_status = patched
