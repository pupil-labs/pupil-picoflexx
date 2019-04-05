import logging

import cv2
import numpy as np
from pyglui import ui

from plugin import Plugin

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
    plugin.menu.append(ui.Switch("_preview_depth", plugin, label="Preview Depth"))

    depth_preview_menu = ui.Growing_Menu("Depth preview settings")
    depth_preview_menu.collapsed = True
    depth_preview_menu.append(
        ui.Info_Text("Set hue and distance ranges for the depth preview.")
    )
    depth_preview_menu.append(
        ui.Slider("_hue_near", plugin, min=0.0, max=1.0, label="Near Hue")
    )
    depth_preview_menu.append(
        ui.Slider("_hue_far", plugin, min=0.0, max=1.0, label="Far Hue")
    )
    depth_preview_menu.append(ui.Button("Fit distance (15th and 85th percentile)", lambda: _fit_distance(plugin)))
    depth_preview_menu.append(
        ui.Slider("_dist_near", plugin, min=0.0, max=4.8, label="Near Distance (m)")
    )
    depth_preview_menu.append(
        ui.Slider("_dist_far", plugin, min=0.2, max=5.0, label="Far Distance (m)")
    )
    depth_preview_menu.append(ui.Switch("_preview_true_depth", plugin, label="Preview using linalg distance"))

    plugin.menu.append(depth_preview_menu)


def _fit_distance(plugin: Plugin):
    if not plugin._recent_depth_frame:
        logger.warning("No recent frame, can't fit hue.")
        return

    if plugin._preview_true_depth:
        depth_data = plugin._recent_depth_frame.true_depth
    else:
        depth_data = plugin._recent_depth_frame._data.z

    plugin._dist_near, plugin._dist_far = np.percentile(depth_data, (15, 85))


def get_hue_color_map(original_depth, hue_near: float, hue_far: float, dist_near: float, dist_far: float):
    depth_values = (original_depth - dist_near) / (dist_far - dist_near)
    depth_values = np.clip(depth_values, 0, 1)
    depth_values = (hue_near + (depth_values * (hue_far - hue_near))) * 255
    depth_values = depth_values.astype(np.uint8)

    hsv = depth_values[:, :, np.newaxis]

    # set saturation and value to 255
    a = np.zeros(hsv.shape, dtype=np.uint8)
    a[:] = 255
    b = np.concatenate((hsv, a, a), axis=2)

    dest = cv2.cvtColor(b, cv2.COLOR_HSV2BGR)

    # blank out missing data
    dest[original_depth == 0] = (0, 0, 0)

    return dest
