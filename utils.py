import logging
import os
import subprocess
from typing import Optional

import cv2
import numpy as np

MICRO_TO_SECONDS = 1e-6

logger = logging.getLogger(__name__)


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
