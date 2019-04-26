import queue
from typing import Optional

from plugin import Plugin
from .frames import DepthDataListener, DepthFrame, IRFrame


class PicoflexxCommon(Plugin):
    def __init__(self, g_pool, *args, **kwargs):
        super(PicoflexxCommon, self).__init__(g_pool, *args, **kwargs)

        self.hue_near = kwargs.get('hue_near', 0.0)
        self.hue_far = kwargs.get('hue_far', 0.75)
        self.dist_near = kwargs.get('dist_near', 0.14)
        self.dist_far = kwargs.get('dist_far', 5.0)
        self.preview_true_depth = kwargs.get('preview_true_depth', False)
        self.preview_depth = kwargs.get('preview_depth', True)

        self._recent_depth_frame = None  # type: Optional[DepthFrame]
        self._recent_frame = None  # type: Optional[IRFrame]
        self.current_exposure = 0  # type: int

        self.queue = queue.Queue(maxsize=1)
        self.data_listener = DepthDataListener(self.queue)

    @property
    def recent_frame(self) -> Optional[IRFrame]:
        return self._recent_frame

    @property
    def recent_depth_frame(self) -> Optional[DepthFrame]:
        return self._recent_depth_frame

    def get_init_dict(self):
        return {
            "preview_depth": self.preview_depth,
            "hue_near": self.hue_near,
            "hue_far": self.hue_far,
            "dist_near": self.dist_near,
            "dist_far": self.dist_far,
            "preview_true_depth": self.preview_true_depth,
        }
