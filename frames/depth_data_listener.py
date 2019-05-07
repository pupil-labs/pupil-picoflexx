import collections
import queue

from picoflexx import roypy
from picoflexx.royale.extension import roypycy
from ..frames.depth_frame import DepthFrame
from ..frames.ir_frame import IRFrame

FramePair = collections.namedtuple("FramePair", ["ir", "depth"])


class DepthDataListener(roypy.IDepthDataListener):
    current_index = 0

    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self._ir_ref = None

        self._data_depth = None
        self._data_ir = None  # type: roypycy.PyIrImage

    def _check_frame(self):
        if self._data_depth is None or self._data_ir is None:
            return
        if roypycy.get_depth_data_ts(self._data_depth) != self._data_ir.timestamp:
            return

        try:
            frame_depth = DepthFrame(self._data_depth)
            frame_ir = IRFrame(self._data_ir)
            frame_depth.index = frame_ir.index = self.current_index
            self.current_index += 1

            self.queue.put(FramePair(ir=frame_ir, depth=frame_depth), block=False)
        except queue.Full:
            pass  # dropping frame pair
        except Exception:
            import traceback

            traceback.print_exc()

        self._data_depth = None
        self._data_ir = None

    def onNewData(self, data):
        self._data_depth = data
        self._check_frame()

    def onNewIrData(self, data: roypycy.PyIrImage):
        self._data_ir = data
        self._check_frame()
