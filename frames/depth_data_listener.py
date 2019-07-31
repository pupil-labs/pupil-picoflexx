import collections
import queue
from typing import Optional

from picoflexx import roypy
from picoflexx.royale.extension import roypycy
from picoflexx.roypy import DepthData
from ..frames.depth_frame import DepthFrame
from ..frames.ir_frame import IRFrame

FramePair = collections.namedtuple("FramePair", ["ir", "depth"])


class DepthDataListener(roypy.IDepthDataListener):
    current_index = 0

    def __init__(self, queue: queue.Queue):
        """
        Instantiate the DepthDataListener with the given queue.

        :param queue: The queue to which FramePairs will be added - usually has
         a size limit of 1
        """

        super().__init__()
        self.queue = queue
        self._ir_ref = None

        self._data_depth = None  # type: Optional[DepthData]
        self._data_ir = None  # type: Optional[roypycy.PyIrImage]

    def _check_frame(self):
        """
        Checks if the next depth and ir frame pair have arrived.

        If this is the case and the queue isn't full, add them to the queue.
        """

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

    def onNewData(self, data: DepthData):
        self._data_depth = data
        self._check_frame()

    def onNewIrData(self, data: roypycy.PyIrImage):
        self._data_ir = data
        self._check_frame()
