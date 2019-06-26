import struct
from typing import Optional


class RrfHelper:
    def __init__(self, rrf_path: str):
        self.num_frames = None
        self.offs_first_frame = None
        self.frames = []
        self.frame_timestamps = []

        self._f = open(rrf_path, 'rb')
        try:
            self._load()
        finally:
            self._f.close()
            self._f = None
            del self._f

    def _read(self, fmt: str, offset: Optional[int] = None):
        """
        Reads data as specified in the format string, optionally seeking in
        the file before reading.

        :param fmt: The struct format string to read
        :param offset: The optional offset to seek to in the file
        :return: The data item read, or a list of such if there were multiple
        """
        if offset is not None:
            self._f.seek(offset)

        size = struct.calcsize(fmt)
        data = self._f.read(size)
        unpacked = struct.unpack(fmt, data)

        if len(unpacked) == 1:
            return unpacked[0]

        return unpacked

    def _load(self):
        # The number of frames is an integer at offset 0x79
        self.num_frames = self._read('<I', 0x79)
        # The byte offset to the first frame in is an integer at offset 0x8A
        self.offs_first_frame = self._read('<I', 0x8A)

        # Each frame is prefixed with it's size,  in testing this has been
        # constant if the frame size ends up being variable, the loop will
        # need to be reworked.
        frame_sz = 0
        for f_idx in range(self.num_frames):
            info = self._read('<IHHHHHQ', self.offs_first_frame + frame_sz * f_idx)
            frame_sz, fps, width, height, unk1, unk2, frame_ts = info
            self.frames.append((frame_ts, fps, width, height, unk1, unk2))
            self.frame_timestamps.append(frame_ts / 1000)
