import struct


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

    def _read(self, fmt: str, offset: int = None):
        if offset is not None:
            self._f.seek(offset)

        size = struct.calcsize(fmt)
        data = self._f.read(size)
        unpacked = struct.unpack(fmt, data)

        if len(unpacked) == 1:
            return unpacked[0]

        return unpacked

    def _load(self):
        self.num_frames = self._read('<I', 0x79)
        self.offs_first_frame = self._read('<I', 0x8A)

        frame_sz = 0
        for f_idx in range(self.num_frames):
            info = self._read('<IHHHHHQ', self.offs_first_frame + frame_sz * f_idx)
            frame_sz, fps, width, height, unk1, unk2, frame_ts = info
            self.frames.append((frame_ts, fps, width, height, unk1, unk2))
            self.frame_timestamps.append(frame_ts / 1000)
