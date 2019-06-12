import zlib

import numpy as np

from picoflexx.frames import IRFrame, DepthFrame

FLAG_IR = 1 << 0
FLAG_POINTCLOUD = 1 << 1
FLAG_NOISE = 1 << 2
FLAG_CONFIDENCE = 1 << 3
FLAG_ALL = FLAG_IR | FLAG_POINTCLOUD | FLAG_NOISE | FLAG_CONFIDENCE

FLAG_STRIP = 1 << 4
FLAG_COMPRESSED = 1 << 5


def _types_from_flags(flags: int):
    types = []

    if flags & FLAG_ALL == 0:
        raise ValueError("No data included in flags?")

    if flags & FLAG_IR != 0:
        types.append(("ir", np.uint8))

    if flags & FLAG_POINTCLOUD != 0:
        types.append(("x", np.float32))
        types.append(("y", np.float32))
        types.append(("z", np.float32))

    if flags & FLAG_NOISE != 0:
        types.append(("noise", np.float32))

    if flags & FLAG_CONFIDENCE != 0:
        types.append(("depthConfidence", np.uint8))

    return np.dtype(types)


def encode_frame(flags: int, ir: IRFrame, depth: DepthFrame):
    data, dtype = [], _types_from_flags(flags)

    if flags & FLAG_IR != 0:
        data.append(ir.data.ravel())  # uint8

    if flags & FLAG_POINTCLOUD != 0:
        data.append(depth._data.x)  # float32
        data.append(depth._data.y)  # float32
        data.append(depth._data.z)  # float32

    if flags & FLAG_NOISE != 0:
        data.append(depth.noise.ravel())  # float32

    if flags & FLAG_CONFIDENCE != 0:
        data.append(depth.confidence.ravel())  # uint8

    array = np.empty(len(data[0]), dtype=dtype)
    for field, d in zip(dtype.names, data):
        array[field] = d

    if flags & FLAG_STRIP != 0:
        old_len = len(array)
        array = array[np.bitwise_or(depth.confidence.ravel() > 15, depth._data.z < 0.1)]
        new_len = len(array)
        print("Stripped: {}/{} = {:.2f}".format(new_len, old_len, new_len / old_len))

    raw_data = array.tostring()

    if flags & FLAG_COMPRESSED != 0:
        obj = zlib.compressobj(level=1)

        compressed_data = obj.compress(raw_data) + obj.flush()

        print("Compressed: {}/{} = {:.2f}".format(len(compressed_data), len(raw_data), len(compressed_data) / len(raw_data)))

        return compressed_data
    else:
        return raw_data


def decode_frame(flags: int, data: bytearray) -> np.ndarray:
    if flags & FLAG_COMPRESSED != 0:
        obj = zlib.decompressobj()
        decompressed_data = obj.decompress(data) + obj.flush()

        print("Decompressed: {}/{} = {:.2f}".format(len(data), len(decompressed_data), len(data) / len(decompressed_data)))
        data = decompressed_data

    if flags & FLAG_STRIP != 0:
        raise NotImplementedError("Decoding FLAG_STRIP")

    dtype = _types_from_flags(flags)

    return np.fromstring(data, dtype=dtype).view(np.recarray)
