# cython: language_level=3, language=c++
# distutils: language=c++
import cython
import numpy as np
from libc.stdint cimport uint16_t, uint32_t, uint8_t

# From https://stackoverflow.com/a/29343772
cdef extern from "swigpyobject.h":
    ctypedef struct SwigPyObject:
        void *ptr

cdef extern from "royale/Vector.hpp" namespace "royale":
    cdef cppclass Vector[T]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        Vector()
        void push_back(T&)
        T& operator[](int)
        T& at(int)
        const T* data()
        iterator begin()
        iterator end()
        size_t count()

cdef extern from "royale/DepthData.hpp" namespace "royale":
    ctypedef struct DepthData:
        int                         version;         # !< version number of the data format
        # std::chrono::microseconds   timeStamp;       # !< timestamp in microseconds precision (time since epoch 1970)
        # StreamId                    streamId;        # !< stream which produced the data
        uint16_t                    width;           # !< width of depth image
        uint16_t                    height;          # !< height of depth image
        Vector[uint32_t]            exposureTimes;   # !< exposureTimes retrieved from CapturedUseCase
        Vector[DepthPoint]          points;          # !< array of points

    ctypedef struct DepthPoint:
        float x;                 # !< X coordinate [meters]
        float y;                 # !< Y coordinate [meters]
        float z;                 # !< Z coordinate [meters]
        float noise;             # !< noise value [meters]
        uint16_t grayValue;      # !< 16-bit gray value
        uint8_t depthConfidence; # !< value from 0 (invalid) to 255 (full confidence)


def get_depth_data(depthdata):
    cdef SwigPyObject *swig_obj = <SwigPyObject*>depthdata.this
    cdef DepthData *mycpp_ptr = <DepthData*?>swig_obj.ptr
    cdef DepthData my_instance = mycpp_ptr[0]

    result = np.zeros((my_instance.points.count(), ), dtype=np.float64)
    cdef double[:] result_view = result

    cdef int i = 0;
    for pt in my_instance.points:
        result_view[i] = pt.z
        i += 1

    return result


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_backend_data(depthdata):
    # Obtain the raw DepthData instance from the Swig object
    cdef SwigPyObject *swig_obj = <SwigPyObject*>depthdata.this
    cdef DepthData *mycpp_ptr = <DepthData*?>swig_obj.ptr
    cdef DepthData my_instance = mycpp_ptr[0]

    data = np.zeros((my_instance.points.count(), ),
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("noise", np.float32),
                ("grayValue", np.uint16),
                ("depthConfidence", np.uint8),
            ],
        ).view(np.recarray)

    # Setup memoryviews
    cdef float[:] r_x = data.x
    cdef float[:] r_y = data.y
    cdef float[:] r_z = data.z
    cdef float[:] r_noise = data.noise
    cdef uint16_t[:] r_grayValue = data.grayValue
    cdef uint8_t[:] r_depthConfidence = data.depthConfidence

    cdef int i = 0;
    for pt in my_instance.points:
        r_x[i] = pt.x
        r_y[i] = pt.y
        r_z[i] = pt.z
        r_noise[i] = pt.noise
        r_grayValue[i] = pt.grayValue
        r_depthConfidence[i] = pt.depthConfidence
        i+=1

    return data
