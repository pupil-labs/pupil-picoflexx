# cython: language_level=3, language=c++
# distutils: language=c++
import cython
import numpy as np
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from libc.stdint cimport uint16_t, uint32_t, uint8_t, int64_t

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

cdef extern from "royale/Pair.hpp" namespace "royale":
    cdef cppclass Pair[T, U]:
        Pair()
        T first
        U second


cdef extern from "royale/IRImage.hpp" namespace "royale":
    cdef struct IRImage:
        int64_t timestamp;
        uint16_t streamId;
        uint16_t width;
        uint16_t height;
        Vector[uint8_t] data;


class PyIrImage:
    def __init__(self, timestamp, stream, width, height, data):
        self.timestamp = timestamp
        self.stream = stream
        self.width = width
        self.height = height
        self.data = data


cdef extern from "royale/IIRImageListener.hpp" namespace "royale":
    cdef cppclass IIRImageListener:
        IIRImageListener()
        void onNewData(IRImage *data)

cdef extern from "roypycy_defs.cpp":
    cdef cppclass PyIRImageListener(IIRImageListener):
        PyIRImageListener(void* obj, object (*callback)(void *irl_l, const IRImage *data))
        void onNewData(IRImage *data)

cdef extern from "royale/LensParameters.hpp" namespace "royale":
    ctypedef struct LensParameters:
        Pair[float, float]     principalPoint;       #!< cx/cy
        Pair[float, float]     focalLength;          #!< fx/fy
        Pair[float, float]     distortionTangential; #!< p1/p2
        Vector[float]          distortionRadial;     #!< k1/k2/k3

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

cdef extern from "royale/ExposureMode.hpp" namespace "royale":
    ctypedef enum ExposureMode:
        MANUAL 'royale::ExposureMode::MANUAL',         # !< Camera exposure mode set to manual
        AUTOMATIC 'royale::ExposureMode::AUTOMATIC'    # !< Camera exposure mode set to automatic

cdef extern from "royale/ICameraDevice.hpp" namespace "royale":
    cdef cppclass ICameraDevice:
        int getExposureMode(ExposureMode &exposureMode, uint16_t streamId)
        int setExposureMode(ExposureMode exposureMode, uint16_t streamId)
        int getLensParameters(LensParameters &params)
        int registerIRImageListener(IIRImageListener *listener)
        int unregisterIRImageListener()


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


cdef c_callback(void *ir_l, IRImage *data) with gil:
    np_data = np.zeros((data.width * data.height, ), np.uint8)

    cdef uint8_t[:] r_data = np_data

    cdef int i = 0;
    for pt in data[0].data:
        r_data[i] = pt
        i+=1

    (<object>ir_l)(PyIrImage(
        data.timestamp,
        data.streamId,
        data.width,
        data.height,
        np_data
    ))


def register_ir_image_listener(camera, callback):
    cdef SwigPyObject *swig_obj = <SwigPyObject*>camera.this
    cdef ICameraDevice **mycpp_ptr = <ICameraDevice**?>swig_obj.ptr

    Py_INCREF(callback)
    cdef PyIRImageListener *ir_l = new PyIRImageListener(<PyObject*>callback, &c_callback)

    mycpp_ptr[0][0].registerIRImageListener(ir_l)

    return <unsigned long>ir_l


def unregister_ir_image_listener(camera, unsigned long ir_l, callback):
    cdef SwigPyObject *swig_obj = <SwigPyObject*>camera.this
    cdef ICameraDevice **mycpp_ptr = <ICameraDevice**?>swig_obj.ptr

    Py_DECREF(callback)
    mycpp_ptr[0][0].unregisterIRImageListener()
    cdef PyIRImageListener* ptr = <PyIRImageListener*> ir_l
    del ptr


def get_lens_parameters(camera):
    cdef SwigPyObject *swig_obj = <SwigPyObject*>camera.this
    cdef ICameraDevice **mycpp_ptr = <ICameraDevice**?>swig_obj.ptr

    cdef LensParameters params;
    mycpp_ptr[0][0].getLensParameters(params)

    return {
        'principalPoint': (params.principalPoint.first, params.principalPoint.second),
        'focalLength': (params.focalLength.first, params.focalLength.second),
        'distortionTangential': (params.distortionTangential.first, params.distortionTangential.second),
        'distortionRadial': tuple(x for x in params.distortionRadial),
    }


def get_exposure_mode(camera):
    cdef SwigPyObject *swig_obj = <SwigPyObject*>camera.this
    cdef ICameraDevice **mycpp_ptr = <ICameraDevice**?>swig_obj.ptr

    cdef ExposureMode exp;
    mycpp_ptr[0][0].getExposureMode(exp, 0)

    return <int>exp


def set_exposure_mode(camera, mode):
    cdef SwigPyObject *swig_obj = <SwigPyObject*>camera.this
    cdef ICameraDevice **mycpp_ptr = <ICameraDevice**?>swig_obj.ptr

    cdef ExposureMode exp_mode;
    if mode == 0:
        exp_mode = ExposureMode.MANUAL
    elif mode == 1:
        exp_mode = ExposureMode.AUTOMATIC
    else:
        raise ValueError('Invalid mode')

    mycpp_ptr[0][0].setExposureMode(exp_mode, 0)


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
