#include <Python.h>
#include "royale/IIRImageListener.hpp"

class PyIRImageListener: public royale::IIRImageListener {
    public:
    PyIRImageListener (void* obj, PyObject* (*callback)(void*, royale::IRImage *data)) {
        this->obj = obj;
        this->callback = callback;
    };
    void onNewData (const royale::IRImage *data) {
        (*this->callback)(this->obj, (royale::IRImage *) data);
    };

    void *obj;
    PyObject* (*callback)(void*, royale::IRImage *data);
};
