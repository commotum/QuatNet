#include <torch/extension.h>
#include "isokawa_layer.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<IsokawaQuaternionLayer>(m, "IsokawaQuaternionLayer")
        .def(py::init<int, int>())
        .def("forward", &IsokawaQuaternionLayer::forward);
} 