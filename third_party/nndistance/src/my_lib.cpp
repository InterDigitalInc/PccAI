#include <torch/torch.h>

#include "cpu_ops.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nnd_forward", &nnd_forward, "nnd_forward");
  m.def("nnd_backward", &nnd_backward, "nnd_backward");
}



