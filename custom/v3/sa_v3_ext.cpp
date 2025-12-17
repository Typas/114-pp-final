#include <torch/extension.h>

torch::Tensor sa_forward_v3(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sa_forward_v3, "Self-Attention forward v3 (WMMA Tensor Core, D=64)");
}