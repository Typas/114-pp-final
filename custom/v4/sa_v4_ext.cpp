#include <torch/extension.h>

torch::Tensor sa_forward_v4(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sa_forward_v4, "Self-Attention forward v4 (Optimized for N=197, D=64)");
}
