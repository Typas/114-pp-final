#include <torch/extension.h>

// CUDA forward declaration
torch::Tensor sa_forward_cuda(torch::Tensor Q,
                              torch::Tensor K,
                              torch::Tensor V,
                              double scale);

torch::Tensor sa_forward(torch::Tensor Q,
                         torch::Tensor K,
                         torch::Tensor V,
                         double scale) {
  TORCH_CHECK(Q.is_cuda(), "Q must be CUDA tensor");
  TORCH_CHECK(K.is_cuda(), "K must be CUDA tensor");
  TORCH_CHECK(V.is_cuda(), "V must be CUDA tensor");
  TORCH_CHECK(Q.scalar_type() == at::kHalf, "Q must be float16");
  TORCH_CHECK(K.scalar_type() == at::kHalf, "K must be float16");
  TORCH_CHECK(V.scalar_type() == at::kHalf, "V must be float16");
  TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
  TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
  TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
  TORCH_CHECK(Q.dim() == 4, "Q shape must be [B,H,N,D]");
  TORCH_CHECK(K.sizes() == Q.sizes(), "K must match Q shape");
  TORCH_CHECK(V.sizes() == Q.sizes(), "V must match Q shape");

  return sa_forward_cuda(Q, K, V, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sa_forward, "Self-Attention forward (CUDA, v0)");
}
