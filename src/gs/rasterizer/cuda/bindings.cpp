#include <torch/extension.h>

// forward.cu
std::vector<torch::Tensor> rasterize_forward_cuda(
    torch::Tensor means2d,
    torch::Tensor inv_cov2d,
    torch::Tensor colors,
    torch::Tensor opacities,
    torch::Tensor gaussian_ids,
    torch::Tensor tile_ranges,
    int H, int W
);

// backward.cu
void rasterize_backward_cuda(
    torch::Tensor means2d,
    torch::Tensor inv_cov2d,
    torch::Tensor colors,
    torch::Tensor opacities,
    torch::Tensor gaussian_ids,
    torch::Tensor tile_ranges,
    torch::Tensor final_T,
    torch::Tensor n_contrib,
    torch::Tensor grad_rendered,
    torch::Tensor grad_means2d,
    torch::Tensor grad_inv_cov2d,
    torch::Tensor grad_colors,
    torch::Tensor grad_opacities,
    int H, int W
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_forward",  &rasterize_forward_cuda,  "Rasterize forward (CUDA)");
    m.def("rasterize_backward", &rasterize_backward_cuda, "Rasterize backward (CUDA)");
}
