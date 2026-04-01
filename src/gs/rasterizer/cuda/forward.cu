#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define BLOCK_SIZE (TILE_SIZE * TILE_SIZE)  // 256 threads per block

__global__ void rasterize_forward_kernel(
    const float* __restrict__ means2d,      // (N, 2)
    const float* __restrict__ inv_cov2d,    // (N, 4)  [a, b, c, d] = [[a,b],[c,d]]
    const float* __restrict__ colors,       // (N, 3)
    const float* __restrict__ opacities,    // (N,)
    const int*   __restrict__ gaussian_ids, // (P,) sorted by tile, depth
    const int*   __restrict__ tile_ranges,  // (n_tiles + 1,)
    float* __restrict__ rendered,           // (3, H, W) output
    float* __restrict__ final_T,           // (H, W)  final transmittance (for backward)
    int*   __restrict__ n_contrib,          // (H, W)  num contributing Gaussians per pixel (for backward)
    int H, int W
) {
    // Each block handles one tile (TILE_SIZE x TILE_SIZE pixels)
    // Each thread handles one pixel within the tile
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int n_tiles_x = (W + TILE_SIZE - 1) / TILE_SIZE;
    int tile_id = tile_y * n_tiles_x + tile_x;

    // Pixel coordinate for this thread
    int px = tile_x * TILE_SIZE + (threadIdx.x % TILE_SIZE);
    int py = tile_y * TILE_SIZE + (threadIdx.x / TILE_SIZE);
    bool inside = (px < W && py < H);

    // Range of sorted Gaussian indices for this tile
    int g_start = tile_ranges[tile_id];
    int g_end   = tile_ranges[tile_id + 1];

    // Per-thread accumulator
    float T = 1.0f;
    float r = 0.0f, g = 0.0f, b = 0.0f;
    int count = 0;

    // Shared memory: load Gaussian data in batches to reduce global memory access
    __shared__ float s_means2d[BLOCK_SIZE * 2];
    __shared__ float s_inv_cov[BLOCK_SIZE * 4];
    __shared__ float s_colors[BLOCK_SIZE * 3];
    __shared__ float s_opacity[BLOCK_SIZE];

    int n_rounds = (g_end - g_start + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int round = 0; round < n_rounds; round++) {
        // Cooperative loading: each thread loads one Gaussian's data
        int load_idx = g_start + round * BLOCK_SIZE + threadIdx.x;
        if (load_idx < g_end) {
            int gid = gaussian_ids[load_idx];
            s_means2d[threadIdx.x * 2 + 0] = means2d[gid * 2 + 0];
            s_means2d[threadIdx.x * 2 + 1] = means2d[gid * 2 + 1];
            s_inv_cov[threadIdx.x * 4 + 0] = inv_cov2d[gid * 4 + 0];
            s_inv_cov[threadIdx.x * 4 + 1] = inv_cov2d[gid * 4 + 1];
            s_inv_cov[threadIdx.x * 4 + 2] = inv_cov2d[gid * 4 + 2];
            s_inv_cov[threadIdx.x * 4 + 3] = inv_cov2d[gid * 4 + 3];
            s_colors[threadIdx.x * 3 + 0]  = colors[gid * 3 + 0];
            s_colors[threadIdx.x * 3 + 1]  = colors[gid * 3 + 1];
            s_colors[threadIdx.x * 3 + 2]  = colors[gid * 3 + 2];
            s_opacity[threadIdx.x]         = opacities[gid];
        }
        __syncthreads();

        if (inside) {
            int batch_end = min(BLOCK_SIZE, g_end - g_start - round * BLOCK_SIZE);
            for (int j = 0; j < batch_end; j++) {
                float mu_x = s_means2d[j * 2 + 0];
                float mu_y = s_means2d[j * 2 + 1];
                float dx = (float)px - mu_x;
                float dy = (float)py - mu_y;

                // Mahalanobis distance: dx^2 * a + dx*dy * (b+c) + dy^2 * d
                float a = s_inv_cov[j * 4 + 0];
                float b = s_inv_cov[j * 4 + 1];
                float c = s_inv_cov[j * 4 + 2];
                float d = s_inv_cov[j * 4 + 3];
                float maha = a * dx * dx + (b + c) * dx * dy + d * dy * dy;

                float alpha = fminf(s_opacity[j] * expf(-0.5f * maha), 0.99f);
                if (alpha < 1.0f / 255.0f) continue;

                float weight = T * alpha;
                r += weight * s_colors[j * 3 + 0];
                g += weight * s_colors[j * 3 + 1];
                b += weight * s_colors[j * 3 + 2];

                T *= (1.0f - alpha);
                count++;

                // Early termination
                if (T < 1e-4f) break;
            }
        }
        __syncthreads();

        // Early termination for entire warp (heuristic)
        if (T < 1e-4f) break;
    }

    if (inside) {
        int pix_idx = py * W + px;
        rendered[0 * H * W + pix_idx] = r;
        rendered[1 * H * W + pix_idx] = g;
        rendered[2 * H * W + pix_idx] = b;
        final_T[pix_idx] = T;
        n_contrib[pix_idx] = count;
    }
}


std::vector<torch::Tensor> rasterize_forward_cuda(
    torch::Tensor means2d,      // (N, 2) float32
    torch::Tensor inv_cov2d,    // (N, 2, 2) float32
    torch::Tensor colors,       // (N, 3) float32
    torch::Tensor opacities,    // (N,) float32
    torch::Tensor gaussian_ids, // (P,) int32
    torch::Tensor tile_ranges,  // (n_tiles+1,) int32
    int H, int W
) {
    auto device = means2d.device();

    // Ensure contiguous and reshape inv_cov2d to (N, 4) for flat access
    means2d = means2d.contiguous();
    auto inv_cov_flat = inv_cov2d.reshape({-1, 4}).contiguous();
    colors = colors.contiguous();
    opacities = opacities.contiguous();
    gaussian_ids = gaussian_ids.contiguous();
    tile_ranges = tile_ranges.contiguous();

    auto rendered = torch::zeros({3, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto final_T  = torch::zeros({H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto n_contrib = torch::zeros({H, W}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    int n_tiles_x = (W + TILE_SIZE - 1) / TILE_SIZE;
    int n_tiles_y = (H + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid(n_tiles_x, n_tiles_y);
    dim3 block(BLOCK_SIZE);

    rasterize_forward_kernel<<<grid, block>>>(
        means2d.data_ptr<float>(),
        inv_cov_flat.data_ptr<float>(),
        colors.data_ptr<float>(),
        opacities.data_ptr<float>(),
        gaussian_ids.data_ptr<int>(),
        tile_ranges.data_ptr<int>(),
        rendered.data_ptr<float>(),
        final_T.data_ptr<float>(),
        n_contrib.data_ptr<int>(),
        H, W
    );

    return {rendered, final_T, n_contrib};
}
