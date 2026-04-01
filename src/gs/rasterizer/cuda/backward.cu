#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define BLOCK_SIZE (TILE_SIZE * TILE_SIZE)

__global__ void rasterize_backward_kernel(
    const float* __restrict__ means2d,      // (N, 2)
    const float* __restrict__ inv_cov2d,    // (N, 4)
    const float* __restrict__ colors,       // (N, 3)
    const float* __restrict__ opacities,    // (N,)
    const int*   __restrict__ gaussian_ids, // (P,)
    const int*   __restrict__ tile_ranges,  // (n_tiles + 1,)
    const float* __restrict__ final_T,      // (H, W)
    const int*   __restrict__ n_contrib,    // (H, W)
    const float* __restrict__ grad_rendered,// (3, H, W)
    float* __restrict__ grad_means2d,       // (N, 2)
    float* __restrict__ grad_inv_cov2d,     // (N, 4)
    float* __restrict__ grad_colors,        // (N, 3)
    float* __restrict__ grad_opacities,     // (N,)
    int H, int W
) {
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int n_tiles_x = (W + TILE_SIZE - 1) / TILE_SIZE;
    int tile_id = tile_y * n_tiles_x + tile_x;

    int px = tile_x * TILE_SIZE + (threadIdx.x % TILE_SIZE);
    int py = tile_y * TILE_SIZE + (threadIdx.x / TILE_SIZE);
    bool inside = (px < W && py < H);

    int g_start = tile_ranges[tile_id];
    int g_end   = tile_ranges[tile_id + 1];

    // Load per-pixel state
    float T_final = 1.0f;
    float dr = 0.0f, dg = 0.0f, db = 0.0f;
    int pix_idx = 0;

    if (inside) {
        pix_idx = py * W + px;
        T_final = final_T[pix_idx];
        dr = grad_rendered[0 * H * W + pix_idx];
        dg = grad_rendered[1 * H * W + pix_idx];
        db = grad_rendered[2 * H * W + pix_idx];
    }

    // We traverse Gaussians in reverse order.
    // Need to reconstruct T at each step. Use the relation:
    //   T_after_last = final_T
    //   T_before_i = T_after_i / (1 - alpha_i)
    //
    // Also accumulate: accum = sum_{j>i} (T_j * alpha_j * color_j)
    // which appears in the gradient of alpha_i:
    //   d_alpha_i = T_i * (d_rendered . color_i) - (1/(1-alpha_i)) * (d_rendered . accum_after_i)
    //
    // Simpler formulation (from 3DGS paper):
    //   d_alpha_i = T_i * (d_rendered . color_i - (rendered_after_i . d_rendered) / (1 - alpha_i))
    //   where rendered_after_i = sum_{j>i} w_j * color_j

    // Shared memory for cooperative loading (reverse traversal)
    __shared__ float s_means2d[BLOCK_SIZE * 2];
    __shared__ float s_inv_cov[BLOCK_SIZE * 4];
    __shared__ float s_colors[BLOCK_SIZE * 3];
    __shared__ float s_opacity[BLOCK_SIZE];
    __shared__ int   s_gids[BLOCK_SIZE];

    float T = T_final;  // Start from final transmittance, reconstruct backwards

    // Accumulated rendered color from Gaussians after current position
    // At the end of forward: rendered = sum_all w_i * c_i, and T = T_final
    // We start from the back, so accum starts at 0
    float accum_r = 0.0f, accum_g = 0.0f, accum_b = 0.0f;

    int n_gaussians = g_end - g_start;
    int n_rounds = (n_gaussians + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int round = n_rounds - 1; round >= 0; round--) {
        // Load batch (same ordering as forward, but we process in reverse within each batch)
        int load_idx = g_start + round * BLOCK_SIZE + threadIdx.x;
        if (load_idx < g_end) {
            int gid = gaussian_ids[load_idx];
            s_gids[threadIdx.x] = gid;
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
            int batch_size = min(BLOCK_SIZE, n_gaussians - round * BLOCK_SIZE);
            // Process in reverse within this batch
            for (int j = batch_size - 1; j >= 0; j--) {
                float mu_x = s_means2d[j * 2 + 0];
                float mu_y = s_means2d[j * 2 + 1];
                float dx = (float)px - mu_x;
                float dy = (float)py - mu_y;

                float a = s_inv_cov[j * 4 + 0];
                float b = s_inv_cov[j * 4 + 1];
                float c = s_inv_cov[j * 4 + 2];
                float d = s_inv_cov[j * 4 + 3];
                float maha = a * dx * dx + (b + c) * dx * dy + d * dy * dy;

                float sigma = s_opacity[j];
                float exp_term = expf(-0.5f * maha);
                float alpha = fminf(sigma * exp_term, 0.99f);
                if (alpha < 1.0f / 255.0f) continue;

                float cr = s_colors[j * 3 + 0];
                float cg = s_colors[j * 3 + 1];
                float cb = s_colors[j * 3 + 2];

                // Reconstruct T_i (transmittance before this Gaussian)
                // T_after = T_before * (1 - alpha)
                // T is currently T_after (transmittance after this Gaussian)
                // So T_before = T / (1 - alpha)
                float one_minus_alpha = 1.0f - alpha;
                float T_i = T / fmaxf(one_minus_alpha, 1e-6f);

                float weight = T_i * alpha;

                // Gradient of color_i:
                // d_color_i = weight * d_rendered
                int gid = s_gids[j];
                atomicAdd(&grad_colors[gid * 3 + 0], weight * dr);
                atomicAdd(&grad_colors[gid * 3 + 1], weight * dg);
                atomicAdd(&grad_colors[gid * 3 + 2], weight * db);

                // Gradient of alpha_i:
                // d_loss/d_alpha_i = T_i * (color_i . d_rendered) - (1/(1-alpha_i)) * (accum_after . d_rendered)
                float d_alpha = T_i * (cr * dr + cg * dg + cb * db)
                              - (1.0f / fmaxf(one_minus_alpha, 1e-6f))
                                * (accum_r * dr + accum_g * dg + accum_b * db);

                // alpha = min(sigma * exp(-0.5 * maha), 0.99)
                // If alpha was clamped to 0.99, gradient is 0
                float d_sigma_exp = (alpha < 0.99f) ? d_alpha : 0.0f;

                // d_alpha / d_sigma = exp_term  (when not clamped)
                // d_alpha / d_maha  = sigma * exp_term * (-0.5) = alpha * (-0.5)
                atomicAdd(&grad_opacities[gid], d_sigma_exp * exp_term);

                float d_maha = d_sigma_exp * sigma * exp_term * (-0.5f);

                // Gradients of inv_cov2d components
                // maha = a*dx*dx + (b+c)*dx*dy + d*dy*dy
                atomicAdd(&grad_inv_cov2d[gid * 4 + 0], d_maha * dx * dx);
                atomicAdd(&grad_inv_cov2d[gid * 4 + 1], d_maha * dx * dy);
                atomicAdd(&grad_inv_cov2d[gid * 4 + 2], d_maha * dx * dy);
                atomicAdd(&grad_inv_cov2d[gid * 4 + 3], d_maha * dy * dy);

                // Gradients of means2d
                // d_maha/d_mu_x = -(2*a*dx + (b+c)*dy)
                // d_maha/d_mu_y = -((b+c)*dx + 2*d*dy)
                float bc = b + c;
                atomicAdd(&grad_means2d[gid * 2 + 0], d_maha * (-(2.0f * a * dx + bc * dy)));
                atomicAdd(&grad_means2d[gid * 2 + 1], d_maha * (-(bc * dx + 2.0f * d * dy)));

                // Update accum (add this Gaussian's contribution)
                accum_r += weight * cr;
                accum_g += weight * cg;
                accum_b += weight * cb;

                // Restore T to T_i (before this Gaussian)
                T = T_i;
            }
        }
        __syncthreads();
    }
}


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
) {
    means2d = means2d.contiguous();
    auto inv_cov_flat = inv_cov2d.reshape({-1, 4}).contiguous();
    colors = colors.contiguous();
    opacities = opacities.contiguous();
    gaussian_ids = gaussian_ids.contiguous();
    tile_ranges = tile_ranges.contiguous();
    final_T = final_T.contiguous();
    n_contrib = n_contrib.contiguous();
    grad_rendered = grad_rendered.contiguous();

    int n_tiles_x = (W + TILE_SIZE - 1) / TILE_SIZE;
    int n_tiles_y = (H + TILE_SIZE - 1) / TILE_SIZE;
    dim3 grid(n_tiles_x, n_tiles_y);
    dim3 block(BLOCK_SIZE);

    rasterize_backward_kernel<<<grid, block>>>(
        means2d.data_ptr<float>(),
        inv_cov_flat.data_ptr<float>(),
        colors.data_ptr<float>(),
        opacities.data_ptr<float>(),
        gaussian_ids.data_ptr<int>(),
        tile_ranges.data_ptr<int>(),
        final_T.data_ptr<float>(),
        n_contrib.data_ptr<int>(),
        grad_rendered.data_ptr<float>(),
        grad_means2d.data_ptr<float>(),
        grad_inv_cov2d.data_ptr<float>(),
        grad_colors.data_ptr<float>(),
        grad_opacities.data_ptr<float>(),
        H, W
    );
}
