import torch
from . import _load_cuda_extension
from .tile_preprocess import tile_preprocess


class CUDARasterizer(torch.autograd.Function):
    """CUDA タイルベース ラスタライザ（autograd.Function）"""

    @staticmethod
    def forward(
        ctx,
        means2d: torch.Tensor,     # (N, 2)
        inv_cov2d: torch.Tensor,   # (N, 2, 2)
        colors: torch.Tensor,      # (N, 3)
        opacities: torch.Tensor,   # (N,)
        depths: torch.Tensor,      # (N,)
        sigma_max: torch.Tensor,   # (N,)
        height: int,
        width: int,
    ) -> torch.Tensor:
        ext = _load_cuda_extension()

        gaussian_ids, tile_ranges = tile_preprocess(
            means2d, depths, sigma_max, height, width, tile_size=16
        )

        # CUDA forward → [rendered, final_T, n_contrib]
        rendered, final_T, n_contrib = ext.rasterize_forward(
            means2d, inv_cov2d, colors, opacities,
            gaussian_ids, tile_ranges,
            height, width,
        )

        ctx.save_for_backward(
            means2d, inv_cov2d, colors, opacities,
            gaussian_ids, tile_ranges, final_T, n_contrib,
        )
        ctx.height = height
        ctx.width = width

        return rendered

    @staticmethod
    def backward(ctx, grad_rendered: torch.Tensor):
        ext = _load_cuda_extension()

        (means2d, inv_cov2d, colors, opacities,
         gaussian_ids, tile_ranges, final_T, n_contrib) = ctx.saved_tensors
        H, W = ctx.height, ctx.width
        N = means2d.shape[0]
        device = means2d.device

        grad_means2d   = torch.zeros_like(means2d)
        grad_inv_cov2d = torch.zeros(N, 4, device=device, dtype=torch.float32)
        grad_colors    = torch.zeros_like(colors)
        grad_opacities = torch.zeros_like(opacities)

        ext.rasterize_backward(
            means2d, inv_cov2d, colors, opacities,
            gaussian_ids, tile_ranges,
            final_T, n_contrib,
            grad_rendered.contiguous(),
            grad_means2d, grad_inv_cov2d,
            grad_colors, grad_opacities,
            H, W,
        )

        grad_inv_cov2d = grad_inv_cov2d.reshape(N, 2, 2)

        # depths, sigma_max, height, width は勾配不要
        return grad_means2d, grad_inv_cov2d, grad_colors, grad_opacities, None, None, None, None
