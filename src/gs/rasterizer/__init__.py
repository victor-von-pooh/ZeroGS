import os
import torch

_cuda_ext = None


def _load_cuda_extension():
    global _cuda_ext
    if _cuda_ext is not None:
        return _cuda_ext

    from torch.utils.cpp_extension import load

    cuda_dir = os.path.join(os.path.dirname(__file__), "cuda")
    _cuda_ext = load(
        name="gs_rasterizer_cuda",
        sources=[
            os.path.join(cuda_dir, "bindings.cpp"),
            os.path.join(cuda_dir, "forward.cu"),
            os.path.join(cuda_dir, "backward.cu"),
        ],
        verbose=True,
    )
    return _cuda_ext


def is_cuda_available() -> bool:
    """CUDA ラスタライザが使用可能かどうかを返す"""
    if not torch.cuda.is_available():
        return False
    try:
        _load_cuda_extension()
        return True
    except Exception:
        return False
