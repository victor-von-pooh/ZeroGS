import torch
from torch import nn


def gaussian_window(size: int, sigma: float) -> torch.Tensor:
    """
    1D ガウシアンカーネルを生成する関数

    Parameters
    ----------
    size: int
        カーネルサイズ
    sigma: float
        ガウシアンの標準偏差

    Returns
    ----------
    kernel: torch.Tensor
        正規化された 1D ガウシアンカーネル
    """
    # 座標の生成
    coords = torch.arange(size, dtype=torch.float32) - size // 2

    # ガウシアンカーネルの計算
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)

    # 正規化
    kernel = kernel / kernel.sum()

    return kernel


def create_ssim_conv(window_size: int, channels: int, device) -> nn.Conv2d:
    """
    SSIM 用の畳み込み層を生成する関数

    Parameters
    ----------
    window_size: int
        ウィンドウサイズ
    channels: int
        チャンネル数
    device:
        デバイス

    Returns
    ----------
    conv: nn.Conv2d
        ガウシアンカーネルを重みとする畳み込み層
    """
    # 1D ガウシアンカーネルから 2D カーネルを生成
    kernel_1d = gaussian_window(window_size, 1.5)
    kernel_2d = kernel_1d.unsqueeze(-1) @ kernel_1d.unsqueeze(0)

    # Conv2d 層を作成
    conv = nn.Conv2d(
        channels, channels, kernel_size=window_size,
        padding=window_size // 2, groups=channels, bias=False
    ).to(device)

    # 重みをガウシアンカーネルで固定
    with torch.no_grad():
        conv.weight.copy_(
            kernel_2d.expand(channels, 1, window_size, window_size)
        )
    conv.weight.requires_grad = False

    return conv


def ssim(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
) -> torch.Tensor:
    """
    SSIM を計算する関数

    Parameters
    ----------
    img1: torch.Tensor
        レンダリング画像
    img2: torch.Tensor
        正解画像
    window_size: int = 11
        ウィンドウサイズ

    Returns
    ----------
    ssim_val: torch.Tensor
        SSIM の値
    """
    # チャンネル数の取得
    c = img1.shape[0]

    # SSIM 用の畳み込み層を生成
    conv = create_ssim_conv(window_size, c, img1.device)

    # バッチ次元を追加
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    # 平均の計算
    mu1 = conv(img1)
    mu2 = conv(img2)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # 分散・共分散の計算
    sigma1_sq = conv(img1 * img1) - mu1_sq
    sigma2_sq = conv(img2 * img2) - mu2_sq
    sigma12 = conv(img1 * img2) - mu1_mu2

    # 安定化定数
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    # SSIM マップの計算
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator

    # 全体の平均
    ssim_val = ssim_map.mean()

    return ssim_val


def gs_loss(
    rendered: torch.Tensor, gt: torch.Tensor, lambda_ssim: float = 0.2
) -> torch.Tensor:
    """
    3DGS の損失関数を計算する関数

    Parameters
    ----------
    rendered: torch.Tensor
        レンダリング画像
    gt: torch.Tensor
        正解画像
    lambda_ssim: float = 0.2
        SSIM の重み

    Returns
    ----------
    loss: torch.Tensor
        損失値
    """
    # L1 Loss
    l1 = torch.abs(rendered - gt).mean()

    # SSIM Loss
    ssim_loss = 1.0 - ssim(rendered, gt)

    # 合計
    loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * ssim_loss

    return loss
