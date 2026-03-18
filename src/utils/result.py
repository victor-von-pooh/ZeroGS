import lpips
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from src.utils.loss import ssim


def plot_training_curve(train_loss_list: list, output_path: str) -> None:
    """
    学習曲線をプロットして保存する関数

    Parameters
    ----------
    train_loss_list: list
        学習過程の損失を格納したリスト
    output_path: str
        プロット画像の保存先パス

    Returns
    -------
    None
    """
    # エポック数のリストを作成
    epochs = list(range(1, len(train_loss_list) + 1))

    # 出力画像の設定
    plt.figure(figsize=(18, 12), tight_layout=True)
    plt.title("Training Loss over Epochs", size=15, color="red")
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # 学習曲線のプロットと凡例の表示
    plt.plot(epochs, train_loss_list, label="Train Loss")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

    # プロットの保存
    plt.savefig(output_path)
    plt.close()


def psnr(rendered: torch.Tensor, gt: torch.Tensor) -> float:
    """
    PSNR を計算する関数

    Parameters
    ----------
    rendered: torch.Tensor
        レンダリング画像
    gt: torch.Tensor
        正解画像

    Returns
    ----------
    psnr_val: float
        PSNR の値
    """
    # MSE の計算
    mse = ((rendered - gt) ** 2).mean().item()

    # MSE が 0 の場合は無限大
    if mse == 0:
        return float("inf")

    # PSNR の計算
    psnr_val = 10.0 * np.log10(1.0 / mse)

    return psnr_val


def lpips_score(
    rendered: torch.Tensor, gt: torch.Tensor, lpips_fn: lpips.LPIPS
) -> float:
    """
    LPIPS を計算する関数

    Parameters
    ----------
    rendered: torch.Tensor
        レンダリング画像
    gt: torch.Tensor
        正解画像
    lpips_fn: lpips.LPIPS
        LPIPS モデル

    Returns
    ----------
    lpips_val: float
        LPIPS の値
    """
    # バッチ次元を追加して [-1, 1] にスケーリング
    rendered_batch = rendered.unsqueeze(0) * 2.0 - 1.0
    gt_batch = gt.unsqueeze(0) * 2.0 - 1.0

    # LPIPS の計算
    lpips_val = lpips_fn(rendered_batch, gt_batch).item()

    return lpips_val


def evaluate(
    model: torch.nn.Module, test_images: dict, test_tensors: dict,
    cameras: dict, device: torch.device, resolution_scale: int = 1
) -> dict:
    """
    評価用画像に対して PSNR / SSIM / LPIPS を計算する関数

    Parameters
    ----------
    model: torch.nn.Module
        学習済みモデル
    test_images: dict
        評価用の画像データ
    test_tensors: dict
        評価用の画像テンソル
    cameras: dict
        カメラデータ
    device: torch.device
        デバイス
    resolution_scale: int = 1
        解像度のスケール

    Returns
    ----------
    metrics: dict
        各メトリクスの平均値
    """
    # LPIPS モデルの初期化
    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    # メトリクスの蓄積用
    psnr_list = []
    ssim_list = []
    lpips_list = []

    # 各評価用画像に対して計算
    for image_id, image_data in test_images.items():
        # カメラデータの取得
        cam = cameras[image_data["camera_id"]]

        # レンダリング解像度の計算
        render_w = cam["width"] // resolution_scale
        render_h = cam["height"] // resolution_scale

        # カメラ内部パラメータのスケーリング
        scaled_params = cam["params"].copy()
        scaled_params[0] /= resolution_scale
        scaled_params[1] /= resolution_scale
        scaled_params[2] /= resolution_scale

        # レンダリング
        with torch.no_grad():
            rendered = model(
                image_data["qvec"], image_data["tvec"],
                scaled_params, render_w, render_h
            )

        # 正解画像のリサイズ
        gt = test_tensors[image_id].to(device)
        resize = transforms.Resize((render_h, render_w))
        gt = resize(gt)

        # 各メトリクスの計算
        psnr_list.append(psnr(rendered, gt))
        ssim_list.append(ssim(rendered, gt).item())
        lpips_list.append(lpips_score(rendered, gt, lpips_fn))

    # 平均値を計算
    metrics = {
        "psnr": np.mean(psnr_list),
        "ssim": np.mean(ssim_list),
        "lpips": np.mean(lpips_list)
    }

    return metrics
