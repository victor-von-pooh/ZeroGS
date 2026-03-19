import os

import lpips
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

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

    # PSNR の計算し, MSE が 0 の場合は無限大
    if mse == 0:
        psnr_val = float("inf")
    else:
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


def export_ply(model: torch.nn.Module, output_path: str) -> None:
    """
    Gaussian の位置と色を PLY ファイルに書き出す関数

    Parameters
    ----------
    model: torch.nn.Module
        学習済みモデル
    output_path: str
        出力先のファイルパス

    Returns
    ----------
    None
    """
    # 位置と SH の DC 項から色を取得
    means = model.means.detach().cpu().numpy()
    sh_dc = model.sh_coeffs.detach().cpu().numpy()[:, 0]
    c0 = 0.28209479177387814
    rgb = np.clip(sh_dc * c0 * 255.0, 0, 255).astype(np.uint8)
    n_points = means.shape[0]

    # PLY ヘッダーとデータを書き出し
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for j in range(n_points):
            x, y, z = means[j]
            r, g, b = rgb[j]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


def save_rendered_images(
    model: torch.nn.Module, test_images: dict, test_tensors: dict,
    cameras: dict, device: torch.device, output_dir: str,
    resolution_scale: int = 1
) -> None:
    """
    評価用画像のレンダリング結果と正解画像を並べて保存する関数

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
    output_dir: str
        出力先ディレクトリ
    resolution_scale: int = 1
        解像度のスケール

    Returns
    ----------
    None
    """
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # 各評価用画像に対してレンダリングして保存
    for image_id, image_data in test_images.items():
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

        # テンソルを画像に変換して横に並べて保存
        rendered_np = (rendered.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        gt_np = (gt.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        combined = np.concatenate([gt_np, rendered_np], axis=1)
        name = image_data["name"].replace(".jpg", ".png")
        Image.fromarray(combined).save(os.path.join(output_dir, name))
