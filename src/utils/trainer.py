import random
from typing import Optional

import torch
from torch import nn, optim
from torchvision import transforms
from tqdm import tqdm

from src.utils.loss import gs_loss


class Options:
    def __init__(self, cfg: dict, model: nn.Module):
        # 最適化器の種類の config を取得
        self.optimizer = cfg["training"]["optimizer"]

        # Adagrad
        self.adagrad = optim.Adagrad(
            params=model.parameters(),
            lr=cfg["training"]["learning_rate"],
            lr_decay=cfg["training"].get("lr_decay", 0),
            weight_decay=cfg["training"].get("weight_decay", 0),
            initial_accumulator_value=cfg["training"].get(
                "initial_accumulator_value", 0
            ),
            eps=cfg["training"].get("eps", 1e-10)
        )

        # Adam
        self.adam = optim.Adam(
            params=model.parameters(),
            lr=cfg["training"]["learning_rate"],
            betas=tuple(cfg["training"].get("adam_betas", [0.9, 0.999])),
            eps=cfg["training"].get("adam_eps", 1e-8),
            weight_decay=cfg["training"].get("adam_weight_decay", 0),
            amsgrad=cfg["training"].get("amsgrad", False),
            foreach=cfg["training"].get("foreach", None),
            maximize=cfg["training"].get("maximize", False),
            capturable=cfg["training"].get("capturable", False),
            differentiable=cfg["training"].get("differentiable", False),
            fused=cfg["training"].get("fused", None)
        )

        # AdamW
        self.adamw = optim.AdamW(
            params=model.parameters(),
            lr=cfg["training"]["learning_rate"],
            betas=tuple(cfg["training"].get("adam_betas", [0.9, 0.999])),
            eps=cfg["training"].get("adam_eps", 1e-8),
            weight_decay=cfg["training"].get("adam_weight_decay", 0.01),
            amsgrad=cfg["training"].get("amsgrad", False),
            maximize=cfg["training"].get("maximize", False),
            foreach=cfg["training"].get("foreach", None),
            capturable=cfg["training"].get("capturable", False),
            differentiable=cfg["training"].get("differentiable", False),
            fused=cfg["training"].get("fused", None)
        )

        # ASGD
        self.asgd = optim.ASGD(
            params=model.parameters(),
            lr=cfg["training"]["learning_rate"],
            lambd=cfg["training"].get("lambd", 1e-4),
            alpha=cfg["training"].get("alpha", 0.75),
            t0=cfg["training"].get("t0", 1e6),
            weight_decay=cfg["training"].get("weight_decay", 0)
        )

        # RAdam
        self.radam = optim.RAdam(
            params=model.parameters(),
            lr=cfg["training"]["learning_rate"],
            betas=tuple(cfg["training"].get("adam_betas", [0.9, 0.999])),
            eps=cfg["training"].get("adam_eps", 1e-8),
            weight_decay=cfg["training"].get("weight_decay", 0),
            decoupled_weight_decay=cfg["training"].get(
                "decoupled_weight_decay", False
            )
        )

        # SGD
        self.sgd = optim.SGD(
            params=model.parameters(),
            lr=cfg["training"]["learning_rate"],
            momentum=cfg["training"].get("momentum", 0),
            dampening=cfg["training"].get("dampening", 0),
            weight_decay=cfg["training"].get("weight_decay", 0),
            nesterov=cfg["training"].get("nesterov", False)
        )

    def getter(self) -> optim.Optimizer:
        # 最適化器の辞書
        opt_dict = {
            "adagrad": self.adagrad, "adam": self.adam,
            "adamw": self.adamw, "asgd": self.asgd,
            "radam": self.radam, "sgd": self.sgd
        }

        # config で指定された最適化器を返す
        optimizer = opt_dict[self.optimizer]

        return optimizer


def train_gs(
    model: nn.Module, optimizer: optim.Optimizer, images: dict,
    image_tensors: dict, cameras: dict, cfg: dict, device: torch.device,
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
) -> tuple[nn.Module, list]:
    """
    3DGS モデルの学習を行う関数

    Parameters
    ----------
    model: nn.Module
        学習するモデル
    optimizer: optim.Optimizer
        最適化器
    images: dict
        parse_images_txt の返り値
    image_tensors: dict
        load_images の返り値
    cameras: dict
        parse_cameras_txt の返り値
    cfg: dict
        config の辞書データ
    device: torch.device
        デバイス
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        学習率スケジューラー

    Returns
    ----------
    model: nn.Module
        学習後のモデル
    train_loss_list: list
        学習過程の損失を格納したリスト
    """
    # 学習設定の取得
    num_iterations = cfg["training"]["iterations"]
    lambda_ssim = cfg["training"].get("lambda_ssim", 0.2)
    resolution_scale = cfg["training"].get("resolution_scale", 1)

    # ADC 設定の取得
    adc_cfg = cfg["training"].get("adc", {})
    adc_start = adc_cfg.get("start_iteration", 500)
    adc_interval = adc_cfg.get("interval", 100)
    adc_stop = adc_cfg.get("stop_iteration", 15000)
    opacity_reset_interval = adc_cfg.get("opacity_reset_interval", 3000)
    grad_threshold = adc_cfg.get("grad_threshold", 0.0002)
    scale_threshold = adc_cfg.get("scale_threshold", 0.01)
    opacity_threshold = adc_cfg.get("opacity_threshold", 0.005)
    max_gaussians = adc_cfg.get("max_gaussians", 100000)

    # ADC 用の勾配蓄積バッファを初期化
    model.setup_adc()

    # 画像 ID のリスト
    image_ids = list(images.keys())

    # 学習用データの損失を格納するリスト
    train_loss_list = []

    # 学習ループの実行
    with tqdm(range(num_iterations)) as pbar:
        for iteration in pbar:
            # イテレーション数の表示
            pbar.set_description(f"Iteration {iteration + 1}")

            # ランダムに画像を選択
            image_id = random.choice(image_ids)
            image_data = images[image_id]
            cam = cameras[image_data["camera_id"]]

            # レンダリング解像度の計算
            render_w = cam["width"] // resolution_scale
            render_h = cam["height"] // resolution_scale

            # カメラ内部パラメータのスケーリング
            scaled_params = cam["params"].copy()
            scaled_params[0] /= resolution_scale
            scaled_params[1] /= resolution_scale
            scaled_params[2] /= resolution_scale

            # 正解画像のリサイズ
            gt = image_tensors[image_id].to(device)
            resize = transforms.Resize((render_h, render_w))
            gt = resize(gt)

            # 勾配の初期化
            optimizer.zero_grad()

            # 順伝播の計算
            rendered = model(
                image_data["qvec"], image_data["tvec"],
                scaled_params, render_w, render_h
            )

            # 損失計算
            loss = gs_loss(rendered, gt, lambda_ssim)

            # 逆伝播の計算とパラメータの更新
            loss.backward()

            # 勾配の蓄積
            if iteration < adc_stop:
                model.accumulate_gradients()

            # パラメータの更新
            optimizer.step()

            # ガウシアンの密化・剪定
            if adc_start <= iteration < adc_stop:
                if (iteration - adc_start) % adc_interval == 0:
                    # 勾配の蓄積に基づいて, ガウシアンを密化・剪定
                    model.densify_and_prune(
                        grad_threshold, scale_threshold,
                        opacity_threshold, max_gaussians
                    )

                    # Optimizer の再構築
                    optimizer = optim.Adam(
                        params=model.parameters(),
                        lr=optimizer.param_groups[0]["lr"]
                    )

            # 不透明度のリセット
            if iteration > 0 and iteration % opacity_reset_interval == 0:
                model.reset_opacities()

            # Scheduler の更新
            if scheduler is not None:
                if isinstance(
                    scheduler, optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step(loss.item())
                else:
                    scheduler.step()

            # 損失を記録
            iteration_loss = loss.item()
            train_loss_list.append(iteration_loss)

    return model, train_loss_list
