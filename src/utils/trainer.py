import random

import torch
from torch import nn, optim
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
    image_tensors: dict, cameras: dict, cfg: dict, device: torch.device
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

            # 正解画像の取得
            gt = image_tensors[image_id].to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 順伝播の計算
            rendered = model(
                image_data["qvec"], image_data["tvec"],
                cam["params"], cam["width"], cam["height"]
            )

            # 損失計算
            loss = gs_loss(rendered, gt, lambda_ssim)

            # 逆伝播の計算とパラメータの更新
            loss.backward()
            optimizer.step()

            # 損失を記録
            iteration_loss = loss.item()
            train_loss_list.append(iteration_loss)

    return model, train_loss_list
