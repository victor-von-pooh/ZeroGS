import random
from logging import Logger
from typing import Optional

import torch
from torch import nn, optim
from torchvision import transforms
from tqdm import tqdm

from src.utils.loss import gs_loss


class Options:
    def __init__(self, cfg: dict, model: nn.Module):
        # 最適化器の種類の config を取得
        self.optimizer_name = cfg["training"]["optimizer"]
        self.cfg = cfg

        # パラメータ別学習率の設定
        lr_default = cfg["training"]["learning_rate"]
        lr_cfg = cfg["training"].get("learning_rates", {})
        param_groups = [
            {
                "params": [model.means],
                "lr": lr_cfg.get("means", lr_default)
            }, {
                "params": [model.sh_coeffs],
                "lr": lr_cfg.get("sh_coeffs", lr_default)
            }, {
                "params": [model.opacities],
                "lr": lr_cfg.get("opacities", lr_default)
            }, {
                "params": [model.scales],
                "lr": lr_cfg.get("scales", lr_default)
            }, {
                "params": [model.rotations],
                "lr": lr_cfg.get("rotations", lr_default)
            }
        ]

        # config で指定された最適化器のみを構築
        self.optimizer = self.build(param_groups)

    def build(self, param_groups: list[dict]) -> optim.Optimizer:
        """
        config に基づいて最適化器を構築するメソッド

        Parameters
        ----------
        param_groups: list[dict]
            パラメータグループ

        Returns
        ----------
        optimizer: optim.Optimizer
            構築された最適化器
        """
        # config から最適化器の種類を取得
        cfg = self.cfg
        lr_default = cfg["training"]["learning_rate"]

        # Adagrad
        if self.optimizer_name == "adagrad":
            return optim.Adagrad(
                params=param_groups, lr=lr_default,
                lr_decay=cfg["training"].get("lr_decay", 0),
                weight_decay=cfg["training"].get("weight_decay", 0),
                initial_accumulator_value=cfg["training"].get(
                    "initial_accumulator_value", 0
                ), eps=cfg["training"].get("eps", 1e-10)
            )
        
        # Adam
        elif self.optimizer_name == "adam":
            return optim.Adam(
                params=param_groups, lr=lr_default, betas=tuple(
                    cfg["training"].get("adam_betas", [0.9, 0.999])
                ), eps=cfg["training"].get("adam_eps", 1e-8),
                weight_decay=cfg["training"].get("adam_weight_decay", 0),
                amsgrad=cfg["training"].get("amsgrad", False),
                foreach=cfg["training"].get("foreach", None),
                maximize=cfg["training"].get("maximize", False),
                capturable=cfg["training"].get("capturable", False),
                differentiable=cfg["training"].get("differentiable", False),
                fused=cfg["training"].get("fused", None)
            )
        
        # AdamW
        elif self.optimizer_name == "adamw":
            return optim.AdamW(
                params=param_groups, lr=lr_default, betas=tuple(
                    cfg["training"].get("adam_betas", [0.9, 0.999])
                ), eps=cfg["training"].get("adam_eps", 1e-8),
                weight_decay=cfg["training"].get("adam_weight_decay", 0.01),
                amsgrad=cfg["training"].get("amsgrad", False),
                maximize=cfg["training"].get("maximize", False),
                foreach=cfg["training"].get("foreach", None),
                capturable=cfg["training"].get("capturable", False),
                differentiable=cfg["training"].get("differentiable", False),
                fused=cfg["training"].get("fused", None)
            )
        
        # ASGD
        elif self.optimizer_name == "asgd":
            return optim.ASGD(
                params=param_groups, lr=lr_default,
                lambd=cfg["training"].get("lambd", 1e-4),
                alpha=cfg["training"].get("alpha", 0.75),
                t0=cfg["training"].get("t0", 1e6),
                weight_decay=cfg["training"].get("weight_decay", 0)
            )
        
        # RAdam
        elif self.optimizer_name == "radam":
            return optim.RAdam(
                params=param_groups, lr=lr_default, betas=tuple(
                    cfg["training"].get("adam_betas", [0.9, 0.999])
                ), eps=cfg["training"].get("adam_eps", 1e-8),
                weight_decay=cfg["training"].get("weight_decay", 0),
                decoupled_weight_decay=cfg["training"].get(
                    "decoupled_weight_decay", False
                )
            )
        
        # SGD
        elif self.optimizer_name == "sgd":
            return optim.SGD(
                params=param_groups, lr=lr_default,
                momentum=cfg["training"].get("momentum", 0),
                dampening=cfg["training"].get("dampening", 0),
                weight_decay=cfg["training"].get("weight_decay", 0),
                nesterov=cfg["training"].get("nesterov", False)
            )
        
        # その他の最適化器は未対応
        else:
            raise ValueError(
                f"未対応の最適化器: {self.optimizer_name}"
            )



def train_gs(
    model: nn.Module, optimizer: optim.Optimizer, images: dict,
    image_tensors: dict, cameras: dict, cfg: dict, device: torch.device,
    logger: Logger, scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
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
                    # ADC 前に optimizer state を保存
                    old_states = []
                    for pg in optimizer.param_groups:
                        p = pg["params"][0]
                        s = optimizer.state.get(p, {})
                        old_states.append(
                            {
                                k: v.clone() if isinstance(v, torch.Tensor)
                                else v for k, v in s.items()
                            }
                        )

                    # 勾配の蓄積に基づいて, ガウシアンを密化・剪定
                    adc_info = model.densify_and_prune(
                        grad_threshold, scale_threshold,
                        opacity_threshold, max_gaussians
                    )
                    keep_mask = adc_info["keep_mask"]
                    clone_mask = adc_info["clone_mask"]
                    split_mask = adc_info["split_mask"]
                    n_clone = adc_info["n_clone"]
                    n_split = adc_info["n_split"]
                    topk = adc_info["topk"]

                    # 現在のガウシアン数のログ出力
                    n_keep = int(keep_mask.sum().item())
                    logger.info(
                        f"ADC [iter {iteration}]: "
                        f"keep={n_keep} clone={n_clone} split={n_split} "
                        f"→ total={model.num_gaussians}"
                    )

                    # optimizer のパラメータ参照を新しいものに差し替え
                    new_params = [
                        model.means, model.sh_coeffs,
                        model.opacities, model.scales, model.rotations,
                    ]
                    for i, p in enumerate(new_params):
                        old_p = optimizer.param_groups[i]["params"][0]
                        # 古いパラメータの state を退避
                        state = optimizer.state.pop(old_p, {})
                        # 参照を差し替え
                        optimizer.param_groups[i]["params"] = [p]
                        # state を新しいパラメータに紐付け
                        assert isinstance(p, torch.Tensor)
                        if state:
                            optimizer.state[p] = state

                    # optimizer state を復元
                    for gi, pg in enumerate(optimizer.param_groups):
                        # 新しいパラメータと古い state を取得
                        p = pg["params"][0]
                        old_s = old_states[gi]
                        if not old_s:
                            continue

                        # 古い state から新しい state を構築
                        new_s = {}
                        for key, val in old_s.items():
                            # テンソルでない値はそのままコピー
                            if not isinstance(val, torch.Tensor):
                                new_s[key] = val
                                continue

                            # スカラーテンソル（step 等）はそのまま
                            if val.dim() == 0:
                                new_s[key] = val.clone()
                                continue

                            # keep_mask で残った部分を取り出す
                            kept = val[keep_mask]

                            # clone_mask で複製する部分を取り出す
                            clone_state = val[clone_mask]

                            # split_mask で分割する部分を取り出す
                            split_src = val[split_mask]
                            split_state = torch.cat(
                                [split_src, split_src], dim=0
                            ) if split_src.shape[0] > 0 else split_src

                            # 新しい state を構築
                            padded = torch.cat(
                                [kept, clone_state, split_state], dim=0
                            )

                            # topk による追加の絞り込み
                            if topk is not None:
                                padded = padded[topk]
                            new_s[key] = padded

                        # 新しい optimizer state に保存
                        optimizer.state[p] = new_s

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
