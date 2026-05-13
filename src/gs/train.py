from pathlib import Path
from datetime import datetime
import time

import numpy as np
import torch

from experiment_tools.set_up import start_experiment
from src.gs.model import GaussianModel
from src.utils.cfg_diff import get_config, get_diff
from src.utils.preprocessing import (
    convert_bin_to_txt, parse_cameras_txt, parse_images_txt,
    parse_points3D_txt, estimate_point_colors,
    split_images, load_images
)
from src.utils.result import (
    plot_training_curve, evaluate,
    export_ply, save_rendered_images
)
from src.utils.trainer import Options, train_gs

# プロジェクトルートを取得
project_root = Path(__file__).resolve().parents[2]

# 実行時刻を取得
now = datetime.now()
date_str = now.strftime("%Y-%m-%d")
time_str = now.strftime("%H-%M-%S")

# config ファイルパス
default_filename = str(project_root / "configs/default/gs.json")
exp_filename = str(project_root / "configs/experiment/gs.json")

# config ファイルの取得
_, cfg, default_str, exp_str = get_config(default_filename, exp_filename)

# output_dir を作成
output_base = Path(project_root) / "outputs" / cfg["phase_type"]
output_dir = output_base / date_str / time_str
output_dir.mkdir(parents=True, exist_ok=True)

# ロガーの初期化
logger = start_experiment(cfg)

# config 差分のログ出力
logger = get_diff(default_str, exp_str, logger)

# デバイス設定
device_str = cfg["device"]
if device_str == "auto":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
else:
    device = torch.device(device_str)
logger.info(f"使用デバイス: {device}")

# データ読み込み
logger.info("データ読み込み")
data_dir = project_root / cfg["data"]["data_dir"]
sparse_dir = data_dir / "sparse" / "0"
images_dir = data_dir / "images"

# データの読み込み
if not (sparse_dir / "cameras.txt").exists():
    logger.info("bin形式のファイルをtxt形式に変換中...")
    convert_bin_to_txt(str(sparse_dir))
    logger.info("変換完了")
cameras = parse_cameras_txt(str(sparse_dir / "cameras.txt"))
images = parse_images_txt(str(sparse_dir / "images.txt"))
points3D = parse_points3D_txt(str(sparse_dir / "points3D.txt"))

# 初期 Gaussian 数の上限
max_initial = cfg["data"].get("max_initial_gaussians", None)
if max_initial and len(points3D) > max_initial:
    import random as _random
    sampled = _random.sample(list(points3D.keys()), max_initial)
    points3D = {k: points3D[k] for k in sampled}
    logger.info(f"初期点群を {max_initial} 点にダウンサンプリング")

# 最適化用画像と評価用画像の分離
test_interval = cfg["data"].get("test_interval", 8)
train_images, test_images = split_images(images, test_interval)
train_tensors = load_images(str(images_dir), train_images)
test_tensors = load_images(str(images_dir), test_images)
logger.info(
    f"データ:\tカメラ数={len(cameras)}\t"
    f"最適化用画像={len(train_images)}\t"
    f"評価用画像={len(test_images)}\t"
    f"点群数={len(points3D)}"
)

# 初期点群の色推定
logger.info("初期点群の色推定")
points3D = estimate_point_colors(points3D, images, cameras, str(images_dir))

# モデル初期化
logger.info("モデル初期化")
sh_degree = cfg.get("sh_degree", 3)
model = GaussianModel(points3D, sh_degree=sh_degree)
model = model.to(device)
logger.info(f"Gaussian 数: {model.num_gaussians}")

# Optimizer 初期化
options = Options(cfg, model)
optimizer = options.optimizer
logger.info(f"Optimizer: {cfg['training']['optimizer']}")
logger.info(f"Learning rate: {cfg['training']['learning_rate']}")

# Scheduler 初期化
scheduler = None
if cfg["training"]["scheduler"]["use"]:
    scheduler_type = cfg["training"]["scheduler"]["type"]
    if scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg["training"]["scheduler"]["step_size"],
            gamma=cfg["training"]["scheduler"]["gamma"]
        )
    elif scheduler_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=cfg["training"]["scheduler"]["mode"],
            factor=cfg["training"]["scheduler"]["factor"],
            patience=cfg["training"]["scheduler"]["patience"],
            min_lr=cfg["training"]["scheduler"]["min_lr"]
        )
    logger.info(f"Scheduler: {scheduler_type}")
else:
    logger.info("Scheduler: なし")

# 学習実行
start_time = time.time()
model, train_loss_list = train_gs(
    model=model, optimizer=optimizer, images=train_images,
    image_tensors=train_tensors, cameras=cameras, cfg=cfg, device=device,
    logger=logger, scheduler=scheduler
)
end_time = time.time()
elapsed_time = end_time - start_time
logger.info("学習完了")
logger.info(f"学習時間: {elapsed_time:.2f}秒")

# 学習曲線の保存
plot_path = output_dir / "training_curve.png"
plot_training_curve(train_loss_list, str(plot_path))
logger.info(f"学習曲線保存先: {plot_path}")

# 損失統計の保存
logger.info("損失統計")
logger.info(f"最終損失: {train_loss_list[-1]:.4f}")
logger.info(f"最小損失: {min(train_loss_list):.4f}")
logger.info(f"平均損失: {np.mean(train_loss_list):.4f}")

# 評価
logger.info("評価実行")
resolution_scale = cfg["data"].get("resolution_scale", 1)
metrics = evaluate(
    model, test_images, test_tensors,
    cameras, device, resolution_scale
)
logger.info(f"PSNR:  {metrics['psnr']:.2f}")
logger.info(f"SSIM:  {metrics['ssim']:.4f}")
logger.info(f"LPIPS: {metrics['lpips']:.4f}")

# モデルの保存
model_path = output_dir / "final_model.pth"
torch.save(model.state_dict(), model_path)
logger.info(f"モデル保存先: {model_path}")

# PLY ファイルの保存
ply_path = output_dir / "gaussians.ply"
export_ply(model, str(ply_path))
logger.info(f"PLY 保存先: {ply_path}")

# レンダリング画像の保存
renders_dir = output_dir / "renders"
save_rendered_images(
    model, test_images, test_tensors, cameras, device,
    str(renders_dir), resolution_scale
)
logger.info(f"レンダリング画像保存先: {renders_dir}")
