from pathlib import Path
from datetime import datetime
import time

import numpy as np
import torch

from experiment_tools.set_up import start_experiment
from src.gs.model import GaussianModel
from src.utils.cfg_diff import get_config, get_diff
from src.utils.preprocessing import (
    parse_cameras_txt, parse_images_txt,
    parse_points3D_txt, estimate_point_colors, load_images
)
from src.utils.result import plot_training_curve
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
cameras = parse_cameras_txt(str(sparse_dir / "cameras.txt"))
images = parse_images_txt(str(sparse_dir / "images.txt"))
points3D = parse_points3D_txt(str(sparse_dir / "points3D.txt"))
image_tensors = load_images(str(images_dir), images)
logger.info(
    f"データ:\tカメラ数={len(cameras)}\t"
    f"画像数={len(images)}\t"
    f"点群数={len(points3D)}"
)

# 初期点群の色推定
logger.info("初期点群の色推定")
points3D = estimate_point_colors(points3D, images, cameras, str(images_dir))

# モデル初期化
logger.info("モデル初期化")
model = GaussianModel(points3D)
model = model.to(device)
logger.info(f"Gaussian 数: {model.num_gaussians}")

# Optimizer 初期化
options = Options(cfg, model)
optimizer = options.getter()
logger.info(f"Optimizer: {cfg['training']['optimizer']}")
logger.info(f"Learning rate: {cfg['training']['learning_rate']}")

# 学習実行
start_time = time.time()
model, train_loss_list = train_gs(
    model=model, optimizer=optimizer,
    images=images, image_tensors=image_tensors,
    cameras=cameras, cfg=cfg, device=device
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
