import os

import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def parse_cameras_txt(path: str) -> dict:
    """
    COLMAP の cameras.txt をパースしてカメラ内部パラメータを取得する関数

    Parameters
    ----------
    path: str
        cameras.txt のパス

    Returns
    ----------
    cameras: dict
        カメラデータの辞書
    """
    # カメラデータを格納する辞書
    cameras = {}

    # cameras.txt を読み込む
    with open(path, "r") as f:
        for line in f:
            # 行をトリムして, コメント行や空行をスキップ
            line = line.strip()
            if line.startswith("#") or len(line) == 0:
                continue

            # 行をスペースで分割して, カメラ ID, モデル, 画像サイズ, パラメータを取得
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(p) for p in parts[4:]], dtype=np.float64)

            # カメラデータを辞書に格納
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }

    return cameras


def parse_images_txt(path: str) -> dict:
    """
    COLMAP の images.txt をパースして画像の外部パラメータを取得する関数

    Parameters
    ----------
    path: str
        images.txt のパス

    Returns
    ----------
    images: dict
        画像データの辞書
    """
    # 画像データを格納する辞書
    images = {}

    # images.txt を読み込む
    with open(path, "r") as f:
        for line in f:
            # 行をトリムして, コメント行や空行をスキップ
            line = line.strip()
            if line.startswith("#") or len(line) == 0:
                continue

            # POINTS2D 行をスキップ
            parts = line.split()
            if len(parts) < 10:
                continue

            # 行をスペースで分割して, 画像 ID, クォータニオン, 平行移動ベクトル, カメラ ID, 画像名を取得
            image_id = int(parts[0])
            qvec = np.array(
                [float(parts[j]) for j in range(1, 5)], dtype=np.float64
            )
            tvec = np.array(
                [float(parts[j]) for j in range(5, 8)], dtype=np.float64
            )
            camera_id = int(parts[8])
            name = parts[9]

            # 画像データを辞書に格納
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }

    return images


def parse_points3D_txt(path: str) -> dict:
    """
    COLMAP の points3D.txt をパースして 3D 点群データを取得する関数

    Parameters
    ----------
    path: str
        points3D.txt のパス

    Returns
    ----------
    points3D: dict
        3D 点群データの辞書
    """
    # 3D 点群データを格納する辞書
    points3D = {}

    # points3D.txt を読み込む
    with open(path, "r") as f:
        for line in f:
            # 行をトリムして, コメント行や空行をスキップ
            line = line.strip()
            if line.startswith("#") or len(line) == 0:
                continue

            # 行をスペースで分割して, 点 ID, 座標, 色, 誤差を取得
            parts = line.split()
            point_id = int(parts[0])
            xyz = np.array(
                [float(parts[j]) for j in range(1, 4)], dtype=np.float64
            )
            rgb = np.array(
                [int(parts[j]) for j in range(4, 7)], dtype=np.uint8
            )
            error = float(parts[7])

            # 3D 点群データを辞書に格納
            points3D[point_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
            }

    return points3D


def load_images(images_dir: str, images: dict) -> dict:
    """
    画像ファイルを読み込んでテンソルに変換する関数

    Parameters
    ----------
    images_dir: str
        画像ディレクトリのパス
    images: dict
        parse_images_txt の返り値

    Returns
    ----------
    image_tensors: dict
        画像 ID をキー, 画像テンソルを値とする辞書
    """
    # テンソル変換
    to_tensor = transforms.ToTensor()

    # 画像テンソルを格納する辞書
    image_tensors = {}

    # 各画像を読み込んでテンソルに変換
    for image_id, image_data in images.items():
        path = os.path.join(images_dir, image_data["name"])
        img = Image.open(path).convert("RGB")
        image_tensors[image_id] = to_tensor(img)

    return image_tensors
