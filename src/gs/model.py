import numpy as np
from scipy.spatial import KDTree
import torch
import torch.nn as nn


class GaussianModel(nn.Module):
    def __init__(self, points3D: dict):
        # 親クラスのコンストラクタを呼び出す
        super(GaussianModel, self).__init__()

        # points3D から numpy 配列を構築
        point_ids = sorted(points3D.keys())
        xyz = np.array(
            [points3D[pid]["xyz"] for pid in point_ids], dtype=np.float32
        )
        rgb = np.array(
            [points3D[pid]["rgb"] for pid in point_ids], dtype=np.float32
        )

        # 位置
        self.means = nn.Parameter(torch.from_numpy(xyz))

        # 色
        self.colors = nn.Parameter(torch.from_numpy(rgb / 255.0))

        # 不透明度
        n = len(point_ids)
        init_opacity = 0.1
        inv_sigmoid_opacity = np.log(init_opacity / (1.0 - init_opacity))
        self.opacities = nn.Parameter(
            torch.full((n, 1), inv_sigmoid_opacity, dtype=torch.float32)
        )

        # スケール
        tree = KDTree(xyz)
        distances, _ = tree.query(xyz, k=4)
        mean_dist = distances[:, 1:].mean(axis=1)
        self.scales = nn.Parameter(
            torch.from_numpy(
                np.log(mean_dist).astype(np.float32)
            ).unsqueeze(-1).expand(-1, 3).clone()
        )

        # 回転
        self.rotations = nn.Parameter(
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
            ).expand(n, -1).clone()
        )

    def forward(
        self, cam_qvec: np.ndarray, cam_tvec: np.ndarray,
        cam_params: np.ndarray, width: int, height: int
    ) -> torch.Tensor:
        """
        順伝播を行う関数

        Parameters
        ----------
        cam_qvec: np.ndarray
            カメラのクォータニオン
        cam_tvec: np.ndarray
            カメラの並進ベクトル
        cam_params: np.ndarray
            カメラの内部パラメータ
        width: int
            画像の幅
        height: int
            画像の高さ

        Returns
        ----------
        rendered: torch.Tensor
            レンダリングされた画像
        """
        # デバイスの取得
        device = self.means.device

        # world-to-camera の回転行列と並進ベクトル
        r_cam = quaternion_to_rotation_matrix(
            torch.tensor(cam_qvec, dtype=torch.float32, device=device)
        )
        t_cam = torch.tensor(cam_tvec, dtype=torch.float32, device=device)

        # カメラ内部パラメータ
        fx = fy = float(cam_params[0])
        cx = float(cam_params[1])
        cy = float(cam_params[2])

        # 対数スケールから復元
        scales = torch.exp(self.scales)
        s = torch.diag_embed(scales)

        # Gaussian の回転行列
        r_gauss = quaternion_to_rotation_matrix(self.rotations)

        # 3D 共分散
        m = r_gauss @ s
        cov3d = m @ m.transpose(-1, -2)

        # Gaussian の平均をカメラ座標系に変換
        means_cam = (r_cam @ self.means.T).T + t_cam

        # カメラの前方にある点のみを使用
        valid_mask = means_cam[:, 2] > 0.01
        means_cam = means_cam[valid_mask]
        cov3d = cov3d[valid_mask]
        colors = self.colors[valid_mask]
        opacities = torch.sigmoid(self.opacities[valid_mask])

        # ピクセル座標の計算
        x = means_cam[:, 0]
        y = means_cam[:, 1]
        z = means_cam[:, 2]
        means2d_x = fx * x / z + cx
        means2d_y = fy * y / z + cy
        means2d = torch.stack([means2d_x, means2d_y], dim=-1)

        # 投影のヤコビアン
        j = torch.zeros(means_cam.shape[0], 2, 3, device=device)
        j[:, 0, 0] = fx / z
        j[:, 0, 2] = -fx * x / (z * z)
        j[:, 1, 1] = fy / z
        j[:, 1, 2] = -fy * y / (z * z)

        # カメラ座標系での 3D 共分散
        w = r_cam.unsqueeze(0)
        cov_cam = w @ cov3d @ w.transpose(-1, -2)

        # 2D 共分散 + 正則化
        cov2d = j @ cov_cam @ j.transpose(-1, -2)
        cov2d = cov2d + 0.3 * torch.eye(2, device=device)

        # 2D 共分散の逆行列
        det = cov2d[:, 0, 0] * cov2d[:, 1, 1]
        det -= cov2d[:, 0, 1] * cov2d[:, 1, 0]
        det = det.clamp(min=1e-8)
        inv_cov2d = torch.zeros_like(cov2d)
        inv_cov2d[:, 0, 0] = cov2d[:, 1, 1] / det
        inv_cov2d[:, 0, 1] = -cov2d[:, 0, 1] / det
        inv_cov2d[:, 1, 0] = -cov2d[:, 1, 0] / det
        inv_cov2d[:, 1, 1] = cov2d[:, 0, 0] / det

        # 深度でソート
        sort_indices = torch.argsort(means_cam[:, 2])
        means2d = means2d[sort_indices]
        inv_cov2d = inv_cov2d[sort_indices]
        cov2d = cov2d[sort_indices]
        colors = colors[sort_indices]
        opacities = opacities[sort_indices]

        # レンダリング用のバッファ
        rendered = torch.zeros(3, height, width, device=device)
        transmittance = torch.ones(height, width, device=device)

        # ピクセルグリッドの作成
        py, px = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij"
        )

        # 各 Gaussian を前方から順にレンダリング
        for i in range(means2d.shape[0]):
            # バウンディングボックスの計算
            radius = 3.0 * torch.sqrt(
                torch.max(cov2d[i, 0, 0], cov2d[i, 1, 1])
            )
            x_min = max(0, int(means2d[i, 0] - radius))
            x_max = min(width, int(means2d[i, 0] + radius) + 1)
            y_min = max(0, int(means2d[i, 1] - radius))
            y_max = min(height, int(means2d[i, 1] + radius) + 1)

            # バウンディングボックスが有効かチェック
            if x_min >= x_max or y_min >= y_max:
                continue

            # バウンディングボックス内のピクセルのみ処理
            dx = px[y_min:y_max, x_min:x_max] - means2d[i, 0]
            dy = py[y_min:y_max, x_min:x_max] - means2d[i, 1]

            # マハラノビス距離
            maha = (
                inv_cov2d[i, 0, 0] * dx * dx
                + (inv_cov2d[i, 0, 1] + inv_cov2d[i, 1, 0]) * dx * dy
                + inv_cov2d[i, 1, 1] * dy * dy
            )

            # ガウシアン重み × 不透明度
            alpha = (opacities[i, 0] * torch.exp(-0.5 * maha)).clamp(max=0.99)

            # 透過率の取得
            t_patch = transmittance[y_min:y_max, x_min:x_max]

            # 色の蓄積
            contribution = t_patch.unsqueeze(0) * \
                           alpha.unsqueeze(0) * colors[i].reshape(3, 1, 1)
            rendered = rendered.clone()
            rendered[:, y_min:y_max, x_min:x_max] = (
                rendered[:, y_min:y_max, x_min:x_max] + contribution
            )

            # 透過率の更新
            new_transmittance = transmittance.clone()
            new_transmittance[y_min:y_max, x_min:x_max] = t_patch * (
                1 - alpha
            )
            transmittance = new_transmittance

        return rendered

    @property
    def num_gaussians(self) -> int:
        """
        Gaussian の数を返す関数

        Parameters
        ----------
        None

        Returns
        ----------
        gaussians: int
            Gaussian の数
        """
        # Gaussian の数は means の行数と同じ
        gaussians = self.means.shape[0]

        return gaussians


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    クォータニオンから回転行列への変換

    Parameters
    ----------
    q: torch.Tensor
        クォータニオン

    Returns
    ----------
    r: torch.Tensor
        回転行列
    """
    # クォータニオンを正規化
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # 回転行列の各要素を計算
    r = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
        ], dim=-1
    ).reshape(*q.shape[:-1], 3, 3)

    return r