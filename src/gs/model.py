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

        # 画像範囲内の Gaussian のみ残す
        margin = 3.0 * torch.sqrt(torch.max(cov2d[:, 0, 0], cov2d[:, 1, 1]))
        visible = (
            (means2d[:, 0] + margin > 0)
            & (means2d[:, 0] - margin < width)
            & (means2d[:, 1] + margin > 0)
            & (means2d[:, 1] - margin < height)
        )
        means2d = means2d[visible]
        inv_cov2d = inv_cov2d[visible]
        colors = colors[visible]
        opacities = opacities[visible]

        # ピクセルグリッドの作成
        py, px = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij"
        )

        # チャンク方式でレンダリング
        n_gaussians = means2d.shape[0]
        chunk_size = 256
        rendered = torch.zeros(3, height, width, device=device)
        running_T = torch.ones(height, width, device=device)

        # チャンクごとに処理
        for start in range(0, n_gaussians, chunk_size):
            # チャンクの終了インデックス
            end = min(start + chunk_size, n_gaussians)

            # チャンク内の差分計算
            dx = px.unsqueeze(0) - means2d[start:end, 0].reshape(-1, 1, 1)
            dy = py.unsqueeze(0) - means2d[start:end, 1].reshape(-1, 1, 1)

            # マハラノビス距離
            ic = inv_cov2d[start:end]
            maha = (
                ic[:, 0, 0].reshape(-1, 1, 1) * dx * dx
                + (ic[:, 0, 1] + ic[:, 1, 0]).reshape(-1, 1, 1) * dx * dy
                + ic[:, 1, 1].reshape(-1, 1, 1) * dy * dy
            )

            # アルファ
            alpha_chunk = (
                opacities[start:end, 0].reshape(-1, 1, 1)
                * torch.exp(-0.5 * maha)
            ).clamp(max=0.99)

            # チャンク内の透過率
            one_minus_alpha = 1.0 - alpha_chunk
            chunk_T = torch.cat(
                [
                    torch.ones(1, height, width, device=device),
                    torch.cumprod(one_minus_alpha[:-1], dim=0)
                ], dim=0
            )

            # running_T を掛けて実際の透過率にする
            actual_T = running_T.unsqueeze(0) * chunk_T

            # 色の蓄積
            weight = actual_T * alpha_chunk
            rendered = rendered + (
                weight.unsqueeze(1) * colors[start:end].reshape(-1, 3, 1, 1)
            ).sum(dim=0)

            # running_T の更新
            running_T = running_T * one_minus_alpha.prod(dim=0)

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