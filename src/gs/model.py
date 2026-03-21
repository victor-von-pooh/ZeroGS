import numpy as np
from scipy.spatial import KDTree
import torch
import torch.nn as nn


class GaussianModel(nn.Module):
    def __init__(self, points3D: dict, sh_degree: int = 3):
        # 親クラスのコンストラクタを呼び出す
        super(GaussianModel, self).__init__()

        # SH の次数と係数数を保存
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2

        # points3D から numpy 配列を構築
        point_ids = sorted(points3D.keys())
        xyz = np.array(
            [points3D[pid]["xyz"] for pid in point_ids], dtype=np.float32
        )
        rgb = np.array(
            [points3D[pid]["rgb"] for pid in point_ids], dtype=np.float32
        )

        # Gaussian の数
        n = len(point_ids)

        # 位置
        self.means = nn.Parameter(torch.from_numpy(xyz))

        # SH 係数 (N, K, 3): DC 項を初期色から初期化, 高次項は 0
        c0 = 0.28209479177387814
        sh_coeffs = torch.zeros(n, self.num_sh_coeffs, 3, dtype=torch.float32)
        sh_coeffs[:, 0] = torch.from_numpy(rgb / 255.0) / c0
        self.sh_coeffs = nn.Parameter(sh_coeffs)

        # 不透明度
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

        # カメラ位置をワールド座標で計算
        cam_pos = -r_cam.T @ t_cam

        # カメラの前方にある点のみを使用
        valid_mask = means_cam[:, 2] > 0.01
        means_cam = means_cam[valid_mask]
        cov3d = cov3d[valid_mask]
        sh_coeffs = self.sh_coeffs[valid_mask]
        opacities = torch.sigmoid(self.opacities[valid_mask])

        # 視線方向の計算
        valid_means = self.means[valid_mask]
        view_dirs = cam_pos.unsqueeze(0) - valid_means
        view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # SH 評価で色を取得
        colors = evaluate_sh(sh_coeffs, view_dirs)

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

    def setup_adc(self):
        """
        ADC 用の勾配蓄積バッファを初期化する関数

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # 2D 位置勾配のノルムの蓄積
        self.grad_accum = torch.zeros(
            self.num_gaussians, device=self.means.device
        )

        # 蓄積回数
        self.grad_count = torch.zeros(
            self.num_gaussians, device=self.means.device
        )

    def accumulate_gradients(self):
        """
        means の勾配ノルムを蓄積する関数

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # means の勾配が存在する場合のみ蓄積
        if self.means.grad is not None:
            grad_norm = self.means.grad.norm(dim=-1)
            self.grad_accum += grad_norm
            self.grad_count += 1

    def densify_and_prune(
        self, grad_threshold: float = 0.0002, scale_threshold: float = 0.01,
        opacity_threshold: float = 0.005, max_gaussians: int = 100000
    ):
        """
        Adaptive Density Control を実行する関数

        Parameters
        ----------
        grad_threshold: float = 0.0002
            勾配の閾値
        scale_threshold: float = 0.01
            スケールの閾値
        opacity_threshold: float = 0.005
            不透明度の閾値
        max_gaussians: int = 100000
            Gaussian の最大数

        Returns
        ----------
        None
        """
        # デバイスの取得
        device = self.means.device

        # 平均勾配の計算
        avg_grad = self.grad_accum / self.grad_count.clamp(min=1)

        # 勾配が閾値を超える Gaussian
        high_grad_mask = avg_grad > grad_threshold

        # スケールの最大値
        max_scale = torch.exp(self.scales).max(dim=-1).values

        # Clone
        clone_mask = high_grad_mask & (max_scale <= scale_threshold)

        # Split
        split_mask = high_grad_mask & (max_scale > scale_threshold)

        # 勾配が大きく, スケールが小さい Gaussian を複製
        if clone_mask.any():
            clone_means = self.means.data[clone_mask]
            clone_sh = self.sh_coeffs.data[clone_mask]
            clone_opacities = self.opacities.data[clone_mask]
            clone_scales = self.scales.data[clone_mask]
            clone_rotations = self.rotations.data[clone_mask]
        else:
            clone_means = torch.empty(0, 3, device=device)
            clone_sh = torch.empty(0, self.num_sh_coeffs, 3, device=device)
            clone_opacities = torch.empty(0, 1, device=device)
            clone_scales = torch.empty(0, 3, device=device)
            clone_rotations = torch.empty(0, 4, device=device)

        # 勾配が大きく, スケールが大きい Gaussian を分割
        if split_mask.any():
            # 分割元のパラメータ
            split_means = self.means.data[split_mask]
            split_sh = self.sh_coeffs.data[split_mask]
            split_opacities = self.opacities.data[split_mask]
            split_scales = self.scales.data[split_mask] - np.log(1.6)
            split_rotations = self.rotations.data[split_mask]

            # スケール方向にランダムにオフセット
            stdev = torch.exp(self.scales.data[split_mask])
            offset = torch.randn_like(split_means) * stdev
            new_means_1 = split_means + offset
            new_means_2 = split_means - offset

            # 分割後のパラメータを結合
            split_means = torch.cat([new_means_1, new_means_2], dim=0)
            split_sh = torch.cat([split_sh, split_sh], dim=0)
            split_opacities = torch.cat(
                [split_opacities, split_opacities], dim=0
            )
            split_scales = torch.cat([split_scales, split_scales], dim=0)
            split_rotations = torch.cat(
                [split_rotations, split_rotations], dim=0
            )
        else:
            # 分割する Gaussian がない場合は空のテンソルを用意
            split_means = torch.empty(0, 3, device=device)
            split_sh = torch.empty(0, self.num_sh_coeffs, 3, device=device)
            split_opacities = torch.empty(0, 1, device=device)
            split_scales = torch.empty(0, 3, device=device)
            split_rotations = torch.empty(0, 4, device=device)

        # 不透明度が閾値以下の Gaussian と分割された Gaussian を削除
        opacity_vals = torch.sigmoid(self.opacities.data[:, 0])
        prune_mask = (opacity_vals < opacity_threshold) | split_mask
        keep_mask = ~prune_mask

        # 残す Gaussian と複製・分割された Gaussian を結合
        new_means = torch.cat(
            [self.means.data[keep_mask], clone_means, split_means], dim=0
        )
        new_sh = torch.cat(
            [self.sh_coeffs.data[keep_mask], clone_sh, split_sh], dim=0
        )
        new_opacities = torch.cat(
            [
                self.opacities.data[keep_mask],
                clone_opacities, split_opacities
            ], dim=0
        )
        new_scales = torch.cat(
            [self.scales.data[keep_mask], clone_scales, split_scales], dim=0
        )
        new_rotations = torch.cat(
            [
                self.rotations.data[keep_mask],
                clone_rotations, split_rotations
            ], dim=0
        )

        # Gaussian の最大数を超えた場合は不透明度の低い順に削除
        if new_means.shape[0] > max_gaussians:
            new_opacity_vals = torch.sigmoid(new_opacities[:, 0])
            topk = torch.topk(new_opacity_vals, max_gaussians).indices
            new_means = new_means[topk]
            new_sh = new_sh[topk]
            new_opacities = new_opacities[topk]
            new_scales = new_scales[topk]
            new_rotations = new_rotations[topk]

        # nn.Parameter として再設定
        self.means = nn.Parameter(new_means)
        self.sh_coeffs = nn.Parameter(new_sh)
        self.opacities = nn.Parameter(new_opacities)
        self.scales = nn.Parameter(new_scales)
        self.rotations = nn.Parameter(new_rotations)

        # 勾配バッファをリセット
        self.setup_adc()

    def reset_opacities(self, new_opacity: float = 0.01):
        """
        全 Gaussian の不透明度をリセットする関数

        Parameters
        ----------
        new_opacity: float = 0.01
            リセット後の不透明度

        Returns
        ----------
        None
        """
        # 逆シグモイド変換した値で上書き
        inv_sigmoid = np.log(new_opacity / (1.0 - new_opacity))
        self.opacities.data.fill_(inv_sigmoid)

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


def evaluate_sh(sh_coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """
    球面調和関数を評価する関数

    Parameters
    ----------
    sh_coeffs: torch.Tensor
        SH 係数
    dirs: torch.Tensor
        正規化された視線方向ベクトル

    Returns
    ----------
    colors: torch.Tensor
        評価された色
    """
    # SH 基底関数の定数
    c0 = 0.28209479177387814
    c1 = 0.4886025119029199
    c2 = [
        1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
        -1.0925484305920792, 0.5462742152960396
    ]
    c3 = [
        -0.5900435899266435, 2.890611442640554, -0.4570457994644658,
        0.3731763325901154, -0.4570457994644658, 1.445305721320277,
        -0.5900435899266435
    ]

    # 方向ベクトルの成分を取得
    x = dirs[:, 0:1]
    y = dirs[:, 1:2]
    z = dirs[:, 2:3]

    # 二乗・積の事前計算
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    # Degree 0
    result = c0 * sh_coeffs[:, 0]

    # Degree 1
    if sh_coeffs.shape[1] > 1:
        result = result + c1 * (
            -y * sh_coeffs[:, 1]
            + z * sh_coeffs[:, 2]
            - x * sh_coeffs[:, 3]
        )

    # Degree 2
    if sh_coeffs.shape[1] > 4:
        result = result + (
            c2[0] * xy * sh_coeffs[:, 4]
            + c2[1] * yz * sh_coeffs[:, 5]
            + c2[2] * (2.0 * zz - xx - yy) * sh_coeffs[:, 6]
            + c2[3] * xz * sh_coeffs[:, 7]
            + c2[4] * (xx - yy) * sh_coeffs[:, 8]
        )

    # Degree 3
    if sh_coeffs.shape[1] > 9:
        result = result + (
            c3[0] * y * (3.0 * xx - yy) * sh_coeffs[:, 9]
            + c3[1] * xy * z * sh_coeffs[:, 10]
            + c3[2] * y * (4.0 * zz - xx - yy) * sh_coeffs[:, 11]
            + c3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coeffs[:, 12]
            + c3[4] * x * (4.0 * zz - xx - yy) * sh_coeffs[:, 13]
            + c3[5] * z * (xx - yy) * sh_coeffs[:, 14]
            + c3[6] * x * (xx - 3.0 * yy) * sh_coeffs[:, 15]
        )

    # [0, 1] にクランプ
    colors = result.clamp(min=0.0, max=1.0)

    return colors
