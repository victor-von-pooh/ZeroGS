from typing import Optional

import numpy as np
from scipy.spatial import KDTree
import torch
import torch.nn as nn


class GaussianRasterizer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.Function, means2d: torch.Tensor,
        inv_cov2d: torch.Tensor, colors: torch.Tensor,
        opacities: torch.Tensor, sigma_max: torch.Tensor,
        height: int, width: int, bg_color: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        ctx: torch.autograd.Function
            自動微分のコンテキスト
        means2d: torch.Tensor
            2D 平均位置
        inv_cov2d: torch.Tensor
            2D 共分散の逆行列
        colors: torch.Tensor
            各 Gaussian の色
        opacities: torch.Tensor
            シグモイド済み不透明度
        sigma_max: torch.Tensor
            バウンディングボックス半径
        height: int
            画像の高さ
        width: int
            画像の幅
        bg_color: torch.Tensor
            背景色

        Returns
        ----------
        rendered: torch.Tensor
            レンダリングされた画像
        """
        # デバイスと Gaussian の数
        device = means2d.device
        n = means2d.shape[0]

        # CPU に移動して numpy でバウンディングボックスを一括計算するため、必要なテンソルを CPU に移動
        means2d_c = means2d.cpu()
        inv_cov2d_c = inv_cov2d.cpu()
        colors_c = colors.cpu()
        opacities_c = opacities.cpu()
        sigma_max_c = sigma_max.cpu()
        bg_color_c = bg_color.detach().cpu()

        # forward では勾配を計算しないため torch.no_grad() コンテキストで実装
        with torch.no_grad():
            # バウンディングボックスを numpy で一括計算
            mu_np = means2d_c.numpy()
            rad_np = sigma_max_c.numpy()
            y0s = np.clip((mu_np[:, 1] - rad_np).astype(int), 0, height)
            y1s = np.clip((mu_np[:, 1] + rad_np).astype(int) + 1, 0, height)
            x0s = np.clip((mu_np[:, 0] - rad_np).astype(int), 0, width)
            x1s = np.clip((mu_np[:, 0] + rad_np).astype(int) + 1, 0, width)

            # ピクセル座標グリッド
            py, px = torch.meshgrid(
                torch.arange(height, dtype=torch.float32) + 0.5,
                torch.arange(width, dtype=torch.float32) + 0.5,
                indexing="ij"
            )
            rendered = torch.zeros(3, height, width)
            running_T = torch.ones(height, width)

            # Gaussian ごとにバウンディングボックス内のピクセルを処理
            for i in range(n):
                # バウンディングボックスの座標を整数に変換
                y0, y1 = int(y0s[i]), int(y1s[i])
                x0, x1 = int(x0s[i]), int(x1s[i])
                if y0 >= y1 or x0 >= x1:
                    continue

                # バウンディングボックス内のピクセルに対してマハラノビス距離を計算
                dx = px[y0:y1, x0:x1] - float(mu_np[i, 0])
                dy = py[y0:y1, x0:x1] - float(mu_np[i, 1])
                ic = inv_cov2d_c[i]
                maha = (
                    ic[0, 0] * dx * dx + (ic[0, 1] + ic[1, 0]) * dx * dy
                    + ic[1, 1] * dy * dy
                )
                alpha = (
                    opacities_c[i] * torch.exp(-0.5 * maha)
                ).clamp(max=0.99)

                # alpha 0.99 以上はほぼ完全に不透明で勾配が消えるため、0.99 以上は 0.99 にクランプして勾配を切る
                alpha = torch.where(
                    alpha > 1.0 / 255.0, alpha, torch.zeros_like(alpha)
                )
                T_p = running_T[y0:y1, x0:x1]

                # レンダリング結果を更新
                rendered[:, y0:y1, x0:x1] += (
                    (T_p * alpha).unsqueeze(0) * colors_c[i].reshape(3, 1, 1)
                )
                running_T[y0:y1, x0:x1] = T_p * (1.0 - alpha)

            # 残り透過率分だけ背景色を加算
            rendered += running_T.unsqueeze(0) * bg_color_c.reshape(3, 1, 1)

        # px, py を backward で再利用するため保存
        ctx.save_for_backward(
            means2d_c, inv_cov2d_c, colors_c, opacities_c, sigma_max_c,
            px, py, bg_color_c
        )
        ctx.bboxes = (y0s, y1s, x0s, x1s)
        ctx.height = height
        ctx.width = width
        ctx.orig_device = device

        # レンダリング結果を元のデバイスに戻す
        return rendered.to(device)

    @staticmethod
    def backward(
        ctx: torch.autograd.Function, d_rendered: torch.Tensor
    ) -> tuple:
        """
        Parameters
        ----------
        ctx: torch.autograd.Function
            自動微分のコンテキスト
        d_rendered: torch.Tensor
            レンダリングされた画像の勾配

        Returns
        ----------
        d_means2d: torch.Tensor
            2D 平均位置の勾配
        d_inv_cov2d: torch.Tensor
            2D 共分散の逆行列の勾配
        d_colors: torch.Tensor
            色の勾配
        d_opacities: torch.Tensor
            不透明度の勾配
        d_sigma_max: torch.Tensor
            バウンディングボックス半径の勾配
        d_height: int
            画像の高さの勾配
        d_width: int
            画像の幅の勾配
        """
        # 保存されたテンソルと情報を取得
        (
            means2d, inv_cov2d, colors, opacities, _,
            px, py, bg_color
        ) = ctx.saved_tensors
        y0s, y1s, x0s, x1s = ctx.bboxes
        mu_np = means2d.numpy()
        height = ctx.height
        width = ctx.width
        orig_device = ctx.orig_device
        n = means2d.shape[0]

        # 勾配の初期化（CPU）
        d_means2d = torch.zeros(n, 2)
        d_inv_cov2d = torch.zeros(n, 2, 2)
        d_colors = torch.zeros(n, 3)
        d_opacities = torch.zeros(n)
        d_sigma_max = torch.zeros(n)

        # d_rendered を CPU に移動
        d_rendered_cpu = d_rendered.cpu()

        # 逆順に走査して勾配を計算するため、forward と同様のループを逆順で実装
        with torch.no_grad():
            # forward を再現して最終透過率を求める
            running_T = torch.ones(height, width)
            for i in range(n):
                y0, y1 = int(y0s[i]), int(y1s[i])
                x0, x1 = int(x0s[i]), int(x1s[i])
                if y0 >= y1 or x0 >= x1:
                    continue
                dx = px[y0:y1, x0:x1] - float(mu_np[i, 0])
                dy = py[y0:y1, x0:x1] - float(mu_np[i, 1])
                ic = inv_cov2d[i]
                maha = (
                    ic[0, 0] * dx * dx + (ic[0, 1] + ic[1, 0]) * dx * dy
                    + ic[1, 1] * dy * dy
                )
                alpha = (
                    opacities[i] * torch.exp(-0.5 * maha)
                ).clamp(max=0.99)
                alpha = torch.where(
                    alpha > 1.0 / 255.0, alpha, torch.zeros_like(alpha)
                )
                running_T[y0:y1, x0:x1] *= (1.0 - alpha)

            # accum_after を初期化
            accum_after = (
                running_T.unsqueeze(0) * bg_color.reshape(3, 1, 1)
            ).clone()

            # Gaussian ごとにバウンディングボックス内のピクセルを処理
            for i in range(n - 1, -1, -1):
                # バウンディングボックスの座標を整数に変換
                y0, y1 = int(y0s[i]), int(y1s[i])
                x0, x1 = int(x0s[i]), int(x1s[i])
                if y0 >= y1 or x0 >= x1:
                    continue

                # alpha を再計算
                dx = px[y0:y1, x0:x1] - float(mu_np[i, 0])
                dy = py[y0:y1, x0:x1] - float(mu_np[i, 1])
                ic = inv_cov2d[i]
                maha = (
                    ic[0, 0] * dx * dx + (ic[0, 1] + ic[1, 0]) * dx * dy
                    + ic[1, 1] * dy * dy
                )
                exp_term = torch.exp(-0.5 * maha)
                sigma_i = opacities[i]
                unclamped = sigma_i * exp_term

                # alpha 0.99 以上はほぼ完全に不透明で勾配が消えるため、0.99 以上は 0.99 にクランプして勾配を切る
                clamp_mask = (
                    (unclamped < 0.99) & (unclamped > 1.0 / 255.0)
                ).float()
                alpha = unclamped.clamp(max=0.99)
                alpha = torch.where(
                    alpha > 1.0 / 255.0, alpha, torch.zeros_like(alpha)
                )

                # 透過率 T_p を計算
                one_minus_alpha = (1.0 - alpha).clamp(min=1e-6)
                T_p = running_T[y0:y1, x0:x1] / one_minus_alpha

                # 勾配のパッチを取得
                d_patch = d_rendered_cpu[:, y0:y1, x0:x1]
                weight = T_p * alpha

                # 直接項のみの色の勾配
                d_colors[i] = (d_patch * weight.unsqueeze(0)).sum(
                    dim=(-1, -2)
                )

                # 直接項 + 間接項の不透明度の勾配
                accum_patch = accum_after[:, y0:y1, x0:x1]
                d_alpha = (
                    T_p * (d_patch * colors[i].reshape(3, 1, 1)).sum(dim=0)
                    - (1.0 / one_minus_alpha) * (d_patch * accum_patch).sum(
                        dim=0
                    )
                )

                # clamp_mask をかけて alpha が 0.99 以上の領域の勾配をゼロにする
                d_unc = d_alpha * clamp_mask
                d_opacities[i] = (d_unc * exp_term).sum()
                d_maha = d_unc * (-0.5 * sigma_i * exp_term)
                d_inv_cov2d[i, 0, 0] = (d_maha * dx * dx).sum()
                d_inv_cov2d[i, 0, 1] = (d_maha * dx * dy).sum()
                d_inv_cov2d[i, 1, 0] = (d_maha * dx * dy).sum()
                d_inv_cov2d[i, 1, 1] = (d_maha * dy * dy).sum()
                d_means2d[i, 0] = (
                    d_maha * (
                        -2.0 * dx * ic[0, 0] - dy * (ic[0, 1] + ic[1, 0])
                    )
                ).sum()
                d_means2d[i, 1] = (
                    d_maha * (
                        -2.0 * dy * ic[1, 1] - dx * (ic[0, 1] + ic[1, 0])
                    )
                ).sum()

                # accum_after を更新
                accum_after[:, y0:y1, x0:x1] += (
                    weight.unsqueeze(0) * colors[i].reshape(3, 1, 1)
                )

                # running_T を T_before に戻す
                running_T[y0:y1, x0:x1] = T_p

        # 勾配を元のデバイスに戻して返す
        return (
            d_means2d.to(orig_device), d_inv_cov2d.to(orig_device),
            d_colors.to(orig_device), d_opacities.to(orig_device),
            d_sigma_max.to(orig_device), None, None, None
        )


class GaussianModel(nn.Module):
    def __init__(self, points3D: dict, sh_degree: int = 3):
        # 親クラスのコンストラクタを呼び出す
        super(GaussianModel, self).__init__()

        # SH の次数と係数数を保存
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2

        # ADC 用の勾配蓄積バッファを初期化
        self.active_sh_degree = 0

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

        # DC 項を初期色から初期化, 高次項は 0
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
        cam_params: np.ndarray, width: int, height: int,
        bg_color: Optional[torch.Tensor] = None
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
        bg_color: Optional[torch.Tensor] = None
            背景色

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
        view_dirs = view_dirs / view_dirs.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)

        # SH 評価で色を取得
        colors = evaluate_sh(sh_coeffs, view_dirs, self.active_sh_degree)

        # ピクセル座標の計算
        x = means_cam[:, 0]
        y = means_cam[:, 1]
        z = means_cam[:, 2]
        means2d_x = fx * x / z + cx
        means2d_y = fy * y / z + cy
        means2d = torch.stack([means2d_x, means2d_y], dim=-1)

        # フラスタム内にある Gaussian のみを残す
        pad_x = width * 0.15
        pad_y = height * 0.15
        frustum_mask = (
            (means2d_x > -pad_x) & (means2d_x < width + pad_x)
            & (means2d_y > -pad_y) & (means2d_y < height + pad_y)
        )
        means_cam = means_cam[frustum_mask]
        cov3d = cov3d[frustum_mask]
        opacities = opacities[frustum_mask]
        colors = colors[frustum_mask]
        means2d = means2d[frustum_mask]

        # カメラ座標系での位置
        x = means_cam[:, 0]
        y = means_cam[:, 1]
        z = means_cam[:, 2]

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

        # フラスタム内の Gaussian のインデックスを取得してソート
        valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
        frustum_indices = valid_indices[frustum_mask]
        sorted_valid_indices = frustum_indices[sort_indices]

        # 2D 共分散の最大固有値から半径を計算して、画像範囲内にある Gaussian のみ残す
        a = cov2d[:, 0, 0]
        b = cov2d[:, 0, 1]
        d = cov2d[:, 1, 1]
        half_trace = (a + d) / 2
        half_diff = (a - d) / 2
        max_eig = half_trace + torch.sqrt(
            half_diff * half_diff + b * b
        )
        sigma_max = 3.0 * torch.sqrt(max_eig.clamp(min=0))
        visible = (
            (means2d[:, 0] + sigma_max > 0)
            & (means2d[:, 0] - sigma_max < width)
            & (means2d[:, 1] + sigma_max > 0)
            & (means2d[:, 1] - sigma_max < height)
        )
        means2d = means2d[visible]
        inv_cov2d = inv_cov2d[visible]
        colors = colors[visible]

        # (N, 1) → (N,) に変換して渡す
        opacities = opacities[visible, 0]
        sigma_max_vis = sigma_max[visible].detach()

        # 勾配を計算するため、means2d に retain_grad() を呼び出して保存
        if means2d.requires_grad:
            means2d.retain_grad()
        self._last_means2d = means2d
        self._last_visible_indices = sorted_valid_indices[visible]

        # 背景色が None の場合は黒にする
        if bg_color is None:
            bg_color = torch.zeros(3, device=device)

        # カスタムラスタライザでレンダリング
        rendered = GaussianRasterizer.apply(
            means2d, inv_cov2d, colors, opacities,
            sigma_max_vis, height, width, bg_color
        )
        assert isinstance(rendered, torch.Tensor)

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
        2D 位置勾配のノルムを蓄積する関数

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # forward で保存された means2d を参照
        if not hasattr(self, "_last_means2d"):
            return
        means2d = self._last_means2d
        if means2d.grad is None:
            return

        # 元 Gaussian インデックスへ蓄積
        grad_norm = means2d.grad.norm(dim=-1)
        idx = self._last_visible_indices
        self.grad_accum.index_add_(0, idx, grad_norm)
        self.grad_count.index_add_(0, idx, torch.ones_like(grad_norm))

    def densify_and_prune(
        self, grad_threshold: float = 0.0002, scale_threshold: float = 0.01,
        opacity_threshold: float = 0.005, max_gaussians: int = 100000,
        max_world_scale: Optional[float] = None
    ) -> dict:
        """
        Adaptive Density Control を実行する関数

        Parameters
        ----------
        grad_threshold: float = 0.0002
            勾配の閾値
        scale_threshold: float = 0.01
            split / clone を分ける scale 閾値
        opacity_threshold: float = 0.005
            不透明度の閾値
        max_gaussians: int = 100000
            Gaussian の最大数
        max_world_scale: Optional[float] = None
            world-space で許容する最大スケール

        Returns
        ----------
        info_data: dict
            optimizer state 再構築に必要な情報
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

        # 不透明度が閾値以下 / 巨大すぎる / 分割された Gaussian を削除
        opacity_vals = torch.sigmoid(self.opacities.data[:, 0])
        prune_mask = (opacity_vals < opacity_threshold) | split_mask
        if max_world_scale is not None:
            big_world_mask = max_scale > max_world_scale
            prune_mask = prune_mask | big_world_mask
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

        # clone / split の追加数を記録
        n_clone = clone_means.shape[0]
        n_split = split_means.shape[0]

        # Gaussian の最大数を超えた場合は不透明度の低い順に削除
        topk_indices = None
        if new_means.shape[0] > max_gaussians:
            new_opacity_vals = torch.sigmoid(new_opacities[:, 0])
            topk_indices = torch.topk(
                new_opacity_vals, max_gaussians
            ).indices
            new_means = new_means[topk_indices]
            new_sh = new_sh[topk_indices]
            new_opacities = new_opacities[topk_indices]
            new_scales = new_scales[topk_indices]
            new_rotations = new_rotations[topk_indices]

        # nn.Parameter として再設定
        self.means = nn.Parameter(new_means)
        self.sh_coeffs = nn.Parameter(new_sh)
        self.opacities = nn.Parameter(new_opacities)
        self.scales = nn.Parameter(new_scales)
        self.rotations = nn.Parameter(new_rotations)

        # 勾配バッファをリセット
        self.setup_adc()

        # optimizer state 再構築に必要な情報を返す
        info_data = {
            "keep_mask": keep_mask, "clone_mask": clone_mask,
            "split_mask": split_mask, "n_clone": n_clone,
            "n_split": n_split, "topk": topk_indices
        }

        return info_data

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


def evaluate_sh(
    sh_coeffs: torch.Tensor, dirs: torch.Tensor, active_degree: int = 3
) -> torch.Tensor:
    """
    球面調和関数を評価する関数

    Parameters
    ----------
    sh_coeffs: torch.Tensor
        SH 係数
    dirs: torch.Tensor
        正規化された視線方向ベクトル
    active_degree: int = 3
        評価に使う SH の最高次数

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

    # 評価する最高次数
    max_degree = min(active_degree, int(sh_coeffs.shape[1] ** 0.5) - 1)

    # Degree 0
    result = c0 * sh_coeffs[:, 0]

    # Degree 1
    if max_degree >= 1 and sh_coeffs.shape[1] > 1:
        result = result + c1 * (
            -y * sh_coeffs[:, 1] + z * sh_coeffs[:, 2]
            - x * sh_coeffs[:, 3]
        )

    # Degree 2
    if max_degree >= 2 and sh_coeffs.shape[1] > 4:
        result = result + (
            c2[0] * xy * sh_coeffs[:, 4] + c2[1] * yz * sh_coeffs[:, 5]
            + c2[2] * (2.0 * zz - xx - yy) * sh_coeffs[:, 6]
            + c2[3] * xz * sh_coeffs[:, 7]
            + c2[4] * (xx - yy) * sh_coeffs[:, 8]
        )

    # Degree 3
    if max_degree >= 3 and sh_coeffs.shape[1] > 9:
        result = result + (
            c3[0] * y * (3.0 * xx - yy) * sh_coeffs[:, 9]
            + c3[1] * xy * z * sh_coeffs[:, 10]
            + c3[2] * y * (4.0 * zz - xx - yy) * sh_coeffs[:, 11]
            + c3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coeffs[:, 12]
            + c3[4] * x * (4.0 * zz - xx - yy) * sh_coeffs[:, 13]
            + c3[5] * z * (xx - yy) * sh_coeffs[:, 14]
            + c3[6] * x * (xx - 3.0 * yy) * sh_coeffs[:, 15]
        )

    # 結果を色として返す
    colors = result

    return colors
