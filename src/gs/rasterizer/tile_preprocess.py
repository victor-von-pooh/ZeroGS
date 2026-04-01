import torch


def tile_preprocess(
    means2d: torch.Tensor,
    depths: torch.Tensor,
    sigma_max: torch.Tensor,
    height: int,
    width: int,
    tile_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    各 Gaussian がどのタイルに属するかを計算し、タイル×深度でソートする。
    CUDA カーネルの forward / backward に渡す前処理。

    Parameters
    ----------
    means2d : torch.Tensor
        2D 投影位置 (N, 2)。device は問わない
    depths : torch.Tensor
        カメラ座標系での z 値 (N,)
    sigma_max : torch.Tensor
        バウンディングボックス半径 (N,)
    height : int
        画像の高さ（ピクセル）
    width : int
        画像の幅（ピクセル）
    tile_size : int
        タイルの一辺のピクセル数（デフォルト 16）

    Returns
    -------
    gaussian_ids : torch.Tensor (int32)
        タイル×深度でソートされた Gaussian ID。shape: (total_pairs,)
    tile_ranges : torch.Tensor (int32)
        各タイルの gaussian_ids 上の [start, end) 範囲。shape: (n_tiles + 1,)
        tile_ranges[t] から tile_ranges[t+1] までが タイル t の Gaussian。
    """
    device = means2d.device

    n_tiles_x = (width  + tile_size - 1) // tile_size
    n_tiles_y = (height + tile_size - 1) // tile_size
    n_tiles   = n_tiles_x * n_tiles_y

    # 各 Gaussian が重なるタイルの範囲（タイル座標）
    tx0 = ((means2d[:, 0] - sigma_max) / tile_size).floor().long().clamp(0, n_tiles_x)
    tx1 = ((means2d[:, 0] + sigma_max) / tile_size).ceil().long().clamp(0, n_tiles_x)
    ty0 = ((means2d[:, 1] - sigma_max) / tile_size).floor().long().clamp(0, n_tiles_y)
    ty1 = ((means2d[:, 1] + sigma_max) / tile_size).ceil().long().clamp(0, n_tiles_y)

    # 各 Gaussian が重なるタイル数
    n_tiles_per = ((tx1 - tx0) * (ty1 - ty0)).clamp(min=0)  # (N,)
    total_pairs = int(n_tiles_per.sum().item())

    if total_pairs == 0:
        return (
            torch.zeros(0, dtype=torch.int32, device=device),
            torch.zeros(n_tiles + 1, dtype=torch.int32, device=device),
        )

    # (tile_id, depth, gaussian_id) のペアを列挙
    # 各 Gaussian が複数タイルにまたがる場合を repeat_interleave で展開する
    #
    # ---- tile_id の列挙 ----
    # N 個の Gaussian それぞれについて [ty0:ty1) × [tx0:tx1) のタイルを展開
    gaussian_idx = torch.repeat_interleave(
        torch.arange(len(means2d), device=device), n_tiles_per
    )  # (total_pairs,)

    # 各 Gaussian に対してタイル座標のオフセット (dty, dtx) を列挙
    offsets = [
        (ty * n_tiles_x + tx)
        for i in range(len(means2d))
        for ty in range(int(ty0[i].item()), int(ty1[i].item()))
        for tx in range(int(tx0[i].item()), int(tx1[i].item()))
    ]
    # 上記は Python ループだが前処理なので許容（学習ループの外）
    tile_ids = torch.tensor(offsets, dtype=torch.int64, device=device)  # (total_pairs,)

    # depth を float32 のビットパターンとして uint32 に変換（ソートキーに使用）
    # 正の depth を仮定（カメラ前方のみ）
    depth_bits = depths[gaussian_idx].view(torch.int32).to(torch.int64) & 0xFFFFFFFF

    # ソートキー = tile_id * 2^32 + depth_bits（タイル優先、同タイル内は深度順）
    sort_keys = tile_ids * (2 ** 32) + depth_bits  # (total_pairs,)

    _, order = sort_keys.sort()
    gaussian_ids = gaussian_idx[order].to(torch.int32)   # (total_pairs,)
    tile_ids_sorted = tile_ids[order]                     # (total_pairs,)

    # tile_ranges: 各タイルの gaussian_ids 上の開始インデックスを計算
    # bincount で各タイルに属するペア数を集計
    counts = torch.bincount(tile_ids_sorted.to(torch.int32), minlength=n_tiles)  # (n_tiles,)
    tile_ranges = torch.zeros(n_tiles + 1, dtype=torch.int32, device=device)
    tile_ranges[1:] = counts.cumsum(0).to(torch.int32)

    return gaussian_ids, tile_ranges
