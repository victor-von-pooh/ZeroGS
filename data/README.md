# data

3DGS の学習に使用するデータを配置するディレクトリ。

## 必要なデータ

[GNN_Colmap](https://github.com/victor-von-pooh/GNN_Colmap) の出力をそのまま配置する。

```
data/
├── images/          # 入力画像群
└── sparse/0/        # GNN_Colmap の出力（COLMAP 互換フォーマット）
    ├── cameras.txt  #   カメラ内部パラメータ（SIMPLE_RADIAL: f, cx, cy, k）
    ├── images.txt   #   カメラ外部パラメータ（QW, QX, QY, QZ, TX, TY, TZ）
    └── points3D.txt #   3D 点群（X, Y, Z, R, G, B=128, ERROR=0, TRACK=空）
```

txt 形式の代わりに COLMAP の bin 形式（`cameras.bin` / `images.bin` / `points3D.bin`）を配置した場合も動作する。学習開始時に自動的に同ディレクトリへ txt 形式へ変換される。

## データの用意

GNN_Colmap の学習・推論を実行し、出力された `sparse/0/` と学習に使用した `images/` を配置する。

## Git 管理

- **追跡対象外** — データファイルは容量が大きいため、Git では管理しない
