# outputs

学習結果の出力先ディレクトリ。

## ディレクトリ構成

実行時刻に基づいてサブディレクトリが自動生成される。

```
outputs/
└── gs/
    └── YYYY-MM-DD/
        └── HH-MM-SS/
            ├── gs.log               # 実験ログ
            ├── training_curve.png   # 学習曲線
            ├── final_model.pth      # 学習済みモデル
            ├── gaussians.ply        # Gaussian の位置と色を PLY 形式で出力
            └── renders/             # 評価用画像のレンダリング結果
                ├── frame_XXXX.png   # 左が正解画像、右がレンダリング画像
                └── ...
```

## Git 管理

- **追跡対象外** — 実験のたびに生成されるため、Git では管理しない
