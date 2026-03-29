# src/utils

ドメインモデルに依存しない汎用ユーティリティ関数群。

## ファイル一覧

| ファイル | 概要 |
|---|---|
| `cfg_diff.py` | config ファイルの読み込みと差分のログ出力 |
| `loss.py` | 損失関数の定義。L1、SSIM、およびそれらの加重和 |
| `preprocessing.py` | COLMAP ファイルのパース（txt / bin 両対応）、bin→txt 変換、画像読み込み、初期色推定、画像分離 |
| `result.py` | 学習曲線のプロット、評価メトリクスの計算、PLY エクスポート、レンダリング画像の保存 |
| `trainer.py` | Optimizer の選択と学習ループの実装 |
