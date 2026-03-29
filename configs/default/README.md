# configs/default

ハイパーパラメータの **デフォルト値（基準値）** を管理するディレクトリ。

## 概要

- 学習実験の基準となるハイパーパラメータを JSON ファイルとして保持する
- このディレクトリ内のファイルは実験ごとに変更しない
- 基準値自体を見直す重要な判断があった場合にのみ更新する

## Git 管理

- **追跡対象** — デフォルト値の変更は実験の基準を変えるイベントであるため、変更履歴を Git で管理する

## ファイル一覧

| ファイル | 概要 |
|---|---|
| `gs.json` | 3D Gaussian Splatting 学習のデフォルトハイパーパラメータ |

## gs.json の主要パラメータ

| キー | デフォルト値 | 概要 |
|---|---|---|
| `sh_degree` | `3` | SH（球面調和関数）の次数。PLY ファイルの `f_rest_*` 数に直結するため、標準 3DGS ビューアとの互換性を保つには `3` を維持する |
| `data.resolution_scale` | `2` | レンダリング解像度のスケールダウン倍率。`2` なら縦横半分の解像度でレンダリングする |
| `data.max_initial_gaussians` | `20000` | 初期点群の上限数。超えた場合はランダムサンプリングで削減する |
| `training.iterations` | `7000` | 学習イテレーション数 |
| `training.learning_rates` | 種別ごとに設定 | Adam 使用時のパラメータ種別ごとの学習率。`means`: 1.6e-4、`sh_coeffs`: 2.5e-3、`opacities`: 5e-2、`scales`: 5e-3、`rotations`: 1e-3 |
| `training.adc.start_iteration` | `500` | ADC（Adaptive Density Control）を開始するイテレーション |
| `training.adc.interval` | `100` | ADC を実行する間隔。実行のたびに Optimizer が再構築されるため、小さくすると収束が不安定になる |
| `training.adc.stop_iteration` | `15000` | ADC を停止するイテレーション |
| `training.adc.opacity_reset_interval` | `3000` | 全 Gaussian の不透明度をリセットする間隔 |
