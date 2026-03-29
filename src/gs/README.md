# src/gs

3D Gaussian Splatting のモデル定義と学習スクリプト。

参考論文: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.14737) (Kerbl et al., 2023)

## ファイル一覧

| ファイル | 概要 |
|---|---|
| `model.py` | `GaussianRasterizer`（カスタム autograd）と `GaussianModel` クラスの定義 |
| `train.py` | 学習のエントリポイント。データ読み込みから学習・評価・エクスポートまでを実行する |

## model.py

### GaussianRasterizer

`torch.autograd.Function` によるカスタム backward のラスタライザ。

- **forward**: Gaussian ごとのバウンディングボックス（3σ 半径）内のピクセルのみを処理することで、メモリ使用量を O(N × パッチサイズ²) に抑える
- **backward**: forward を再実行して `running_T` を復元しながら解析的に勾配を計算する。中間値を保持しないためメモリ効率が高い
- **CPU 強制実行**: Python for-loop ベースのため、MPS / CUDA 環境ではカーネル起動オーバーヘッドが大きくなる。そのためデバイスに関わらずラスタライザは常に CPU で実行し、結果を元のデバイスに戻す

### GaussianModel

Gaussian パラメータ（位置・SH 係数・不透明度・スケール・回転）を `nn.Parameter` として保持し、以下の機能を実装する。

| メソッド / プロパティ | 概要 |
|---|---|
| `forward()` | カメラパラメータを受け取り、投影・SH 評価・可視フィルタリング・深度ソートを経てレンダリング画像を返す |
| `setup_adc()` | ADC 用の勾配蓄積バッファ（`grad_accum`, `grad_count`）を初期化する |
| `accumulate_gradients()` | `means.grad` のノルムを蓄積する。ADC の clone / split 判定に使用する |
| `densify_and_prune()` | 勾配ノルムが閾値を超えた Gaussian を clone または split し、不透明度が低い Gaussian を prune する |
| `reset_opacities()` | 全 Gaussian の不透明度を指定値にリセットする（定期的な opacity reset に使用） |
| `num_gaussians` | 現在の Gaussian 数を返すプロパティ |

## train.py

- COLMAP bin 形式（`cameras.bin` / `images.bin` / `points3D.bin`）を検出した場合、学習開始時に自動的に txt 形式へ変換する
- `max_initial_gaussians` を指定することで初期点群をランダムサンプリングして上限を設ける
- `resolution_scale` で解像度をスケールダウンしてレンダリング解像度を調整する
- Adam に対してパラメータ種別ごとに異なる学習率を設定する（`learning_rates` セクション）
- 学習後に training_curve / モデル重み / PLY / レンダリング画像を `outputs/` に自動保存する
- デバイス `"auto"` 設定では CUDA → CPU の優先順位で選択する
