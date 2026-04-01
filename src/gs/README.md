# src/gs

3D Gaussian Splatting のモデル定義と学習スクリプト。

参考論文: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.14737) (Kerbl et al., 2023)

## ファイル一覧

| ファイル | 概要 |
|---|---|
| `model.py` | `GaussianRasterizer`（カスタム autograd）と `GaussianModel` クラスの定義 |
| `train.py` | 学習のエントリポイント。データ読み込みから学習・評価・エクスポートまでを実行する |

## rasterizer/

CUDA タイルベースラスタライザの実装。CUDA 環境（Google Colab 等）では Python ループの代わりに GPU カーネルでレンダリングを行い、大幅な高速化を実現する。CUDA が利用できない環境では `model.py` の `GaussianRasterizer`（Python 実装）に自動フォールバックする。

### 背景

Python for-loop によるラスタライザは、Gaussian 1 つずつを逐次処理するため GPU の並列性を活かせない。alpha compositing の逐次依存（T_i = Π_{j<i}(1 - α_j)）があるため、PyTorch の標準演算では高速化が困難である。一方、**ピクセル間は独立**であることを利用すると、各ピクセルを 1 GPU スレッドに割り当てて並列処理できる。これが CUDA カーネルの基本方針である。

### アーキテクチャ

```
GaussianModel.forward()
    │
    ├─ 投影・SH 評価・深度ソート（PyTorch、デバイス上）
    │
    └─ ラスタライザ
         ├─ CUDA 環境: CUDARasterizer（rasterizer/rasterizer.py）
         │    ├─ tile_preprocess(): Gaussian→タイル割り当て + 深度ソート
         │    ├─ forward.cu: タイルベース forward カーネル
         │    └─ backward.cu: 逆順再生 backward カーネル
         │
         └─ CPU 環境: GaussianRasterizer（model.py）
              └─ Python for-loop + numpy bbox
```

### タイルベースラスタライザの仕組み

画像を 16×16 ピクセルのタイルに分割し、各タイルに重なる Gaussian のリストを事前に構築する。

1. **前処理**（`tile_preprocess.py`、PyTorch）
   - 各 Gaussian のバウンディングボックス（3σ）からタイル座標を計算
   - (tile_id, depth, gaussian_id) のペアを生成し、tile_id × depth でソート
   - `tile_ranges` テンソルで各タイルの Gaussian リスト範囲を記録

2. **Forward カーネル**（`forward.cu`）
   - 1 CUDA ブロック = 1 タイル（256 スレッド = 16×16 ピクセル）
   - 各スレッドが担当ピクセルの alpha compositing を実行
   - shared memory に Gaussian データをバッチロードしてグローバルメモリアクセスを削減
   - `T < 1e-4` で早期終了（完全に不透明になったピクセルはスキップ）
   - `final_T`（最終透過率）と `n_contrib`（寄与 Gaussian 数）を保存

3. **Backward カーネル**（`backward.cu`）
   - Forward と同じタイル構造を**逆順**に走査
   - `final_T` から `T_i = T / (1 - α_i)` で各ステップの透過率を復元
   - `atomicAdd` でグローバルメモリ上の勾配テンソルに蓄積
   - 解析的勾配: `d_colors`, `d_opacities`, `d_inv_cov2d`, `d_means2d`

### ファイル一覧

| ファイル | 概要 |
|---|---|
| `rasterizer/__init__.py` | CUDA 拡張の JIT コンパイルと `is_cuda_available()` |
| `rasterizer/rasterizer.py` | `CUDARasterizer(torch.autograd.Function)` ラッパー |
| `rasterizer/tile_preprocess.py` | Gaussian→タイル割り当て + ソート（PyTorch） |
| `rasterizer/cuda/forward.cu` | CUDA forward カーネル |
| `rasterizer/cuda/backward.cu` | CUDA backward カーネル |
| `rasterizer/cuda/bindings.cpp` | pybind11 バインディング |

### 期待される性能

| 環境 | ラスタライザ | 速度（目安） |
|---|---|---|
| MacBook CPU | Python for-loop | ~12 秒/iter |
| Colab T4 GPU + CUDA カーネル | タイルベース並列 | ~0.1〜0.3 秒/iter |

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
